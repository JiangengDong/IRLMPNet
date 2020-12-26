""" Learn a policy using TD3 for the reach task"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gym
from kinodyn_envs import DifferentialDriveFreeEnv, DifferentialDrive1OrderFreeEnv

from typing import Dict, Tuple
from tqdm import tqdm
import glob
import copy
import imageio


def weighSync(target_model: torch.nn.Module, source_model: torch.nn.Module, tau: float = 0.001) -> None:
    tau_2 = 1 - tau
    for (source_param, target_param) in zip(source_model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * source_param.data + tau_2 * target_param.data)


class Replay:
    def __init__(self, buffer_size: int, init_length: int, state_dim: int, action_dim: int, env: gym.Env):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        self.states = np.zeros(shape=(buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(shape=(buffer_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros(shape=(buffer_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros(shape=(buffer_size,), dtype=np.float32)
        self.dones = np.zeros(shape=(buffer_size,), dtype=np.float32)

        # circular queue
        self.size = 0
        self.buffer_size = buffer_size
        self.next_sample_index = 0

        # running average
        self.count = 0
        self.cum_sum = 0.0
        self.cum_sum2 = 0.0
        self.reward_mean = 0.0
        self.reward_std = 0.0

        self.random_init(env, init_length)

    def buffer_add(self, exp: Dict) -> None:
        """
        A function to add a dictionary to the buffer
        param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """
        if exp["states"].ndim > 1:
            raise RuntimeError("Please feed one entry at a time")

        index = self.next_sample_index
        self.states[index] = exp["states"]
        self.actions[index] = exp["actions"]
        self.rewards[index] = exp["rewards"]
        self.next_states[index] = exp["next_states"]
        self.dones[index] = exp["dones"]

        self.next_sample_index = (index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

        # update running average
        self.count += 1
        self.cum_sum += exp["rewards"]
        self.cum_sum2 += exp["rewards"]**2
        if self.count > 10:
            self.reward_mean = self.cum_sum / self.count 
            self.reward_std = np.sqrt(self.cum_sum2 / self.count - self.reward_mean**2)

    def buffer_sample(self, n: int) -> Dict:
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        indices = np.random.choice(self.size, n)
        return {"states": self.states[indices],
                "actions": self.actions[indices],
                "rewards": (self.rewards[indices] - self.reward_mean) / self.reward_std, # normalized reward
                "next_states": self.next_states[indices],
                "dones": self.dones[indices]}

    def random_init(self, env, init_length):
        state = env.reset()
        for _ in range(init_length):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            self.buffer_add({
                "states": state,
                "actions": action,
                "next_states": next_state,
                "rewards": reward,
                "dones": done
            })

            if done:
                state = env.reset()
            else:
                state = next_state


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()

        hidden_dim_1 = 512
        hidden_dim_2 = 512

        self.fc1 = nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, action_dim)

        self.fc1.weight.data.uniform_(-1 / np.sqrt(state_dim), 1 / np.sqrt(state_dim))
        self.fc2.weight.data.uniform_(-1 / np.sqrt(hidden_dim_1), 1 / np.sqrt(hidden_dim_1))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass
        param: state: The state of the environment
        """
        x = state
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x) # TODO: remove this layer

        return x

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()

        hidden_dim = 512
        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the critic
        """
        x = torch.cat([state, action], dim=-1)
        return self.net.forward(x)


class TD3:
    def __init__(
            self,
            env: gym.Env,
            action_dim: int,
            state_dim: int,
            critic_lr: float = 3e-4,
            actor_lr: float = 3e-4,
            gamma: float = 0.99,
            batch_size: int = 100,
    ):
        """
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.evaluate_env = copy.deepcopy(env)

        # Create a actor and actor_target
        self.actor = Actor(state_dim, action_dim).cuda()
        self.actor_target = Actor(state_dim=state_dim, action_dim=action_dim).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad_(False)

        # Create a critic and critic_target object
        self.critic1 = Critic(state_dim, action_dim).cuda()
        self.critic1_target = Critic(state_dim=state_dim, action_dim=action_dim).cuda()
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_target.requires_grad_(False)
        self.critic2 = Critic(state_dim, action_dim).cuda()
        self.critic2_target = Critic(state_dim=state_dim, action_dim=action_dim).cuda()
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_target.requires_grad_(False)

        # Define the optimizer for the actor
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # Define the optimizer for the critic
        self.optimizer_critic = torch.optim.Adam(list(self.critic1.parameters())+list(self.critic2.parameters()), lr=critic_lr)

        # define a replay buffer
        self.ReplayBuffer = Replay(buffer_size=100000, init_length=1000, state_dim=state_dim, action_dim=action_dim, env=env)

    def update_target_networks(self) -> None:
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic1_target, self.critic1)
        weighSync(self.critic2_target, self.critic2)

    def update_network(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A function to update the function just once
        """
        # sample from replay buffer and unpack
        batch = self.ReplayBuffer.buffer_sample(self.batch_size)
        states = torch.from_numpy(batch["states"]).cuda()
        actions = torch.from_numpy(batch["actions"]).cuda()
        next_states = torch.from_numpy(batch["next_states"]).cuda()
        rewards = torch.from_numpy(batch["rewards"]).cuda()
        dones = torch.from_numpy(batch["dones"]).cuda()

        # train critic
        next_actions = self.actor_target.forward(next_states)
        epsilon = torch.randn_like(next_actions) * 0.1
        epsilon = torch.clamp(epsilon, -0.2, 0.2)
        next_actions = torch.clamp(next_actions + epsilon, -1.0, 1.0)
        Q1_next = self.critic1_target.forward(next_states, next_actions).squeeze()
        Q2_next = self.critic2_target.forward(next_states, next_actions).squeeze()
        Q_next = torch.min(Q1_next, Q2_next)
        Q_backup = rewards + self.gamma * (1 - dones) * Q_next
        Q1_predict = self.critic1.forward(states, actions).squeeze()
        Q2_predict = self.critic2.forward(states, actions).squeeze()
        critic_loss = F.mse_loss(Q1_predict, Q_backup) + F.mse_loss(Q2_predict, Q_backup)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # train actor
        actions = self.actor.forward(states)
        actor_loss = - self.critic1.forward(states, actions).squeeze().mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return actor_loss.detach().cpu(), critic_loss.detach().cpu()

    def sample_action(self, state: np.ndarray, stochastic=True) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state.astype(np.float32)).cuda()
            action_tensor = self.actor.forward(state_tensor)
            action_mean = action_tensor.cpu().numpy()
        if stochastic:
            noise = np.random.randn(2) * 0.33
            action = np.clip(action_mean + noise, -1, 1)
        else:
            action = action_mean
        return action.astype(np.float32)

    def train(self, num_steps: int, log_dir: str) -> nn.Module:
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        # bring to local scope to increase speed
        env = self.env
        replay_buffer = self.ReplayBuffer

        # add logger and remove previous curve
        for filename in glob.glob(os.path.join(log_dir, "events.*")):
            os.remove(filename)
        writer = SummaryWriter(log_dir)

        state = env.reset()
        for iter in tqdm(range(num_steps)):
            action = self.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.buffer_add({
                "states": state,
                "actions": action,
                "next_states": next_state,
                "rewards": reward,
                "dones": done
            })
            if done:
                state = env.reset()
            else:
                state = next_state

            actor_loss, critic_loss = self.update_network()
            self.update_target_networks()

            writer.add_scalar("actor_loss", actor_loss, iter)
            writer.add_scalar("critic_loss", critic_loss, iter)

            if iter % 500 == 0:
                writer.add_scalar("return", self.evaluate(), iter)

        return self.actor_target

    def evaluate(self) -> float:
        env = self.evaluate_env
        T = 150
        returns = np.zeros((5, ))
        rewards = np.zeros((T,))
        for k in range(5):
            t = 0
            state = env.reset()
            for t in range(T):
                action = self.sample_action(state, False)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                rewards[t] = reward
                if done:
                    break

            gamma = self.gamma
            for i in reversed(range(t)):
                rewards[i] += gamma * rewards[i + 1]
            returns[k] = rewards[0]

        return np.mean(returns)


def generate_movie(env: gym.Env, policy: nn.Module, gif_path: str):
    env.reset()

    imgs = []
    with torch.no_grad():
        state = torch.from_numpy(env.reset().astype(np.float32)).cuda()
        for t in range(150):
            action_tensor = policy.forward(state)
            action = action_tensor.cpu().numpy()
            state, reward, done, info = env.step(action)
            state = torch.from_numpy(state.astype(np.float32)).cuda()
            imgs.append(env.render("rgb_array"))
            if done:
                break
    imageio.mimsave(gif_path, imgs, fps=10, subrectangles=True)


if __name__ == "__main__":
    # Define the environment
    env = DifferentialDrive1OrderFreeEnv()
    task = "train"

    if task == "train":
        TD3_object = TD3(
            env=env,
            state_dim=6,
            action_dim=2,
            critic_lr=1e-3,
            actor_lr=1e-3,
            gamma=0.99,
            batch_size=100,
        )
        # Train the policy
        log_dir = os.path.join("./data", "log", "rl", "car_free-TD3")
        model_filename = os.path.join("./data", "pytorch_model", "rl", "car_free-TD3.pt")
        os.makedirs(os.path.split(model_filename)[0], exist_ok=True)
        policy = TD3_object.train(200000, log_dir)
        torch.save(policy, model_filename)

    policy = torch.load(model_filename).cuda()
    img_filename = os.path.join("data", "img", "differential_free-TD3-episode{}.gif")
    for i in tqdm(range(10)):
        generate_movie(env, policy, img_filename.format(i))
