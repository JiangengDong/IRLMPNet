import torch
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import List
from models import TransitionModel, ObservationEncoder
from envs import Car1OrderEnv
from tqdm import tqdm, trange

from planner import MLPPlanner, MLPPolicy


def load_data():
    starts = np.load("data/car1order/start_goal/train_starts.npy")
    goals = np.load("data/car1order/start_goal/train_goals.npy")
    mask = np.zeros((5000, ), dtype=np.bool8)
    trajs = []
    with np.load("data/car1order/train_traj/train_traj.npz") as data:
        for i in trange(5000):
            if "traj%d" % i in data:
                mask[i] = True
                trajs.append(data["traj%d" % i])
    print("number of valid trajs: ", len(trajs))

    return starts[mask], goals[mask], trajs


def convert_to_hidden(starts: np.ndarray,
                      goals: np.ndarray,
                      trajs: List[np.ndarray]):
    env = Car1OrderEnv(2000, 2)
    transition_model = torch.jit.load("data/car1order/rl_result/test2/torchscript/transition_model.pth").cuda()
    encoder = torch.jit.load("data/car1order/rl_result/test2/torchscript/observation_encoder.pth").cuda()
    belief_size = 256
    state_size = 32

    beliefs = []
    states = []
    actions_tensor = []

    num_trajs = len(trajs)

    for traj_idx in trange(num_trajs, smoothing=0.01):
        start = starts[traj_idx].astype(np.float32)
        goal = goals[traj_idx].astype(np.float32)
        traj = trajs[traj_idx].astype(np.float32)
        # reset
        env.reset()
        normalized_start = (start - env.state_center) / env.state_range
        normalized_goal = (goal - env.state_center) / env.state_range
        env.state = start
        env.goal = goal
        env.normalized_state = normalized_start
        env.normalized_goal = normalized_goal
        # prepare all actions
        actions = (traj[:-1, 3:5] - env.control_center) / env.control_range
        T = actions.shape[0]

        # correct belief and state with initial observation
        belief = torch.zeros(1, belief_size, device="cuda")
        posterior_state = torch.zeros(1, state_size, device="cuda")
        action = torch.zeros(1, 2, device="cuda")
        observation = env.get_ob().cuda()
        belief, _, _, _, posterior_state, _, _ = transition_model.forward(
            posterior_state,
            action.unsqueeze(dim=0),
            belief,
            encoder(observation).unsqueeze(dim=0))
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)

        for t in trange(T, leave=False):
            observation, _, _, _ = env.step(actions[t])
            observation = observation.cuda()
            action = torch.from_numpy(actions[t]).cuda().reshape((1, 2))

            beliefs.append(belief.detach())
            states.append(posterior_state.detach())
            actions_tensor.append(action.detach())

            belief, _, _, _, posterior_state, _, _ = transition_model.forward(
                posterior_state,
                action.unsqueeze(dim=0),
                belief,
                encoder(observation).unsqueeze(dim=0))
            belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)

    return torch.cat(beliefs, dim=0), torch.cat(states, dim=0), torch.cat(actions_tensor, dim=0)


if os.path.exists("data/car1order/rl_result/test2/MLP/dataset.pt"):
    temp = torch.load("data/car1order/rl_result/test2/MLP/dataset.pt")
    beliefs = temp["beliefs"]
    states = temp["states"]
    actions = temp["actions"]
else:
    beliefs, states, actions = convert_to_hidden(*load_data())
    torch.save({"beliefs": beliefs, "states": states, "actions": actions}, "data/car1order/rl_result/test2/MLP/dataset.pt")

mlp = MLPPolicy(256, 32, 2)
optimizer = torch.optim.Adam(mlp.parameters())
writer = SummaryWriter("data/car1order/rl_result/test2/MLP/tensorboard")
dataset = TensorDataset(beliefs, states, actions)
dataloader = DataLoader(dataset, 4, True)

i = 0
for _ in range(1):
    for (beliefs, states, actions) in tqdm(dataloader):
        loss = F.mse_loss(mlp(beliefs, states), actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            writer.add_scalar("MLP_loss", loss, i)
            torch.save(mlp.state_dict(), "data/car1order/rl_result/test2/checkpoint/MLP.pth")
        i += 1


torch.save(mlp.state_dict(), "data/car1order/rl_result/test2/checkpoint/MLP.pth")
