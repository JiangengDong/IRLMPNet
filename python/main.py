"""The entrance of the PlaNet algorithm.

Notations:
* h: belief, the deterministic part of the latent state
* s: state, the stochastic part of the latent state
* a: action, calculated using MPC planner
* o: environment observation
* x: environment internal state (unknown)
* r: reward
* z: done
"""
import argparse
import logging
import os
from math import inf
import json

import numpy as np
import torch
from typing import Dict, Tuple, List

from torch import nn, optim, jit
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from envs import Car1OrderEnv, Env
from memory import ExperienceReplay
from models import (ObservationEncoder, ObservationDecoder, RewardModel,
                    TransitionModel, bottle)
from planner import MPCPlanner
from visual import visualize_global_map, visualize_local_map


def get_args() -> argparse.Namespace:
    # Hyperparameters
    parser = argparse.ArgumentParser(description='PlaNet')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')

    # environment configs
    parser.add_argument('--max-episode-length', type=int, default=500, metavar='T', help='Max episode length')
    parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--obstacle-index', type=int, default=0, help='Obstacle index')

    # replay buffer configs
    parser.add_argument('--experience-size', type=int, default=10000, metavar='D', help='Experience replay size')

    # network configs
    parser.add_argument('--activation-function', type=str, default='relu', choices=dir(F), help='Model activation function')
    parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='H', help='Hidden size')
    parser.add_argument('--belief-size', type=int, default=256, metavar='H', help='Belief/hidden size')
    parser.add_argument('--state-size', type=int, default=32, metavar='Z', help='State/latent size')

    # MPC planner configs
    parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
    parser.add_argument('--planning-horizon', type=int, default=12, metavar='H', help='Planning horizon distance')
    parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
    parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
    parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')

    # hyperparameters
    parser.add_argument('--episodes', type=int, default=200, metavar='E', help='Total number of episodes')
    parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
    parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
    parser.add_argument('--batch-size', type=int, default=10, metavar='B', help='Batch size')
    parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
    parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
    parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
    parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1',
                        help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
    parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
    parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate')
    parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS',
                        help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)')
    parser.add_argument('--adam-epsilon', type=float, default=1e-4, metavar='ε', help='Adam optimiser epsilon value')
    parser.add_argument('--grad-clip-norm', type=float, default=1000, metavar='C', help='Gradient clipping norm')

    # test configs
    parser.add_argument('--test', action='store_true', help='Test only')
    parser.add_argument('--test-interval', type=int, default=25,  metavar='I', help='Test interval (episodes)')
    parser.add_argument('--test-episodes', type=int, default=1, metavar='E', help='Number of test episodes')

    # checkpoint configs
    parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
    parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')

    # continue training
    parser.add_argument('--result-dir', type=str, help='Default value is set according to ID. Override with this option.')
    parser.add_argument('--checkpoint-dir', type=str, help='Default value is set according to ID. Override with this option.')
    parser.add_argument('--checkpoint-path', type=str, help='Default value is set according to ID. Override with this option.')
    parser.add_argument('--tensorboard-dir', type=str, help='Default value is set according to ID. Override with this option.')
    parser.add_argument('--torchscript-dir', type=str, help='Default value is set according to ID. Override with this option.')
    parser.add_argument("--video-dir", type=str, help='Default value is set according to ID. Override with this option.')

    args = parser.parse_args()

    return args


def postprocess_args(args: argparse.Namespace) -> argparse.Namespace:
    # check validity of args and add additional args
    args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)

    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    args.result_dir = args.result_dir or os.path.join('data/car1order/rl_result', args.id)
    args.checkpoint_dir = args.checkpoint_dir or os.path.join(args.result_dir, "checkpoint")
    args.tensorboard_dir = args.tensorboard_dir or os.path.join(args.result_dir, "tensorboard")
    args.torchscript_dir = args.torchscript_dir or os.path.join(args.result_dir, "torchscript")
    args.video_dir = args.video_dir or os.path.join(args.result_dir, "video")

    record_path = os.path.join(args.checkpoint_dir, "checkpoint")
    if os.path.exists(record_path):
        with open(record_path) as f:
            args.checkpoint_path = args.checkpoint_path or os.path.join(args.checkpoint_dir, f.readline().rstrip('\n'))

    print('Options')
    for k, v in vars(args).items():
        print('\t' + k + ': ' + str(v))

    return args


def save_args(args: argparse.Namespace):
    args_dict = args.__dict__.copy()
    args_dict.pop("device")
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)


def setup_workdir(args: argparse.Namespace):
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.torchscript_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    for filename in os.listdir(args.tensorboard_dir):
        os.remove(os.path.join(args.tensorboard_dir, filename))


def setup_seed(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == torch.device('cuda'):
        torch.cuda.manual_seed(args.seed)
    # TODO: set dynamic system/env seed here


def setup_env(args: argparse.Namespace) -> Env:
    env = Car1OrderEnv(
        args.max_episode_length,
        args.action_repeat
    )
    return env


def setup_replay(args: argparse.Namespace, env: Env) -> ExperienceReplay:
    D = ExperienceReplay(
        args.experience_size,
        env.observation_size,
        env.action_size,
        args.device
    )
    # Initialise dataset D with random seed episodes
    for _ in range(1, args.seed_episodes + 1):
        observation, done = env.reset(), False
        while not done:
            action = env.sample_random_action()
            next_observation, _, done, info = env.step(action)
            D.append(observation, action, info["reward_dist"], info["reward_coll"], done)
            observation = next_observation

    return D


def setup_models(args: argparse.Namespace, env: Env) -> Tuple[Tuple[nn.Module, nn.Module, nn.Module, nn.Module],
                                                              Tuple[optim.Optimizer, optim.Optimizer],
                                                              List[torch.nn.parameter.Parameter]]:
    # Initialise model parameters randomly
    transition_model = TransitionModel(
        args.belief_size,
        args.state_size,
        env.action_size,
        args.hidden_size,
        args.embedding_size,
        args.activation_function
    ).to(device=args.device)

    observation_model = ObservationDecoder(
        args.belief_size,
        args.state_size,
        args.embedding_size
    ).to(device=args.device)

    reward_model = RewardModel(
        args.belief_size,
        args.state_size,
        args.hidden_size
    ).to(device=args.device)

    encoder = ObservationEncoder(
        args.embedding_size,
    ).to(device=args.device)

    param_list = (
        list(transition_model.parameters()) +
        list(observation_model.parameters()) +
        list(encoder.parameters())
    )

    transition_optimizer = optim.Adam(param_list, args.learning_rate, eps=args.adam_epsilon)
    reward_optimizer = optim.Adam(reward_model.parameters(), args.learning_rate, eps=args.adam_epsilon)

    # load parameters
    if args.checkpoint_path is not None:
        if os.path.exists(args.checkpoint_path):
            model_dicts = torch.load(args.checkpoint_path)
            transition_model.load_state_dict(model_dicts['transition_model'])
            observation_model.load_state_dict(model_dicts['observation_model'])
            reward_model.load_state_dict(model_dicts['reward_model'])
            encoder.load_state_dict(model_dicts['encoder'])
            transition_optimizer.load_state_dict(model_dicts['transition_optimizer'])
            reward_optimizer.load_state_dict(model_dicts['reward_optimizer'])
        else:
            logging.warning("Model weight file: {} does not exist".format(args.checkpoint_path))

    return (transition_model, observation_model, reward_model, encoder), (transition_optimizer, reward_optimizer), param_list


def setup_planner(args: argparse.Namespace, env: Env, transition_model: nn.Module, reward_model: nn.Module) -> nn.Module:
    planner = MPCPlanner(
        env.action_size,
        args.planning_horizon,
        args.optimisation_iters,
        args.candidates,
        args.top_candidates,
        transition_model,
        reward_model,
        env.action_range[0],
        env.action_range[1]
    )
    return planner


def setup(args: argparse.Namespace) -> Tuple[Env,
                                             ExperienceReplay,
                                             Tuple[nn.Module, nn.Module, nn.Module, nn.Module],
                                             Tuple[optim.Optimizer, optim.Optimizer],
                                             List[nn.parameter.Parameter],
                                             nn.Module]:
    if not args.test:
        setup_seed(args)
        setup_workdir(args)
        save_args(args)
    env = setup_env(args)
    D = setup_replay(args, env)
    models, optimizers, param_list = setup_models(args, env)
    planner = setup_planner(args, env, models[0], models[2])

    return env, D, models, optimizers, param_list, planner


def collect_experience(args: argparse.Namespace,
                       env: Env,
                       models: Tuple[nn.Module, nn.Module, nn.Module, nn.Module],
                       planner: nn.Module,
                       explore: bool = True,
                       desc: str = "Collecting episode") -> Dict[str, List[torch.Tensor]]:
    """collect an episode by applying policy on the real env.
    """
    # unpack models
    transition_model, _, _, encoder = models
    # storage
    experience = {
        "belief": [],
        "state": [],
        "action": [],
        "observation": [],
        "reward_dist": [],
        "reward_coll": [],
        "done": []
    }
    with torch.no_grad():
        # h[-1], s[-1], a[-1], o[0]
        belief = torch.zeros(1, args.belief_size, device=args.device)
        posterior_state = torch.zeros(1, args.state_size, device=args.device)
        action = torch.zeros(1, env.action_size, device=args.device)
        observation = env.reset()

        for _ in trange(args.max_episode_length // args.action_repeat, leave=False, desc=desc):
            # h[t] = f(h[t-1], a[t-1])
            # s[t] ~ Prob(s|h[t])
            # action and observation need extra time dimension because transition model uses batch operation
            belief, _, _, _, posterior_state, _, _ = transition_model.forward(
                posterior_state,
                action.unsqueeze(dim=0),
                belief,
                encoder(observation.to(device=args.device)).unsqueeze(dim=0))
            belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)

            # a[t] = pi(h[t], s[t]) + noise
            # action is bounded by action range
            action = planner(belief, posterior_state)
            if explore:
                action += args.action_noise * torch.randn_like(action)
            action.clamp_(min=env.action_range[0], max=env.action_range[1])

            # o[t+1] ~ Prob(o|x[t], a[t]), r[t+1], z[t+1]
            next_observation, _, done, info = env.step(action[0].cpu())

            # save h[t], s[t], a[t], o[t], r[t+1], z[t+1]
            experience["belief"].append(belief)
            experience["state"].append(posterior_state)
            experience["action"].append(action.cpu())
            experience["observation"].append(observation)
            experience["reward_dist"].append(info["reward_dist"])
            experience["reward_coll"].append(info["reward_coll"])
            experience["done"].append(done)

            if done:
                break
            else:
                observation = next_observation

    return experience


def test(args: argparse.Namespace, env: Env, models: Tuple[nn.Module, nn.Module, nn.Module, nn.Module], planner: nn.Module):
    for model in models:
        model.eval()

    # unpack models
    _, observation_model, _, _ = models
    # collect an episode
    with torch.no_grad():
        experience = collect_experience(args, env, models, planner, False, desc="Collecting experience 0")
        # get observations and predictions
        observations = torch.cat(experience["observation"], dim=0)
        beliefs = torch.cat(experience["belief"], dim=0)
        states = torch.cat(experience["state"], dim=0)
        predictions = observation_model.forward(beliefs.to(args.device), states.to(args.device))
        # visualize them
        visualize_local_map(os.path.join(args.video_dir, "observation.mp4"), observations)
        visualize_local_map(os.path.join(args.video_dir, "prediction.mp4"), predictions)
        visualize_global_map(os.path.join(args.video_dir, "global.mp4"), args.obstacle_index, observations, predictions)

    for model in models:
        model.train()


def train(args: argparse.Namespace,
          env: Env,
          D: ExperienceReplay,
          models: Tuple[nn.Module, nn.Module, nn.Module, nn.Module],
          optimizer: Tuple[optim.Optimizer, optim.Optimizer],
          param_list: List[nn.parameter.Parameter],
          planner: nn.Module):
    # auxilliary tensors
    global_prior = Normal(
        torch.zeros(args.batch_size, args.state_size, device=args.device),
        torch.ones(args.batch_size, args.state_size, device=args.device)
    )  # Global prior N(0, I)
    # Allowed deviation in KL divergence
    free_nats = torch.full((1, ), args.free_nats, dtype=torch.float32, device=args.device)
    summary_writter = SummaryWriter(args.tensorboard_dir)

    # unpack models
    transition_model, observation_model, reward_model, encoder = models
    transition_optimizer, reward_optimizer = optimizer

    for idx_episode in trange(args.episodes, leave=False, desc="Episode"):
        for idx_train in trange(args.collect_interval, leave=False, desc="Training"):
            # Draw sequence chunks {(o[t], a[t], r[t+1], z[t+1])} ~ D uniformly at random from the dataset
            # The first two dimensions of the tensors are L (chunk size) and n (batch size)
            # We want to use o[t+1] to correct the error of the transition model,
            # so we need to convert the sequence to {(o[t+1], a[t], r[t+1], z[t+1])}
            observations, actions, rewards_dist, rewards_coll, nonterminals = D.sample(args.batch_size, args.chunk_size)
            # Create initial belief and state for time t = 0
            init_belief = torch.zeros(args.batch_size, args.belief_size, device=args.device)
            init_state = torch.zeros(args.batch_size, args.state_size, device=args.device)
            # Transition model forward
            # deterministic: h[t+1] = f(h[t], a[t])
            # prior:         s[t+1] ~ Prob(s|h[t+1])
            # posterior:     s[t+1] ~ Prob(s|h[t+1], o[t+1])
            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(
                init_state,
                actions[:-1],
                init_belief,
                bottle(encoder, (observations[1:], )),
                nonterminals[:-1]
            )

            # observation loss
            predictions = bottle(observation_model, (beliefs, posterior_states))
            visual_loss = F.mse_loss(
                predictions[:, :, :3*64*64],
                observations[1:, :, :3*64*64]
            ).mean()
            symbol_loss = F.mse_loss(
                predictions[:, :, 3*64*64:],
                observations[1:, :, 3*64*64:]
            ).mean()
            observation_loss = visual_loss + symbol_loss

            # KL divergence loss. Minimize the difference between posterior and prior
            kl_loss = torch.max(
                kl_divergence(
                    Normal(posterior_means, posterior_std_devs),
                    Normal(prior_means, prior_std_devs)
                ).sum(dim=2),
                free_nats
            ).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
            if args.global_kl_beta != 0:
                kl_loss += args.global_kl_beta * kl_divergence(
                    Normal(posterior_means, posterior_std_devs),
                    global_prior
                ).sum(dim=2).mean(dim=(0, 1))

            # overshooting loss
            if args.overshooting_kl_beta != 0:
                overshooting_vars = []  # Collect variables for overshooting to process in batch
                for t in range(1, args.chunk_size - 1):
                    d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
                    # Use t_ and d_ to deal with different time indexing for latent states
                    t_, d_ = t - 1, d - 1
                    # Calculate sequence padding so overshooting terms can be calculated in one batch
                    seq_pad = (0, 0, 0, 0, 0, t - d + args.overshooting_distance)
                    # Store
                    # * a[t:d],
                    # * z[t+1:d+1]
                    # * r[t+1:d+1]
                    # * h[t]
                    # * s[t] prior
                    # * E[s[t:d]] posterior
                    # * Var[s[t:d]] posterior
                    # * mask:
                    #       the last few sequences do not have enough length,
                    #       so we pad it with 0 to the same length as previous sequence for batch operation,
                    #       and use mask to indicate invalid variables.
                    overshooting_vars.append(
                        (F.pad(actions[t:d], seq_pad),
                         F.pad(nonterminals[t:d], seq_pad),
                         F.pad(rewards_dist[t:d], seq_pad[2:]),
                         beliefs[t_],
                         prior_states[t_],
                         F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad),
                         F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1),
                         F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device), seq_pad)
                         )
                    )  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences

                overshooting_vars = tuple(zip(*overshooting_vars))
                # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
                beliefs, prior_states, prior_means, prior_std_devs = transition_model(
                    torch.cat(overshooting_vars[4], dim=0),
                    torch.cat(overshooting_vars[0], dim=1),
                    torch.cat(overshooting_vars[3], dim=0),
                    None,
                    torch.cat(overshooting_vars[1], dim=1)
                )
                seq_mask = torch.cat(overshooting_vars[7], dim=1)
                # Calculate overshooting KL loss with sequence mask
                kl_loss += (1 / args.overshooting_distance) * args.overshooting_kl_beta * torch.max(
                    (kl_divergence(
                        Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)),
                        Normal(prior_means, prior_std_devs)
                    ) * seq_mask).sum(dim=2),
                    free_nats
                ).mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)

            # TODO: add learning rate schedule
            # Update model parameters
            transition_optimizer.zero_grad()
            loss = observation_loss * 200 + kl_loss
            loss.backward()
            nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
            transition_optimizer.step()

            # reward loss
            rewards_dist_predict, rewards_coll_predict = bottle(reward_model.raw, (beliefs.detach(), posterior_states.detach()))
            reward_loss = F.mse_loss(
                rewards_dist_predict,
                rewards_dist[:-1],
                reduction='mean'
            ) + F.binary_cross_entropy(
                rewards_coll_predict,
                rewards_coll[:-1],
                reduction='mean'
            )
            reward_optimizer.zero_grad()
            reward_loss.backward()
            reward_optimizer.step()

            # add tensorboard log
            global_step = idx_train + idx_episode * args.collect_interval
            summary_writter.add_scalar("observation_loss", observation_loss, global_step)
            summary_writter.add_scalar("reward_loss", reward_loss, global_step)
            summary_writter.add_scalar("kl_loss", kl_loss, global_step)

        for idx_collect in trange(1, leave=False, desc="Collecting"):
            experience = collect_experience(args, env, models, planner, True, desc="Collecting experience {}".format(idx_collect))
            T = len(experience["observation"])
            for idx_step in range(T):
                D.append(experience["observation"][idx_step],
                         experience["action"][idx_step],
                         experience["reward_dist"][idx_step],
                         experience["reward_coll"][idx_step],
                         experience["done"][idx_step])

        # Checkpoint models
        if (idx_episode + 1) % args.checkpoint_interval == 0:
            record_path = os.path.join(args.checkpoint_dir, "checkpoint")
            checkpoint_path = os.path.join(args.checkpoint_dir, 'models_%d.pth' % (idx_episode+1))
            torch.save(
                {
                    'transition_model': transition_model.state_dict(),
                    'observation_model': observation_model.state_dict(),
                    'reward_model': reward_model.state_dict(),
                    'encoder': encoder.state_dict(),
                    'transition_optimizer': transition_optimizer.state_dict(),
                    'reward_optimizer': reward_optimizer.state_dict()
                },
                checkpoint_path)
            with open(record_path, "w") as f:
                f.write('models_%d.pth' % (idx_episode+1))
            planner.save(os.path.join(args.torchscript_dir, "mpc_planner.pth"))
            transition_model.save(os.path.join(args.torchscript_dir, "transition_model.pth"))
            reward_model.save(os.path.join(args.torchscript_dir, "reward_model.pth"))
            observation_model.save(os.path.join(args.torchscript_dir, "observation_decoder.pth"))
            encoder.save(os.path.join(args.torchscript_dir, "observation_encoder.pth"))

    summary_writter.close()


def main():
    args = postprocess_args(get_args())
    env, D, models, optimiser, param_list, planner = setup(args)
    if not args.test:
        train(args, env, D, models, optimiser, param_list, planner)
    test(args, env, models, planner)


if __name__ == "__main__":
    main()
