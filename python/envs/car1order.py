"""A wrapper environment for the C++ class `Car1OrderSystem`.

`Car1OrderSystem` is a C++ class for a 1st order car dynamic system which has methods for propogation, collision checking and sampling.

`Car1OrderEnv` is a wrapper environment for `Car1OrderSystem` that adds the following features:
1. All the observations are normalized to [-0.5, -0.5].
1. All the actions are unnormalized from [-0.5, 0.5] to the real value.
1. Calculate rewards based on observations and actions.
1. Set a maximum for steps.
1. Repeat every input action for several times.
"""
import numpy as np
import torch
from typing import Tuple, Dict

from .base import Env
from .KinoDynSys import Car1OrderSystem


class Car1OrderEnv(Env):
    def __init__(self, max_episode_length, action_repeat, symbolic=False):
        super().__init__()
        self.system = Car1OrderSystem()
        # fetch config from cpp part
        self.state_dim = self.system.state_dim
        self.state_upper_bound = self.system.state_upper_bound
        self.state_lower_bound = self.system.state_lower_bound
        self.control_dim = self.system.control_dim
        self.control_lower_bound = self.system.control_lower_bound
        self.control_upper_bound = self.system.control_upper_bound

        self.state_range = (self.state_upper_bound - self.state_lower_bound)
        self.state_center = (self.state_upper_bound + self.state_lower_bound)/2.0
        self.control_range = (self.control_upper_bound - self.control_lower_bound)
        self.control_center = (self.control_upper_bound + self.control_lower_bound)/2.0

        # allocate space for inner states
        self.state = np.zeros((self.state_dim, ), dtype=np.float32)
        self.normalized_state = self.state.copy()
        self.state_next = np.zeros((self.state_dim, ), dtype=np.float32)
        self.goal = np.zeros((self.state_dim, ), dtype=np.float32)
        self.normalized_goal = self.goal.copy()

        self.step_count = 0
        self.done = False

        # other config designated from outside
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.symbolic = symbolic

        # allocate a space for sampling action. This shouldn't be accessed from outside
        self.__action = np.zeros(shape=(self.control_dim, ), dtype=np.float32)

    def get_ob(self) -> torch.Tensor:
        # return 1*6 tensor as normalized observation (range: [-0.5, 0.5])
        local_map = self.system.get_local_map(self.state).flatten() - 0.5
        local_map = np.tile(local_map, [3])
        local_map = torch.from_numpy(local_map).unsqueeze(0)
        if not self.symbolic:
            goal_map = self.system.get_local_map(self.goal).flatten() - 0.5
            goal_map = np.tile(goal_map, [3])
            goal_map = torch.from_numpy(goal_map).unsqueeze(0)
            return torch.cat([local_map, goal_map], dim=1)
        else:
            state = torch.from_numpy(np.concatenate([self.normalized_state, self.normalized_goal])).unsqueeze(0)
            return torch.cat([local_map, state], dim=1)

    def reset(self) -> torch.Tensor:
        # random start and goal. They are assumed to be easy to sample
        while True:
            if self.system.sample_valid_state(self.state) and self.system.sample_valid_state(self.goal) and np.linalg.norm(self.state-self.goal) < 20:
                # if self.system.sample_valid_state(self.state) and self.system.sample_valid_state(self.goal):
                self.normalized_state = (self.state - self.state_center) / self.state_range
                self.normalized_goal = (self.goal - self.state_center) / self.state_range
                break
        self.step_count = 0
        self.done = False

        return self.get_ob()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict[str, float]]:
        # input action is assumed to be (2, ) ndarray and normalized
        assert action.shape == (self.control_dim, )
        # unnormalize the action
        action = action*self.control_range + self.control_center

        reward_dist = 0.0
        for _ in range(self.action_repeat):
            # should not step after done
            if self.done:
                pass
            self.step_count += 1
            is_collided = False
            # take mini steps to check collision
            self.system.propagate(self.state, action, self.state_next, 0.1)
            if self.system.is_valid_state(self.state_next):
                self.state[:] = self.state_next[:]
            else:
                is_collided = True
                break

            # check done and reward
            self.normalized_state = (self.state - self.state_center)/self.state_range
            diff = self.system.diff(self.normalized_state, self.normalized_goal)
            dist = np.linalg.norm(diff*np.array([1.0, 1.0, 0.1]))
            reward_dist += - dist + 0.5
            if self.step_count == self.max_episode_length or dist < 0.01 or is_collided:
                self.done = True

        return self.get_ob(), (reward_dist + (-10 if is_collided else 0)), self.done, {"reward_dist": reward_dist, "reward_coll": float(is_collided)}

    def render(self, mode="rgb_array"):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    @property
    def observation_size(self) -> int:
        return 3*64*64+3*64*64

    @property
    def action_size(self) -> int:
        return 2

    @property
    def action_range(self) -> Tuple[float, float]:
        return -0.5, 0.5

    def sample_random_action(self) -> torch.Tensor:
        self.system.sample_valid_control(self.__action)
        return torch.from_numpy((self.__action - self.control_center)/self.control_range)
