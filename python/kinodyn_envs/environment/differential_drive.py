import warnings

import numpy as np
import gym
from PIL import Image

from kinodyn_envs.system.differential_drive import DifferentialDrive
from kinodyn_envs.visual.differential_drive import plot_differential_drive


class DifferentialDriveFreeEnv(gym.Env):
    spec = {"dt": 0.1, "max_step": 150}
    action_space = gym.spaces.Box(low=DifferentialDrive.control_bound[:, 0],
                                  high=DifferentialDrive.control_bound[:, 1])
    observation_space = gym.spaces.Box(low=np.tile(DifferentialDrive.state_bound[:, 0], 2),
                                       high=np.tile(DifferentialDrive.state_bound[:, 1], 2))

    def __init__(self):
        super(DifferentialDriveFreeEnv, self).__init__()

        self._system = DifferentialDrive()
        self._state = self.observation_space.sample()
        self._goal = self.observation_space.sample()
        self._step_count = 0
        self._done = False

        self.background = Image.new("RGB", (2300, 2300), color=(255, 255, 255))

        self.reset()

    def step(self, action: np.ndarray):
        assert action.shape == (self._system.control_dim, )

        # should not step after done
        if self._done:
            warnings.warn("Stepping after done will generate useless rewards.", RuntimeWarning)
        # try to move forward
        self._step_count += 1
        new_state = self._system.propagate(self._state, action, self.spec["dt"])
        if self._system.is_valid(new_state):
            self._state = new_state

        # check if done
        diff = self._system.diff(self._state, self._goal) 
        dist = np.linalg.norm(diff*np.array([1.0, 1.0, 0.1, 0.0, 0.0]))
        if self._step_count == self.spec["max_step"] or dist < 0.01:
            self._done = True
        
        reward = - np.log(dist+0.01) + 0.1*np.abs(self._state[3]) # - 0.5*np.abs(self._state[4])

        return np.concatenate([self._state, self._goal]), reward, self._done, {}




    def reset(self):
        # random start
        for _ in range(50):
            ob = np.random.rand(10)*2-1
            self._state, self._goal = np.split(ob, 2)
            if self._system.is_valid(self._state) and self._system.is_valid(self._goal):
                break
        else:  # start and goal should be close
            self._state = np.zeros((self._system.state_dim, ), dtype=np.float32)
            self._target = np.zeros((self._system.state_dim, ), dtype=np.float32)
            warnings.warn("Fail to sample start or goal. Use zero instead.", RuntimeWarning)
        self._step_count = 0
        self._done = False

        return np.concatenate([self._state, self._goal])

    def render(self, mode="rgb_array"):
        if mode == "human":
            raise NotImplementedError
        elif mode == "rgb_array":
            img = self.background.copy()
            pos = self._state[:3]*[1.0, 1.0, np.pi]
            plot_differential_drive(img, pos, (0, 0, 0), dx=0.002)
            pos = self._goal[:3]*[1.0, 1.0, np.pi]
            plot_differential_drive(img, pos, (255, 0, 0), dx=0.002)
            return np.array(img)
        else:
            raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)


class DifferentialDriveFreeSparseEnv(gym.Env):
    spec = {"dt": 0.1, "max_step": 150}
    action_space = gym.spaces.Box(low=DifferentialDrive.control_bound[:, 0],
                                  high=DifferentialDrive.control_bound[:, 1])
    observation_space = gym.spaces.Box(low=np.tile(DifferentialDrive.state_bound[:, 0], 2),
                                       high=np.tile(DifferentialDrive.state_bound[:, 1], 2))

    def __init__(self):
        super(DifferentialDriveFreeSparseEnv, self).__init__()

        self._system = DifferentialDrive()
        self._state = self.observation_space.sample()
        self._goal = self.observation_space.sample()
        self._step_count = 0
        self._done = False

        self.background = Image.new("RGB", (2300, 2300), color=(255, 255, 255))

        self.reset()

    def step(self, action: np.ndarray):
        assert action.shape == (self._system.control_dim, )

        # should not step after done
        if self._done:
            warnings.warn("Stepping after done will generate useless rewards.", RuntimeWarning)
        # try to move forward
        self._step_count += 1
        new_state = self._system.propagate(self._state, action, self.spec["dt"])
        if self._system.is_valid(new_state):
            self._state = new_state

        # check if done
        diff = self._system.diff(self._state, self._goal)
        dist = np.linalg.norm(diff[:3])
        if self._step_count == self.spec["max_step"] or dist < 0.05:
            self._done = True
        
        reward = - (dist > 0.05).astype(np.float32)

        return np.concatenate([self._state, self._goal]), reward, self._done, {}


    def reset(self):
        # random start
        for _ in range(50):
            ob = np.random.rand(10)*2-1
            self._state, self._goal = np.split(ob, 2)
            if self._system.is_valid(self._state) and self._system.is_valid(self._goal):
                break
        else:  # start and goal should be close
            self._state = np.zeros((self._system.state_dim, ), dtype=np.float32)
            self._target = np.zeros((self._system.state_dim, ), dtype=np.float32)
            warnings.warn("Fail to sample start or goal. Use zero instead.", RuntimeWarning)
        self._step_count = 0
        self._done = False

        return np.concatenate([self._state, self._goal])

    def render(self, mode="rgb_array"):
        if mode == "human":
            raise NotImplementedError
        elif mode == "rgb_array":
            img = self.background.copy()
            pos = self._state[:3]*[1.0, 1.0, np.pi]
            plot_differential_drive(img, pos, (0, 0, 0), dx=0.002)
            pos = self._goal[:3]*[1.0, 1.0, np.pi]
            plot_differential_drive(img, pos, (255, 0, 0), dx=0.002)
            return np.array(img)
        else:
            raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
