from KinoDynSys import Car1OrderSystem
import numpy as np


class Car1OrderEnv:
    def __init__(self):
        self.system = Car1OrderSystem()
        # fetch config from cpp part
        self.state_dim = self.system.state_dim
        self.state_upper_bound = self.system.state_upper_bound
        self.state_lower_bound = self.system.state_lower_bound
        self.control_dim = self.system.control_dim
        self.control_lower_bound = self.system.control_lower_bound
        self.control_upper_bound = self.system.control_upper_bound

        self.state_half_range = (self.state_upper_bound - self.state_lower_bound)/2.0
        self.state_center = (self.state_upper_bound + self.state_lower_bound)/2.0
        self.control_half_range = (self.control_upper_bound - self.control_lower_bound)/2.0
        self.control_center = (self.control_upper_bound + self.control_lower_bound)/2.0

        self._state = np.zeros((self.state_dim, ), dtype=np.float32)
        self._normalized_state = self._state.copy()
        self._state_next = np.zeros((self.state_dim, ), dtype=np.float32)
        self._goal = np.zeros((self.state_dim, ), dtype=np.float32)
        self._normalized_goal = self._goal.copy()

        self._step_count = 0
        self._done = False

    def step(self, action: np.ndarray):
        assert action.shape == (self.control_dim, )

        # should not step after done
        if self._done:
            # warnings.warn("Stepping after done will generate useless rewards.", RuntimeWarning)
            pass
        # try to move forward
        self._step_count += 1
        action = (action - self.control_center)/self.control_half_range
        is_collided = False
        for _ in range(50):
            self.system.propagate(self._state, action, self._state_next, 0.002)
            if self.system.is_valid_state(self._state_next):
                self._state[:] = self._state_next[:]
            else:
                is_collided = True
                break

        # check if done
        diff = self.system.diff(self._state, self._goal)
        dist = np.linalg.norm(diff*np.array([1.0, 1.0, 0.1]))
        if self._step_count == 150 or dist < 0.01:
            self._done = True

        # calculate normalized observation
        self._normalized_state = (self._state - self.state_center)/self.state_half_range
        local_map = self.system.get_local_map(self._state)
        ob = np.concatenate([self._normalized_state, self._normalized_goal, local_map.flatten()])

        # calculate reward
        reward = (- dist - 0.2)*0.1 - (1 if is_collided else 0)

        return ob, reward, self._done, {}

    def reset(self):
        # random start
        for _ in range(50):
            if self.system.sample_valid_state(self._state) or self.system.sample_valid_state(self._goal):
                self._normalized_state = (self._state - self.state_center) / self.state_half_range
                self._normalized_goal = (self._goal - self.state_center) / self.state_half_range
                break
        else:  # start and goal should be close
            self._state = np.zeros((self.state_dim, ), dtype=np.float32)
            self._goal = np.zeros((self.state_dim, ), dtype=np.float32)
            self._normalized_state = (self._state - self.state_center) / self.state_half_range
            self._normalized_goal = (self._goal - self.state_center) / self.state_half_range
        self._step_count = 0
        self._done = False

        local_map = self.system.get_local_map(self._state)
        # ob = {"state": self._normalized_state, "goal": self._normalized_goal, "local_map": local_map}
        ob = np.concatenate([self._normalized_state, self._normalized_goal, local_map.flatten()])

        return ob

    def render(self, mode="rgb_array"):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        pass

if __name__ == "__main__":
    a = Car1OrderEnv()
    ac = np.array([0.1, 0.2], dtype=np.float32)
    a.step(ac)
