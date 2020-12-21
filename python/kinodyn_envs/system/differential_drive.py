import numpy as np


class DifferentialDrive:
    """A dynamic model of the differential car. 

    This class defines the range and topology of the differential car's state space and control space, and provides basic operations. 
    It also defines the dynamic model, so it can be used to get the trajectory. 

    This class is designed to be stateless, which means it does not have memory about the previous input. 

    Note: 
        Many `assert` clauses are used in this class to ensure a proper input. You can optimize it out with `python -O your_script.py`.
    """

    # state-space-related parameters
    state_dim = 5  # [x, y, theta, v, omega]
    state_bound = np.array([
        [-2.0, 2.0],  # x range
        [-2.0, 2.0],  # y range
        [-1.0, 1.0],    # theta range
        [-1.0, 1.0],    # v range
        [-1.0, 1.0]  # omega range
    ], dtype=np.float32)
    is_circular = np.array([False, False, True, False, False])

    # control-space-related parameters
    control_dim = 2  # [alpha, beta]
    control_bound = np.array([
        [-1.0, 1.0],
        [-1.0, 1.0]
    ], dtype=np.float32)

    # integration-related parameters
    integrate_dt = 0.02

    def __init__(self):
        # auxiliary, intermediate results
        self._state_range = self.state_bound[:, 1] - self.state_bound[:, 0]
        self._state_bias = self.state_bound[:, 0]

    def enforce_bound(self, state: np.ndarray) -> np.ndarray:
        assert state.shape == (self.state_dim, )
        state = state.copy()

        # bring variable to local scope to increase speed
        circular = self.is_circular
        not_circular = np.logical_not(circular)

        # clip or wrap
        state[circular] = np.remainder(state[circular] - self._state_bias[circular], self._state_range[circular]) + self._state_bias[circular]
        state[not_circular] = np.clip(state[not_circular], self.state_bound[not_circular, 0], self.state_bound[not_circular, 1])

        return state

    def diff(self, state1: np.ndarray, state2: np.ndarray) -> np.ndarray:
        """Calculate distance between two states, taking state space topology into consideration.

        Note: 
            state1 and state2 are assumed to be valid.
        """
        assert state1.shape == (self.state_dim, ) and state2.shape == (self.state_dim, )

        # bring variable to local scope to increase speed
        circular = self.is_circular

        diff = state1 - state2
        diff[circular] = np.remainder(diff[circular] + self._state_range[circular] / 2.0, self._state_range[circular]) - self._state_range[circular] / 2.0

        return diff

    def distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        return np.linalg.norm(self.diff(state1, state2))

    def is_valid(self, state: np.ndarray) -> bool:
        assert state.shape == (self.state_dim, )
        return np.logical_and(state >= self.state_bound[:, 0], state <= self.state_bound[:, 1]).all()

    def function(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Dynamic model of the differential car

        $$\dot{x} = f(x, u)$$
        """
        assert state.shape == (self.state_dim, ) and control.shape == (self.control_dim, )

        deriv = np.zeros_like(state)
        deriv[0] = state[3] * np.cos(state[2]*np.pi)
        deriv[1] = state[3] * np.sin(state[2]*np.pi)
        deriv[2] = state[4]
        deriv[3] = control[0]
        deriv[4] = control[1]
        return deriv

    def propagate(self, state: np.ndarray, control: np.ndarray, duration: float) -> np.ndarray:
        assert state.shape == (self.state_dim, ) and control.shape == (self.control_dim, ) and duration > 0

        # bring to local scope to increase speed
        dt = self.integrate_dt

        # numerical integration with Euler method
        n_steps, remainder = np.divmod(duration, self.integrate_dt)
        for _ in range(n_steps.astype(np.int)):
            state = self.function(state, control)*dt + state
            state = self.enforce_bound(state)
        if not np.isclose(remainder, 0):
            state = self.function(state, control)*remainder + state
            state = self.enforce_bound(state)
        # TODO: add prediction-correction Euler & Runge-Kutta Method

        return state
