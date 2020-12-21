# kinodyn_envs

This package provides several simple environments for kino-dynamic motion planning. 

## Submodules

### system

This is where the state space, control space and system dynamics are defined. Mechanical systems such as differential drive, quadrotor, acrobot and cartpole are implemented here. 

The classes are designed to be stateless, in other words, are ignorant about any previous input. They are only namespaces to organize system parameters and dynamics. This is designed for easier integration with collision checking: if a new state is in collision, you can roll back without side effect.

Because of the same reason, these classes also don't know the geometric details about the robots. Only abstract states are considered here.

TODO: move this part to cpp for better efficiency.

### visual

This module provides a batch of plotting functions and classes for different robots and obstacles. 

### environment

This module is the main entrance of this package. It provides simulation environments that integrate system dynamics, collision checking, visualization together, and use the gym-style API. 

## Acknowledgement 

The design of this library is inspired by the OMPL library, Linjun Lee's mpc-mpnet-py repository and Leon Dai's dynamic_env package.