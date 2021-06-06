# RLMPNet

This project aims to combine [MPNet], [SST], and [PlaNet] together to solve kinodynamic motion planning (KMP) problem. Waypoints are generated with [MPNet], which is then used as the input of [PlaNet]'s Cross-Entropy Method (CEM) policy to get the control. These two steps are repeated following [SST]'s pattern until a path is found.

(This project was called `IRLMPNet` because the original plan was to use [EAIRL]. You can still find the name `IRLMPNet` in many parts of this project.)

## Requirements

There is a docker image on the **ARCLab DL1** server which packs all the necessary libraries. You can also install the following libraries on your own computer.

**Note**: it seems that both the pip version of PyTorch and libTorch cannot be linked correctly to this project. Compiling from source solves this problem.

* [Conda] 4.10.1
* [PyTorch] 1.8.1 (compile from source recommended)
* [OMPL] 1.4.2 (compile from source required)

## Project structure

This project consists of four parts.

* **System**: This part provides the interfaces for the dynamic systems. Every system follows OMPL's convention, consisting of a state space, a control space, a collision checker, and a propagator. The part is written in C++, while a python binding is provided through [pybind11] so that both ends share the same system. The source code lies in [src/system](src/system).

* **PlaNet**: This part is a variation of [PlaNet] algorithm, consisting of an observation encoder, an observation decoder, a latent transition model, and a reward model. The source code lies in [python/](python/) folder. For more details, check [python/README.md](python/README.md).

* **MPNet**: This part is a MPNet model from Linjun's MPC-MPNet project. We didn't train the model by our own.

* **RL-MPNet**: This part is a motion planner following OMPL's convention. It is written in C++ for speed. The source code lies in [src/planner](src/planner).

## Build C++ part

We use [CMake] as the build system in this project. Use the following commands to build this project.

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=<path_to_torch>
make
# or "make -j n", where n is the number of parallel jobs
```

Two targets are provided.

* The first one is "KinoDynSys", which is a python binding library for the dynamic systems. The generated file will be automatically copied to [python/envs/](python/envs/) folder.

* The second one is "IRLMPNet", an executable file that tests the planner's performance. The entrance of this project is [src/main.cpp](src/main.cpp).

[MPNet]: https://sites.google.com/view/mpnet
[SST]: https://arxiv.org/abs/1407.2896
[PlaNet]: https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html
[Conda]: https://docs.conda.io/en/latest/miniconda.html
[PyTorch]: https://pytorch.org/
[OMPL]: https://ompl.kavrakilab.org/
[pybind11]: https://pybind11.readthedocs.io/en/stable/index.html
[EAIRL]: https://sites.google.com/view/eairl
[CMake]: https://cmake.org/
