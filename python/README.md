# PlaNet

## Environment

We only use 1st order differential car now.

### Car1order

* State space: [𝑥, 𝑦, 𝜃], invisible to the PlaNet algorithm
  * Real range: −25 < 𝑥 < 25, −35 < 𝑦 < 35, −𝜋 < 𝜃 < 𝜋
  * Normalized range: −0.5 < 𝑥,𝑦,𝜃 <0.5
* Observation space:
  * 𝑂^𝑙: local map, 64x64x3 image consists of 0 / 1
  * 𝑂^𝑔: goal map
* Control space: 𝑎=[𝑣, 𝜔]
  * Real range: −1 < 𝑣 < 2, −1 < 𝜔 < 1
  * Normalized range: −0.5< 𝑣,𝜔 <0.5
* Ground truth transition model: Integration (RK4)
* Reward function:
  * 𝑟^𝑑: distance between normalized states and normalized goal
  * 𝑟^𝑐: 0 if not collided, otherwise 1

## Components

This algorithm consists of four trainable models and a CEM algorithm.

* Observation encoder: convert observation to embedding
* Observation decoder: extract embedding from latent state, and reconstruct observation
* Latent transition model: a latent space transition model. When an action is given, the model predicts the next latent state; when the observation embedding is provided, the latent state is corrected accordingly
* Reward model: an approximation of the ground truth reward function. The input is the latent state.
* CEM algorithm: check Linjun's [MPC-MPNet] paper for more details.

[MPC-MPNet]: https://arxiv.org/abs/2101.06798
