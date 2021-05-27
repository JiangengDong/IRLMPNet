from models_groundtruth import TransitionModel, RewardModel, ObservationEncoder
from planner import MPCPlanner
import torch
from envs import Car1OrderEnv
from visual import visualize_global_map, visualize_local_map, visualize_planning_result

transition_model = TransitionModel(
    3,
    6,
    2,
    0,
    0,
    "none"
).to(device="cuda")

reward_model = RewardModel(
    0,
    6,
    0
).to(device="cuda")

encoder = ObservationEncoder(
    0,
).to(device="cuda")

planner = MPCPlanner(
    2,
    100,
    4,
    10,
    3,
    transition_model,
    reward_model,
    -0.5,
    0.5
)

env = Car1OrderEnv(
    500,
    2
)


def collect_experience():
    """collect an episode by applying policy on the real env.
    """
    # storage
    experience = {
        "belief": [],
        "state": [],
        "action": [],
        "observation": [],
        "reward": [],
        "done": []
    }
    with torch.no_grad():
        # h[-1], s[-1], a[-1], o[0]
        belief = torch.zeros(1, 3, device="cuda")
        posterior_state = torch.zeros(1, 6, device="cuda")
        action = torch.zeros(1, 2, device="cuda")
        observation = env.reset()

        for t in range(500 // 2):
            # h[t] = f(h[t-1], a[t-1])
            # s[t] ~ Prob(s|h[t])
            # action and observation need extra time dimension because transition model uses batch operation
            belief, _, _, _, posterior_state, _, _ = transition_model.forward(
                posterior_state,
                action.unsqueeze(dim=0),
                belief,
                encoder(observation.to(device="cuda")).unsqueeze(dim=0))
            belief, posterior_state = belief.squeeze(
                dim=0), posterior_state.squeeze(dim=0)

            # a[t] = pi(h[t], s[t]) + noise
            # action is bounded by action range
            action = planner(belief, posterior_state)
            action.clamp_(min=env.action_range[0], max=env.action_range[1])

            # o[t+1] ~ Prob(o|x[t], a[t]), r[t+1], z[t+1]
            next_observation, reward, done = env.step(action[0].cpu())

            # save h[t], s[t], a[t], o[t], r[t+1], z[t+1]
            experience["belief"].append(belief)
            experience["state"].append(posterior_state)
            experience["action"].append(action.cpu())
            experience["observation"].append(observation)
            experience["reward"].append(reward)
            experience["done"].append(done)

            if done:
                break
            else:
                observation = next_observation

    return experience


if __name__ == "__main__":
    with torch.no_grad():
        experience = collect_experience()
        observations = torch.cat(experience["observation"], dim=0)
        # visualize_local_map("observation.mp4", observations)
        visualize_global_map("global.mp4", 0, observations, observations)
