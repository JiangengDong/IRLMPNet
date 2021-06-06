from math import inf
import torch
from torch import jit, nn


# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(jit.ScriptModule):
    __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

    def __init__(self,
                 action_size: int,
                 planning_horizon: int,
                 optimisation_iters: int,
                 candidates: int,
                 top_candidates: int,
                 transition_model: nn.Module,
                 reward_model: nn.Module,
                 min_action: float = -inf,
                 max_action: float = inf):
        super().__init__()
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates

    @jit.script_method
    def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B, H, Z = belief.size(0), belief.size(1), state.size(1)
        belief = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H)
        state = state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device)
        action_std_dev = torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)
        for _ in range(self.optimisation_iters):
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            actions = (action_mean +
                       action_std_dev *
                       torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)
                       ).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
            actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
            # Sample next states
            beliefs, states, _, _ = self.transition_model(state, actions, belief)
            # Calculate expected returns (technically sum of rewards over planning horizon)
            returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
            # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
            topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
            # Update belief with new means and standard deviations
            action_mean = best_actions.mean(dim=2, keepdim=True)
            action_std_dev = best_actions.std(dim=2, unbiased=False, keepdim=True)
        # Return first action mean µ_t
        return action_mean[0].squeeze(dim=1)


class MLPPolicy(nn.Module):
    def __init__(self, belief_size, state_size, action_size):
        super().__init__()
        self.planner = nn.Sequential(
            nn.Linear(belief_size + state_size, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(),
            nn.Linear(32, action_size), nn.Sigmoid()
        )

    def forward(self, belief, state):
        return self.planner.forward(torch.cat([belief, state], dim=1)) - 0.5


class MLPPlanner(jit.ScriptModule):
    __constants__ = ['action_size', 'planning_horizon', 'candidates']

    def __init__(self, action_size, planning_horizon, candidates, transition_model, reward_model, policy):
        super().__init__()
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size = action_size
        self.candidates = candidates
        self.planning_horizon = planning_horizon
        self.policy = policy

    @jit.script_method
    def forward(self, belief, state):
        H, Z = belief.size(1), state.size(1)
        belief = belief.expand(self.candidates, H)
        state = state.expand(self.candidates, Z)
        self.policy.train()
        initial_actions = self.policy(torch.cat([belief, state], dim=1))
        self.policy.eval()
        actions = initial_actions.unsqueeze(0)
        rewards = []
        for _ in range(self.planning_horizon):
            beliefs, states, _, _ = self.transition_model(state, actions, belief)
            belief, state = beliefs.squeeze(0), states.squeeze(0)
            rewards.append(self.reward_model(belief, state))
            actions = self.policy(torch.cat([belief, state], dim=1))
        rewards = torch.sum(torch.cat(rewards, dim=0), dim=0)
        best_idx = rewards.argmax(dim=-1)
        return initial_actions[best_idx]
