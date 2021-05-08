from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F


# Wraps the input tuple for a function to process a time*batch*features sequence in batch*features (assumes one output)
def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


class TransitionModel(jit.ScriptModule):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    @jit.script_method
    def forward(
        self,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_belief: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        nonterminals: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1

        beliefs = [torch.empty(0)] * T
        prior_states = [torch.empty(0)] * T
        prior_means = [torch.empty(0)] * T
        prior_std_devs = [torch.empty(0)] * T
        posterior_states = [torch.empty(0)] * T
        posterior_means = [torch.empty(0)] * T
        posterior_std_devs = [torch.empty(0)] * T

        beliefs[0] = prev_belief
        prior_states[0] = prev_state
        posterior_states[0] = prev_state

        # Loop over time sequence
        for t in range(T - 1):
            # Select appropriate previous state
            _state = prior_states[t] if observations is None else posterior_states[t]
            _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])

        # Return new hidden states
        hidden = [torch.stack(beliefs[1:], dim=0),
                  torch.stack(prior_states[1:], dim=0),
                  torch.stack(prior_means[1:], dim=0),
                  torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0),
                       torch.stack(posterior_means[1:], dim=0),
                       torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden


class ObservationModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, embedding_size):
        super().__init__()
        self.belief_size = belief_size
        self.state_size = state_size
        self.embedding_size = embedding_size

        self.stem = nn.Linear(belief_size+state_size, embedding_size)
        self.visual_input = nn.Linear(embedding_size-6, 256)
        self.visual = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2)
        )
        self.symbol = nn.Sequential(
            nn.Identity(),
        )

    @jit.script_method
    def forward(self, belief, state):
        """
        belief: n*belief_size torch tensor
        state: n*state_size torch tensor
        """
        hidden = self.stem(torch.cat([belief, state], dim=1))
        hidden1, hidden2 = torch.split(hidden, [self.embedding_size-6, 6], dim=1)

        hidden1 = self.visual_input(hidden1)
        visual = self.visual(hidden1.view(-1, 256, 1, 1))

        symbol = self.symbol(hidden2)

        return torch.cat([visual.view(-1, 3*64*64), symbol], dim=1)


class ObservationEncoder(jit.ScriptModule):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU()
        )
        self.visual_output = nn.Linear(1024, embedding_size-6)
        self.symbol_encoder = nn.Sequential(
            nn.Identity()
        )

    @jit.script_method
    def forward(self, observation):
        """
        observation: n*observation_size tensor
        """
        visual, symbol = torch.split(observation, [3*64*64, 6], dim=1)

        hidden1 = self.visual_encoder(visual.view(-1, 3, 64, 64))
        hidden1 = self.visual_output(hidden1.view(-1, 1024))

        hidden2 = self.symbol_encoder(symbol)

        return torch.cat([hidden1, hidden2], dim=1)


class RewardModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size):
        super().__init__()
        self.belief_size = belief_size
        self.state_size = state_size
        self.hidden_size = hidden_size

        self.network = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    @jit.script_method
    def forward(self, belief, state):
        reward = self.network(torch.cat([belief, state], dim=1)).squeeze()
        return reward
