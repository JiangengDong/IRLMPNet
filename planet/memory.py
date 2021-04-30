import numpy as np
import torch


class ExperienceReplay():
    def __init__(self, size, observation_size, action_size, device):
        self.device = device
        self.size = size
        self.observations = np.empty((size, observation_size), dtype=np.float32)
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size, ), dtype=np.float32)
        self.nonterminals = np.empty((size, 1), dtype=np.float32)
        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total

    def append(self, observation, action, reward, done):
        self.observations[self.idx] = observation.numpy()
        self.actions[self.idx] = action.numpy()
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

    # Returns an index for a valid single sequence chunk uniformly sampled from the memory
    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, (self.size if self.full else self.idx) - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        return (self.observations[vec_idxs].reshape(L, n, -1), # TODO: why
                self.actions[vec_idxs].reshape(L, n, -1),
                self.rewards[vec_idxs].reshape(L, n),
                self.nonterminals[vec_idxs].reshape(L, n, 1))

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample(self, n, L):
        idxs = np.asarray([self._sample_idx(L) for _ in range(n)])
        batch = self._retrieve_batch(idxs, n, L)
        return [torch.as_tensor(item).to(device=self.device) for item in batch]
