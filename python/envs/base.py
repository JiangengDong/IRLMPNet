import torch
from typing import Dict, Tuple


class Env:
    def get_ob(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict[str, float]]:
        raise NotImplementedError

    @property
    def observation_size(self) -> int:
        raise NotImplementedError

    @property
    def action_size(self) -> int:
        raise NotImplementedError

    @property
    def action_range(self) -> Tuple[float, float]:
        raise NotImplementedError

    def sample_random_action(self) -> torch.Tensor:
        raise NotImplementedError
