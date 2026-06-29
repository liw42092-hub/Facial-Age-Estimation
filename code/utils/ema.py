from __future__ import annotations

import copy
import math

import torch


class ModelEMA:
    """Exponential moving average with a short warm-up for stable validation."""

    def __init__(self, model, decay: float = 0.999, updates: int = 0):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between 0 and 1.")
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = updates
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        model_state = model.state_dict()
        for name, ema_value in self.model.state_dict().items():
            source_value = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(source_value)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.model.load_state_dict(state_dict)
