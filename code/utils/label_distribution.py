from __future__ import annotations

import torch


def make_gaussian_label(age, num_ages: int = 101, sigma: float = 1.0) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero.")
    age_value = torch.as_tensor(age, dtype=torch.float32)
    age_indices = torch.arange(num_ages, dtype=torch.float32)
    distribution = torch.exp(-((age_indices - age_value) ** 2) / (2 * sigma**2))
    return distribution / distribution.sum().clamp_min(1e-12)
