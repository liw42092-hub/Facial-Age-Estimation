from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class AgeCalibrator:
    def __init__(self, x_thresholds, y_thresholds):
        self.x_thresholds = np.asarray(x_thresholds, dtype=np.float64)
        self.y_thresholds = np.asarray(y_thresholds, dtype=np.float64)
        if self.x_thresholds.ndim != 1 or len(self.x_thresholds) < 2:
            raise ValueError("Calibration requires at least two one-dimensional thresholds.")

    @classmethod
    def from_json(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["x_thresholds"], payload["y_thresholds"])

    def apply_numpy(self, ages):
        values = np.asarray(ages, dtype=np.float64)
        return np.interp(values, self.x_thresholds, self.y_thresholds).clip(0.0, 100.0)

    def apply_tensor(self, ages: torch.Tensor) -> torch.Tensor:
        calibrated = self.apply_numpy(ages.detach().cpu().numpy())
        return torch.as_tensor(calibrated, device=ages.device, dtype=ages.dtype)
