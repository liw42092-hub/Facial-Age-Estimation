from __future__ import annotations

import torch


def logits_to_age(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    age_indices = torch.arange(logits.size(1), device=logits.device, dtype=logits.dtype)
    return torch.sum(probabilities * age_indices.unsqueeze(0), dim=1)


def outputs_to_age(outputs: torch.Tensor, method: str) -> torch.Tensor:
    if method == "regression":
        return outputs.squeeze(1).clamp(0.0, 100.0)
    if method == "coral":
        return torch.sigmoid(outputs).sum(dim=1)
    return logits_to_age(outputs)


def forward_with_tta(model, images: torch.Tensor, method: str, enabled: bool = False) -> torch.Tensor:
    outputs = model(images)
    if not enabled:
        return outputs
    flipped_outputs = model(torch.flip(images, dims=(3,)))
    if method == "regression":
        return 0.5 * (outputs + flipped_outputs)
    if method == "coral":
        probabilities = 0.5 * (
            torch.sigmoid(outputs) + torch.sigmoid(flipped_outputs)
        )
        return torch.logit(probabilities.clamp(1e-6, 1.0 - 1e-6))
    probabilities = 0.5 * (
        torch.softmax(outputs, dim=1) + torch.softmax(flipped_outputs, dim=1)
    )
    return torch.log(probabilities.clamp_min(1e-8))


def compute_metrics(pred_age: torch.Tensor, true_age: torch.Tensor) -> dict[str, float]:
    absolute_error = torch.abs(pred_age.float() - true_age.float())
    return {
        "MAE": absolute_error.mean().item(),
        "Acc@1": (absolute_error <= 1).float().mean().item(),
        "Acc@3": (absolute_error <= 3).float().mean().item(),
    }


class MetricAccumulator:
    def __init__(self):
        self.absolute_error_sum = 0.0
        self.acc1_count = 0
        self.acc3_count = 0
        self.sample_count = 0

    def update(self, pred_age: torch.Tensor, true_age: torch.Tensor) -> None:
        absolute_error = torch.abs(pred_age.detach().float() - true_age.detach().float())
        self.absolute_error_sum += absolute_error.sum().item()
        self.acc1_count += int((absolute_error <= 1).sum().item())
        self.acc3_count += int((absolute_error <= 3).sum().item())
        self.sample_count += absolute_error.numel()

    def compute(self) -> dict[str, float]:
        if self.sample_count == 0:
            raise RuntimeError("Cannot compute metrics without samples.")
        return {
            "MAE": self.absolute_error_sum / self.sample_count,
            "Acc@1": self.acc1_count / self.sample_count,
            "Acc@3": self.acc3_count / self.sample_count,
        }
