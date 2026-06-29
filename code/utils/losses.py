from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.metrics import outputs_to_age


def _weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    if sample_weights is None:
        return values.mean()
    weights = sample_weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def compute_loss(
    outputs: torch.Tensor,
    ages: torch.Tensor,
    soft_labels: torch.Tensor,
    method: str,
    mae_weight: float = 0.1,
    variance_weight: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if method in {"dldl", "dldlv2"}:
        log_probs = F.log_softmax(outputs, dim=1)
        probabilities = torch.softmax(outputs, dim=1)
        kl_per_sample = F.kl_div(log_probs, soft_labels, reduction="none").sum(dim=1)
        pred_age = outputs_to_age(outputs, method)
        mae_per_sample = torch.abs(pred_age - ages)
        age_indices = torch.arange(outputs.size(1), device=outputs.device, dtype=outputs.dtype)
        variance_per_sample = torch.sum(
            probabilities * (age_indices.unsqueeze(0) - pred_age.unsqueeze(1)) ** 2,
            dim=1,
        )
        loss_per_sample = kl_per_sample + mae_weight * mae_per_sample
        if method == "dldlv2":
            loss_per_sample = loss_per_sample + variance_weight * variance_per_sample
        loss = _weighted_mean(loss_per_sample, sample_weights)
        parts = {
            "kl": _weighted_mean(kl_per_sample, sample_weights).item(),
            "mae_aux": _weighted_mean(mae_per_sample, sample_weights).item(),
            "variance": _weighted_mean(variance_per_sample, sample_weights).item(),
        }
    elif method == "coral":
        thresholds = torch.arange(outputs.size(1), device=ages.device).unsqueeze(0)
        levels = (ages.unsqueeze(1) > thresholds).to(dtype=outputs.dtype)
        bce_per_sample = F.binary_cross_entropy_with_logits(outputs, levels, reduction="none").mean(dim=1)
        loss = _weighted_mean(bce_per_sample, sample_weights)
        pred_age = outputs_to_age(outputs, method)
        parts = {"ordinal_bce": loss.item()}
    elif method == "classification":
        ce_per_sample = F.cross_entropy(outputs, ages.long(), reduction="none")
        loss = _weighted_mean(ce_per_sample, sample_weights)
        pred_age = outputs_to_age(outputs, method)
        parts = {"cross_entropy": loss.item()}
    elif method == "regression":
        raw_age = outputs.squeeze(1)
        l1_per_sample = torch.abs(raw_age - ages)
        loss = _weighted_mean(l1_per_sample, sample_weights)
        pred_age = raw_age.clamp(0.0, 100.0)
        parts = {"l1": loss.item()}
    else:
        raise ValueError(f"Unknown training method: {method}")
    return loss, pred_age, parts
