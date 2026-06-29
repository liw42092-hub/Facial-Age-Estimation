from __future__ import annotations

import torch
import torch.nn.functional as F


def probabilities_to_age(probabilities: torch.Tensor) -> torch.Tensor:
    ages = torch.arange(
        probabilities.size(1), device=probabilities.device, dtype=probabilities.dtype
    )
    return torch.sum(probabilities * ages.unsqueeze(0), dim=1)


def farl_age_loss(
    logits: torch.Tensor,
    ages: torch.Tensor,
    soft_labels: torch.Tensor,
    mae_weight: float = 0.2,
    distribution_weight: float = 0.2,
    label_smoothing: float = 0.05,
):
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities_to_age(probabilities)
    cross_entropy = F.cross_entropy(logits, ages.long(), label_smoothing=label_smoothing)
    expected_mae = F.l1_loss(predictions, ages)
    distribution = F.kl_div(
        F.log_softmax(logits, dim=1), soft_labels, reduction="batchmean"
    )
    loss = cross_entropy + mae_weight * expected_mae + distribution_weight * distribution
    return loss, predictions, {
        "cross_entropy": cross_entropy.detach().item(),
        "mae_aux": expected_mae.detach().item(),
        "distribution": distribution.detach().item(),
    }


def farl_local_age_loss(
    local_logits: torch.Tensor,
    ages: torch.Tensor,
    soft_labels: torch.Tensor,
    mae_weight: float = 0.0,
    distribution_weight: float = 0.0,
    label_smoothing: float = 0.0,
):
    batch_size, patch_count, age_count = local_logits.shape
    flat_logits = local_logits.reshape(batch_size * patch_count, age_count)
    repeated_ages = ages.repeat_interleave(patch_count)
    repeated_soft_labels = soft_labels.repeat_interleave(patch_count, dim=0)
    return farl_age_loss(
        flat_logits,
        repeated_ages,
        repeated_soft_labels,
        mae_weight=mae_weight,
        distribution_weight=distribution_weight,
        label_smoothing=label_smoothing,
    )


def symmetric_kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.clamp_min(1e-8)
    second = second.clamp_min(1e-8)
    return 0.5 * (
        torch.sum(first * (first.log() - second.log()), dim=1)
        + torch.sum(second * (second.log() - first.log()), dim=1)
    )


def _head_probabilities(model, features: torch.Tensor, patch_features: torch.Tensor | None, head: str, local_blend: float):
    global_probabilities = torch.softmax(model.logits_from_features(features, head), dim=1)
    if local_blend <= 0 or patch_features is None:
        return global_probabilities
    local_logits = model.local_logits_from_tokens(patch_features, head)
    local_probabilities = torch.softmax(local_logits, dim=2).mean(dim=1)
    return (1.0 - local_blend) * global_probabilities + local_blend * local_probabilities


def _spatial_views(images: torch.Tensor, tta: str = "flip", crop_size: int = 224) -> list[torch.Tensor]:
    if tta == "none":
        return [images]
    if tta == "flip":
        return [images, torch.flip(images, dims=(3,))]
    if tta not in {"five_crop", "ten_crop"}:
        raise ValueError(f"Unknown TTA mode: {tta}")

    _, _, height, width = images.shape
    crop = min(crop_size, height, width)
    offsets = [
        (0, 0),
        (0, width - crop),
        (height - crop, 0),
        (height - crop, width - crop),
        ((height - crop) // 2, (width - crop) // 2),
    ]
    if height == crop and width == crop:
        offsets = [((height - crop) // 2, (width - crop) // 2)]
    else:
        offsets = list(dict.fromkeys((max(0, t), max(0, l)) for t, l in offsets))

    views = [images[:, :, top : top + crop, left : left + crop] for top, left in offsets]
    if tta == "ten_crop":
        views.extend(torch.flip(view, dims=(3,)) for view in list(views))
    return views


def _average_probabilities(model, images: torch.Tensor, head: str, local_blend: float, tta: str, crop_size: int):
    probabilities = []
    for view in _spatial_views(images, tta=tta, crop_size=crop_size):
        features, patch_features = model.encode_with_tokens(view)
        probabilities.append(_head_probabilities(model, features, patch_features, head, local_blend))
    return torch.stack(probabilities, dim=0).mean(dim=0)


def predict_with_flip_routing(
    model,
    images: torch.Tensor,
    mode: str = "route",
    local_blend: float = 0.2,
    tta: str = "flip",
    crop_size: int = 224,
):
    if local_blend < 0 or local_blend > 1:
        raise ValueError("local_blend must be in [0, 1].")
    features, patch_features = model.encode_with_tokens(images)
    flipped_features, flipped_patch_features = model.encode_with_tokens(torch.flip(images, dims=(3,)))

    vanilla = _head_probabilities(model, features, patch_features, "vanilla", local_blend)
    vanilla_flip = _head_probabilities(model, flipped_features, flipped_patch_features, "vanilla", local_blend)
    vanilla_average = 0.5 * (vanilla + vanilla_flip)
    if mode == "vanilla":
        probabilities = _average_probabilities(model, images, "vanilla", local_blend, tta, crop_size)
        return probabilities_to_age(probabilities), probabilities, None

    balanced = _head_probabilities(model, features, patch_features, "balanced", local_blend)
    balanced_flip = _head_probabilities(model, flipped_features, flipped_patch_features, "balanced", local_blend)
    balanced_average = 0.5 * (balanced + balanced_flip)
    if mode == "balanced":
        probabilities = _average_probabilities(model, images, "balanced", local_blend, tta, crop_size)
        return probabilities_to_age(probabilities), probabilities, None
    if mode != "route":
        raise ValueError(f"Unknown prediction mode: {mode}")

    use_balanced = symmetric_kl(balanced, balanced_flip) < symmetric_kl(vanilla, vanilla_flip)
    if tta == "flip":
        vanilla_probabilities = vanilla_average
        balanced_probabilities = balanced_average
    else:
        vanilla_probabilities = _average_probabilities(model, images, "vanilla", local_blend, tta, crop_size)
        balanced_probabilities = _average_probabilities(model, images, "balanced", local_blend, tta, crop_size)
    probabilities = torch.where(use_balanced.unsqueeze(1), balanced_probabilities, vanilla_probabilities)
    return probabilities_to_age(probabilities), probabilities, use_balanced


def cumulative_class_mae(predictions: list[float], targets: list[int]) -> float:
    per_age = []
    for age in sorted(set(targets)):
        errors = [abs(pred - target) for pred, target in zip(predictions, targets) if target == age]
        if errors:
            per_age.append(sum(errors) / len(errors))
    if not per_age:
        raise RuntimeError("Cannot compute CMAE without predictions.")
    return sum(per_age) / len(per_age)
