from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.utkface import build_datasets
from models.farl_age_model import FaRLAgeEstimator
from utils.farl import predict_with_flip_routing
from utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit monotonic age calibration for one or more FaRL-GLAE checkpoints."
    )
    parser.add_argument("--data_dir", default="./data/UTKFace")
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
        help="One or more FaRL checkpoints. Multiple checkpoints are probability-averaged before calibration.",
    )
    parser.add_argument("--output", default="./results/calibration.json")
    parser.add_argument("--mode", choices=("route", "vanilla", "balanced"), default="route")
    parser.add_argument("--local_blend", type=float, default=None)
    parser.add_argument("--tta", choices=("none", "flip", "five_crop", "ten_crop"), default="flip")
    parser.add_argument("--tta_crop_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--limit_val_samples", type=int, default=None)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "farl_vit_b16_glae":
        raise RuntimeError(f"Expected a FaRL-GLAE checkpoint: {checkpoint_path}")
    model = FaRLAgeEstimator().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, checkpoint


def checkpoint_has_local_heads(checkpoint: dict) -> bool:
    state = checkpoint.get("model_state_dict", {})
    return any(key.startswith("vanilla_local_head.") for key in state) and any(
        key.startswith("balanced_local_head.") for key in state
    )


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaded = [load_model(path, device) for path in args.checkpoint]
    models = [item[0] for item in loaded]
    checkpoint = loaded[0][1]
    if args.local_blend is None:
        local_blend = checkpoint.get("local_blend")
        if local_blend is None:
            local_blend = 0.2 if checkpoint_has_local_heads(checkpoint) else 0.0
    else:
        local_blend = args.local_blend

    datasets, _ = build_datasets(
        args.data_dir,
        sigma=checkpoint.get("sigma", 1.0),
        seed=checkpoint.get("split_seed", 42),
        val_ratio=checkpoint.get("val_ratio", 0.1),
        test_ratio=checkpoint.get("test_ratio", 0.1),
        verify_images=not args.skip_verify,
        image_size=checkpoint.get("image_size", 256),
        transform_profile=checkpoint.get("transform_profile", "farl"),
    )
    if args.limit_val_samples is not None:
        datasets["val"] = Subset(
            datasets["val"], range(min(args.limit_val_samples, len(datasets["val"])))
        )
    loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    predictions = []
    targets = []
    with torch.inference_mode():
        for images, ages, _, _ in tqdm(loader, desc="calibrate-farl", ascii=True):
            images = images.to(device, non_blocking=True)
            probability_views = []
            for model in models:
                _, probabilities, _ = predict_with_flip_routing(
                    model,
                    images,
                    mode=args.mode,
                    local_blend=local_blend,
                    tta=args.tta,
                    crop_size=args.tta_crop_size,
                )
                probability_views.append(probabilities)
            probabilities = torch.stack(probability_views, dim=0).mean(dim=0)
            age_values = torch.arange(
                probabilities.size(1), device=probabilities.device, dtype=probabilities.dtype
            )
            predictions.append(torch.sum(probabilities * age_values.unsqueeze(0), dim=1).cpu())
            targets.append(ages.float())

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    calibrator = IsotonicRegression(y_min=0.0, y_max=100.0, out_of_bounds="clip")
    calibrated = calibrator.fit_transform(predictions.numpy(), targets.numpy())
    calibrated_tensor = torch.from_numpy(calibrated).float()

    payload = {
        "checkpoint": [str(Path(path).resolve()) for path in args.checkpoint],
        "split": "validation",
        "mode": args.mode,
        "tta": args.tta,
        "tta_crop_size": args.tta_crop_size,
        "local_blend": local_blend,
        "x_thresholds": calibrator.X_thresholds_.tolist(),
        "y_thresholds": calibrator.y_thresholds_.tolist(),
        "raw_validation_metrics": compute_metrics(predictions, targets),
        "calibrated_validation_metrics": compute_metrics(calibrated_tensor, targets),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Raw validation metrics:", payload["raw_validation_metrics"])
    print("Calibrated validation metrics:", payload["calibrated_validation_metrics"])
    print(f"Calibration saved to: {output_path}")


if __name__ == "__main__":
    main()
