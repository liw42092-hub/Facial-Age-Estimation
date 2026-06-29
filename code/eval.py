from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.utkface import build_datasets
from models.farl_age_model import FaRLAgeEstimator
from utils.farl import cumulative_class_mae, predict_with_flip_routing
from utils.metrics import MetricAccumulator
from utils.calibration import AgeCalibrator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a FaRL-GLAE age checkpoint.")
    parser.add_argument("--data_dir", default="./data/UTKFace")
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        default=["./model/best_model.pth"],
        help="One or more FaRL checkpoints. Multiple checkpoints are probability-averaged.",
    )
    parser.add_argument("--output_csv", default="./results/eval_predictions.csv")
    parser.add_argument("--mode", choices=("route", "vanilla", "balanced"), default="route")
    parser.add_argument("--local_blend", type=float, default=None)
    parser.add_argument("--tta", choices=("none", "flip", "five_crop", "ten_crop"), default="flip")
    parser.add_argument("--tta_crop_size", type=int, default=224)
    parser.add_argument("--calibration", default=None, help="Optional isotonic calibration JSON from calibrate_farl.py")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--limit_test_samples", type=int, default=None)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "farl_vit_b16_glae":
        raise RuntimeError(f"This evaluator requires a FaRL-GLAE checkpoint: {checkpoint_path}")
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
    calibrator = AgeCalibrator.from_json(args.calibration) if args.calibration else None

    datasets, scan_summary = build_datasets(
        args.data_dir,
        sigma=checkpoint.get("sigma", 1.0),
        seed=checkpoint.get("split_seed", 42),
        val_ratio=checkpoint.get("val_ratio", 0.1),
        test_ratio=checkpoint.get("test_ratio", 0.1),
        verify_images=not args.skip_verify,
        image_size=checkpoint.get("image_size", 256),
        transform_profile=checkpoint.get("transform_profile", "farl"),
    )
    if args.limit_test_samples is not None:
        datasets["test"] = Subset(
            datasets["test"], range(min(args.limit_test_samples, len(datasets["test"])))
        )
    loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    metrics = MetricAccumulator()
    rows = []
    routed_balanced = 0
    print(
        f"Device: {device} | mode: {args.mode} | tta: {args.tta} "
        f"| checkpoints: {len(models)} | samples: {len(datasets['test'])}"
    )
    print(f"Dataset scan: {scan_summary}")
    with torch.inference_mode():
        for images, ages, _, paths in tqdm(loader, desc="test-farl", ascii=True):
            images = images.to(device, non_blocking=True)
            ages = ages.to(device, dtype=torch.float32, non_blocking=True)
            probabilities_list = []
            selected = None
            for model_index, model in enumerate(models):
                _, probabilities, model_selected = predict_with_flip_routing(
                    model,
                    images,
                    mode=args.mode,
                    local_blend=local_blend,
                    tta=args.tta,
                    crop_size=args.tta_crop_size,
                )
                probabilities_list.append(probabilities)
                if model_index == 0:
                    selected = model_selected
            probabilities = torch.stack(probabilities_list, dim=0).mean(dim=0)
            age_values = torch.arange(
                probabilities.size(1), device=probabilities.device, dtype=probabilities.dtype
            )
            raw_predictions = torch.sum(probabilities * age_values.unsqueeze(0), dim=1)
            predictions = calibrator.apply_tensor(raw_predictions) if calibrator is not None else raw_predictions
            metrics.update(predictions, ages)
            if selected is not None:
                routed_balanced += int(selected.sum().item())
            confidences = probabilities.max(dim=1).values
            for path, true_age, pred_age, raw_pred_age, confidence in zip(
                paths,
                ages.cpu().tolist(),
                predictions.cpu().tolist(),
                raw_predictions.cpu().tolist(),
                confidences.cpu().tolist(),
            ):
                rows.append(
                    {
                        "image_path": path,
                        "true_age": int(true_age),
                        "pred_age": f"{pred_age:.4f}",
                        "raw_pred_age": f"{raw_pred_age:.4f}",
                        "rounded_pred_age": int(round(pred_age)),
                        "error": f"{abs(pred_age - true_age):.4f}",
                        "confidence": f"{confidence:.6f}",
                    }
                )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = metrics.compute()
    predictions = [float(row["pred_age"]) for row in rows]
    targets = [int(row["true_age"]) for row in rows]
    result["CMAE"] = cumulative_class_mae(predictions, targets)
    result["CS@5"] = sum(abs(pred - age) <= 5 for pred, age in zip(predictions, targets)) / len(rows)
    result["balanced_route_ratio"] = routed_balanced / len(rows) if args.mode == "route" else 0.0

    grouped = {}
    for start_age in range(0, 101, 10):
        end_age = min(100, start_age + 9)
        selected_rows = [row for row in rows if start_age <= int(row["true_age"]) <= end_age]
        if not selected_rows:
            continue
        errors = [float(row["error"]) for row in selected_rows]
        grouped[f"{start_age:02d}-{end_age:02d}"] = {
            "count": len(errors),
            "MAE": sum(errors) / len(errors),
            "Acc@1": sum(error <= 1 for error in errors) / len(errors),
            "Acc@3": sum(error <= 3 for error in errors) / len(errors),
        }

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "overall": result,
                "by_age_group": grouped,
                "mode": args.mode,
                "tta": args.tta,
                "checkpoint": args.checkpoint,
                "calibration": args.calibration,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Test MAE {result['MAE']:.4f} | CMAE {result['CMAE']:.4f} "
        f"| Acc@1 {result['Acc@1']:.3%} | Acc@3 {result['Acc@3']:.3%} "
        f"| CS@5 {result['CS@5']:.3%}"
    )
    if args.mode == "route":
        print(f"Balanced head selected for {result['balanced_route_ratio']:.2%} of samples.")
    print("Age-group MAE:", {key: round(value["MAE"], 3) for key, value in grouped.items()})
    print(f"Predictions: {output_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
