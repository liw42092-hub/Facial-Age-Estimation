from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.utkface import build_datasets
from models.farl_age_model import FaRLAgeEstimator
from utils.calibration import AgeCalibrator
from utils.farl import cumulative_class_mae, predict_with_flip_routing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run final FaRL age-estimation inference and generate report figures."
    )
    parser.add_argument("--data_dir", default="./data/UTKFace")
    parser.add_argument("--checkpoint", default="./model/best_model.pth")
    parser.add_argument("--output_dir", default="./results/final")
    parser.add_argument("--split", choices=("val", "test", "both"), default="test")
    parser.add_argument("--mode", choices=("route", "vanilla", "balanced"), default="route")
    parser.add_argument("--tta", choices=("none", "flip", "five_crop", "ten_crop"), default="flip")
    parser.add_argument("--tta_crop_size", type=int, default=224)
    parser.add_argument("--local_blend", type=float, default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--training_log", default="./model/training_log.csv")
    return parser.parse_args()


def load_model(checkpoint_path: str | Path, device: torch.device):
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


def compute_metrics(predictions: list[float], targets: list[int]) -> dict[str, float]:
    errors = [abs(pred - target) for pred, target in zip(predictions, targets)]
    squared = [(pred - target) ** 2 for pred, target in zip(predictions, targets)]
    thresholds = {f"CS@{limit}": sum(error <= limit for error in errors) / len(errors) for limit in range(1, 11)}
    return {
        "count": len(errors),
        "MAE": sum(errors) / len(errors),
        "RMSE": (sum(squared) / len(squared)) ** 0.5,
        "MedianAE": sorted(errors)[len(errors) // 2],
        "Acc@1": thresholds["CS@1"],
        "Acc@3": thresholds["CS@3"],
        "CS@5": thresholds["CS@5"],
        "CMAE": cumulative_class_mae(predictions, targets),
        **thresholds,
    }


def group_metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped = {}
    for start_age in range(0, 101, 10):
        end_age = min(100, start_age + 9)
        selected = [row for row in rows if start_age <= int(row["true_age"]) <= end_age]
        if not selected:
            continue
        errors = [float(row["error"]) for row in selected]
        grouped[f"{start_age:02d}-{end_age:02d}"] = {
            "count": len(errors),
            "MAE": sum(errors) / len(errors),
            "Acc@1": sum(error <= 1 for error in errors) / len(errors),
            "Acc@3": sum(error <= 3 for error in errors) / len(errors),
            "CS@5": sum(error <= 5 for error in errors) / len(errors),
        }
    return grouped


@torch.inference_mode()
def run_split(model, dataset, split_name: str, args, checkpoint, device, output_dir: Path, calibrator):
    if args.limit_samples is not None:
        dataset = Subset(dataset, range(min(args.limit_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    if args.local_blend is None:
        local_blend = checkpoint.get("local_blend")
        if local_blend is None:
            local_blend = 0.2 if checkpoint_has_local_heads(checkpoint) else 0.0
    else:
        local_blend = args.local_blend
    rows = []
    routed_balanced = 0
    routed_total = 0
    for images, ages, _, paths in tqdm(loader, desc=f"infer-{split_name}", ascii=True):
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, dtype=torch.float32, non_blocking=True)
        raw_predictions, probabilities, selected = predict_with_flip_routing(
            model,
            images,
            mode=args.mode,
            local_blend=local_blend,
            tta=args.tta,
            crop_size=args.tta_crop_size,
        )
        predictions = calibrator.apply_tensor(raw_predictions) if calibrator is not None else raw_predictions
        if selected is not None:
            routed_balanced += int(selected.sum().item())
            routed_total += selected.numel()
        confidences = probabilities.max(dim=1).values
        for index, (path, true_age, pred_age, raw_pred_age, confidence) in enumerate(
            zip(paths, ages.cpu().tolist(), predictions.cpu().tolist(), raw_predictions.cpu().tolist(), confidences.cpu().tolist())
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
                    "routed_head": "balanced" if selected is not None and bool(selected[index].item()) else "vanilla",
                }
            )

    csv_path = output_dir / f"{split_name}_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    predictions = [float(row["pred_age"]) for row in rows]
    targets = [int(row["true_age"]) for row in rows]
    metrics = compute_metrics(predictions, targets)
    metrics["balanced_route_ratio"] = routed_balanced / routed_total if routed_total else 0.0
    grouped = group_metrics(rows)
    metrics_path = output_dir / f"{split_name}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "overall": metrics,
                "by_age_group": grouped,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "data_dir": str(Path(args.data_dir).resolve()),
                "split": split_name,
                "mode": args.mode,
                "tta": args.tta,
                "local_blend": local_blend,
                "calibration": args.calibration,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_split_figures(rows, grouped, split_name, output_dir)
    return metrics, csv_path, metrics_path


def make_split_figures(rows: list[dict], grouped: dict[str, dict[str, float]], split_name: str, output_dir: Path):
    true_ages = [int(row["true_age"]) for row in rows]
    pred_ages = [float(row["pred_age"]) for row in rows]
    errors = [float(row["error"]) for row in rows]

    plt.figure(figsize=(6, 6))
    plt.scatter(true_ages, pred_ages, s=8, alpha=0.35)
    plt.plot([0, 100], [0, 100], color="black", linewidth=1)
    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    plt.title(f"{split_name}: predicted vs true age")
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_pred_vs_true.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=40, color="#4C78A8", alpha=0.9)
    plt.xlabel("Absolute error")
    plt.ylabel("Count")
    plt.title(f"{split_name}: absolute error distribution")
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_error_histogram.png", dpi=200)
    plt.close()

    labels = list(grouped.keys())
    mae_values = [grouped[label]["MAE"] for label in labels]
    counts = [grouped[label]["count"] for label in labels]
    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(labels, mae_values, color="#59A14F")
    plt.xlabel("Age group")
    plt.ylabel("MAE")
    plt.title(f"{split_name}: MAE by age group")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_age_group_mae.png", dpi=200)
    plt.close()

    thresholds = list(range(1, 16))
    cs_values = [sum(error <= threshold for error in errors) / len(errors) for threshold in thresholds]
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, cs_values, marker="o", color="#E15759")
    plt.xlabel("Error threshold (years)")
    plt.ylabel("Cumulative score")
    plt.ylim(0.0, 1.0)
    plt.title(f"{split_name}: cumulative score curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_cumulative_score.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.hist(true_ages, bins=range(0, 102, 2), alpha=0.55, label="True age", color="#4C78A8")
    plt.hist(pred_ages, bins=range(0, 102, 2), alpha=0.55, label="Predicted age", color="#F28E2B")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title(f"{split_name}: age distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_age_distribution.png", dpi=200)
    plt.close()


def make_training_curve(training_log: str | Path, output_dir: Path):
    path = Path(training_log)
    if not path.is_file():
        return None
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["global_epoch"] = len(rows) + 1
            rows.append(row)
    if not rows:
        return None

    epochs = [row["global_epoch"] for row in rows]
    val_mae = [float(row["val_MAE"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    acc3 = [float(row["val_Acc@3"]) for row in rows]
    phases = [row["phase"] for row in rows]

    plt.figure(figsize=(9, 5))
    axis = plt.gca()
    axis.plot(epochs, val_mae, marker="o", label="Val MAE", color="#4C78A8")
    axis.set_xlabel("Logged epoch")
    axis.set_ylabel("Val MAE")
    axis.grid(alpha=0.25)
    twin = axis.twinx()
    twin.plot(epochs, train_loss, marker="s", label="Train loss", color="#F28E2B", alpha=0.75)
    twin.set_ylabel("Train loss")
    for idx, phase in enumerate(phases[1:], start=1):
        if phase != phases[idx - 1]:
            axis.axvline(idx + 0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
            axis.text(idx + 0.55, min(val_mae), phase, fontsize=9, va="bottom")
    lines, labels = axis.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    axis.legend(lines + twin_lines, labels + twin_labels, loc="upper right")
    plt.title("Training curve")
    plt.tight_layout()
    curve_path = output_dir / "training_curve.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, acc3, marker="o", color="#59A14F")
    plt.xlabel("Logged epoch")
    plt.ylabel("Val Acc@3")
    plt.title("Validation Acc@3 curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    acc_path = output_dir / "training_acc3_curve.png"
    plt.savefig(acc_path, dpi=200)
    plt.close()
    return curve_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_model(args.checkpoint, device)
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

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Dataset scan: {scan_summary}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")

    selected_splits = ["val", "test"] if args.split == "both" else [args.split]
    summary = {}
    for split_name in selected_splits:
        metrics, csv_path, metrics_path = run_split(
            model, datasets[split_name], split_name, args, checkpoint, device, output_dir, calibrator
        )
        summary[split_name] = metrics
        print(
            f"{split_name} MAE {metrics['MAE']:.4f} | RMSE {metrics['RMSE']:.4f} "
            f"| Acc@1 {metrics['Acc@1']:.3%} | Acc@3 {metrics['Acc@3']:.3%} "
            f"| CS@5 {metrics['CS@5']:.3%}"
        )
        print(f"{split_name} predictions: {csv_path}")
        print(f"{split_name} metrics: {metrics_path}")

    curve_path = make_training_curve(args.training_log, output_dir)
    if curve_path is not None:
        print(f"Training curve: {curve_path}")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
