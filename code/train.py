from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from datasets.utkface import build_datasets
from models.farl_age_model import FaRLAgeEstimator, load_farl_visual_weights
from utils.farl import farl_age_loss, farl_local_age_loss, predict_with_flip_routing
from utils.metrics import MetricAccumulator
from utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a FaRL ViT age estimator with GLAE-style balanced routing."
    )
    parser.add_argument("--data_dir", default="./data/UTKFace")
    parser.add_argument(
        "--weights",
        default="./pretrained/FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
        help="Official FaRL checkpoint; ignored when --resume is used",
    )
    parser.add_argument("--save_dir", default="./checkpoints/farl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--balanced_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--balanced_lr", type=float, default=2e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-6)
    parser.add_argument("--unfreeze_blocks", type=int, default=0, choices=range(0, 13))
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=3,
        help="Train only the new MLP head before fine-tuning the selected final ViT blocks",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mae_weight", type=float, default=0.0)
    parser.add_argument("--distribution_weight", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--local_weight", type=float, default=0.15)
    parser.add_argument("--local_blend", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=0.005)
    parser.add_argument("--balance_bin_size", type=int, default=5)
    parser.add_argument("--balance_strength", type=float, default=0.75)
    parser.add_argument("--max_balance_weight", type=float, default=10.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init_age_checkpoint", default=None, help="Warm-start age heads from a previous FaRL run")
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--limit_train_samples", type=int, default=None)
    parser.add_argument("--limit_val_samples", type=int, default=None)
    return parser.parse_args()


def make_loader(dataset, batch_size, shuffle, workers, pin_memory, sampler=None, seed=42):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def dataset_ages(dataset) -> list[int]:
    if isinstance(dataset, Subset):
        return [dataset.dataset.records[index].age for index in dataset.indices]
    return [record.age for record in dataset.records]


def make_balanced_sampler(dataset, bin_size, strength, max_weight, seed):
    ages = dataset_ages(dataset)
    bins = [age // bin_size for age in ages]
    counts = Counter(bins)
    largest = max(counts.values())
    weights = [min(max_weight, (largest / counts[age_bin]) ** strength) for age_bin in bins]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return sampler, (min(weights), max(weights), len(counts))


def run_train_epoch(model, loader, optimizer, scheduler, scaler, device, args, phase):
    model.train()
    metrics = MetricAccumulator()
    total_loss = 0.0
    sample_count = 0
    use_amp = device.type == "cuda" and not args.no_amp
    head = "vanilla" if phase == "vanilla" else "balanced"
    freeze_features = phase == "balanced"
    use_local = args.local_weight > 0

    progress = tqdm(loader, desc=f"train-{phase}", leave=False, ascii=True)
    for images, ages, soft_labels, _ in progress:
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, dtype=torch.float32, non_blocking=True)
        soft_labels = soft_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if freeze_features:
                with torch.no_grad():
                    if use_local:
                        features, patch_features = model.encode_with_tokens(images)
                    else:
                        features = model.encode(images)
                        patch_features = None
                logits = model.logits_from_features(features, head)
                local_logits = model.local_logits_from_tokens(patch_features, head) if use_local else None
            else:
                if use_local:
                    logits, local_logits = model(images, head=head, return_local=True)
                else:
                    logits = model(images, head=head)
                    local_logits = None
            loss, predictions, _ = farl_age_loss(
                logits,
                ages,
                soft_labels,
                mae_weight=args.mae_weight,
                distribution_weight=args.distribution_weight,
                label_smoothing=args.label_smoothing,
            )
            if local_logits is not None:
                local_loss, _, _ = farl_local_age_loss(
                    local_logits,
                    ages,
                    soft_labels,
                    mae_weight=args.mae_weight,
                    distribution_weight=args.distribution_weight,
                    label_smoothing=args.label_smoothing,
                )
                loss = loss + args.local_weight * local_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if not use_amp or scaler.get_scale() >= previous_scale:
            scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.detach().item() * batch_size
        sample_count += batch_size
        metrics.update(predictions, ages)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    result = metrics.compute()
    result["loss"] = total_loss / sample_count
    return result


@torch.inference_mode()
def run_validation(model, loader, device, mode, local_blend):
    model.eval()
    metrics = MetricAccumulator()
    balanced_count = 0
    sample_count = 0
    for images, ages, _, _ in tqdm(loader, desc=f"val-{mode}", leave=False, ascii=True):
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, dtype=torch.float32, non_blocking=True)
        predictions, _, selected = predict_with_flip_routing(
            model, images, mode=mode, local_blend=local_blend
        )
        metrics.update(predictions, ages)
        if selected is not None:
            balanced_count += int(selected.sum().item())
            sample_count += selected.numel()
    result = metrics.compute()
    result["balanced_route_ratio"] = balanced_count / sample_count if sample_count else 0.0
    return result


def checkpoint_payload(model, args, phase, epoch, best_mae, optimizer=None, scheduler=None, scaler=None):
    return {
        "model_type": "farl_vit_b16_glae",
        "model_state_dict": model.state_dict(),
        "phase": phase,
        "epoch": epoch,
        "best_val_mae": best_mae,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "image_size": args.image_size,
        "sigma": args.sigma,
        "split_seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "transform_profile": "farl",
        "adaptive_routing": phase == "balanced",
        "local_weight": args.local_weight,
        "local_blend": args.local_blend,
        "config": vars(args),
    }


def write_history(path: Path, history: list[dict]):
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def make_stage_one_optimizer(model, args, steps_per_epoch):
    model.unfreeze_last_blocks(args.unfreeze_blocks)
    model.vanilla_head.requires_grad_(True)
    model.vanilla_local_head.requires_grad_(True)
    model.balanced_head.requires_grad_(False)
    model.balanced_local_head.requires_grad_(False)
    groups = [
        {
            "params": list(model.vanilla_head.parameters()) + list(model.vanilla_local_head.parameters()),
            "lr": args.lr,
        }
    ]
    max_lrs = [args.lr]
    trainable_encoder = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    if trainable_encoder:
        groups.append({"params": trainable_encoder, "lr": args.backbone_lr})
        max_lrs.append(args.backbone_lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    return optimizer, scheduler


def configure_stage_one_epoch(model, args, epoch):
    model.vanilla_head.requires_grad_(True)
    model.vanilla_local_head.requires_grad_(True)
    model.balanced_head.requires_grad_(False)
    model.balanced_local_head.requires_grad_(False)
    if args.unfreeze_blocks == 0 or epoch <= args.freeze_epochs:
        model.freeze_encoder()
    else:
        model.unfreeze_last_blocks(args.unfreeze_blocks)


def make_stage_two_optimizer(model, args, steps_per_epoch):
    model.requires_grad_(False)
    model.balanced_head.requires_grad_(True)
    model.balanced_local_head.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        list(model.balanced_head.parameters()) + list(model.balanced_local_head.parameters()),
        lr=args.balanced_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.balanced_lr,
        epochs=args.balanced_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    return optimizer, scheduler


def restore_training_state(checkpoint, optimizer, scheduler, scaler):
    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])


def load_compatible_age_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()
    compatible = {
        name: value
        for name, value in source_state.items()
        if not name.startswith("encoder.") and name in target_state and target_state[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(target_state)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    datasets, scan_summary = build_datasets(
        args.data_dir,
        sigma=args.sigma,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        verify_images=not args.skip_verify,
        image_size=args.image_size,
        transform_profile="farl",
    )
    if args.limit_train_samples is not None:
        datasets["train"] = Subset(
            datasets["train"], range(min(args.limit_train_samples, len(datasets["train"])))
        )
    if args.limit_val_samples is not None:
        datasets["val"] = Subset(
            datasets["val"], range(min(args.limit_val_samples, len(datasets["val"])))
        )

    train_loader = make_loader(
        datasets["train"], args.batch_size, True, args.num_workers, device.type == "cuda", seed=args.seed
    )
    val_loader = make_loader(
        datasets["val"], args.batch_size, False, args.num_workers, device.type == "cuda", seed=args.seed
    )
    balanced_sampler, balance_info = make_balanced_sampler(
        datasets["train"],
        args.balance_bin_size,
        args.balance_strength,
        args.max_balance_weight,
        args.seed,
    )
    balanced_loader = make_loader(
        datasets["train"],
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
        sampler=balanced_sampler,
        seed=args.seed,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = FaRLAgeEstimator().to(device)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if resume_checkpoint.get("model_type") != "farl_vit_b16_glae":
            raise RuntimeError("--resume is not a FaRL-GLAE checkpoint.")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=False)
        print(f"Resuming {resume_checkpoint['phase']} phase from: {args.resume}")
    else:
        load_info = load_farl_visual_weights(model, args.weights)
        print(f"Loaded official FaRL visual tower: {load_info}")
        if args.init_age_checkpoint:
            loaded, total = load_compatible_age_checkpoint(model, args.init_age_checkpoint, device)
            print(f"Warm-started {loaded}/{total} compatible tensors from: {args.init_age_checkpoint}")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Dataset scan: {scan_summary}")
    print("Split sizes:", {name: len(dataset) for name, dataset in datasets.items()})
    print(
        f"Balanced sampler: {balance_info[2]} bins, weights "
        f"{balance_info[0]:.2f}-{balance_info[1]:.2f}"
    )

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    started_at = time.time()
    resume_phase = resume_checkpoint.get("phase") if resume_checkpoint else None

    if resume_phase != "balanced":
        optimizer, scheduler = make_stage_one_optimizer(model, args, len(train_loader))
        start_epoch = 1
        best_mae = float("inf")
        no_improve = 0
        if resume_checkpoint is not None:
            start_epoch = int(resume_checkpoint["epoch"]) + 1
            best_mae = float(resume_checkpoint["best_val_mae"])
            restore_training_state(resume_checkpoint, optimizer, scheduler, scaler)

        for epoch in range(start_epoch, args.epochs + 1):
            configure_stage_one_epoch(model, args, epoch)
            train_result = run_train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, args, "vanilla"
            )
            val_result = run_validation(model, val_loader, device, "vanilla", args.local_blend)
            improved = val_result["MAE"] < best_mae - args.min_delta
            if improved:
                best_mae = val_result["MAE"]
                no_improve = 0
                torch.save(
                    checkpoint_payload(model, args, "vanilla", epoch, best_mae),
                    save_dir / "best_vanilla.pth",
                )
            else:
                no_improve += 1
            torch.save(
                checkpoint_payload(model, args, "vanilla", epoch, best_mae, optimizer, scheduler, scaler),
                save_dir / "last_model.pth",
            )
            row = {
                "phase": "vanilla",
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_MAE": train_result["MAE"],
                "val_MAE": val_result["MAE"],
                "val_Acc@1": val_result["Acc@1"],
                "val_Acc@3": val_result["Acc@3"],
                "balanced_route_ratio": 0.0,
            }
            history.append(row)
            write_history(save_dir / "training_log.csv", history)
            print(
                f"Vanilla {epoch:03d}/{args.epochs} | loss {train_result['loss']:.4f} "
                f"| val MAE {val_result['MAE']:.3f} | Acc@1 {val_result['Acc@1']:.2%} "
                f"| Acc@3 {val_result['Acc@3']:.2%} | no-improve {no_improve}"
            )
            if no_improve >= args.early_stopping_patience:
                print("Early stopping vanilla phase.")
                break

        best_vanilla = torch.load(save_dir / "best_vanilla.pth", map_location=device, weights_only=False)
        model.load_state_dict(best_vanilla["model_state_dict"])
        model.reset_balanced_head()
        best_route_mae = best_vanilla["best_val_mae"]
        torch.save(
            checkpoint_payload(model, args, "balanced", 0, best_route_mae),
            save_dir / "best_model.pth",
        )
        balanced_start_epoch = 1
    else:
        best_route_mae = float(resume_checkpoint["best_val_mae"])
        balanced_start_epoch = int(resume_checkpoint["epoch"]) + 1

    if args.balanced_epochs > 0 and balanced_start_epoch <= args.balanced_epochs:
        optimizer, scheduler = make_stage_two_optimizer(model, args, len(balanced_loader))
        no_improve = 0
        if resume_phase == "balanced":
            restore_training_state(resume_checkpoint, optimizer, scheduler, scaler)
        for epoch in range(balanced_start_epoch, args.balanced_epochs + 1):
            train_result = run_train_epoch(
                model, balanced_loader, optimizer, scheduler, scaler, device, args, "balanced"
            )
            val_result = run_validation(model, val_loader, device, "route", args.local_blend)
            improved = val_result["MAE"] < best_route_mae - args.min_delta
            if improved:
                best_route_mae = val_result["MAE"]
                no_improve = 0
                torch.save(
                    checkpoint_payload(model, args, "balanced", epoch, best_route_mae),
                    save_dir / "best_model.pth",
                )
            else:
                no_improve += 1
            torch.save(
                checkpoint_payload(model, args, "balanced", epoch, best_route_mae, optimizer, scheduler, scaler),
                save_dir / "last_model.pth",
            )
            row = {
                "phase": "balanced",
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_MAE": train_result["MAE"],
                "val_MAE": val_result["MAE"],
                "val_Acc@1": val_result["Acc@1"],
                "val_Acc@3": val_result["Acc@3"],
                "balanced_route_ratio": val_result["balanced_route_ratio"],
            }
            history.append(row)
            write_history(save_dir / "training_log.csv", history)
            print(
                f"Balanced {epoch:03d}/{args.balanced_epochs} | loss {train_result['loss']:.4f} "
                f"| routed val MAE {val_result['MAE']:.3f} | Acc@1 {val_result['Acc@1']:.2%} "
                f"| Acc@3 {val_result['Acc@3']:.2%} "
                f"| routed-balanced {val_result['balanced_route_ratio']:.1%}"
            )
            if no_improve >= max(3, args.early_stopping_patience // 2):
                print("Early stopping balanced phase.")
                break

    elapsed_hours = (time.time() - started_at) / 3600
    print(f"Training complete in {elapsed_hours:.2f} h. Best routed val MAE: {best_route_mae:.3f}")
    print(f"Best checkpoint: {save_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
