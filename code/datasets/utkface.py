from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

from utils.label_distribution import make_gaussian_label


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class FaceRecord:
    path: str
    age: int


def parse_age(path: Path, max_age: int = 100) -> int | None:
    try:
        age = int(path.name.split("_", maxsplit=1)[0])
    except (ValueError, IndexError):
        return None
    return age if 0 <= age <= max_age else None


def _image_is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, UnidentifiedImageError):
        return False


def _dataset_signature(paths: Iterable[Path]) -> dict[str, int]:
    count = 0
    total_size = 0
    newest_mtime_ns = 0
    for path in paths:
        stat = path.stat()
        count += 1
        total_size += stat.st_size
        newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)
    return {
        "count": count,
        "total_size": total_size,
        "newest_mtime_ns": newest_mtime_ns,
    }


def scan_utkface(
    data_dir: str | Path,
    max_age: int = 100,
    verify_images: bool = True,
    use_cache: bool = True,
) -> tuple[list[FaceRecord], dict[str, int]]:
    """Scan UTKFace once, validate labels/images, and cache the resulting index."""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"UTKFace directory does not exist: {root}. "
            "Pass the folder containing files such as 25_0_2_*.jpg."
        )

    image_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No supported images were found under: {root}")

    signature = _dataset_signature(image_paths)
    cache_path = root / ".utkface_index.json"
    if use_cache and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cache.get("signature") == signature
                and cache.get("max_age") == max_age
                and cache.get("verified") >= verify_images
            ):
                records = [FaceRecord(**item) for item in cache["records"]]
                return records, cache["summary"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    records: list[FaceRecord] = []
    invalid_name_or_age = 0
    unreadable = 0
    for path in image_paths:
        age = parse_age(path, max_age=max_age)
        if age is None:
            invalid_name_or_age += 1
            continue
        if verify_images and not _image_is_valid(path):
            unreadable += 1
            continue
        records.append(FaceRecord(path=str(path), age=age))

    if not records:
        raise RuntimeError(f"All images under {root} were filtered out.")

    summary = {
        "total_images": len(image_paths),
        "valid_images": len(records),
        "invalid_name_or_age": invalid_name_or_age,
        "unreadable_images": unreadable,
    }
    if use_cache:
        payload = {
            "signature": signature,
            "max_age": max_age,
            "verified": verify_images,
            "summary": summary,
            "records": [record.__dict__ for record in records],
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return records, summary


def split_records(
    records: list[FaceRecord],
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[FaceRecord]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    test_size = int(len(shuffled) * test_ratio)
    val_size = int(len(shuffled) * val_ratio)
    train_end = len(shuffled) - val_size - test_size
    val_end = len(shuffled) - test_size
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def build_transforms(train: bool, image_size: int = 224, profile: str = "default"):
    if profile == "farl":
        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.80, 1.0),
                        ratio=(0.95, 1.05),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.01)],
                        p=0.5,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(CLIP_MEAN, CLIP_STD),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    if profile != "default":
        raise ValueError(f"Unknown transform profile: {profile}")
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(
                    degrees=7,
                    translate=(0.04, 0.04),
                    scale=(0.94, 1.06),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.03),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.12, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class UTKFaceDataset(Dataset):
    def __init__(self, records: list[FaceRecord], transform, sigma: float = 1.0, num_ages: int = 101):
        if not records:
            raise ValueError("UTKFaceDataset received an empty record list.")
        self.records = records
        self.transform = transform
        self.sigma = sigma
        self.num_ages = num_ages

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(record.path) as image:
                image = image.convert("RGB")
                image_tensor = self.transform(image)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Failed to load image: {record.path}") from exc

        soft_label = make_gaussian_label(record.age, num_ages=self.num_ages, sigma=self.sigma)
        return image_tensor, float(record.age), soft_label, record.path


def build_datasets(
    data_dir: str | Path,
    sigma: float = 1.0,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    verify_images: bool = True,
    image_size: int = 224,
    transform_profile: str = "default",
):
    records, summary = scan_utkface(data_dir, verify_images=verify_images)
    splits = split_records(records, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    datasets = {
        "train": UTKFaceDataset(
            splits["train"], build_transforms(True, image_size, transform_profile), sigma=sigma
        ),
        "val": UTKFaceDataset(
            splits["val"], build_transforms(False, image_size, transform_profile), sigma=sigma
        ),
        "test": UTKFaceDataset(
            splits["test"], build_transforms(False, image_size, transform_profile), sigma=sigma
        ),
    }
    return datasets, summary
