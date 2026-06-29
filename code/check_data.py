from __future__ import annotations

import argparse
from collections import Counter

from datasets.utkface import scan_utkface, split_records


def parse_args():
    parser = argparse.ArgumentParser(description="Validate UTKFace files and report age statistics.")
    parser.add_argument("--data_dir", default="./data/UTKFace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_verify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    records, summary = scan_utkface(args.data_dir, verify_images=not args.skip_verify)
    splits = split_records(records, seed=args.seed)
    age_counts = Counter(record.age for record in records)
    print("Scan summary:", summary)
    print("Split sizes:", {name: len(items) for name, items in splits.items()})
    print(f"Age range: {min(age_counts)}-{max(age_counts)}")
    print("Most common ages:", age_counts.most_common(10))
    sparse = sorted(age for age, count in age_counts.items() if count < 10)
    print(f"Ages with fewer than 10 samples ({len(sparse)}): {sparse}")
    print("Dataset is ready for training.")


if __name__ == "__main__":
    main()
