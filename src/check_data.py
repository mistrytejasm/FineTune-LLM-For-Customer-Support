"""
check_data.py — Inspect the processed dataset before training.

Usage:
    python src/check_data.py

Shows: split sizes, one full example per split, length statistics,
and verifies zero placeholder contamination.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_from_disk
from config import DATA_PROCESSED, DATA_RAW


def main():
    print("=" * 65)
    print("LOADING PROCESSED DATASET (chat format for training):")
    print("=" * 65)
    ds = load_from_disk(DATA_PROCESSED)
    print(ds)
    print()

    for split in ["train", "validation", "test"]:
        print(f"--- {split.upper()} EXAMPLE ---")
        print(json.dumps(ds[split][0], indent=2, ensure_ascii=False))
        print()

    # Quick quality stats
    train = ds["train"]
    placeholder_count = sum(1 for ex in train if "{{" in ex["messages"][2]["content"])
    user_lens  = [len(ex["messages"][1]["content"]) for ex in train]
    asst_lens  = [len(ex["messages"][2]["content"]) for ex in train]

    print("=" * 65)
    print("DATASET HEALTH CHECK:")
    print("=" * 65)
    print(f"  Placeholder contamination : {placeholder_count} ({100*placeholder_count//len(train)}%)")
    print(f"  User msg  — min/avg/max chars: {min(user_lens)} / {sum(user_lens)//len(user_lens)} / {max(user_lens)}")
    print(f"  Asst resp — min/avg/max chars: {min(asst_lens)} / {sum(asst_lens)//len(asst_lens)} / {max(asst_lens)}")
    print(f"  Approx > 256 tokens: {sum(1 for a in asst_lens if a > 800)} examples")
    print()

    print("=" * 65)
    print("LOADING RAW DATASET (original columns for evaluation):")
    print("=" * 65)
    ds_raw = load_from_disk(DATA_RAW)
    print(ds_raw)
    print()
    print("RAW COLUMN NAMES:", ds_raw["train"].column_names)
    print("RAW EXAMPLE:", json.dumps({k: str(v)[:120] for k, v in ds_raw["train"][0].items()}, indent=2))


if __name__ == "__main__":
    main()
