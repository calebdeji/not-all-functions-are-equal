"""
prepare_novel_apks.py

Two-pass preparation for novel-family obfuscation evaluation:

  Pass 1 — Stage originals for obfuscation
    For each novel-family APK that is NOT yet fully obfuscated across all three
    methods, copy the original from raw_format/higraph/{year}/ into
    raw_format/higraph_novel/{year}/ so that obfuscate_higraph.sh processes it.
    APKs already fully obfuscated are skipped (nothing to do).

  Pass 2 — Collect already-obfuscated novel APKs
    For each novel-family APK that is already obfuscated (per method), copy it
    from obfuscated_higraph/{method}/{year}/ into
    obfuscated_higraph_novel/{method}/{year}/.
    This gives apk_to_higraph.py a clean, novel-only input folder per method.

Run order:
  1. python -m src.data.prepare_novel_apks          # stage + collect
  2. bash obfuscate_higraph.sh                     # obfuscate the staged originals
  3. python -m src.data.prepare_novel_apks --collect_only  # collect newly obfuscated
  4. python -m src.data.apk_to_higraph --apk_root obfuscated_higraph_novel/arithmetic_branch ...

Usage:
  python -m src.data.prepare_novel_apks              # full run (stage + collect)
  python -m src.data.prepare_novel_apks --dry_run    # preview without copying
  python -m src.data.prepare_novel_apks --collect_only  # skip staging, only collect
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.core.utils import Utils

PROJECT_ROOT    = Path(Utils.get_project_root())
SPLIT_FILE      = PROJECT_ROOT / "dataset_config_split" / "higraph" / "trial_config_v3" / "no_overlap_split.json"
TRUTH_DIR       = PROJECT_ROOT / "dataset_refined_truth" / "higraph"
APK_ROOT        = PROJECT_ROOT / "raw_format" / "higraph"
STAGE_ROOT      = PROJECT_ROOT / "raw_format" / "higraph_novel"
OBFUSCATED_ROOT = PROJECT_ROOT / "obfuscated_higraph"
COLLECT_ROOT    = PROJECT_ROOT / "obfuscated_higraph_novel"

METHODS = ["arithmetic_branch", "call_indirection", "reorder"]


def obfuscated_path(method: str, year: str, hash_val: str) -> Path:
    return OBFUSCATED_ROOT / method / year / f"{hash_val}.apk"


def all_obfuscated(hash_val: str, year: str) -> bool:
    return all(obfuscated_path(m, year, hash_val).exists() for m in METHODS)


def load_novel_hashes() -> dict:
    """Returns {year: [(hash, family), ...]} for novel families only."""
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    novel_families = set(split["novel"])
    print(f"Novel families ({len(novel_families)}): {sorted(novel_families)}\n")

    result = {}
    for csv_path in sorted(TRUTH_DIR.glob("labelled_samples_with_family_*.csv")):
        year = csv_path.stem.replace("labelled_samples_with_family_", "")
        df   = pd.read_csv(csv_path)
        rows = df[df["family"].str.lower().isin(novel_families)]
        if not rows.empty:
            result[year] = [
                (row["hash"].strip().lower(), row["family"].strip().lower())
                for _, row in rows.iterrows()
            ]
    return result


def pass1_stage(novel_hashes: dict, dry_run: bool):
    """Copy originals that still need obfuscation into the staging folder."""
    print("=== Pass 1: Staging originals for obfuscation ===")
    total_found = total_staged = total_done = total_missing = 0

    for year, samples in sorted(novel_hashes.items()):
        year_staged = year_done = year_missing = 0

        for hash_val, family in samples:
            total_found += 1

            if all_obfuscated(hash_val, year):
                year_done += 1
                continue

            src = APK_ROOT / year / f"{hash_val}.apk"
            dst = STAGE_ROOT / year / f"{hash_val}.apk"

            if not src.exists():
                year_missing += 1
                print(f"  [WARN] src missing: {year}/{hash_val}.apk ({family})")
                continue

            if not dst.exists():
                if dry_run:
                    print(f"  [dry]  stage {year}/{hash_val}.apk ({family})")
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

            year_staged += 1

        total_staged += year_staged
        total_done   += year_done
        total_missing += year_missing

        print(f"  [{year}] {len(samples)} samples — "
              f"{year_done} already obfuscated, "
              f"{year_staged} staged, "
              f"{year_missing} src missing")

    print()
    print(f"  Total: {total_staged} staged, {total_done} already done, {total_missing} missing")
    if total_staged > 0:
        print(f"  → Run: bash obfuscate_higraph.sh")
    print()


def pass2_collect(novel_hashes: dict, dry_run: bool):
    """Copy already-obfuscated novel APKs into obfuscated_higraph_novel/{method}/{year}/."""
    print("=== Pass 2: Collecting obfuscated novel APKs ===")

    for method in METHODS:
        total_collected = total_missing = 0
        print(f"\n  [{method}]")

        for year, samples in sorted(novel_hashes.items()):
            year_collected = year_missing = 0

            for hash_val, family in samples:
                src = obfuscated_path(method, year, hash_val)
                dst = COLLECT_ROOT / method / year / f"{hash_val}.apk"

                if not src.exists():
                    year_missing += 1
                    continue

                if not dst.exists():
                    if dry_run:
                        print(f"    [dry]  collect {method}/{year}/{hash_val}.apk")
                    else:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)

                year_collected += 1

            total_collected += year_collected
            total_missing   += year_missing
            print(f"    [{year}] {year_collected} collected, {year_missing} not yet obfuscated")

        print(f"  {method} total: {total_collected} collected, {total_missing} still missing")

    print()
    print(f"  Collected novel APKs → {COLLECT_ROOT}")
    print(f"  → Next: run apk_to_higraph.py --apk_root obfuscated_higraph_novel/<method>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview without copying anything")
    parser.add_argument("--collect_only", action="store_true",
                        help="Skip Pass 1 (staging), only run Pass 2 (collect obfuscated)")
    args = parser.parse_args()

    novel_hashes = load_novel_hashes()

    if not args.collect_only:
        pass1_stage(novel_hashes, args.dry_run)

    pass2_collect(novel_hashes, args.dry_run)


if __name__ == "__main__":
    main()
