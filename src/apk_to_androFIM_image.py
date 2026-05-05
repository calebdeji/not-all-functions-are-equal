"""
apk_to_androFIM_image.py

Convert raw APKs to colour images for the AndroFIM few-shot baseline.

Pipeline (per APK):
  1. Unpack APK (ZIP) and extract classes.dex bytecode.
     Falls back to concatenating all multidex .dex files if classes.dex
     is absent.
  2. Read raw bytes; pad to multiple of 3 with zeros.
  3. Treat every 3 consecutive bytes as one RGB pixel (values already 0-255).
  4. Arrange pixels into a 2D image of fixed width IMAGE_WIDTH (256 px).
  5. Save as PNG to out_dir/{year}/{sha256}.png.

Manifest CSV (androFIM_manifest.csv) is written incrementally and maps
sha256 -> year, family, image_path for use by the dataset loader.

Output:
    dataset/higraph/androfim/{year}/{sha256}.png
    dataset/higraph/androfim/androFIM_manifest.csv
    processed_data/androFIM_images/conversion.log

Usage:
    python src/apk_to_androFIM_image.py
    python src/apk_to_androFIM_image.py --years 2019 2020 2021
    python src/apk_to_androFIM_image.py --workers 8
"""

import argparse
import csv
import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_WIDTH = 256  # fixed image width in pixels; height varies with file size


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("apk_to_image")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_hash_family_map(csv_dir: Path) -> dict:
    """Returns dict: sha256 -> (year_str, family_str)"""
    mapping = {}
    for csv_path in sorted(csv_dir.glob("labelled_samples_with_family_*.csv")):
        year = csv_path.stem.replace("labelled_samples_with_family_", "")
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                mapping[row["hash"].strip().lower()] = (year, row["family"].strip())
    return mapping


def apk_to_image(apk_path: Path, out_path: Path) -> tuple[bool, str]:
    """
    Extract Dalvik bytecode from APK, convert to an RGB image, save to out_path.
    Returns (True, "") on success or (False, reason) on failure.
    """
    try:
        with zipfile.ZipFile(apk_path, "r") as zf:
            names = set(zf.namelist())
            if "classes.dex" in names:
                raw_bytes = zf.read("classes.dex")
            else:
                # Multidex: concatenate all .dex files in order
                dex_files = sorted(n for n in names if n.endswith(".dex"))
                if not dex_files:
                    return False, "no .dex file found"
                raw_bytes = b"".join(zf.read(d) for d in dex_files)
    except zipfile.BadZipFile as e:
        return False, f"bad zip: {e}"
    except Exception as e:
        return False, str(e)

    if len(raw_bytes) < 3:
        return False, "dex too small (<3 bytes)"

    # Pad to a multiple of 3 so every byte maps cleanly to an RGB channel
    n = len(raw_bytes)
    pad = (-n) % 3
    if pad:
        raw_bytes = raw_bytes + b"\x00" * pad

    # (n_pixels, 3) — each row is one RGB pixel
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
    n_pixels = arr.shape[0]

    # Arrange into 2D image of fixed width; zero-pad the last row if needed
    n_rows = (n_pixels + IMAGE_WIDTH - 1) // IMAGE_WIDTH
    total = n_rows * IMAGE_WIDTH
    if total > n_pixels:
        arr = np.vstack([arr, np.zeros((total - n_pixels, 3), dtype=np.uint8)])
    arr = arr.reshape(n_rows, IMAGE_WIDTH, 3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_path, format="PNG")
    return True, ""


def process_one(sha, year, family, apk_root, out_dir):
    apk_path = apk_root / year / f"{sha}.apk"
    out_path  = out_dir  / year / f"{sha}.png"

    if out_path.exists():
        return sha, "skipped", year, family, str(out_path)
    if not apk_path.exists():
        return sha, "missing", year, family, ""

    ok, reason = apk_to_image(apk_path, out_path)
    if ok:
        return sha, "ok", year, family, str(out_path)
    return sha, f"failed: {reason}", year, family, ""


def main():
    parser = argparse.ArgumentParser(
        description="Convert HiGraph APKs to AndroFIM colour images"
    )
    parser.add_argument("--apk_root", default=None,
                        help="Root dir with year-subdirs of APKs. Defaults to raw_format/higraph.")
    parser.add_argument("--csv_dir",  default=None,
                        help="Dir with labelled_samples_with_family_*.csv. "
                             "Defaults to dataset_refined_truth/higraph.")
    parser.add_argument("--out_dir",  default=None,
                        help="Output directory for PNG images. Defaults to dataset/higraph/androfim.")
    parser.add_argument("--years", nargs="+", default=None,
                        help="Years to process (default: all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker threads (default: 4)")
    args = parser.parse_args()

    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from utils import Utils
    project_root = Path(__file__).resolve().parent.parent
    apk_root = Path(args.apk_root) if args.apk_root is not None \
               else Path(Utils.get_raw_apk_root('higraph'))
    csv_dir  = Path(args.csv_dir)  if args.csv_dir  is not None \
               else Path(Utils.get_refined_truth_root('higraph'))
    out_dir  = Path(args.out_dir)  if args.out_dir  is not None \
               else Path(Utils.get_androfim_root('higraph'))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(out_dir / "conversion.log"))

    hash_family = load_hash_family_map(csv_dir)
    logger.info(f"Loaded {len(hash_family)} hash->family mappings")

    if args.years:
        hashes = {sha: (yr, fam) for sha, (yr, fam) in hash_family.items()
                  if yr in set(args.years)}
    else:
        hashes = hash_family

    logger.info(f"APKs to process: {len(hashes)} | workers: {args.workers}")
    logger.info(f"Image width: {IMAGE_WIDTH}px | Output: {out_dir}")

    manifest_path = out_dir / "androFIM_manifest.csv"
    total = n_ok = n_skip = n_fail = n_missing = 0

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=["sha256", "year", "family", "image_path"])
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one, sha, yr, fam, apk_root, out_dir): sha
                for sha, (yr, fam) in hashes.items()
            }
            for future in as_completed(futures):
                sha, status, year, family, img_path = future.result()
                total += 1

                if status == "skipped":
                    n_skip += 1
                elif status == "ok":
                    n_ok += 1
                    logger.info(f"OK   {sha[:16]}  {year}  {family}")
                elif status == "missing":
                    n_missing += 1
                    logger.warning(f"MISS {sha[:16]}  {year}  — APK not found")
                else:
                    n_fail += 1
                    logger.warning(f"FAIL {sha[:16]}  {year}  {family}  — {status}")

                if img_path:
                    writer.writerow({
                        "sha256": sha, "year": year,
                        "family": family, "image_path": img_path,
                    })
                    mf.flush()

                if total % 500 == 0:
                    logger.info(
                        f"Progress {total}/{len(hashes)} | "
                        f"ok={n_ok}  skip={n_skip}  fail={n_fail}  missing={n_missing}"
                    )

    logger.info("=" * 60)
    logger.info(f"Total   : {total}")
    logger.info(f"OK      : {n_ok}")
    logger.info(f"Skipped : {n_skip}")
    logger.info(f"Failed  : {n_fail}")
    logger.info(f"Missing : {n_missing}")
    logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
