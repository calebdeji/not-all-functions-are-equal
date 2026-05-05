"""
download_apks.py

Downloads HiGraph APKs from AndroZoo, organised by year.

Output structure:
    raw_format/higraph/{year}/{sha256}.apk

Log file:
    raw_format/higraph/download.log

Skips already-downloaded files so the script is safe to resume.

Usage:
    python src/download_apks.py --api_key YOUR_KEY
    python src/download_apks.py --api_key YOUR_KEY --year 2017       # single year
    python src/download_apks.py --api_key YOUR_KEY --workers 4       # parallel downloads
    python src/download_apks.py --api_key YOUR_KEY --delay 1.0       # 1s between requests
"""

import argparse
import csv
import glob
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from utils import Utils as _Utils

ANDROZOO_URL = "https://androzoo.uni.lu/api/download"
HIGRAPH_DIR  = _Utils.get_refined_truth_root('higraph')
OUTPUT_DIR   = _Utils.get_raw_apk_root('higraph')


def setup_logger():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "download.log")
    logger = logging.getLogger("download_apks")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_hashes(year=None):
    pattern = (
        os.path.join(HIGRAPH_DIR, f"higraph_labelled_with_family_index_{year}.csv")
        if year
        else os.path.join(HIGRAPH_DIR, "higraph_labelled_with_family_index_*.csv")
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found matching: {pattern}")

    hashes = {}
    for fpath in files:
        yr = os.path.basename(fpath) \
               .replace("higraph_labelled_with_family_index_", "") \
               .replace(".csv", "")
        with open(fpath) as f:
            for row in csv.DictReader(f):
                sha256 = row["hash"].strip().lower()
                family = row["family"].strip()
                hashes[sha256] = (yr, family)

    return hashes


def download_apk(sha256, year, family, api_key, output_dir, retries=3, delay=0.0):
    """Download one APK. Returns (sha256, status, message)."""
    out_path = os.path.join(output_dir, year, f"{sha256}.apk")

    if delay > 0:
        time.sleep(delay)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return sha256, "skipped", "already exists"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                ANDROZOO_URL,
                params={"apikey": api_key, "sha256": sha256},
                stream=True,
                timeout=60,
            )
            if resp.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                size_kb = os.path.getsize(out_path) // 1024
                return sha256, "ok", f"{size_kb} KB"
            elif resp.status_code == 404:
                return sha256, "failed", "404 not found"
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == retries:
                return sha256, "failed", str(e)
            time.sleep(2 ** attempt)

    return sha256, "failed", "failed after retries"


def main(args):
    logger = setup_logger()

    hashes = load_hashes(args.year)
    total = len(hashes)

    already = sum(
        1 for sha256, (yr, _) in hashes.items()
        if os.path.exists(os.path.join(OUTPUT_DIR, yr, f"{sha256}.apk"))
        and os.path.getsize(os.path.join(OUTPUT_DIR, yr, f"{sha256}.apk")) > 0
    )

    logger.info(f"Total samples : {total}")
    logger.info(f"Already done  : {already}")
    logger.info(f"To download   : {total - already}")
    logger.info(f"Workers: {args.workers} | Delay: {args.delay}s | Output: {OUTPUT_DIR}/{{year}}/")

    by_year = defaultdict(list)
    for sha256, (yr, family) in hashes.items():
        by_year[yr].append((sha256, family))

    total_success = total_failed = total_skipped = 0

    for yr in sorted(by_year.keys()):
        year_samples = by_year[yr]
        year_success = year_failed = year_skip = 0

        logger.info(f"── Year {yr} | {len(year_samples)} samples ──────────────────────────")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_apk, sha256, yr, family, args.api_key, OUTPUT_DIR,
                    delay=args.delay
                ): (sha256, family)
                for sha256, family in year_samples
            }
            for future in as_completed(futures):
                sha256, family = futures[future]
                _, status, msg = future.result()

                if status == "skipped":
                    year_skip += 1
                    logger.debug(f"SKIP | {yr} | {family} | {sha256} | {msg}")
                elif status == "ok":
                    year_success += 1
                    logger.info(f"OK   | {yr} | {family} | {sha256} | {msg}")
                else:
                    year_failed += 1
                    logger.warning(f"FAIL | {yr} | {family} | {sha256} | {msg}")

        logger.info(
            f"Year {yr} done — downloaded: {year_success} | skipped: {year_skip} | failed: {year_failed}"
        )
        total_success += year_success
        total_failed  += year_failed
        total_skipped += year_skip

    logger.info("=" * 60)
    logger.info(f"Total downloaded : {total_success}")
    logger.info(f"Total skipped    : {total_skipped}")
    logger.info(f"Total failed     : {total_failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="AndroZoo API key")
    parser.add_argument("--year",    default=None,  help="Download a single year only")
    parser.add_argument("--workers", type=int,   default=2,   help="Parallel download threads (default 2)")
    parser.add_argument("--delay",   type=float, default=0.0, help="Seconds to wait before each download (default 0)")
    args = parser.parse_args()
    main(args)
