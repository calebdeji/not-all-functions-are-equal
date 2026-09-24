"""
count_placeholder_apks.py

Counts HiGraph APKs that use the placeholder node (a single node with a zero
feature vector) for functions whose CFG could not be recovered.

Placeholder condition (mirrors src/parse_dataset.py, HiGraphDataset):
  a function index i is a placeholder iff str(i) is NOT a key in the APK's
  `control_flow_edges` map. In that case parse_dataset represents the function's
  CFG as `torch.zeros(1, node_dim - 1)` with type feature 0 and complexity 0.

Placeholders arise for functions with no recoverable intra-procedural CFG:
system/library API calls, and Androguard-incomplete extractions on obfuscated
or packed code (paper §III-A).

Reads raw JSONs directly (no torch / PyG), so it is fast and dependency-light.

Usage:
  python -m src.analysis.count_placeholder_apks
  python -m src.analysis.count_placeholder_apks --graph_root dataset/higraph/graph
  python -m src.analysis.count_placeholder_apks --years 2019 2020 2021 --out_csv placeholder_per_apk.csv
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def cfg_map(json_data):
    """Return the control_flow_edges as a dict (handles dict or JSON-string)."""
    cfe = json_data.get('control_flow_edges', {})
    if isinstance(cfe, dict):
        return cfe
    try:
        return json.loads(cfe)
    except (json.JSONDecodeError, TypeError):
        return {}


def scan_apk(path):
    """Return per-APK placeholder stats, or None if the JSON is unusable/empty."""
    try:
        with open(path) as f:
            j = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    fc = j.get('function_count', 0)
    if not fc:                      # empty graph — parse_dataset skips these
        return None

    m = cfg_map(j)
    with_cfg = sum(1 for i in range(fc) if str(i) in m)
    placeholders = fc - with_cfg

    return {
        'hash': j.get('hash', os.path.basename(path)),
        'year': str(j.get('year', '')),
        'family': j.get('family', ''),
        'function_count': fc,
        'placeholder_count': placeholders,
        'placeholder_frac': placeholders / fc,
    }


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def summarize(rows, label):
    n_apk = len(rows)
    if n_apk == 0:
        print(f"  [{label}] no usable APKs")
        return
    with_ph = sum(1 for r in rows if r['placeholder_count'] > 0)
    all_ph = sum(1 for r in rows if r['placeholder_count'] == r['function_count'])
    none_ph = sum(1 for r in rows if r['placeholder_count'] == 0)
    tot_funcs = sum(r['function_count'] for r in rows)
    tot_ph = sum(r['placeholder_count'] for r in rows)
    fracs = sorted(r['placeholder_frac'] for r in rows)

    def q(p):
        if not fracs:
            return 0.0
        idx = min(len(fracs) - 1, int(p * (len(fracs) - 1)))
        return fracs[idx]

    mean_frac = sum(fracs) / len(fracs)

    print(f"  [{label}]")
    print(f"    APKs                                : {n_apk}")
    print(f"    APKs with >=1 placeholder function  : {with_ph}  ({pct(with_ph, n_apk):.2f}%)")
    print(f"    APKs with NO placeholder function   : {none_ph}  ({pct(none_ph, n_apk):.2f}%)")
    print(f"    APKs entirely placeholder (no CFG)  : {all_ph}  ({pct(all_ph, n_apk):.2f}%)")
    print(f"    Functions total                     : {tot_funcs:,}")
    print(f"    Placeholder functions total         : {tot_ph:,}  ({pct(tot_ph, tot_funcs):.2f}%)")
    print(f"    Placeholder fraction per APK        : "
          f"mean {mean_frac:.3f}  median {q(0.5):.3f}  "
          f"p90 {q(0.9):.3f}  max {fracs[-1]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_root', default='dataset/higraph/graph',
                        help='Root containing {year}/raw/*.json')
    parser.add_argument('--years', nargs='+', default=None,
                        help='Optional list of years to include (default: all)')
    parser.add_argument('--out_csv', default=None,
                        help='Optional path to write per-APK placeholder counts')
    args = parser.parse_args()

    year_dirs = sorted(glob.glob(os.path.join(args.graph_root, '*')))
    year_dirs = [d for d in year_dirs if os.path.isdir(os.path.join(d, 'raw'))]
    if args.years:
        wanted = set(args.years)
        year_dirs = [d for d in year_dirs if os.path.basename(d) in wanted]

    if not year_dirs:
        print(f"No year/raw directories found under {args.graph_root}")
        return

    all_rows = []
    per_year = defaultdict(list)
    skipped = 0

    for ydir in year_dirs:
        year = os.path.basename(ydir)
        for path in glob.glob(os.path.join(ydir, 'raw', '*.json')):
            row = scan_apk(path)
            if row is None:
                skipped += 1
                continue
            row['year'] = row['year'] or year
            all_rows.append(row)
            per_year[year].append(row)

    print(f"Scanned {len(all_rows)} APKs across {len(year_dirs)} year(s); "
          f"skipped {skipped} empty/unreadable.\n")

    print("=== Per year ===")
    for year in sorted(per_year):
        summarize(per_year[year], year)
    print("\n=== Overall ===")
    summarize(all_rows, 'ALL')

    if args.out_csv:
        with open(args.out_csv, 'w') as f:
            f.write('hash,year,family,function_count,placeholder_count,placeholder_frac\n')
            for r in all_rows:
                f.write(f"{r['hash']},{r['year']},{r['family']},"
                        f"{r['function_count']},{r['placeholder_count']},"
                        f"{r['placeholder_frac']:.6f}\n")
        print(f"\nPer-APK CSV written → {args.out_csv}")


if __name__ == '__main__':
    main()
