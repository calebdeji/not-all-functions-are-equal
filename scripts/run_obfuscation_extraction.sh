#!/usr/bin/env bash
# run_obfuscation_extraction.sh
#
# Extracts hierarchical graphs (FCG + CFGs) from obfuscated novel-family APKs
# for all three obfuscation methods using apk_to_higraph.py.
#
# Prerequisites:
#   1. python -m src.data.prepare_novel_apks          -- stage originals
#   2. bash obfuscate_higraph.sh                 -- obfuscate staged APKs
#   3. python -m src.data.prepare_novel_apks --collect_only  -- collect into novel folder
#
# Output:
#   dataset/obfuscated_higraph_arithmetic_branch/graph/{year}/raw/{hash}.json
#   dataset/obfuscated_higraph_call_indirection/graph/{year}/raw/{hash}.json
#   dataset/obfuscated_higraph_reorder/graph/{year}/raw/{hash}.json
#
# USAGE:
#   bash run_obfuscation_extraction.sh               # all three methods
#   bash run_obfuscation_extraction.sh arithmetic_branch  # single method

set -euo pipefail

METHODS=("arithmetic_branch" "call_indirection" "reorder")
PROCESSES=$(python3 -c "import multiprocessing; print(multiprocessing.cpu_count())")

FILTER=${1:-all}

for METHOD in "${METHODS[@]}"; do
    if [[ "$FILTER" != "all" && "$FILTER" != "$METHOD" ]]; then
        continue
    fi

    APK_ROOT="obfuscated_higraph_novel/${METHOD}"
    OUT_DIR="dataset/obfuscated_higraph_${METHOD}/graph"

    if [ ! -d "$APK_ROOT" ]; then
        echo "[skip] ${METHOD} — ${APK_ROOT} not found, run prepare_novel_apks.py first"
        continue
    fi

    echo "========================================"
    echo "  Method   : ${METHOD}"
    echo "  APK root : ${APK_ROOT}"
    echo "  Output   : ${OUT_DIR}"
    echo "  Start    : $(date '+%H:%M:%S')"
    echo "========================================"

    python -m src.data.apk_to_higraph \
        --dataset  higraph \
        --apk_root "$APK_ROOT" \
        --out_dir  "$OUT_DIR" \
        --processes "$PROCESSES"

    echo "--- Done: ${METHOD}  ($(date '+%H:%M:%S')) ---"
    echo ""
done

echo "=== Obfuscation graph extraction complete ==="
echo "Next: implement and run episodic_obfuscated_evaluate.py"
