#!/usr/bin/env bash
# run_obfuscation_famcf_androfim_extraction.sh
#
# Extracts FAMCF features and AndroFIM images from the obfuscated novel-family
# APKs for all three obfuscation methods.
#
# Prerequisites:
#   - obfuscated_higraph_novel/{method}/{year}/*.apk must exist
#     (run prepare_novel_apks.py + obfuscate_higraph.sh first)
#   - dataset/higraph/famcf/vocab.pkl must exist (clean corpus extraction done)
#
# Output per method:
#   dataset/obfuscated_famcf_{method}/features.pkl   (encoded with clean vocab)
#   dataset/obfuscated_androfim_{method}/             (images + manifest CSV)
#
# Usage:
#   bash run_obfuscation_famcf_androfim_extraction.sh               # all methods
#   bash run_obfuscation_famcf_androfim_extraction.sh arithmetic_branch

set -euo pipefail

METHODS=("arithmetic_branch" "call_indirection" "reorder")
CLEAN_VOCAB="dataset/higraph/famcf/vocab.pkl"
CSV_DIR="dataset_refined_truth/higraph"

FILTER=${1:-all}

if [ ! -f "$CLEAN_VOCAB" ]; then
    echo "ERROR: Clean vocab not found at $CLEAN_VOCAB"
    echo "       Run extract_famcf_features.py on the full corpus first."
    exit 1
fi

for METHOD in "${METHODS[@]}"; do
    if [[ "$FILTER" != "all" && "$FILTER" != "$METHOD" ]]; then
        continue
    fi

    APK_ROOT="obfuscated_higraph_novel/${METHOD}"
    FAMCF_OUT="dataset/obfuscated_famcf_${METHOD}"
    ANDROFIM_OUT="dataset/obfuscated_androfim_${METHOD}"

    if [ ! -d "$APK_ROOT" ]; then
        echo "[skip] ${METHOD} — ${APK_ROOT} not found"
        continue
    fi

    echo "========================================"
    echo "  Method    : ${METHOD}"
    echo "  APK root  : ${APK_ROOT}"
    echo "  FAMCF out : ${FAMCF_OUT}"
    echo "  FIM out   : ${ANDROFIM_OUT}"
    echo "  Start     : $(date '+%H:%M:%S')"
    echo "========================================"

    echo "--- FAMCF: stage 1 (extract raw features) ---"
    python -m src.data.extract_famcf_features \
        --apk_root  "$APK_ROOT" \
        --csv_dir   "$CSV_DIR" \
        --out_dir   "$FAMCF_OUT" \
        --stage     1

    echo "--- FAMCF: stage 2 (encode with clean vocab) ---"
    python -m src.data.extract_famcf_features \
        --apk_root      "$APK_ROOT" \
        --csv_dir       "$CSV_DIR" \
        --out_dir       "$FAMCF_OUT" \
        --stage         2 \
        --existing_vocab "$CLEAN_VOCAB"

    echo "--- AndroFIM: convert APKs to images ---"
    python -m src.data.apk_to_androFIM_image \
        --apk_root "$APK_ROOT" \
        --csv_dir  "$CSV_DIR" \
        --out_dir  "$ANDROFIM_OUT"

    echo "--- Done: ${METHOD}  ($(date '+%H:%M:%S')) ---"
    echo ""
done

echo "=== Obfuscated FAMCF + AndroFIM extraction complete ==="
echo "Next: run episodic_obfuscated_evaluate.py (it will pick up the new features)"
