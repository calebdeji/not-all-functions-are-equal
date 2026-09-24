#!/usr/bin/env bash
# run_temporal_evaluation.sh
#
# Evaluates all models on the temporal test split (2019-2021 samples,
# all families regardless of training overlap) for the NDSS 2026
# temporal evaluation section.
#
# Requires: run_temporal_training.sh to have completed first.
#
# Usage:
#   bash run_temporal_evaluation.sh
#   bash run_temporal_evaluation.sh 1    # 1-shot only
#   bash run_temporal_evaluation.sh 5    # 5-shot only

set -euo pipefail

CONFIG="trial_config_v3_higraph_temporal"
N_EPISODES=500
SEED=42
SHOTS=${1:-"1 5"}

echo "========================================"
echo "[temporal eval] config: ${CONFIG}"
echo "========================================"

python -m src.eval.episodic_evaluate \
    --config     "$CONFIG" \
    --n_episodes "$N_EPISODES" \
    --n_shots    $SHOTS \
    --seed       "$SEED"

echo ""
echo "=== Temporal evaluation complete: ${CONFIG} ==="
