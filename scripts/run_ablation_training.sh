#!/usr/bin/env bash
# run_ablation_training.sh
#
# Trains the four SDL ablation variants using the same config/dataset as
# run_training.sh. Requires the backbone to already be trained (step 2 of
# run_training.sh must have completed first).
#
# Variants:
#   sdl_no_learnable_param  — σ(sim_in − sim_out)              0 params
#   sdl_complexity_prior    — σ(norm_complexity)               0 params
#   sdl_without_structure   — σ(λa·sim_in − λb·sim_out)        2 params
#   sdl_pagerank            — σ(λa·sim_in + λb·PR − sim_out)   2 params
#
# Also trains sdl_complexity on the instance-only backbone, for the granular
# component ablation (see scripts/run_rebuttal_ablation_training.sh).
#
# Usage:
#   bash scripts/run_ablation_training.sh                      # all variants
#   bash scripts/run_ablation_training.sh sdl_complexity_prior # single variant

set -euo pipefail

CONFIGS=("trial_config_v3_higraph_no_overlap" "trial_config_v3_maldroid_no_overlap" "trial_config_v3_kronodroid_no_overlap" )


SEED=42
STAGE=${1:-all}

ABLATION_VARIANTS=(
    sdl_no_learnable_param
    sdl_complexity_prior
    sdl_without_structure
    sdl_pagerank
)

INSTANCE_ONLY_VARIANTS=(
    sdl_complexity
)

run_variant() {
    local VARIANT=$1
    local CONFIG=$2
    local EXTRA_FLAG=${3:-}
    echo "========================================"
    echo "[ablation] ${VARIANT}  [${CONFIG}]  ${EXTRA_FLAG}"
    echo "========================================"
    python -m src.train.episodic_train \
        --config  "$CONFIG" \
        --variant "$VARIANT" \
        --seed    "$SEED" \
        $EXTRA_FLAG
    echo "--- Done: ${VARIANT} ---"
}

for CONFIG in "${CONFIGS[@]}"; do
  if [[ "$STAGE" == "all" ]]; then
    for VARIANT in "${ABLATION_VARIANTS[@]}"; do
        run_variant "$VARIANT" "$CONFIG"
    done
    for VARIANT in "${INSTANCE_ONLY_VARIANTS[@]}"; do
        run_variant "$VARIANT" "$CONFIG" "--instance_only"
    done
  else
    run_variant "$STAGE" "$CONFIG"
  fi
done

echo ""
echo "=== Ablation training complete across all configs ==="
