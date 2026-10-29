#!/usr/bin/env bash
# run_rebuttal_ablation_training.sh
#
# Trains all granular component ablation models for the NDSS 2026 rebuttal.
# Answers Reviewer #3235A Weakness 2: isolates contribution of CFG hierarchy,
# pretraining losses, and SDL weighting. See ablation_plans.txt for full design.
#
# GROUP A — Architecture/Pretraining ablation (frozen, no episodic fine-tuning):
#   A1 — Flat FCG    + full pretrain  + mean pool (frozen)  [backbone trained here]
#   A2 — Hierarchical + instance only + mean pool (frozen)  [whole_graph_backbone — EXISTS]
#   A3 — Hierarchical + set only      + mean pool (frozen)  [set_only_backbone — trained here]
#   A4 — Hierarchical + full pretrain + mean pool (frozen)  [pretrained_backbone — EXISTS]
#
# GROUP B — Episodic fine-tuning built on top of each Group A backbone:
#   B_A1_sdl  — Flat FCG    + full pretrain + SDL FT        [trained here]
#   B_A2_sdl  — Hier + instance only + SDL FT               [trained here]
#   B_A3_sdl  — Hier + set only + SDL FT                    [trained here]
#   B_A4_sdl  — Hier + full pretrain + SDL FT               [sdl_complexity — EXISTS]
#
# Currently implemented modules: A1, A3, A_instance_only, B_A1_sdl, B_A2_sdl, B_A3_sdl,
# B_instance_only_sdl.  (mean-pool fine-tuned heads are intentionally not included.)
#
# RESUME SUPPORT:
#   Completed steps are recorded in PROGRESS_FILE. Re-running the script after
#   an interruption will skip already-completed steps automatically.
#   To reset progress and start fresh: bash scripts/run_rebuttal_ablation_training.sh reset
#
# Usage:
#   bash run_rebuttal_ablation_training.sh          # all configs (resumable)
#   bash run_rebuttal_ablation_training.sh higraph  # single config keyword (resumable)
#   bash scripts/run_rebuttal_ablation_training.sh reset    # wipe progress file and exit

set -euo pipefail

CONFIGS=(
    "trial_config_v3_higraph_no_overlap"
    "trial_config_v3_maldroid_no_overlap"
    "trial_config_v3_kronodroid_no_overlap"
)

SEED=42
FILTER=${1:-all}
PROGRESS_FILE="rebuttal_ablation_progress.txt"

# ---- Resume helpers ----
is_done()   { grep -qxF "$1:$2" "$PROGRESS_FILE" 2>/dev/null; }
mark_done() { echo "$1:$2" >> "$PROGRESS_FILE"; }

if [[ "$FILTER" == "reset" ]]; then
    rm -f "$PROGRESS_FILE"
    echo "Progress file cleared. Run the script again to start fresh."
    exit 0
fi

run_step() {
    local CONFIG=$1
    local STEP=$2
    shift 2
    if is_done "$CONFIG" "$STEP"; then
        echo "--- [skip] ${STEP} already done for ${CONFIG} ---"
        return 0
    fi
    echo "--- ${STEP} [${CONFIG}] ---"
    "$@"
    mark_done "$CONFIG" "$STEP"
    echo "--- Done: ${STEP} ---"
}

run_config() {
    local CONFIG=$1

    echo "========================================"
    echo "[rebuttal ablation] config: ${CONFIG}"
    echo "========================================"

    run_step "$CONFIG" "flat_fcg_backbone" \
        python -m src.pretrain.train_back_bone --config "$CONFIG" --flat_fcg

    run_step "$CONFIG" "set_only_backbone" \
        python -m src.pretrain.train_back_bone --config "$CONFIG" --set_only


    run_step "$CONFIG" "flat_fcg_sdl_complexity" \
        python -m src.train.episodic_train --config "$CONFIG" --variant sdl_complexity --seed "$SEED" --flat_fcg


    run_step "$CONFIG" "whole_graph_sdl_complexity" \
        python -m src.train.episodic_train --config "$CONFIG" --variant sdl_complexity --seed "$SEED" --whole_graph


    run_step "$CONFIG" "set_only_sdl_complexity" \
        python -m src.train.episodic_train --config "$CONFIG" --variant sdl_complexity --seed "$SEED" --set_only

    run_step "$CONFIG" "instance_only_backbone" \
        python -m src.pretrain.train_back_bone --config "$CONFIG" --instance_only


    run_step "$CONFIG" "instance_only_sdl_complexity" \
        python -m src.train.episodic_train --config "$CONFIG" --variant sdl_complexity --seed "$SEED" --instance_only

    echo "========================================"
    echo "[rebuttal ablation] finished: ${CONFIG}"
    echo "========================================"
}

for CONFIG in "${CONFIGS[@]}"; do
    if [[ "$FILTER" == "all" ]] || [[ "$CONFIG" == *"$FILTER"* ]]; then
        run_config "$CONFIG"
    fi
done

echo ""
echo "=== Rebuttal ablation training complete ==="
echo "Next: wire new variants into episodic_evaluate.py, then run evaluation."
