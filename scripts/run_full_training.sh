#!/usr/bin/env bash
# run_full_training.sh
#
# Master training script — trains everything in dependency order for HiGraph no-overlap.
#
#   Phase 1 — Backbones (must complete before any episodic head)
#     1.  Whole-graph backbone          (instance-level NT-Xent only)
#     2.  Full backbone                 (instance + set-level NT-Xent)
#     3.  Flat-FCG backbone             (no CFG hierarchy, full pretraining)
#     4.  Set-only backbone             (set-level NT-Xent only)
#     5.  Instance-only backbone        (instance-level NT-Xent, hierarchical)
#
#   Phase 2 — Standalone models
#     6.  FCGAT
#     7.  FAMCF — permission
#     8.  FAMCF — api_call
#     9.  FAMCF — opcode
#    10.  AndroFIM DINO pretrain
#    11.  AndroFIM fine-tune
#
#   Phase 3 — Episodic heads on main backbone
#    12.  sdl_degree
#    13.  sdl_complexity
#    14.  sdl_pagerank
#    15.  self_attention
#    17.  gmt
#    18.  gmpool
#    19.  topk_pool
#    20.  diff_pool
#    21.  sdl_no_learnable_param
#    22.  sdl_complexity_prior
#    23.  sdl_with_intra_param
#    24.  sdl_without_structure
#
#   Phase 4 — Episodic heads on ablation backbones
#    26.  flat_fcg    + sdl_complexity
#    28.  set_only    + sdl_complexity
#    30.  instance_only + sdl_complexity
#    32.  whole_graph + sdl_complexity
#
# TOTAL: 32 steps
#
# RESUME SUPPORT:
#   Completed steps are logged in PROGRESS_FILE. Re-running skips finished steps.
#   To reset:  bash run_full_training.sh reset
#
# USAGE:
#   bash run_full_training.sh       # run all 32 steps (resumable)
#   bash run_full_training.sh reset # wipe progress and exit

set -euo pipefail

CONFIG="trial_config_v3_higraph_no_overlap"
DATASET="higraph"
SEED=42
GPUS=1
WORKERS=4
PROGRESS_FILE="full_training_progress.txt"
MANIFEST="dataset/${DATASET}/androfim/androFIM_manifest.csv"

TOTAL_STEPS=32
_step=0

# ---- Resume helpers ----
is_done()   { grep -qxF "$1" "$PROGRESS_FILE" 2>/dev/null; }
mark_done() { echo "$1" >> "$PROGRESS_FILE"; }

if [[ "${1:-}" == "reset" ]]; then
    rm -f "$PROGRESS_FILE"
    echo "Progress file cleared. Run the script again to start fresh."
    exit 0
fi

# ---- Step runner ----
# Usage: run_step <key> <description> <command...>
run_step() {
    local KEY=$1
    local DESC=$2
    shift 2
    _step=$(( _step + 1 ))

    if is_done "$KEY"; then
        echo "--- [skip ${_step}/${TOTAL_STEPS}] ${DESC} ---"
        return 0
    fi

    local T0
    T0=$(date +%s)
    echo ""
    echo "========================================"
    echo "  [${_step}/${TOTAL_STEPS}] ${DESC}"
    echo "  Start: $(date '+%H:%M:%S')"
    echo "========================================"

    "$@"

    local T1
    T1=$(date +%s)
    local ELAPSED=$(( T1 - T0 ))
    mark_done "$KEY"
    echo "--- Done [${_step}/${TOTAL_STEPS}]: ${DESC}  ($(printf '%dm%02ds' $(( ELAPSED / 60 )) $(( ELAPSED % 60 )))) ---"
}

# ========================================================================
# Phase 1 — Backbones
# ========================================================================

run_step "whole_graph_backbone" \
    "Whole-graph backbone (instance-level NT-Xent)" \
    python -m src.pretrain.train_whole_graph_backbone --config "$CONFIG"

run_step "full_backbone" \
    "Full backbone (instance + set-level NT-Xent)" \
    python -m src.pretrain.train_back_bone --config "$CONFIG"

run_step "flat_fcg_backbone" \
    "Flat-FCG backbone (no CFG hierarchy, full pretraining)" \
    python -m src.pretrain.train_back_bone --config "$CONFIG" --flat_fcg

run_step "set_only_backbone" \
    "Set-only backbone (set-level NT-Xent only)" \
    python -m src.pretrain.train_back_bone --config "$CONFIG" --set_only

run_step "instance_only_backbone" \
    "Instance-only backbone (instance-level NT-Xent, hierarchical)" \
    python -m src.pretrain.train_back_bone --config "$CONFIG" --instance_only

# ========================================================================
# Phase 2 — Standalone models
# ========================================================================

run_step "fcgat" \
    "FCGAT" \
    python -m src.train.train_fcgat --config "$CONFIG"

run_step "famcf_permission" \
    "FAMCF — permission" \
    python -m src.train.train_famcf --config "$CONFIG" --data_dir "dataset/${DATASET}" --feature_type permission

run_step "famcf_api_call" \
    "FAMCF — api_call" \
    python -m src.train.train_famcf --config "$CONFIG" --data_dir "dataset/${DATASET}" --feature_type api_call

run_step "famcf_opcode" \
    "FAMCF — opcode" \
    python -m src.train.train_famcf --config "$CONFIG" --data_dir "dataset/${DATASET}" --feature_type opcode

run_step "androfim_pretrain" \
    "AndroFIM DINO pretrain" \
    python -m torch.distributed.launch \
        --nproc_per_node="$GPUS" \
        src/pretrain/main_dino.py \
        --arch                resnet12 \
        --manifest            "$MANIFEST" \
        --config              "$CONFIG" \
        --epochs              100 \
        --batch_size_per_gpu  64 \
        --optimizer           sgd \
        --lr                  0.03 \
        --global_crops_scale  0.4 1.0 \
        --local_crops_number  6 \
        --local_crops_scale   0.05 0.4 \
        --num_workers         "$WORKERS" \
        --saveckp_freq        20

run_step "androfim_finetune" \
    "AndroFIM fine-tune" \
    python -m src.train.finetune_androfim \
        --dataset     "$DATASET" \
        --manifest    "$MANIFEST" \
        --config      "$CONFIG" \
        --epochs      40 \
        --batch_size  64 \
        --lr          0.03 \
        --dropout     0.1 \
        --mixup_alpha 2.0 \
        --workers     "$WORKERS"

# ========================================================================
# Phase 3 — Episodic heads on main backbone
# ========================================================================
# gmt gmpool topk_pool diff_pool \
for VARIANT in \
    sdl_degree sdl_complexity sdl_pagerank self_attention \
    gmt gmpool topk_pool \
    sdl_no_learnable_param sdl_complexity_prior sdl_without_structure
do
    run_step "episodic_${VARIANT}" \
        "Episodic head: ${VARIANT}" \
        python -m src.train.episodic_train --config "$CONFIG" --variant "$VARIANT" --seed "$SEED"
done

# ========================================================================
# Phase 4 — Episodic heads on ablation backbones
# ========================================================================

for PREFIX in flat_fcg set_only instance_only whole_graph; do
    for VARIANT in sdl_complexity; do
        run_step "${PREFIX}_${VARIANT}" \
            "Ablation episodic: ${PREFIX} + ${VARIANT}" \
            python -m src.train.episodic_train --config "$CONFIG" --variant "$VARIANT" --seed "$SEED" --"${PREFIX}"
    done
done

echo ""
echo "========================================"
echo "=== Full training complete (${TOTAL_STEPS} steps) ==="
echo "=== Config: ${CONFIG} ==="
echo "========================================"
echo "Next: bash run_full_evaluation.sh"
