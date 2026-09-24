#!/usr/bin/env bash
# run_temporal_training.sh
#
# Trains backbone + all episodic heads on the temporal split (train 2012-2018,
# test 2019-2021) for the NDSS 2026 temporal evaluation.
#
# Steps:
#   1.  backbone         — GraphCL pretraining (instance + set-level NT-Xent)
#                          Also serves as 'pretrained_backbone' at eval time.
#   2.  sdl_degree       — episodic SDL (degree) head
#   3.  sdl_complexity   — episodic SDL (complexity) head
#   4.  sdl_pagerank     — episodic SDL (pagerank) head
#   5.  self_attention   — episodic self-attention head
#   6.  gmt              — graph multiset transformer head
#   7.  gmpool           — graph multiset pooling head
#   8.  topk_pool        — top-k pooling head
#   9.  fcgat            — standalone FCGAT baseline
#   10. famcf            — flat-feature FAMCF baseline (3 feature types)
#   11. androfim_pretrain — DINO self-supervised pretrain on APK images
#   12. androfim_finetune — AndroFIM ResNet-12 fine-tune on base families
#
# RESUME SUPPORT:
#   Completed steps are recorded in PROGRESS_FILE. Re-running after an
#   interruption skips already-completed steps automatically.
#   To reset: bash run_temporal_training.sh reset
#
# Usage:
#   bash run_temporal_training.sh          # run all steps (resumable)
#   bash run_temporal_training.sh reset    # clear progress and exit

set -euo pipefail

CONFIG="trial_config_v3_higraph_temporal"
DATASET="higraph"
MANIFEST="dataset/${DATASET}/androfim/androFIM_manifest.csv"
GPUS=2
WORKERS=8
SEED=42
PROGRESS_FILE="temporal_training_progress.txt"

# ---- Resume helpers ----
is_done()   { grep -qxF "$1:$2" "$PROGRESS_FILE" 2>/dev/null; }
mark_done() { echo "$1:$2" >> "$PROGRESS_FILE"; }

if [[ "${1:-}" == "reset" ]]; then
    rm -f "$PROGRESS_FILE"
    echo "Progress file cleared. Run the script again to start fresh."
    exit 0
fi

run_step() {
    local STEP=$1
    shift
    if is_done "$CONFIG" "$STEP"; then
        echo "--- [skip] ${STEP} already done for ${CONFIG} ---"
        return 0
    fi
    echo "--- ${STEP} [${CONFIG}] ---"
    "$@"
    mark_done "$CONFIG" "$STEP"
    echo "--- Done: ${STEP} ---"
}

echo "========================================"
echo "[temporal] config: ${CONFIG}"
echo "========================================"

echo "--- Regenerating temporal split (force overwrite) ---"
python -m src.data.generate_splits --config "$CONFIG" --force

# 1. GraphCL backbone (also serves as pretrained_backbone at eval time)
run_step "backbone" \
    python -m src.pretrain.train_whole_graph_backbone --config "$CONFIG"

# 2-4. SDL episodic heads
# run_step "sdl_degree" \
#     python -m src.train.episodic_train --config "$CONFIG" --variant sdl_degree --seed "$SEED"

# run_step "sdl_complexity" \
#     python -m src.train.episodic_train --config "$CONFIG" --variant sdl_complexity --seed "$SEED"

# run_step "sdl_pagerank" \
#     python -m src.train.episodic_train --config "$CONFIG" --variant sdl_pagerank --seed "$SEED"

# # 5-8. Graph pooling baselines (BASELINE_REGISTRY)
# run_step "self_attention" \
#     python -m src.train.episodic_train --config "$CONFIG" --variant self_attention --seed "$SEED"

# run_step "gmt" \
#     python -m src.train.episodic_train --config "$CONFIG" --variant gmt --seed "$SEED"

# run_step "gmpool" \
#     python -m src.train.episodic_train --config "$CONFIG" --variant gmpool --seed "$SEED"

# run_step "topk_pool" \
#     python -m src.train.episodic_train --config "$CONFIG" --variant topk_pool --seed "$SEED"

# # 9. FCGAT
# run_step "fcgat" \
#     python -m src.train.train_fcgat --config "$CONFIG"

# # 10. FAMCF (three feature types)
# for FT in permission api_call opcode; do
#     run_step "famcf_${FT}" \
#         python -m src.train.train_famcf --config "$CONFIG" --data_dir "dataset/${DATASET}" --feature_type "$FT"
# done

# # 11. AndroFIM DINO pretrain
# run_step "androfim_pretrain" \
#     python -m torch.distributed.launch \
#         --nproc_per_node="$GPUS" \
#         src/pretrain/main_dino.py \
#         --arch                resnet12 \
#         --manifest            "$MANIFEST" \
#         --config              "$CONFIG" \
#         --epochs              100 \
#         --batch_size_per_gpu  64 \
#         --optimizer           sgd \
#         --lr                  0.03 \
#         --global_crops_scale  0.4 1.0 \
#         --local_crops_number  6 \
#         --local_crops_scale   0.05 0.4 \
#         --num_workers         "$WORKERS" \
#         --saveckp_freq        20

# # 12. AndroFIM fine-tune (uses checkpoint.pth saved by step 11)
# run_step "androfim_finetune" \
#     python -m src.train.finetune_androfim \
#         --dataset     "$DATASET" \
#         --manifest    "$MANIFEST" \
#         --config      "$CONFIG" \
#         --epochs      40 \
#         --batch_size  64 \
#         --lr          0.03 \
#         --dropout     0.1 \
#         --mixup_alpha 2.0 \
#         --workers     "$WORKERS"

echo ""
echo "=== Temporal training complete: ${CONFIG} ==="
echo "Note: 'pretrained_backbone' at eval time reuses the backbone checkpoint from step 1."
echo "Next: bash run_temporal_evaluation.sh"
