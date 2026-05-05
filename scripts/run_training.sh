#!/usr/bin/env bash
# run_training.sh
# Full training pipeline in dependency order:
#   0. Generate splits                (generate_splits.py — must run first)
#   1. Graph-level instance backbone  (train_whole_graph_backbone.py — NT-Xent on graph embeddings only)
#   2. Instance + set-level backbone  (train_back_bone.py — NT-Xent on graph + set-level node features)
#   3. Episodic heads                 (self_attention, sdl_degree, sdl_complexity) — requires step 2
#   4. FAMCF encoders                 (permission, api_call, opcode)
#   5. AndroFIM                       (DINO pretrain → fine-tune)
#
# Prerequisites for step 5 — run once before this script:
#   bash run_extract_features.sh --csv <csv> --apk_dir <dir> --name higraph --stage androfim
#
# Usage:
#   bash run_training.sh                   # full pipeline
#   bash run_training.sh splits
#   bash run_training.sh instance_backbone
#   bash run_training.sh full_backbone
#   bash run_training.sh episodic
#   bash run_training.sh famcf
#   bash run_training.sh androfim
#   bash run_training.sh fcgat

set -euo pipefail

# ─── Config ───────────────────────────────────────────────────────────────────
STAGE=${1:-all}
CONFIG="trial_config_v3_higraph_no_overlap"
DATASET="higraph"
GPUS=1
WORKERS=4

MANIFEST="dataset/${DATASET}/androfim/androFIM_manifest.csv"

# ─── 0. Generate splits ───────────────────────────────────────────────────────
run_splits() {
    echo "========================================"
    echo "[0] Generate splits  [${CONFIG}]"
    echo "========================================"
    python src/generate_splits.py \
        --config "$CONFIG"
    echo "--- Done ---"
}

# ─── 1. Graph-level instance backbone ────────────────────────────────────────
run_instance_backbone() {
    echo "========================================"
    echo "[1] Graph-level instance backbone  [${CONFIG}]"
    echo "========================================"
    python src/train_whole_graph_backbone.py \
        --config       "$CONFIG"
    echo "--- Done ---"
}

# ─── 2. Instance + set-level backbone ─────────────────────────────────────────
run_full_backbone() {
    echo "========================================"
    echo "[2] Instance + set-level backbone  [${CONFIG}]"
    echo "========================================"
    python src/train_back_bone.py \
        --config       "$CONFIG"
    echo "--- Done ---"
}

# ─── 3. Episodic heads ────────────────────────────────────────────────────────
run_episodic() {
    for VARIANT in self_attention sdl_degree sdl_complexity; do
        echo "========================================"
        echo "[3] Episodic — ${VARIANT}  [${CONFIG}]"
        echo "========================================"
        python src/episodic_train.py \
            --config       "$CONFIG" \
            --variant      "$VARIANT" \
            --seed         42
        echo "--- Done: ${VARIANT} ---"
    done
}


# ─── 4. FAMCF encoders ────────────────────────────────────────────────────────
run_famcf() {
    for FT in permission api_call opcode; do
        echo "========================================"
        echo "[4] FAMCF — ${FT}  [${CONFIG}]"
        echo "========================================"
        python src/train_famcf.py \
            --config       "$CONFIG" \
            --data_dir     "dataset/${DATASET}" \
            --feature_type "$FT"
        echo "--- Done: ${FT} ---"
    done
}

# ─── 5. AndroFIM (DINO pretrain + finetune) ───────────────────────────────────
run_androfim() {
    echo "========================================"
    echo "[5a] AndroFIM DINO pretrain  [${CONFIG}]"
    echo "========================================"
    python -m torch.distributed.launch \
        --nproc_per_node="$GPUS" \
        src/main_dino.py \
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
    echo "--- Pretrain done ---"

    echo "========================================"
    echo "[5b] AndroFIM fine-tune  [${CONFIG}]"
    echo "========================================"
    python src/finetune_androfim.py \
        --dataset     "$DATASET" \
        --manifest    "$MANIFEST" \
        --config      "$CONFIG" \
        --epochs      40 \
        --batch_size  64 \
        --lr          0.03 \
        --dropout     0.1 \
        --mixup_alpha 2.0 \
        --workers     "$WORKERS"
    echo "--- Finetune done ---"
}

# ─── 6.  Adapted FCGAT training (TODO) ────────────────────────────────────────────────────────
run_fcgat() {
    echo "========================================"
    echo "[6] Adapted FCGAT  [${CONFIG}]"
    echo "========================================"
    python src/train_fcgat.py \
        --config       "$CONFIG"
    echo "--- Done ---"
}

# ─── Dispatch ─────────────────────────────────────────────────────────────────
case "$STAGE" in
    all)
        run_splits
        run_instance_backbone
        run_full_backbone
        run_episodic
        run_famcf
        run_androfim
        run_fcgat
        ;;
    splits)            run_splits ;;
    instance_backbone) run_instance_backbone ;;
    full_backbone)     run_full_backbone ;;
    episodic)          run_episodic ;;
    famcf)             run_famcf ;;
    androfim)          run_androfim ;;
    fcgat)             run_fcgat ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 [all|instance_backbone|full_backbone|episodic|famcf|androfim|fcgat]"
        exit 1
        ;;
esac

echo ""
echo "=== Training complete: stage=${STAGE}  config=${CONFIG} ==="
