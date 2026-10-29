#!/usr/bin/env bash
# Trains GMT, GMPool, and TopK pooling episodic heads on the HiGraph no-overlap split.
# Backbone must already be trained before running this script.
#
# USAGE:
#   bash run_graph_pooling_training.sh              # sequential
#   bash run_graph_pooling_training.sh --parallel   # all three in parallel
#   bash run_graph_pooling_training.sh reset        # clear progress and exit
#
# RESUME SUPPORT (sequential only):
#   Completed steps are logged in PROGRESS_FILE. Re-running skips finished steps.

set -euo pipefail

CONFIG="trial_config_v3_higraph_no_overlap"
SEED=42
PROGRESS_FILE="graph_pooling_training_progress.txt"
VARIANTS=("gmt" "gmpool" "topk_pool" "diff_pool")
PARALLEL=false

for arg in "$@"; do
    case "$arg" in
        --parallel) PARALLEL=true ;;
        reset)
            rm -f "$PROGRESS_FILE"
            echo "Progress file cleared."
            exit 0
            ;;
    esac
done

# ---- Sequential helpers ----
is_done()   { grep -qxF "$1" "$PROGRESS_FILE" 2>/dev/null; }
mark_done() { echo "$1" >> "$PROGRESS_FILE"; }

run_step() {
    local VARIANT=$1
    local KEY="${CONFIG}:${VARIANT}"
    if is_done "$KEY"; then
        echo "--- [skip] ${VARIANT} already done ---"
        return 0
    fi
    echo "========================================"
    echo "  Variant: ${VARIANT}"
    echo "  Config : ${CONFIG}"
    echo "========================================"
    python -m src.train.episodic_train \
        --config  "$CONFIG" \
        --variant "$VARIANT" \
        --seed    "$SEED"
    mark_done "$KEY"
    echo "--- Done: ${VARIANT} ---"
}

# ---- Parallel helpers ----
run_parallel() {
    local VARIANT=$1
    local LOG="graph_pooling_${VARIANT}.log"
    echo "  Launching ${VARIANT} → ${LOG}"
    python -m src.train.episodic_train \
        --config  "$CONFIG" \
        --variant "$VARIANT" \
        --seed    "$SEED" \
        > "$LOG" 2>&1 &
}

# ---- Main ----
if $PARALLEL; then
    echo "=== Parallel mode: launching all variants ==="
    PIDS=()
    for VARIANT in "${VARIANTS[@]}"; do
        run_parallel "$VARIANT"
        PIDS+=("$!:$VARIANT")
    done

    echo "  PIDs: ${PIDS[*]}"
    echo "  Waiting for all jobs to finish..."

    EXIT_CODE=0
    for ENTRY in "${PIDS[@]}"; do
        PID="${ENTRY%%:*}"
        VARIANT="${ENTRY##*:}"
        if wait "$PID"; then
            echo "  [done] ${VARIANT}"
        else
            echo "  [FAILED] ${VARIANT} (exit $?) — see graph_pooling_${VARIANT}.log"
            EXIT_CODE=1
        fi
    done

    echo ""
    echo "=== Graph pooling training complete (parallel) ==="
    exit $EXIT_CODE
else
    for VARIANT in "${VARIANTS[@]}"; do
        run_step "$VARIANT"
    done
    echo ""
    echo "=== Graph pooling training complete ==="
fi
