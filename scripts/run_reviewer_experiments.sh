#!/usr/bin/env bash
#
# run_reviewer_experiments.sh
#
# Runs the three NDSS-revision experiments added for the reviewer response, in
# order. All three target HiGraph (no-overlap) and require the obfuscated HiGraph
# graphs (dataset/obfuscated_higraph_{method}/graph) to be present for steps 1
# and 3 — produced by obfuscate_higraph.sh + run_obfuscation_extraction.sh.
#
#   STEP 1 — Matched-function weight stability
#            src/analysis/measure_weight_stability.py
#            Clean-support / clean-vs-obf-query per-function SDL weights on the
#            SAME functions of the SAME APK. Reports Spearman, clean-anchored
#            top-k retention (ret@5/10), coverage. Output:
#            results/interpretability/weight_stability_{method}.csv + summary.
#
#   STEP 2 — Random-init control (no contrastive pretraining), TRAIN + EVAL
#            episodic_train --random_init_backbone  (SDL)
#            then episodic_evaluate.py --ablation_only, which loads
#            random_init_sdl_complexity alongside the
#            rest of the granular ablation ladder.
#
#   STEP 3 — Clean-support / obfuscated-query accuracy (evasion threat model)
#            src/robustness/episodic_obfuscated_query_evaluate.py
#            Δ = obf-query − clean, isolating the query-side obfuscation effect.
#
# RESUME: completed steps are recorded in PROGRESS_FILE; re-running skips them.
#   bash run_reviewer_experiments.sh          # run all (resumable)
#   bash run_reviewer_experiments.sh reset    # clear progress and exit

set -euo pipefail

CONFIG="trial_config_v3_higraph_no_overlap"
SEED=42
N_EPISODES=500
SHOTS="1 5"
N_QUERY=15
STABILITY_EPISODES=200
WEIGHTS_EXPORT_DIR="metric_data_ablation_temp/"

PROGRESS_FILE="reviewer_experiments_progress.txt"

is_done()   { grep -qxF "$1" "$PROGRESS_FILE" 2>/dev/null; }
mark_done() { echo "$1" >> "$PROGRESS_FILE"; }

if [[ "${1:-run}" == "reset" ]]; then
    rm -f "$PROGRESS_FILE"
    echo "Progress file cleared. Run the script again to start fresh."
    exit 0
fi

run_step() {
    local STEP=$1
    shift
    if is_done "$STEP"; then
        echo "--- [skip] ${STEP} already done ---"
        return 0
    fi
    echo "=========================================================="
    echo ">>> ${STEP}"
    echo "=========================================================="
    "$@"
    mark_done "$STEP"
    echo "--- Done: ${STEP} ---"
    echo
}

echo "Config: ${CONFIG}   Seed: ${SEED}"
echo

# ---------------------------------------------------------------------------
# STEP 1 — Matched-function weight stability under obfuscation
# ---------------------------------------------------------------------------
run_step "1_weight_stability" \
    python -m src.analysis.measure_weight_stability \
        --config "$CONFIG" \
        --n_episodes "$STABILITY_EPISODES" \
        --k_shot 5 \
        --seed "$SEED"

# ---------------------------------------------------------------------------
# STEP 2 — Random-init control: train (SDL), then evaluate
# ---------------------------------------------------------------------------
run_step "2a_random_init_train_sdl" \
    python -m src.train.episodic_train \
        --config "$CONFIG" --variant sdl_complexity --seed "$SEED" \
        --random_init_backbone

# Evaluation loads the full granular-ablation ladder INCLUDING the new
# random_init_* rows (episodic_evaluate.py picks them up automatically).
run_step "2c_ablation_evaluation" \
    python -m src.eval.episodic_evaluate \
        --config "$CONFIG" \
        --n_query "$N_QUERY" --n_episodes "$N_EPISODES" \
        --n_shots $SHOTS \
        --ablation_only --export_weights_dir "$WEIGHTS_EXPORT_DIR"

# ---------------------------------------------------------------------------
# STEP 3 — Clean-support / obfuscated-query accuracy (evasion)
# ---------------------------------------------------------------------------
run_step "3_clean_support_obf_query" \
    python -m src.robustness.episodic_obfuscated_query_evaluate \
        --config "$CONFIG" \
        --n_episodes "$N_EPISODES" \
        --n_shots $SHOTS \
        --n_query "$N_QUERY" \
        --seed "$SEED"

echo "=========================================================="
echo "=== All reviewer experiments complete ==="
echo "  Step 1 weight stability : results/interpretability/weight_stability_*.csv"
echo "  Step 2 random-init table: trained_models/.../evaluation_logs/ (ablation eval)"
echo "  Step 3 query-side obf   : trained_models/.../evaluation_logs/evaluate_obfuscated_query_*.log"
echo "=========================================================="
