#!/usr/bin/env bash
# run_full_evaluation.sh
#
# Master evaluation script — evaluates ALL models on the exact same set of episodes
# using --all_models, so results are directly comparable with no episode-set mismatch.
#
# Models evaluated in one pass:
#   Episodic heads    : sdl_complexity, sdl_degree, sdl_pagerank, self_attention,
#                       gmt, gmpool, topk_pool, diff_pool,
#                       sdl_no_learnable_param, sdl_complexity_prior,
#                       sdl_with_intra_param, sdl_without_structure
#   Pretrained frozen : pretrained_backbone, whole_graph_backbone,
#                       flat_fcg_backbone, set_only_backbone, instance_only_backbone
#   Group B ablations : flat_fcg_sdl_complexity, set_only_sdl_complexity,
#                       instance_only_sdl_complexity, whole_graph_sdl_complexity
#   Baselines         : FCGAT, FAMCF (permission / api_call / opcode), AndroFIM
#
# Missing checkpoints are skipped gracefully (not an error).
#
# USAGE:
#   bash run_full_evaluation.sh

set -euo pipefail

CONFIG="trial_config_v3_higraph_no_overlap"

echo "========================================"
echo "  Full evaluation — all models"
echo "  Config : ${CONFIG}"
echo "  Start  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python -m src.eval.episodic_evaluate \
    --config        "$CONFIG" \
    --n_query       15 \
    --n_episodes    500 \
    --n_shots       1 5 \
    --all_models

echo ""
echo "========================================"
echo "=== Full evaluation complete ==="
echo "=== Config: ${CONFIG} ==="
echo "========================================"
