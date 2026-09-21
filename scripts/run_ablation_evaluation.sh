CONFIGS=("trial_config_v3_higraph_no_overlap" "trial_config_v3_maldroid_no_overlap" "trial_config_v3_kronodroid_no_overlap") 

for CONFIG in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Evaluating all models  [$CONFIG]"
    echo "========================================"
    python -m src.eval.episodic_evaluate \
        --config "$CONFIG" \
        --n_query 15 --n_episodes 500 \
        --skip_episode_train --n_shots 1 5 \
        --ablation_only --export_weights_dir metric_data_ablation_temp/
done