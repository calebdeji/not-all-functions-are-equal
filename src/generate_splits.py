"""
generate_splits.py

Generates and saves the family train/val/test split for a given config,
without running any training.  The split strategy is read from the config.

Splits are saved to:
  dataset_config_split/higraph/{config_id}/no_overlap_split.json
  dataset_config_split/maldroid/{config_id}/split.json
  dataset_config_split/kronodroid/{config_id}/split.json
  
  
If the split file already exists it is left untouched (splits are
deterministic so re-generating is safe, but we avoid overwriting to protect
any downstream baselines that already loaded them).

Usage:
  python src/generate_splits.py --config trial_config_v3
  python src/generate_splits.py --config trial_config_v3_maldroid_no_overlap
"""

import argparse

from experiment_configs import ExperimentConfig
from utils import Utils


def generate(config_name: str):
    getter = getattr(ExperimentConfig, f"get_{config_name}", None)
    if getter is None:
        raise ValueError(f"No ExperimentConfig.get_{config_name}() found.")

    config = getter()
    print(f"[generate_splits] config={config_name}  "
          f"dataset={config.dataset_name}  split_type={config.dataset_split_type.value}")
    Utils.get_splits(config)
    print(f"[generate_splits] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="ExperimentConfig preset name (e.g. trial_config_v3_maldroid_no_overlap)")
    args = parser.parse_args()

    try:
        generate(args.config)
    except Exception as e:
        print(f"[generate_splits] ERROR: {e}")
