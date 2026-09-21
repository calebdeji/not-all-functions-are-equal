"""
generate_splits.py

Generates and saves the family train/val/test split for a given config,
without running any training.  The split strategy is read from the config.

Splits are saved to:
  dataset_config_split/higraph/{config_id}/no_overlap_split.json
  dataset_config_split/higraph/{config_id}/temporal_split.json
  dataset_config_split/maldroid/{config_id}/split.json
  dataset_config_split/kronodroid/{config_id}/split.json

If the split file already exists it is left untouched (splits are
deterministic so re-generating is safe, but we avoid overwriting to protect
any downstream baselines that already loaded them).  Pass --force to delete
the existing file and regenerate.

Usage:
  python -m src.data.generate_splits --config trial_config_v3_higraph_no_overlap
  python -m src.data.generate_splits --config trial_config_v3_higraph_temporal
  python -m src.data.generate_splits --config trial_config_v3_maldroid_no_overlap
  python -m src.data.generate_splits --config trial_config_v3_higraph_no_overlap --force
"""

import argparse
from pathlib import Path

from src.core.experiment_configs import ExperimentConfig
from src.core.utils import Utils, _FLAT_DATASET_NAMES


def generate(config_name: str, force: bool = False):
    getter = getattr(ExperimentConfig, f"get_{config_name}", None)
    if getter is None:
        raise ValueError(f"No ExperimentConfig.get_{config_name}() found.")

    config = getter()
    split_type = config.dataset_split_type.value
    print(f"[generate_splits] config={config_name}  "
          f"dataset={config.dataset_name}  split_type={split_type}")

    if force:
        split_filename = "split.json" if config.dataset_name in _FLAT_DATASET_NAMES \
                         else f"{split_type}_split.json"
        split_file = (
            Path(Utils.get_project_root())
            / "dataset_config_split"
            / config.dataset_name
            / config.id
            / split_filename
        )
        if split_file.exists():
            split_file.unlink()
            print(f"[generate_splits] Deleted existing split: {split_file}")

    Utils.get_splits(config)
    print(f"[generate_splits] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="ExperimentConfig preset name (e.g. trial_config_v3_maldroid_no_overlap)")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing split file and regenerate from scratch")
    args = parser.parse_args()

    try:
        generate(args.config, force=args.force)
    except Exception as e:
        print(f"[generate_splits] ERROR: {e}")
