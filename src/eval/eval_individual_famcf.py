"""
eval_individual_famcf.py

Evaluates each FAMCF encoder (permission, api_call, opcode) individually
alongside the full soft-voting ensemble, using the same config, data paths,
episode construction, and seed as episodic_evaluate.py.

Results are directly comparable to the main evaluation table.

Usage:
  python src/eval_individual_famcf.py --config trial_config_v3
"""

import argparse
import json
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import scipy.stats as st
import torch
from tqdm import tqdm

from src.core.experiment_configs import ExperimentConfig
from src.core.utils import Utils
from src.train.train_famcf import MLPEncoder
from src.core.parse_dataset import HiGraphData
from src.core.episodic_models import _scatter_mean
from src.eval.episodic_evaluate import _build_famcf_hash_to_features
from src.train.episodic_train import extract_episode, build_episode_batches


# ---------------------------------------------------------------------------
# Per-episode evaluation — mirrors _evaluate_famcf_episode but tracks
# per-encoder predictions before summing logits for the ensemble.
# ---------------------------------------------------------------------------

FEAT_KEYS = {
    "permission": "perm_vec",
    "api_call":   "api_vec",
    "opcode":     "opc_vec",
}


@torch.no_grad()
def _evaluate_episode(famcf_encoders, sup_data, qry_data, hash_to_features, device):
    """
    Returns {ft: (preds, labels), ..., 'ensemble': (preds, labels)}
    or None if any sample's features are missing.
    """
    sup_hashes = sup_data.hash if isinstance(sup_data.hash, list) else [sup_data.hash]
    qry_hashes = qry_data.hash if isinstance(qry_data.hash, list) else [qry_data.hash]

    sup_entries = [hash_to_features.get(h) for h in sup_hashes]
    qry_entries = [hash_to_features.get(h) for h in qry_hashes]
    if any(e is None for e in sup_entries) or any(e is None for e in qry_entries):
        return None

    y_spt = sup_data.family_class
    y_qry = qry_data.family_class

    per_encoder = {}
    logits_sum  = None

    for ft, encoder in famcf_encoders.items():
        key = FEAT_KEYS[ft]
        sup_feats = torch.tensor(
            np.stack([e[key] for e in sup_entries]), dtype=torch.float32
        ).to(device)
        qry_feats = torch.tensor(
            np.stack([e[key] for e in qry_entries]), dtype=torch.float32
        ).to(device)

        z_spt = encoder(sup_feats)
        z_qry = encoder(qry_feats)

        prototypes = _scatter_mean(z_spt, y_spt, dim=0)
        logits = -torch.cdist(z_qry, prototypes)
        logits_sum = logits if logits_sum is None else logits_sum + logits

        per_encoder[ft] = (logits.argmax(dim=1).cpu(), y_qry.cpu())

    per_encoder["ensemble"] = (logits_sum.argmax(dim=1).cpu(), y_qry.cpu())
    return per_encoder


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='trial_config_v3')
    parser.add_argument('--n_way',      type=int, default=5)
    parser.add_argument('--n_shots',    type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--n_query',    type=int, default=15)
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    getter = getattr(ExperimentConfig, f'get_{args.config}')
    params = getter()
    device = Utils.get_device()

    log_dir = os.path.join(
        Utils.get_project_root(), 'trained_models',
        params.id, params.dataset_name,
        params.dataset_split_type.value, 'evaluation_logs',
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'eval_individual_famcf.log')
    logger = Utils.setup_logger('eval_individual_famcf', log_path)

    logger.info(f'Config: {params.id} | Dataset: {params.dataset_name} | '
                f'Split: {params.dataset_split_type.value} | '
                f'Seed: {args.seed} | Episodes: {args.n_episodes}')

    # ── Load FAMCF encoders — identical to episodic_evaluate.py ──────────────
    famcf_encoders = {}
    famcf_save_dir = Utils.get_famcf_save_dir(
        params.id, params.dataset_name, params.dataset_split_type.value
    )
    famcf_features_pkl = os.path.join(
        Utils.get_project_root(), 'dataset', params.dataset_name, 'famcf', 'features.pkl'
    )
    _short = {"permission": "per", "api_call": "api", "opcode": "opc"}
    for ft, short in _short.items():
        enc_path = os.path.join(famcf_save_dir, f'encoder_{short}.pth')
        cfg_path = os.path.join(famcf_save_dir, f'model_config_{ft}.json')
        if os.path.exists(enc_path) and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            enc = MLPEncoder(cfg['input_dim'], cfg['hidden_dim'], cfg['output_dim']).to(device)
            enc.load_state_dict(torch.load(enc_path, map_location=device, weights_only=True))
            enc.eval()
            famcf_encoders[ft] = enc
            logger.info(f'  Loaded FAMCF encoder: {ft}')
        else:
            logger.info(f'  FAMCF {ft} not found at {enc_path}, skipping')

    if not famcf_encoders:
        raise RuntimeError(f'No FAMCF encoders found in {famcf_save_dir}')

    if not os.path.exists(famcf_features_pkl):
        raise FileNotFoundError(f'features.pkl not found at {famcf_features_pkl}')

    hash_to_features = _build_famcf_hash_to_features(famcf_features_pkl)
    logger.info(f'  Loaded features.pkl: {len(hash_to_features):,} entries')

    # ── Load test dataset — identical to episodic_evaluate.py ────────────────
    _, _, test_ds = Utils.get_splits(params)

    from torch.utils.data import Subset, ConcatDataset
    inner = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
    if isinstance(inner, ConcatDataset):
        all_labels = torch.cat([ds.get_labels() for ds in inner.datasets])
    else:
        all_labels = inner.get_labels()
    subset_labels = all_labels[torch.tensor(test_ds.indices, dtype=torch.long)]

    indices_by_class = defaultdict(list)
    for i, lbl in enumerate(subset_labels.tolist()):
        indices_by_class[lbl].append(i)

    logger.info(f'Test families: {len(indices_by_class)} | '
                f'Test samples: {len(subset_labels)}')

    encoder_names = list(famcf_encoders.keys()) + ["ensemble"]

    # ── Run episodes ──────────────────────────────────────────────────────────
    for n_shot in args.n_shots:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        valid_classes = {c: idxs for c, idxs in indices_by_class.items()
                         if len(idxs) >= n_shot + args.n_query}

        if len(valid_classes) < args.n_way:
            logger.warning(f'{n_shot}-shot: not enough valid classes, skipping')
            continue

        all_accs = defaultdict(list)
        completed = attempts = 0

        pbar = tqdm(total=args.n_episodes,
                    desc=f'{args.n_way}-way {n_shot}-shot')

        while completed < args.n_episodes and attempts < args.n_episodes * 3:
            attempts += 1
            s_idx, q_idx, sampled = extract_episode(
                valid_classes, args.n_way, n_shot, args.n_query
            )
            if s_idx is None:
                continue

            sup_data, qry_data = build_episode_batches(
                test_ds, s_idx, q_idx, sampled, device
            )

            result = _evaluate_episode(
                famcf_encoders, sup_data, qry_data, hash_to_features, device
            )
            if result is None:
                continue

            for name, (preds, labels) in result.items():
                all_accs[name].append((preds == labels).float().mean().item())

            completed += 1
            pbar.update(1)
            pbar.set_postfix({n: f'{np.mean(v):.3f}' for n, v in all_accs.items()})

        pbar.close()

        parts = []
        for name in encoder_names:
            accs = all_accs[name]
            if not accs:
                continue
            mean_acc = float(np.mean(accs))
            ci95 = float(st.t.interval(
                0.95, df=len(accs) - 1, loc=mean_acc, scale=st.sem(accs)
            )[1] - mean_acc) if len(accs) > 1 else 0.0
            parts.append(f'{name}: {mean_acc * 100:.2f}% ± {ci95 * 100:.2f}%')

        logger.info(
            f'{args.n_way}-way {n_shot}-shot ({completed} eps) | ' + ' | '.join(parts)
        )


if __name__ == '__main__':
    main()