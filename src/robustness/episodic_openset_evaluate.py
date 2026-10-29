"""
episodic_openset_evaluate.py

Open-set few-shot evaluation: 5 support families + intruder queries from
unseen families. Measures how well max-prototype-similarity scores separate
known from unknown, with threshold calibrated on the validation split.

Episode structure:
  Support   : K shots x N families  (same as closed-set)
  Known Q   : n_query per family from the N support families
  Intruder Q: n_intruder APKs from families NOT in the current N

Scoring (per model type):
  SDL / SA / pool heads : max cosine similarity to graph prototypes
  FCGAT / backbone      : max cosine similarity to L2-normalised prototypes
  FAMCF                 : max summed negative-cdist logit across 3 encoders
  AndroFIM              : negative minimum Euclidean distance to any prototype

Threshold calibration (on validation split):
  Collect per-model scores for all known queries across n_cal episodes.
  Set threshold = (1 - tpr_target) percentile of those scores.
  At this threshold, tpr_target fraction of validation known queries are accepted.

Reported metrics per model per k-shot:
  known_acc  -- closed-set accuracy on known queries (no threshold applied)
  AUROC      -- AUC-ROC for known vs intruder (threshold-free)
  FPR@95TPR  -- intruder acceptance rate at calibrated threshold
  open_f1    -- binary macro-F1 (accepted=known / rejected=unknown) at threshold

Usage:
  python src/episodic_openset_evaluate.py --config trial_config_v3_higraph_no_overlap
  python src/episodic_openset_evaluate.py --config trial_config_v3_higraph_no_overlap \\
      --n_shots 1 5 --n_intruder 15 --n_episodes 500 --n_cal 200
"""

import argparse
import json as _json
import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import Subset, ConcatDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.core.backbone import Backbone
from src.core.experiment_configs import ExperimentConfig
from src.core.utils import Utils, ModelParameters
from src.core.episodic_models import _scatter_mean
from src.train.episodic_train import (
    BASELINE_REGISTRY,
    create_model,
    extract_episode,
    build_episode_batches,
)
from src.eval.episodic_evaluate import (
    OnTheFlyFewShotEvaluator,
    _build_famcf_hash_to_features,
    _build_androfim_hash_to_path,
)
from src.train.train_famcf import MLPEncoder
from src.eval.eval_androfim import (
    load_backbone as _load_androfim_backbone,
    generate_positions as _androfim_generate_positions,
    make_crops as _androfim_make_crops,
    IMAGE_SIZE as _ANDROFIM_IMAGE_SIZE,
    CROP_SIZE  as _ANDROFIM_CROP_SIZE,
    N_CROPS    as _ANDROFIM_N_CROPS,
)
from src.core.parse_dataset import HiGraphData


PROJECT_ROOT = Path(Utils.get_project_root())


# ---------------------------------------------------------------------------
# Open-set evaluator (SDL / SA / pooling heads)
# ---------------------------------------------------------------------------

class OpenSetEvaluator(OnTheFlyFewShotEvaluator):
    """
    Extends OnTheFlyFewShotEvaluator with open-set scoring.

    The new method evaluate_openset_task mirrors evaluate_task but processes
    known and intruder queries separately, returning max-prototype similarity
    scores alongside predictions.
    """

    @torch.no_grad()
    def evaluate_openset_task(self, support_data, query_known_data, query_intruder_data):
        """
        Single open-set episode.

        Returns:
          known_preds     : (n_known,)    -- predicted episode-local class
          known_labels    : (n_known,)    -- true episode-local class
          known_scores    : (n_known,)    -- max prototype cosine similarity
          intruder_scores : (n_intruder,) -- max prototype cosine similarity
        """
        self.model.eval()
        support_data        = support_data.to(self.device)
        query_known_data    = query_known_data.to(self.device)
        query_intruder_data = query_intruder_data.to(self.device)

        # ---- Phase 1: Support (labels known) --------------------------------
        sup_node_embs, sup_fcg_batch = self.model.get_node_embeddings(support_data)
        sup_node_labels = torch.repeat_interleave(
            support_data.family_class, support_data.num_functions
        )
        sup_weights = self._compute_support_weights(
            sup_node_embs, sup_node_labels, support_data, sup_fcg_batch
        )

        _, labels_inv = torch.unique(sup_node_labels, return_inverse=True)
        node_protos = F.normalize(
            _scatter_mean(sup_node_embs * sup_weights.view(-1, 1), labels_inv, dim=0),
            p=2, dim=-1,
        )

        sup_graph_embs = self._pool_nodes(sup_node_embs, sup_fcg_batch, sup_weights, data=support_data)
        graph_protos   = F.normalize(
            _scatter_mean(sup_graph_embs, support_data.family_class, dim=0),
            p=2, dim=-1,
        )

        # ---- Phase 2: Query embedding helper --------------------------------
        def _embed(qdata):
            n_embs, n_batch = self.model.get_node_embeddings(qdata)
            w = self._compute_query_weights(n_embs, node_protos, qdata, fcg_batch=n_batch)
            g_embs = self._pool_nodes(n_embs, n_batch, w, data=qdata)
            sims   = torch.matmul(g_embs, graph_protos.t())
            return sims.argmax(dim=1), sims.max(dim=1).values

        known_preds,    known_scores    = _embed(query_known_data)
        _intruder_preds, intruder_scores = _embed(query_intruder_data)

        return (
            known_preds.cpu(),
            query_known_data.family_class.cpu(),
            known_scores.cpu(),
            intruder_scores.cpu(),
        )


# ---------------------------------------------------------------------------
# Open-set scoring helpers for standalone baselines
# ---------------------------------------------------------------------------

@torch.no_grad()
def _fcgat_openset_task(fcgat, sup_data, qry_known, qry_intruder, device):
    """FCGAT open-set: cosine similarity to L2-normalised prototypes."""
    sup_data    = sup_data.to(device)
    qry_known   = qry_known.to(device)
    qry_intruder = qry_intruder.to(device)

    z_spt, _ = fcgat.embed(sup_data)
    z_kn,  _ = fcgat.embed(qry_known)
    z_it,  _ = fcgat.embed(qry_intruder)

    protos = _scatter_mean(z_spt, sup_data.family_class, dim=0)
    protos = F.normalize(protos, p=2, dim=1)

    kn_sims = z_kn @ protos.t()
    it_sims = z_it @ protos.t()

    return (
        kn_sims.argmax(dim=1).cpu(),
        qry_known.family_class.cpu(),
        kn_sims.max(dim=1).values.cpu(),
        it_sims.max(dim=1).values.cpu(),
    )


@torch.no_grad()
def _backbone_openset_task(backbone, sup_data, qry_known, qry_intruder, device):
    """Pretrained backbone open-set: cosine similarity to L2-normalised prototypes."""
    sup_data    = sup_data.to(device)
    qry_known   = qry_known.to(device)
    qry_intruder = qry_intruder.to(device)

    z_spt, _ = backbone(sup_data,    return_node_embeddings=False)
    z_kn,  _ = backbone(qry_known,   return_node_embeddings=False)
    z_it,  _ = backbone(qry_intruder, return_node_embeddings=False)

    z_spt = F.normalize(z_spt, p=2, dim=1)
    z_kn  = F.normalize(z_kn,  p=2, dim=1)
    z_it  = F.normalize(z_it,  p=2, dim=1)

    protos = _scatter_mean(z_spt, sup_data.family_class, dim=0)
    protos = F.normalize(protos, p=2, dim=1)

    kn_sims = z_kn @ protos.t()
    it_sims = z_it @ protos.t()

    return (
        kn_sims.argmax(dim=1).cpu(),
        qry_known.family_class.cpu(),
        kn_sims.max(dim=1).values.cpu(),
        it_sims.max(dim=1).values.cpu(),
    )


@torch.no_grad()
def _famcf_openset_task(famcf_encoders, sup_data, qry_known, qry_intruder,
                         hash_to_features, device):
    """
    FAMCF open-set: max summed negative-cdist logit across 3 encoders.
    Returns None if any sample is missing from features dict.
    """
    feat_keys = {"permission": "perm_vec", "api_call": "api_vec", "opcode": "opc_vec"}

    def _hashes(batch):
        return batch.hash if isinstance(batch.hash, list) else [batch.hash]

    sup_hashes = _hashes(sup_data)
    kn_hashes  = _hashes(qry_known)
    it_hashes  = _hashes(qry_intruder)

    sup_entries = [hash_to_features.get(h) for h in sup_hashes]
    kn_entries  = [hash_to_features.get(h) for h in kn_hashes]
    it_entries  = [hash_to_features.get(h) for h in it_hashes]

    if any(e is None for e in sup_entries + kn_entries + it_entries):
        return None

    y_spt = sup_data.family_class

    kn_logits_sum = None
    it_logits_sum = None

    for ft, encoder in famcf_encoders.items():
        key = feat_keys[ft]
        sup_feats = torch.tensor(
            np.stack([e[key] for e in sup_entries]), dtype=torch.float32
        ).to(device)
        kn_feats = torch.tensor(
            np.stack([e[key] for e in kn_entries]), dtype=torch.float32
        ).to(device)
        it_feats = torch.tensor(
            np.stack([e[key] for e in it_entries]), dtype=torch.float32
        ).to(device)

        z_spt = encoder(sup_feats)
        z_kn  = encoder(kn_feats)
        z_it  = encoder(it_feats)

        protos = _scatter_mean(z_spt, y_spt, dim=0)

        kn_logits = -torch.cdist(z_kn, protos)
        it_logits = -torch.cdist(z_it, protos)

        kn_logits_sum = kn_logits if kn_logits_sum is None else kn_logits_sum + kn_logits
        it_logits_sum = it_logits if it_logits_sum is None else it_logits_sum + it_logits

    return (
        kn_logits_sum.argmax(dim=1).cpu(),
        qry_known.family_class.cpu(),
        kn_logits_sum.max(dim=1).values.cpu(),
        it_logits_sum.max(dim=1).values.cpu(),
    )


@torch.no_grad()
def _androfim_openset_task(androfim_backbone, sup_data, qry_known, qry_intruder,
                            hash_to_path, device):
    """
    AndroFIM open-set: negative minimum Euclidean distance to any prototype.
    Score is higher for queries that are close to a known prototype.
    Returns None if any APK image is missing from hash_to_path.
    """
    def _hashes(batch):
        return batch.hash if isinstance(batch.hash, list) else [batch.hash]

    sup_hashes = _hashes(sup_data)
    kn_hashes  = _hashes(qry_known)
    it_hashes  = _hashes(qry_intruder)

    sup_paths = [hash_to_path.get(h) for h in sup_hashes]
    kn_paths  = [hash_to_path.get(h) for h in kn_hashes]
    it_paths  = [hash_to_path.get(h) for h in it_hashes]

    if any(p is None for p in sup_paths + kn_paths + it_paths):
        return None

    sup_labels    = sup_data.family_class.cpu().tolist()
    kn_labels     = qry_known.family_class.cpu().tolist()
    unique_labels = sorted(set(sup_labels))
    n_support     = len(sup_paths)
    b, k = _ANDROFIM_CROP_SIZE, _ANDROFIM_N_CROPS

    def _classify_and_score(query_paths):
        preds, scores = [], []
        for q_path in query_paths:
            positions = _androfim_generate_positions(
                _ANDROFIM_IMAGE_SIZE, _ANDROFIM_IMAGE_SIZE, b, k
            )

            q_crops = _androfim_make_crops(q_path, positions, b).to(device)
            q_feats = androfim_backbone(q_crops)
            q_mean  = q_feats.mean(0)

            all_s_crops = torch.stack([
                _androfim_make_crops(p, positions, b) for p in sup_paths
            ]).to(device)
            flat_feats  = androfim_backbone(all_s_crops.view(n_support * k, 3, b, b))
            s_feats     = flat_feats.view(n_support, k, -1)

            distances = torch.norm(q_feats.unsqueeze(0) - s_feats, dim=2)
            weights   = F.softmax(-distances, dim=1)
            s_vecs    = (weights.unsqueeze(2) * s_feats).sum(1)

            all_vecs = torch.cat([s_vecs, q_mean.unsqueeze(0)], dim=0)
            center   = all_vecs.mean(0)
            s_norm   = F.normalize(s_vecs - center, dim=1)
            q_norm   = F.normalize(q_mean - center, dim=0)

            protos = torch.stack([
                s_norm[[i for i, l in enumerate(sup_labels) if l == c]].mean(0)
                for c in unique_labels
            ])
            dists = torch.norm(q_norm.unsqueeze(0) - protos, dim=1)
            preds.append(unique_labels[dists.argmin().item()])
            scores.append(-dists.min().item())

        return preds, scores

    kn_preds, kn_scores = _classify_and_score(kn_paths)
    _,        it_scores = _classify_and_score(it_paths)

    return (
        torch.tensor(kn_preds,  dtype=torch.long),
        torch.tensor(kn_labels, dtype=torch.long),
        torch.tensor(kn_scores, dtype=torch.float),
        torch.tensor(it_scores, dtype=torch.float),
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _get_ibc(ds):
    """Build {class_id: [subset_indices]} from a Subset(ConcatDataset) or plain dataset."""
    if isinstance(ds, Subset):
        inner = ds.dataset
        all_labels = (
            torch.cat([d.get_labels() for d in inner.datasets])
            if isinstance(inner, ConcatDataset)
            else inner.get_labels()
        )
        labels = all_labels[torch.tensor(ds.indices, dtype=torch.long)]
    else:
        labels = ds.get_labels()
    ibc = defaultdict(list)
    for i, lbl in enumerate(labels.tolist()):
        ibc[lbl].append(i)
    return ibc


def _intruder_batch(dataset, intruder_idx: list, device):
    """Load intruder samples; family_class is set to -1 (unknown sentinel)."""
    loader = DataLoader(Subset(dataset, intruder_idx), batch_size=len(intruder_idx), shuffle=False)
    batch  = next(iter(loader))
    batch.family_class = torch.full((len(intruder_idx),), -1, dtype=torch.long)
    return batch.to(device)


def _sample_intruders(intruder_pool: dict, n_intruder: int) -> list:
    """Sample n_intruder indices uniformly from all intruder-pool classes."""
    pool = [i for idxs in intruder_pool.values() for i in idxs]
    if not pool:
        return []
    return np.random.choice(pool, min(n_intruder, len(pool)), replace=False).tolist()


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def _run_episodes(
    evaluators: dict,
    dataset,
    ibc: dict,
    n_way: int,
    k_shot: int,
    n_query: int,
    n_intruder: int,
    n_episodes: int,
    device,
    label: str,
    fcgat=None,
    pretrained_backbones: dict = None,
    famcf_encoders: dict = None,
    hash_to_features: dict = None,
    androfim_backbone=None,
    hash_to_path: dict = None,
) -> dict:
    """
    Run open-set episodes for one k_shot value.

    Returns {model_name: {known_scores, intruder_scores, known_correct, known_total}}.
    """
    valid = {c: v for c, v in ibc.items() if len(v) >= k_shot}
    if len(valid) < n_way:
        all_names = (
            list(evaluators.keys())
            + (['fcgat'] if fcgat is not None else [])
            + (list(pretrained_backbones.keys()) if pretrained_backbones else [])
            + (['famcf'] if famcf_encoders else [])
            + (['androfim'] if androfim_backbone else [])
        )
        return {name: {"known_scores": [], "intruder_scores": [],
                       "known_correct": 0, "known_total": 0}
                for name in all_names}

    def _empty_rec():
        return {"known_scores": [], "intruder_scores": [], "known_correct": 0, "known_total": 0}

    records = {name: _empty_rec() for name in evaluators}
    if fcgat is not None:
        records['fcgat'] = _empty_rec()
    if pretrained_backbones:
        for name in pretrained_backbones:
            records[name] = _empty_rec()
    if famcf_encoders and hash_to_features:
        records['famcf'] = _empty_rec()
    if androfim_backbone and hash_to_path:
        records['androfim'] = _empty_rec()

    completed = attempts = 0
    pbar = tqdm(total=n_episodes, desc=f'[{label}] {n_way}-way {k_shot}-shot open-set')

    while completed < n_episodes and attempts < n_episodes * 3:
        attempts += 1

        s_idx, q_idx, sampled = extract_episode(valid, n_way, k_shot, n_query)
        if s_idx is None:
            continue

        intruder_pool = {c: v for c, v in ibc.items() if c not in set(sampled.tolist())}
        intruder_idx  = _sample_intruders(intruder_pool, n_intruder)
        if not intruder_idx:
            continue

        sup_data, qry_known = build_episode_batches(dataset, s_idx, q_idx, sampled, device)
        qry_intruder        = _intruder_batch(dataset, intruder_idx, device)

        # SDL / SA / pooling heads
        for name, ev in evaluators.items():
            preds, labels, kn_s, it_s = ev.evaluate_openset_task(
                sup_data, qry_known, qry_intruder
            )
            r = records[name]
            r["known_scores"].extend(kn_s.tolist())
            r["intruder_scores"].extend(it_s.tolist())
            r["known_correct"] += (preds == labels).sum().item()
            r["known_total"]   += len(labels)

        # FCGAT
        if fcgat is not None:
            result = _fcgat_openset_task(fcgat, sup_data, qry_known, qry_intruder, device)
            preds, labels, kn_s, it_s = result
            r = records['fcgat']
            r["known_scores"].extend(kn_s.tolist())
            r["intruder_scores"].extend(it_s.tolist())
            r["known_correct"] += (preds == labels).sum().item()
            r["known_total"]   += len(labels)

        # Pretrained backbones
        if pretrained_backbones:
            for name, bb in pretrained_backbones.items():
                result = _backbone_openset_task(bb, sup_data, qry_known, qry_intruder, device)
                preds, labels, kn_s, it_s = result
                r = records[name]
                r["known_scores"].extend(kn_s.tolist())
                r["intruder_scores"].extend(it_s.tolist())
                r["known_correct"] += (preds == labels).sum().item()
                r["known_total"]   += len(labels)

        # FAMCF
        if famcf_encoders and hash_to_features:
            result = _famcf_openset_task(
                famcf_encoders, sup_data, qry_known, qry_intruder, hash_to_features, device
            )
            if result is not None:
                preds, labels, kn_s, it_s = result
                r = records['famcf']
                r["known_scores"].extend(kn_s.tolist())
                r["intruder_scores"].extend(it_s.tolist())
                r["known_correct"] += (preds == labels).sum().item()
                r["known_total"]   += len(labels)

        # AndroFIM
        if androfim_backbone is not None and hash_to_path is not None:
            result = _androfim_openset_task(
                androfim_backbone, sup_data, qry_known, qry_intruder, hash_to_path, device
            )
            if result is not None:
                preds, labels, kn_s, it_s = result
                r = records['androfim']
                r["known_scores"].extend(kn_s.tolist())
                r["intruder_scores"].extend(it_s.tolist())
                r["known_correct"] += (preds == labels).sum().item()
                r["known_total"]   += len(labels)

        completed += 1
        pbar.update(1)

    pbar.close()
    return records


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate(
    evaluators: dict,
    val_ds,
    val_ibc: dict,
    n_way: int,
    k_shot: int,
    n_query: int,
    n_intruder: int,
    n_cal: int,
    device,
    logger: logging.Logger,
    tpr_target: float = 0.95,
    fcgat=None,
    pretrained_backbones: dict = None,
    famcf_encoders: dict = None,
    hash_to_features: dict = None,
    androfim_backbone=None,
    hash_to_path: dict = None,
) -> dict:
    """
    Run open-set episodes on validation, find per-model rejection threshold.
    At the returned threshold, tpr_target fraction of validation known queries
    are accepted (i.e. score >= threshold).

    Returns {model_name: threshold}.
    """
    logger.info(f'  Running {n_cal} calibration episodes ({k_shot}-shot)...')
    records = _run_episodes(
        evaluators, val_ds, val_ibc,
        n_way, k_shot, n_query, n_intruder, n_cal, device, 'cal',
        fcgat=fcgat,
        pretrained_backbones=pretrained_backbones,
        famcf_encoders=famcf_encoders,
        hash_to_features=hash_to_features,
        androfim_backbone=androfim_backbone,
        hash_to_path=hash_to_path,
    )
    thresholds = {}
    for name, rec in records.items():
        ks = rec["known_scores"]
        if not ks:
            thresholds[name] = 0.0
            logger.info(f'    {name:<30} no validation scores — threshold set to 0.0')
            continue
        t = float(np.percentile(ks, (1.0 - tpr_target) * 100.0))
        thresholds[name] = t
        logger.info(f'    {name:<30} threshold = {t:.4f}  ({len(ks)} known scores)')
    return thresholds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(records: dict, thresholds: dict) -> dict:
    """
    From pooled scores across all test episodes compute four open-set metrics.

    known_acc : closed-set accuracy on known queries (no threshold applied)
    auroc     : AUC-ROC separating known (1) from intruder (0) by max similarity
    fpr95     : fraction of intruders accepted when threshold yields ~95% TPR
    open_f1   : binary macro-F1 (known vs unknown) with the calibrated threshold
    """
    out = {}
    for name, rec in records.items():
        ks  = np.array(rec["known_scores"],    dtype=np.float32)
        its = np.array(rec["intruder_scores"], dtype=np.float32)

        if len(ks) == 0 or len(its) == 0:
            out[name] = None
            continue

        known_acc = rec["known_correct"] / max(rec["known_total"], 1)

        all_scores = np.concatenate([ks, its])
        all_labels = np.concatenate([np.ones(len(ks), dtype=int),
                                     np.zeros(len(its), dtype=int)])
        auroc = float(roc_auc_score(all_labels, all_scores))

        t     = thresholds.get(name, 0.0)
        fpr95 = float((its >= t).mean())

        binary_pred = (all_scores >= t).astype(int)
        open_f1     = float(f1_score(all_labels, binary_pred, average='macro'))

        out[name] = {
            "known_acc": known_acc,
            "auroc":     auroc,
            "fpr95":     fpr95,
            "open_f1":   open_f1,
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Open-set few-shot evaluation')
    parser.add_argument('--config',          default='trial_config_v3_higraph_no_overlap',
                        help='ExperimentConfig method suffix')
    parser.add_argument('--n_way',           type=int, default=5)
    parser.add_argument('--n_shots',         type=int, nargs='+', default=[1, 5])
    parser.add_argument('--n_query',         type=int, default=15,
                        help='Known queries per family per episode')
    parser.add_argument('--n_intruder',      type=int, default=15,
                        help='Intruder queries per episode')
    parser.add_argument('--n_episodes',      type=int, default=500,
                        help='Test episodes')
    parser.add_argument('--n_cal',           type=int, default=200,
                        help='Calibration episodes on validation')
    parser.add_argument('--tpr_target',      type=float, default=0.95,
                        help='Target TPR for threshold calibration')
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--androfim_ckpt',   default=None,
                        help='Path to AndroFIM finetuned ResNet-12 checkpoint. '
                             'Defaults to models/androFIM/<split>/finetuned_backbone.pth.')
    parser.add_argument('--androfim_manifest', default=None,
                        help='Path to androFIM_manifest.csv. '
                             'Defaults to dataset/<dataset_name>/androfim/androFIM_manifest.csv.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params: ModelParameters = getattr(ExperimentConfig, f'get_{args.config}')()
    device = Utils.get_device()

    log_dir = os.path.join(
        Utils.get_project_root(), 'trained_models',
        params.id, params.dataset_name,
        params.dataset_split_type.value, 'evaluation_logs', 'unfrozen',
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f'evaluate_openset_{time.strftime("%Y%m%d_%H%M%S")}.log'
    )
    logger = Utils.setup_logger('openset_eval', log_path)

    logger.info(f'Config: {params.id}  Device: {device}  Seed: {args.seed}')
    logger.info(
        f'n_way={args.n_way}  n_shots={args.n_shots}  n_query={args.n_query}  '
        f'n_intruder={args.n_intruder}  n_episodes={args.n_episodes}  '
        f'n_cal={args.n_cal}  tpr_target={args.tpr_target}'
    )

    # ---- Backbone -----------------------------------------------------------
    backbone = Backbone(
        cfg_dim=params.cfg_dim,
        func_feat_dim=params.func_feat_dim,
        cfg_hidden_channels=params.cfg_hidden_channels,
        fcg_hidden_channels=params.fcg_hidden_channels,
        final_channels=params.final_channels,
        dropout_value=params.dropout,
    ).to(device)

    bb_path = Utils.get_backbone_model_saved_path(params)
    if not os.path.exists(bb_path):
        raise FileNotFoundError(f'Backbone checkpoint not found: {bb_path}')
    backbone.load_state_dict(torch.load(bb_path, map_location=device))
    backbone.eval()
    logger.info(f'Loaded backbone: {bb_path}')

    # ---- Load SDL/SA/pooling episodic models --------------------------------
    evaluators = {}
    for variant in BASELINE_REGISTRY:
        ckpt = Utils.get_episodic_model_path(params, variant, frozen=False)
        if not os.path.exists(ckpt):
            logger.info(f'  Skipping {variant} — no checkpoint')
            continue
        model = create_model(variant, backbone, fcg_dim=params.final_channels).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        evaluators[variant] = OpenSetEvaluator(model=model, device=device)
        logger.info(f'  Loaded {variant}')

    if not evaluators:
        logger.error('No episodic models loaded. Run episodic training first.')
        return

    logger.info(f'Loaded {len(evaluators)} episodic models: {list(evaluators.keys())}')

    # ---- FCGAT --------------------------------------------------------------
    fcgat = None
    try:
        from src.core.fcgat_model import StandaloneFCGAT
        fcgat_path = Utils.get_fcgat_model_saved_path(params)
        if os.path.exists(fcgat_path):
            fcgat = StandaloneFCGAT(
                cfg_dim=params.cfg_dim,
                func_feat_dim=params.func_feat_dim,
            ).to(device)
            fcgat.load_state_dict(torch.load(fcgat_path, map_location=device), strict=False)
            fcgat.eval()
            logger.info('  Loaded FCGAT')
        else:
            logger.info(f'  FCGAT not found at {fcgat_path}, skipping')
    except Exception as e:
        logger.info(f'  FCGAT skipped: {e}')

    # ---- Pretrained backbone (no episodic head) -----------------------------
    pretrained_backbones = {}
    pretrained_path = Utils.get_backbone_model_saved_path(params)
    if os.path.exists(pretrained_path):
        bb = Backbone(
            cfg_dim=params.cfg_dim,
            func_feat_dim=params.func_feat_dim,
            cfg_hidden_channels=params.cfg_hidden_channels,
            fcg_hidden_channels=params.fcg_hidden_channels,
            final_channels=params.final_channels,
            dropout_value=params.dropout,
        ).to(device)
        bb.load_state_dict(torch.load(pretrained_path, map_location=device))
        bb.eval()
        pretrained_backbones['pretrained_backbone'] = bb
        logger.info('  Loaded pretrained backbone')
    else:
        logger.info(f'  Pretrained backbone not found at {pretrained_path}, skipping')

    # ---- FAMCF encoders + features lookup -----------------------------------
    famcf_encoders   = {}
    hash_to_features = None
    famcf_save_dir   = Utils.get_famcf_save_dir(
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
                cfg = _json.load(f)
            enc = MLPEncoder(cfg['input_dim'], cfg['hidden_dim'], cfg['output_dim']).to(device)
            enc.load_state_dict(torch.load(enc_path, map_location=device))
            enc.eval()
            famcf_encoders[ft] = enc
            logger.info(f'  Loaded FAMCF encoder: {ft}')
        else:
            logger.info(f'  FAMCF {ft} encoder not found at {enc_path}, skipping')
    if famcf_encoders and os.path.exists(famcf_features_pkl):
        hash_to_features = _build_famcf_hash_to_features(famcf_features_pkl)
        logger.info(f'  Loaded FAMCF features ({len(hash_to_features):,} samples)')
    elif famcf_encoders:
        logger.warning(
            f'  FAMCF encoders loaded but features.pkl not found at {famcf_features_pkl} — skipping FAMCF'
        )
        famcf_encoders = {}
    if not famcf_encoders:
        famcf_encoders   = None
        hash_to_features = None

    # ---- AndroFIM backbone + image lookup -----------------------------------
    androfim_backbone = None
    hash_to_path      = None
    _split            = params.dataset_split_type.value
    _default_ckpt     = os.path.join(
        Utils.get_androfim_save_dir(params.id, params.dataset_name, _split),
        'finetuned_backbone.pth',
    )
    androfim_ckpt_path     = Path(args.androfim_ckpt if args.androfim_ckpt else _default_ckpt)
    androfim_manifest_path = (
        Path(args.androfim_manifest)
        if args.androfim_manifest
        else Path(Utils.get_androfim_manifest(params.dataset_name))
    )
    if androfim_ckpt_path.exists() and androfim_manifest_path.exists():
        androfim_backbone = _load_androfim_backbone(str(androfim_ckpt_path), device=device)
        img_root     = androfim_manifest_path.parent
        flat         = False
        hash_to_path = _build_androfim_hash_to_path(androfim_manifest_path, img_root, flat=flat)
        if hash_to_path:
            logger.info(f'  Loaded AndroFIM backbone ({len(hash_to_path):,} images in manifest)')
        else:
            logger.warning('  AndroFIM skipped — no image files found on disk')
            androfim_backbone = None
            hash_to_path      = None
    else:
        logger.info(
            f'  AndroFIM skipped — '
            f'ckpt exists={androfim_ckpt_path.exists()}, '
            f'manifest exists={androfim_manifest_path.exists()}'
        )

    # ---- Dataset splits -----------------------------------------------------
    _, val_ds, test_ds = Utils.get_splits(params)
    val_ibc  = _get_ibc(val_ds)
    test_ibc = _get_ibc(test_ds)
    logger.info(f'Validation : {len(val_ds):,} samples  {len(val_ibc)} families')
    logger.info(f'Test       : {len(test_ds):,} samples  {len(test_ibc)} families')

    # ---- Per k-shot: calibrate on val, evaluate on test ---------------------
    for k_shot in args.n_shots:
        logger.info(f'\n{"="*70}')
        logger.info(f'  {args.n_way}-way {k_shot}-shot open-set')
        logger.info(f'{"="*70}')

        logger.info('\n[Calibration — validation split]')
        thresholds = calibrate(
            evaluators, val_ds, val_ibc,
            args.n_way, k_shot, args.n_query, args.n_intruder,
            args.n_cal, device, logger, args.tpr_target,
            fcgat=fcgat,
            pretrained_backbones=pretrained_backbones if pretrained_backbones else None,
            famcf_encoders=famcf_encoders,
            hash_to_features=hash_to_features,
            androfim_backbone=androfim_backbone,
            hash_to_path=hash_to_path,
        )

        logger.info('\n[Test — test split]')
        test_records = _run_episodes(
            evaluators, test_ds, test_ibc,
            args.n_way, k_shot, args.n_query, args.n_intruder,
            args.n_episodes, device, 'test',
            fcgat=fcgat,
            pretrained_backbones=pretrained_backbones if pretrained_backbones else None,
            famcf_encoders=famcf_encoders,
            hash_to_features=hash_to_features,
            androfim_backbone=androfim_backbone,
            hash_to_path=hash_to_path,
        )

        metrics = compute_metrics(test_records, thresholds)

        logger.info(f'\n{"="*70}')
        logger.info(f'  Results — {args.n_way}-way {k_shot}-shot  '
                    f'({args.n_episodes} episodes, {args.n_intruder} intruders/ep)')
        logger.info(f'{"="*70}')
        hdr = (f'{"Model":<30}  {"Known Acc":>10}  {"AUROC":>7}  '
               f'{"FPR@95TPR":>10}  {"Open-F1":>8}')
        logger.info(hdr)
        logger.info('-' * len(hdr))
        for name in sorted(metrics):
            m = metrics[name]
            if m is None:
                logger.info(f'{name:<30}  (no data)')
                continue
            logger.info(
                f'{name:<30}  {m["known_acc"] * 100:>9.2f}%  '
                f'{m["auroc"]:>7.4f}  '
                f'{m["fpr95"]:>10.4f}  '
                f'{m["open_f1"]:>8.4f}'
            )

    logger.info(f'\nLog saved: {log_path}')


if __name__ == '__main__':
    main()
