"""
episodic_evaluate.py

Few-shot evaluation using OnTheFlyFewShotEvaluator.

Protocol (identical to episodic_train.py forward pass):
  Phase 1 (Support, labels known):
    - Get node embeddings from backbone
    - Compute weights → build node prototypes + graph prototypes
  Phase 2 (Query, labels unknown):
    - Get node embeddings from backbone
    - Compute weights using node prototypes as pseudo-labels
    - Pool to graph level → classify by cosine similarity to graph prototypes

This evaluator is used:
  1. During training for validation (called from episodic_train.py)
  2. Post-training on test years for final results

Usage:
  python -m src.episodic_evaluate --config trial_config_v3 --variant self_attention
"""

import argparse
import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from backbone import Backbone
from experiment_configs import ExperimentConfig
from utils import Utils, ModelParameters, _FLAT_DATASET_NAMES
from parse_dataset import HiGraphDataset, HiGraphData, get_dataset_root
from episodic_models import (
    TrainingUtils,
    SelfAttentionModel,
    SDLDegreeModel,
    SDLComplexityModel,
    SDLWithoutLearnableParamModel,
    SDLComplexityPriorModel,
    SDLWithoutStructureModel,
    _scatter_mean,
    _scatter_softmax,
)
from episodic_train import (
    BASELINE_REGISTRY,
    MODEL_REGISTRY,
    create_model,
    extract_episode,
    build_episode_batches,
    ABLATION_REGISTRY,
    SDL_REGISTRY,
)

from eval_androfim import (
    run_episode   as _androfim_run_episode,
    load_backbone as _load_androfim_backbone,
    CROP_SIZE     as _ANDROFIM_CROP_SIZE,
    N_CROPS       as _ANDROFIM_N_CROPS,
)
from train_famcf import MLPEncoder


# ---------------------------------------------------------------------------
# Resource tracking
# ---------------------------------------------------------------------------

class ResourceTracker:
    """
    Tracks wall time, GPU time, and peak GPU memory per named operation.
    Usage:
        with tracker.track('model_name'):
            result = model(...)
    """

    def __init__(self, device):
        self.device   = device
        self.use_cuda = torch.cuda.is_available() and str(device).startswith('cuda')
        self._wall_ms  = defaultdict(list)
        self._gpu_ms   = defaultdict(list)
        self._peak_mem = defaultdict(list)

    class _Ctx:
        def __init__(self, tracker, name):
            self._t = tracker
            self.name = name

        def __enter__(self):
            if self._t.use_cuda:
                torch.cuda.reset_peak_memory_stats(self._t.device)
                self._e0 = torch.cuda.Event(enable_timing=True)
                self._e1 = torch.cuda.Event(enable_timing=True)
                self._e0.record()
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            t = self._t
            t._wall_ms[self.name].append((time.perf_counter() - self._t0) * 1000)
            if t.use_cuda:
                self._e1.record()
                torch.cuda.synchronize()
                t._gpu_ms[self.name].append(self._e0.elapsed_time(self._e1))
                t._peak_mem[self.name].append(
                    torch.cuda.max_memory_allocated(t.device) / 1024 ** 2
                )

    def track(self, name):
        return self._Ctx(self, name)

    def summary(self):
        """Returns {name: {avg_wall_ms, avg_gpu_ms, avg_peak_mem_mb, ...}}."""
        out = {}
        for name in self._wall_ms:
            w = self._wall_ms[name]
            entry = {
                'n':            len(w),
                'avg_wall_ms':  float(np.mean(w)),
                'std_wall_ms':  float(np.std(w)),
                'total_wall_s': float(np.sum(w) / 1000),
            }
            if name in self._gpu_ms:
                g = self._gpu_ms[name]
                m = self._peak_mem[name]
                entry.update({
                    'avg_gpu_ms':      float(np.mean(g)),
                    'std_gpu_ms':      float(np.std(g)),
                    'avg_peak_mem_mb': float(np.mean(m)),
                    'max_peak_mem_mb': float(max(m)),
                })
            out[name] = entry
        return out


def _count_parameters(model, backbone=None):
    """
    Returns (total, head_only).
    head_only = total minus backbone params (if backbone is provided).
    """
    total = sum(p.numel() for p in model.parameters())
    if backbone is not None:
        backbone_count = sum(p.numel() for p in backbone.parameters())
        return total, total - backbone_count
    return total, total


def _build_androfim_hash_to_path(manifest_path, img_root, flat: bool = False):
    """Return {sha256: abs_image_path} from the AndroFIM manifest CSV.
    Only includes entries whose image file exists at the constructed path.
    """
    import pandas as pd
    df = pd.read_csv(manifest_path)
    result = {}
    missing = 0
    for _, row in df.iterrows():
        filename = f"{row.sha256}.png"
        if flat:
            path = str(img_root / "images" / filename)
        else:
            path = str(img_root / str(row.year) / filename)
        if os.path.exists(path):
            result[str(row.sha256)] = path
        else:
            missing += 1
    if missing:
        print(f'[WARNING] AndroFIM: {missing} manifest entries have no image on disk — skipped')
    return result


def _format_resource_report(tracker, models, backbone, fcgat, episode_train_only, config_id, dataset_split_type, pretrained_backbones=None, androfim_backbone=None, famcf_encoders=None):
    """Build a human-readable resource report string."""
    use_cuda = tracker.use_cuda
    lines = []
    lines.append('=' * 70)
    lines.append(f'RESOURCE ANALYSIS — {config_id} | {dataset_split_type}')
    lines.append('=' * 70)

    # ---- Parameter counts ----
    lines.append('\nParameters:')
    all_named = list(models.items()) + list((episode_train_only or {}).items())
    if fcgat is not None:
        all_named.append(('fcgat', fcgat))
    if pretrained_backbones:
        for name, bb in pretrained_backbones.items():
            all_named.append((name, bb))
    if androfim_backbone is not None:
        all_named.append(('androfim', androfim_backbone))
    if famcf_encoders:
        for ft, enc in famcf_encoders.items():
            all_named.append((f'famcf_{ft}', enc))

    backbone_count = sum(p.numel() for p in backbone.parameters()) if backbone is not None else 0

    famcf_names = {f'famcf_{ft}' for ft in (famcf_encoders or {})}
    standalone_names = {'fcgat', 'androfim'} | famcf_names | (set(pretrained_backbones) if pretrained_backbones else set())
    for name, model in all_named:
        if name in standalone_names:
            total, head = _count_parameters(model)
        else:
            total, head = _count_parameters(model, backbone)
        suffix = '(episode-train only)' if name in (episode_train_only or {}) else ''
        lines.append(
            f'  {name:<30} | total: {total:>10,} | head: {head:>10,}  {suffix}'
        )
    if backbone_count:
        lines.append(f'  {"(shared backbone)":<30} | params: {backbone_count:>10,}')

    # ---- Per-episode timing ----
    summ = tracker.summary()
    if not summ:
        lines.append('\n(No timing data collected)')
        return '\n'.join(lines)

    lines.append('\nPer-episode inference time:')
    lines.append(f'  {"Model":<35} | {"wall (ms)":<18} | {"gpu (ms)":<18} | {"peak mem (MB)"}')
    lines.append('  ' + '-' * 85)

    episode_trained_keys = [k for k in summ if k.endswith('_episode_trained')]
    inference_keys       = [k for k in summ if not k.endswith('_episode_trained')]

    def _row(name, s):
        wall = f'{s["avg_wall_ms"]:.2f} ± {s["std_wall_ms"]:.2f}'
        gpu  = f'{s["avg_gpu_ms"]:.2f} ± {s["std_gpu_ms"]:.2f}' if use_cuda else 'N/A'
        mem  = f'{s["avg_peak_mem_mb"]:.1f} (max {s["max_peak_mem_mb"]:.1f})' if use_cuda else 'N/A'
        return f'  {name:<35} | {wall:<18} | {gpu:<18} | {mem}'

    for key in sorted(inference_keys):
        lines.append(_row(key, summ[key]))

    if episode_trained_keys:
        lines.append('\nPer-episode time (episode-trained — inner-loop training + inference):')
        lines.append(f'  {"Model":<35} | {"wall (ms)":<18} | {"gpu (ms)":<18} | {"peak mem (MB)"}')
        lines.append('  ' + '-' * 85)
        for key in sorted(episode_trained_keys):
            lines.append(_row(key, summ[key]))

    lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# OnTheFlyFewShotEvaluator
# ---------------------------------------------------------------------------

class OnTheFlyFewShotEvaluator:
    """
    Nearest-prototype few-shot evaluator. No linear classifier, no OT.
    Uses the model's own pooling strategy at inference time.
    """

    def __init__(self, model, device, export_weights=False):
        self.model = model
        self.device = device
        self.export_weights = export_weights
        self._weight_records = []
        self._current_k_shot = None

    @torch.no_grad()
    def evaluate_task(self, support_data, query_data):
        """
        Single N-way K-shot task evaluation.
        Returns (predictions, true_labels) as CPU tensors.
        """
        self.model.eval()
        support_data = support_data.to(self.device)
        query_data = query_data.to(self.device)

        # ---- PHASE 1: Support (labels known) --------------------------------
        sup_node_embs, sup_fcg_batch = self.model.get_node_embeddings(support_data)
        sup_node_labels = torch.repeat_interleave(
            support_data.family_class, support_data.num_functions
        )

        sup_weights = self._compute_support_weights(
            sup_node_embs, sup_node_labels, support_data, sup_fcg_batch
        )

        if self.export_weights:
            self._record_weights(sup_node_embs, sup_weights, sup_fcg_batch, sup_node_labels, support_data)

        # Node prototypes for query pseudo-labelling
        _, labels_inverse = torch.unique(sup_node_labels, return_inverse=True)
        weighted_sup = sup_node_embs * sup_weights.view(-1, 1)
        node_prototypes = _scatter_mean(weighted_sup, labels_inverse, dim=0)
        node_prototypes = F.normalize(node_prototypes, p=2, dim=-1)

        # Graph prototypes for classification
        sup_graph_embs = self._pool_nodes(sup_node_embs, sup_fcg_batch, sup_weights)
        graph_prototypes = _scatter_mean(sup_graph_embs, support_data.family_class, dim=0)
        graph_prototypes = F.normalize(graph_prototypes, p=2, dim=-1)

        # ---- PHASE 2: Query (pseudo-labels from node prototypes) ------------
        query_node_embs, query_fcg_batch = self.model.get_node_embeddings(query_data)

        query_weights = self._compute_query_weights(
            query_node_embs, node_prototypes, query_data
        )

        query_graph_embs = self._pool_nodes(
            query_node_embs, query_fcg_batch, query_weights
        )

        # Classify by cosine similarity to graph prototypes
        similarities = torch.matmul(query_graph_embs, graph_prototypes.t())
        preds = similarities.argmax(dim=1)

        return preds.cpu(), query_data.family_class.cpu()

    def evaluate_episodes(self, dataset, n_way, k_shot, n_query, n_episodes):
        """
        Run `n_episodes` N-way K-shot tasks and return mean accuracy.
        Used for validation during training and for final test evaluation.
        """
        self._current_k_shot = k_shot
        # Fast label extraction via labels.pt — avoid loading each graph from disk
        from torch.utils.data import ConcatDataset, Subset
        if isinstance(dataset, Subset):
            inner = dataset.dataset
            if isinstance(inner, ConcatDataset):
                all_labels = torch.cat([ds.get_labels() for ds in inner.datasets])
            else:
                all_labels = inner.get_labels()
            subset_labels = all_labels[torch.tensor(dataset.indices, dtype=torch.long)]
        else:
            subset_labels = dataset.get_labels()

        indices_by_class = defaultdict(list)
        for i, lbl in enumerate(subset_labels.tolist()):
            indices_by_class[lbl].append(i)

        # Filter classes with enough samples
        valid_classes = {c: idxs for c, idxs in indices_by_class.items()
                         if len(idxs) >= k_shot}

        correct = total = 0
        completed = attempts = 0

        pbar = tqdm(total=n_episodes, desc=f'{n_way}-way {k_shot}-shot eval')
        while completed < n_episodes and attempts < n_episodes * 3:
            attempts += 1
            s_idx, q_idx, sampled = extract_episode(
                valid_classes, n_way, k_shot, n_query
            )
            if s_idx is None:
                continue

            sup_data, qry_data = build_episode_batches(
                dataset, s_idx, q_idx, sampled, self.device
            )
            preds, labels = self.evaluate_task(sup_data, qry_data)
            correct += (preds == labels).sum().item()
            total += len(labels)
            completed += 1
            pbar.update(1)
            pbar.set_postfix({'acc': f'{correct / max(total, 1):.3f}'})
        pbar.close()

        return correct / max(total, 1)

    # ---- Internal helpers ---------------------------------------------------

    def _compute_support_weights(self, node_embs, node_labels, data, fcg_batch):
        model = self.model


        if isinstance(model, SelfAttentionModel):
            return TrainingUtils.generate_task_specific_weights(
                node_embs, node_labels, fcg_batch, temp=5.0,
            )

        if isinstance(model, SDLDegreeModel):
            return TrainingUtils.generate_sdl_degree_weights(
                node_embeddings=node_embs,
                node_labels=node_labels,
                edge_index=data.fcg_edge_index,
                lambda_a_val=model.raw_lambda_a,
                lambda_b_val=model.raw_lambda_b,
            )

        if isinstance(model, SDLComplexityModel):
            raw_complexity = data.function_features[:, 1].to(node_embs.device)
            return TrainingUtils.generate_sdl_complexity_weights(
                node_embeddings=node_embs,
                node_labels=node_labels,
                raw_complexity=raw_complexity,
                lambda_a_val=model.raw_lambda_a,
                lambda_b_val=model.raw_lambda_b,
            )


        if isinstance(model, SDLWithoutLearnableParamModel):
            return TrainingUtils.generate_task_specific_weights(
                node_embs, node_labels, fcg_batch, temp=5.0,
            )

        if isinstance(model, SDLComplexityPriorModel):
            raw_complexity = data.function_features[:, 1].to(node_embs.device)
            return TrainingUtils.generate_sdl_complexity_prior_weights(raw_complexity)


        if isinstance(model, SDLWithoutStructureModel):
            return TrainingUtils.generate_sdl_without_structure_weights(
                node_embeddings=node_embs,
                node_labels=node_labels,
                lambda_a_val=model.raw_lambda_a,
                lambda_b_val=model.raw_lambda_b,
            )

        raise ValueError(f'Unknown model type: {type(model).__name__}')

    def _compute_query_weights(self, query_node_embs, node_prototypes, data):
        model = self.model

        if isinstance(model, SelfAttentionModel):
            return TrainingUtils.generate_evaluation_weights(
                query_node_embs, node_prototypes, temp=5.0,
            )

        if isinstance(model, SDLDegreeModel):
            return TrainingUtils.generate_sdl_degree_evaluation_weights(
                query_node_embeddings=query_node_embs,
                support_prototypes=node_prototypes,
                edge_index=data.fcg_edge_index,
                lambda_a_val=model.raw_lambda_a,
                lambda_b_val=model.raw_lambda_b,
            )

        if isinstance(model, SDLComplexityModel):
            raw_complexity = data.function_features[:, 1].to(query_node_embs.device)
            return TrainingUtils.generate_sdl_complexity_evaluation_weights(
                query_node_embeddings=query_node_embs,
                support_prototypes=node_prototypes,
                raw_complexity=raw_complexity,
                lambda_a_val=model.raw_lambda_a,
                lambda_b_val=model.raw_lambda_b,
            )


        if isinstance(model, SDLWithoutLearnableParamModel):
            return TrainingUtils.generate_evaluation_weights(
                query_node_embs, node_prototypes, temp=5.0,
            )

        if isinstance(model, SDLComplexityPriorModel):
            raw_complexity = data.function_features[:, 1].to(query_node_embs.device)
            return TrainingUtils.generate_sdl_complexity_prior_weights(raw_complexity)

        if isinstance(model, SDLWithoutStructureModel):
            return TrainingUtils.generate_sdl_without_structure_evaluation_weights(
                query_node_embeddings=query_node_embs,
                support_prototypes=node_prototypes,
                lambda_a_val=model.raw_lambda_a,
                lambda_b_val=model.raw_lambda_b,
            )

        raise ValueError(f'Unknown model type: {type(model).__name__}')

    def _record_weights(self, node_embs, sup_weights, batch, node_labels, support_data):
        model = self.model
        is_sdl = isinstance(model, (SDLComplexityModel, SDLDegreeModel))
        is_sa  = isinstance(model, SelfAttentionModel)
        if not (is_sdl or is_sa):
            return

        if is_sa:
            raw_w = model.attentive_pooling.attn_mlp(node_embs).squeeze(-1).detach()
        else:
            raw_w = sup_weights.detach()

        softmax_w = _scatter_softmax(raw_w, batch).detach()

        num_functions = support_data.num_functions.tolist()
        hashes_per_node = [h for h, nf in zip(support_data.hash, num_functions) for _ in range(nf)]
        func_idx = torch.cat([torch.arange(nf) for nf in num_functions])
        n_nodes = raw_w.size(0)

        self._weight_records.append({
            'raw_weight':     raw_w.cpu(),
            'softmax_weight': softmax_w.cpu(),
            'complexity':     support_data.function_features[:, 1].cpu(),
            'is_native':      support_data.function_features[:, 0].cpu(),
            'family':         node_labels.cpu(),
            'k_shot':         torch.full((n_nodes,), self._current_k_shot or 0, dtype=torch.long),
            'func_idx':       func_idx,
            'apk_hash':       hashes_per_node,
        })

    def save_weight_records(self, path):
        if not self._weight_records:
            return
        tensor_keys = ['raw_weight', 'softmax_weight', 'complexity', 'is_native',
                       'family', 'k_shot', 'func_idx']
        combined = {k: torch.cat([r[k] for r in self._weight_records]) for k in tensor_keys}
        combined['apk_hash'] = [h for r in self._weight_records for h in r['apk_hash']]
        torch.save(combined, path)
        print(f'Weight records saved → {path}  ({len(self._weight_records)} episodes, {combined["family"].size(0)} nodes)')

    def _pool_nodes(self, node_embs, batch, weights):
        """Pool nodes using model-specific strategy."""
        if isinstance(self.model, SelfAttentionModel):
            # Self-attention uses its own MLP; external weights not used
            return self.model.forward_attention(node_embeddings=node_embs, batch=batch)

        return self.model.forward_attention(
            node_embeddings=node_embs, batch=batch, weights=weights
        )




# ---------------------------------------------------------------------------
# FCGAT episode helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_fcgat_episode(fcgat, sup_data, qry_data, device):
    """Nearest-prototype classification for StandaloneFCGAT on one episode."""
    sup_data = sup_data.to(device)
    qry_data = qry_data.to(device)
    z_spt, _ = fcgat.embed(sup_data)   # already L2-normalised, (n_spt, D)
    z_qry, _ = fcgat.embed(qry_data)   # (n_qry, D)
    y_spt = sup_data.family_class
    y_qry = qry_data.family_class
    prototypes = _scatter_mean(z_spt, y_spt, dim=0)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    preds = (z_qry @ prototypes.t()).argmax(dim=1)
    return preds.cpu(), y_qry.cpu()


# ---------------------------------------------------------------------------
# Pretrained backbone episode helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_backbone_episode(backbone, sup_data, qry_data, device):
    """
    Nearest-prototype accuracy using a pretrained backbone's graph-level embedding.
    Covers both:
      - whole_graph_backbone  (instance-level NT-Xent only, no episodic fine-tuning)
      - pretrained_backbone   (instance + set-level NT-Xent, no episodic fine-tuning)
    """
    sup_data = sup_data.to(device)
    qry_data = qry_data.to(device)
    z_spt, _ = backbone(sup_data, return_node_embeddings=False)
    z_qry, _ = backbone(qry_data, return_node_embeddings=False)
    z_spt = F.normalize(z_spt, p=2, dim=1)
    z_qry = F.normalize(z_qry, p=2, dim=1)
    prototypes = _scatter_mean(z_spt, sup_data.family_class, dim=0)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    preds = (z_qry @ prototypes.t()).argmax(dim=1)
    return preds.cpu(), qry_data.family_class.cpu()




# ---------------------------------------------------------------------------
# AndroFIM episode helper
# ---------------------------------------------------------------------------

def _evaluate_androfim_episode(androfim_backbone, sup_data, qry_data,
                                hash_to_path, device):
    """
    Run multi-crop ProtoNet (eval_androfim.run_episode) on the same episode
    sampled for HiGraph models.

    sup_data / qry_data are PyG Batch objects; .hash is a list of SHA256 strings
    (strings are not moved to device, so they remain accessible after .to(device)).

    Returns (preds, labels) tensors, or None if any image in the episode is missing
    (e.g. that APK was not converted to an image yet).
    """
    sup_hashes = sup_data.hash if isinstance(sup_data.hash, list) else [sup_data.hash]
    qry_hashes = qry_data.hash if isinstance(qry_data.hash, list) else [qry_data.hash]

    support_paths = [hash_to_path.get(h) for h in sup_hashes]
    query_paths   = [hash_to_path.get(h) for h in qry_hashes]
    if any(p is None for p in support_paths) or any(p is None for p in query_paths):
        return None  # missing image(s) — skip this episode

    support_labels = sup_data.family_class.cpu().tolist()
    query_labels   = qry_data.family_class.cpu().tolist()

    preds_list, labels_list = _androfim_run_episode(
        support_paths, support_labels,
        query_paths,   query_labels,
        androfim_backbone, device,
        b=_ANDROFIM_CROP_SIZE,
        k=_ANDROFIM_N_CROPS,
    )
    return torch.tensor(preds_list, dtype=torch.long), torch.tensor(labels_list, dtype=torch.long)


# ---------------------------------------------------------------------------
# FAMCF episode helper
# ---------------------------------------------------------------------------

def _build_famcf_hash_to_features(features_pkl_path):
    """Return {sha256: {"perm_vec", "api_vec", "opc_vec"}} from features.pkl."""
    import pickle
    with open(features_pkl_path, "rb") as f:
        return pickle.load(f)


@torch.no_grad()
def _evaluate_famcf_episode(famcf_encoders, sup_data, qry_data, hash_to_features, device):
    """
    Soft-voting prototypical classification across three FAMCF encoders
    (permission, api_call, opcode) on the same episode as the graph models.

    Returns accuracy float, or None if any sample's features are missing.
    """
    sup_hashes = sup_data.hash if isinstance(sup_data.hash, list) else [sup_data.hash]
    qry_hashes = qry_data.hash if isinstance(qry_data.hash, list) else [qry_data.hash]

    sup_entries = [hash_to_features.get(h) for h in sup_hashes]
    qry_entries = [hash_to_features.get(h) for h in qry_hashes]
    if any(e is None for e in sup_entries) or any(e is None for e in qry_entries):
        return None

    y_spt = sup_data.family_class
    y_qry = qry_data.family_class

    feat_keys = {
        "permission": "perm_vec",
        "api_call":   "api_vec",
        "opcode":     "opc_vec",
    }

    logits_sum = None
    for ft, encoder in famcf_encoders.items():
        key = feat_keys[ft]
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

    preds = logits_sum.argmax(dim=1).cpu()
    return preds, y_qry.cpu()


# ---------------------------------------------------------------------------
# Unified evaluation pipeline — all models on the same episodes
# ---------------------------------------------------------------------------

_SDL_TYPES = (SDLDegreeModel, SDLComplexityModel)


def _run_unified_episodes(evaluators, fcgat, dataset, indices_by_class,
                           n_way, n_shots, n_query, n_episodes, device,
                           label_prefix, logger,
                           tracker=None,
                           pretrained_backbones=None,
                           androfim_backbone=None, hash_to_path=None,
                           famcf_encoders=None, hash_to_features=None,
                           idx_to_family=None):
    """
    Run episodes for every model in `evaluators` (and optionally fcgat) on the
    SAME support/query split.  Returns {k_shot: {model_name: (mean_acc, ci95)}}.
    
    tracker            : ResourceTracker or None — records per-episode timing/memory.
    """
    

    results = {}
    for k_shot in n_shots:
        valid_classes = {c: idxs for c, idxs in indices_by_class.items()
                         if len(idxs) >= k_shot}
        if len(valid_classes) < n_way:
            logger.warning(f'  {k_shot}-shot: not enough valid classes, skipping')
            continue

        all_accs = defaultdict(list)
        family_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {model: {family_id: [correct, total]}}
        completed = attempts = 0
        pbar = tqdm(total=n_episodes,
                    desc=f'  [{label_prefix}] {n_way}-way {k_shot}-shot')

        while completed < n_episodes and attempts < n_episodes * 3:
            attempts += 1
            s_idx, q_idx, sampled = extract_episode(valid_classes, n_way, k_shot, n_query)
            if s_idx is None:
                continue

            sup_data, qry_data = build_episode_batches(dataset, s_idx, q_idx, sampled, device)

            # Collect (preds, labels) for every model so we can track per-family accuracy.
            episode_results = {}

            for name, ev in evaluators.items():
                with (tracker.track(name) if tracker else nullcontext()):
                    preds, labels = ev.evaluate_task(sup_data, qry_data)
                all_accs[name].append((preds == labels).float().mean().item())
                episode_results[name] = (preds, labels)


            if fcgat is not None:
                with (tracker.track('fcgat') if tracker else nullcontext()):
                    preds, labels = _evaluate_fcgat_episode(fcgat, sup_data, qry_data, device)
                all_accs['fcgat'].append((preds == labels).float().mean().item())
                episode_results['fcgat'] = (preds, labels)

            if pretrained_backbones:
                for name, bb in pretrained_backbones.items():
                    with (tracker.track(name) if tracker else nullcontext()):
                        preds, labels = _evaluate_backbone_episode(bb, sup_data, qry_data, device)
                    all_accs[name].append((preds == labels).float().mean().item())
                    episode_results[name] = (preds, labels)

            if androfim_backbone is not None and hash_to_path is not None:
                with (tracker.track('androfim') if tracker else nullcontext()):
                    result = _evaluate_androfim_episode(
                        androfim_backbone, sup_data, qry_data, hash_to_path, device
                    )
                if result is not None:
                    preds, labels = result
                    all_accs['androfim'].append((preds == labels).float().mean().item())
                    episode_results['androfim'] = (preds, labels)

            if famcf_encoders is not None and hash_to_features is not None:
                with (tracker.track('famcf') if tracker else nullcontext()):
                    result = _evaluate_famcf_episode(
                        famcf_encoders, sup_data, qry_data, hash_to_features, device
                    )
                if result is not None:
                    preds, labels = result
                    all_accs['famcf'].append((preds == labels).float().mean().item())
                    episode_results['famcf'] = (preds, labels)

            # Accumulate per-family correct/total counts.
            for name, (ep_preds, ep_labels) in episode_results.items():
                for p, l in zip(ep_preds.tolist(), ep_labels.tolist()):
                    fid = int(sampled[l])
                    family_stats[name][fid][1] += 1
                    if p == l:
                        family_stats[name][fid][0] += 1

            completed += 1
            pbar.update(1)
            pbar.set_postfix({n: f'{np.mean(a):.3f}' for n, a in all_accs.items()})

        pbar.close()

        shot_results = {}
        for name, accs in all_accs.items():
            if not accs:
                continue
            mean_acc = float(np.mean(accs))
            ci95 = float(st.t.interval(
                0.95, df=len(accs) - 1, loc=mean_acc, scale=st.sem(accs),
            )[1] - mean_acc) if len(accs) > 1 else 0.0
            shot_results[name] = (mean_acc, ci95)
            logger.info(
                f'  [{name}] {n_way}-way {k_shot}-shot: '
                f'{mean_acc * 100:.2f}% ± {ci95 * 100:.2f}%  ({len(accs)} eps)'
            )

        # Per-family accuracy breakdown
        logger.info(f'\n  --- Per-family accuracy [{n_way}-way {k_shot}-shot] ---')
        for name in sorted(family_stats):
            fam_dict = family_stats[name]
            parts = []
            for fid in sorted(fam_dict):
                correct, total = fam_dict[fid]
                fname = idx_to_family.get(fid, str(fid)) if idx_to_family else str(fid)
                parts.append(f'{fname}: {correct}/{total} ({100 * correct / total:.1f}%)')
            logger.info(f'  [{name}] ' + '  |  '.join(parts))

        results[k_shot] = shot_results
    return results


def run_unified_evaluation(
    episodic_models: dict,
    fcgat,
    params: ModelParameters,
    logger: logging.Logger,
    device,
    n_way: int = 5,
    n_shots: list = None,
    n_query: int = 15,
    n_episodes: int = 200,
    pretrained_backbones: dict = None,
    androfim_backbone=None,
    hash_to_path: dict = None,
    famcf_encoders: dict = None,
    hash_to_features: dict = None,
    export_weights_dir: str = None,
):
    """
    Evaluate all models on the exact same episodes for a fair comparison.

    episodic_models    : {variant_name: model}
    fcgat              : StandaloneFCGAT or None
    episode_train_only : {variant_name: model} — no global checkpoint; only
                         participates in the _episode_trained inner-loop variant.

    Returns (results, tracker).
    """
    from utils import DatasetType
    if n_shots is None:
        n_shots = [1, 5]
    

    tracker = ResourceTracker(device)

    _EXPORT_VARIANTS = {'sdl_complexity', 'sdl_degree', 'self_attention'}
    evaluators = {
        name: OnTheFlyFewShotEvaluator(
            model=model, device=device,
            export_weights=(export_weights_dir is not None and name in _EXPORT_VARIANTS),
        )
        for name, model in episodic_models.items()
    }

    def _build_idx_to_family(ds):
        from torch.utils.data import ConcatDataset, Subset as TSubset
        inner = ds.dataset if isinstance(ds, TSubset) else ds
        if isinstance(inner, ConcatDataset):
            inner = inner.datasets[0]
        try:
            mapping = inner.get_family_mapping()
            return {v: k for k, v in mapping.items()}
        except Exception:
            return None

    results = {}

    if params.dataset_split_type == DatasetType.NoOverlap:
        logger.info(f'Dataset split type: {params.dataset_split_type.value} — evaluating on held-out test split')
        _, _, test_ds = Utils.get_splits(params)

        from torch.utils.data import ConcatDataset
        
        inner = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
        if isinstance(inner, ConcatDataset):
            all_labels = torch.cat([ds.get_labels() for ds in inner.datasets])
        else:
            all_labels = inner.get_labels()
        subset_labels = all_labels[torch.tensor(test_ds.indices, dtype=torch.long)]

        indices_by_class = defaultdict(list)
        for i, lbl in enumerate(subset_labels.tolist()):
            indices_by_class[lbl].append(i)

        n_families = len(indices_by_class)
        logger.info(f'Test split — families: {n_families}  samples: {len(subset_labels)}')

        if n_families < n_way:
            logger.warning(f'Skipping: only {n_families} families (need {n_way})')
            return results, tracker

        results['test_split'] = _run_unified_episodes(
            evaluators, fcgat, test_ds, indices_by_class,
            n_way, n_shots, n_query, n_episodes, device, 'test_split', logger,
            tracker=tracker,
            pretrained_backbones=pretrained_backbones,
            androfim_backbone=androfim_backbone, hash_to_path=hash_to_path,
            famcf_encoders=famcf_encoders, hash_to_features=hash_to_features,
            idx_to_family=_build_idx_to_family(test_ds),
        )

    else:
        logger.error(f'Dataset split type: {params.dataset_split_type.value} — not supported')

        

    if export_weights_dir is not None:
        os.makedirs(export_weights_dir, exist_ok=True)
        for name in _EXPORT_VARIANTS:
            if name in evaluators:
                path = os.path.join(export_weights_dir, f'weight_records_{name}.pt')
                evaluators[name].save_weight_records(path)

    return results, tracker


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    from utils import DatasetType
    from fcgat_model import StandaloneFCGAT

    parser = argparse.ArgumentParser(description='Unified few-shot evaluation — all models on same episodes')
    parser.add_argument('--config', default='trial_config_v3')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shots', type=int, nargs='+', default=[1, 5])
    parser.add_argument('--n_query', type=int, default=15)
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--androfim_ckpt', default=None,
                        help='Path to AndroFIM finetuned ResNet-12 checkpoint.')
    parser.add_argument('--androfim_manifest', default=None,
                        help='Path to androFIM_manifest.csv. Defaults to dataset/<dataset_name>/androfim/androFIM_manifest.csv.')
    parser.add_argument('--export_weights_dir', default=None,
                        help='Directory to save per-function weight records (.pt) for sdl_complexity, '
                             'sdl_degree, and self_attention. One file per variant.')
    parser.add_argument('--ablation_only', action='store_true',
                        help='Only evaluate ablation models (no FCGAT, AndoFIM, or FAMCF).')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    getter = getattr(ExperimentConfig, f'get_{args.config}')
    params: ModelParameters = getter()
    device = Utils.get_device()

    frozen_str = 'frozen' if args.freeze_backbone else 'unfrozen'
    log_dir = os.path.join(
        Utils.get_project_root(), 'trained_models',
        params.id, params.dataset_name, params.dataset_split_type.value, 'evaluation_logs', frozen_str,
    )
    os.makedirs(log_dir, exist_ok=True)
    
    now = time.strftime('%Y%m%d_%H%M%S')
    
    log_path = os.path.join(log_dir,  f'evaluate_{now}.log' if not args.ablation_only else f'evaluate_ablation_{now}.log')
    logger = Utils.setup_logger(f'unified_eval_{frozen_str}_{params.dataset_split_type.value}', log_path)

    logger.info(f'Config: {params.id}  Dataset split type: {params.dataset_split_type.value}  '
                f'Seed: {args.seed}  Device: {device}')

    # Shared backbone (all episodic heads use the same frozen backbone)
    backbone = Backbone(
        cfg_dim=params.cfg_dim,
        func_feat_dim=params.func_feat_dim,
        cfg_hidden_channels=params.cfg_hidden_channels,
        fcg_hidden_channels=params.fcg_hidden_channels,
        final_channels=params.final_channels,
        dropout_value=params.dropout,
    ).to(device)



    # Load all available episodic head variants
    episodic_models = {}
    episode_train_only = {}
    REGISTRY = {**ABLATION_REGISTRY, **SDL_REGISTRY} if args.ablation_only else BASELINE_REGISTRY
    for variant in REGISTRY.keys():
        model_path = Utils.get_episodic_model_path(params, variant, args.freeze_backbone)
        if not os.path.exists(model_path):
            
            logger.info(f'  Skipping {variant} — no checkpoint at {model_path}')
            continue
        model = create_model(variant, backbone, fcg_dim=params.final_channels).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        episodic_models[variant] = model
        logger.info(f'  Loaded {variant}')

    # Load FCGAT if available
    fcgat = None
    fcgat_path = Utils.get_fcgat_model_saved_path(params)
    if os.path.exists(fcgat_path) and not args.ablation_only:
        fcgat = StandaloneFCGAT(
            cfg_dim=params.cfg_dim,
            func_feat_dim=params.func_feat_dim,
        ).to(device)
        fcgat.load_state_dict(torch.load(fcgat_path, map_location=device), strict=False)
        fcgat.eval()
        logger.info(f'  Loaded FCGAT')
    else:
        logger.info(f'  FCGAT not found at {fcgat_path}, skipping')

    # Load pretrained backbones (no episodic fine-tuning)
    pretrained_backbones = {}

    whole_graph_path = Utils.get_whole_graph_backbone_model_saved_path(params)
    if os.path.exists(whole_graph_path) and not args.ablation_only:
        bb = Backbone(
            cfg_dim=params.cfg_dim,
            func_feat_dim=params.func_feat_dim,
            cfg_hidden_channels=params.cfg_hidden_channels,
            fcg_hidden_channels=params.fcg_hidden_channels,
            final_channels=params.final_channels,
            dropout_value=params.dropout,
        ).to(device)
        bb.load_state_dict(torch.load(whole_graph_path, map_location=device))
        bb.eval()
        pretrained_backbones['whole_graph_backbone'] = bb
        logger.info('  Loaded whole-graph backbone (instance-level NT-Xent only)')
    else:
        logger.info(f'  Whole-graph backbone not found at {whole_graph_path}, skipping')

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
        logger.info('  Loaded pretrained backbone (instance + set-level NT-Xent)')
    else:
        logger.info(f'  Pretrained backbone not found at {pretrained_path}, skipping')

    # Load FAMCF encoders + build hash→features lookup
    famcf_encoders = {} if not args.ablation_only else None
    hash_to_features = None
    famcf_save_dir = Utils.get_famcf_save_dir(params.id, params.dataset_name, params.dataset_split_type.value)
    famcf_features_pkl = os.path.join(
        Utils.get_project_root(), 'dataset', params.dataset_name, 'famcf', 'features.pkl'
    )
    _short = {"permission": "per", "api_call": "api", "opcode": "opc"}
    if not args.ablation_only:
        for ft, short in _short.items():
            enc_path = os.path.join(famcf_save_dir, f'encoder_{short}.pth')
            cfg_path = os.path.join(famcf_save_dir, f'model_config_{ft}.json')
            if os.path.exists(enc_path) and os.path.exists(cfg_path):
                import json as _json
                with open(cfg_path) as _f:
                    cfg = _json.load(_f)
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
        logger.warning(f'  FAMCF encoders loaded but features.pkl not found at {famcf_features_pkl} — skipping FAMCF')
        famcf_encoders = {}
    if not famcf_encoders:
        famcf_encoders = None
        hash_to_features = None

    # Load AndroFIM backbone + build hash→path lookup
    androfim_backbone = None
    hash_to_path = None
    _androfim_split = params.dataset_split_type.value
    _default_ckpt   = os.path.join(
        Utils.get_androfim_save_dir(params.id, params.dataset_name, _androfim_split),
        'finetuned_backbone.pth',
    )
    androfim_ckpt_path     = Path(args.androfim_ckpt if args.androfim_ckpt is not None else _default_ckpt)
    androfim_manifest_path = Path(args.androfim_manifest) if args.androfim_manifest is not None \
                             else Path(Utils.get_androfim_manifest(params.dataset_name))
    if androfim_ckpt_path.exists() and androfim_manifest_path.exists() and not args.ablation_only:
        androfim_backbone = _load_androfim_backbone(str(androfim_ckpt_path), device=device)
        img_root  = androfim_manifest_path.parent
        flat = params.dataset_split_type == DatasetType.NoOverlap and params.dataset_name != 'higraph'
        hash_to_path = _build_androfim_hash_to_path(androfim_manifest_path, img_root, flat=flat)
        if hash_to_path:
            logger.info(f'  Loaded AndroFIM backbone ({len(hash_to_path):,} images in manifest)')
        else:
            logger.warning('  AndroFIM skipped — no image files found on disk')
            androfim_backbone = None
            hash_to_path = None
    else:
        logger.info(
            f'  AndroFIM skipped — '
            f'ckpt exists={androfim_ckpt_path.exists()}, '
            f'manifest exists={androfim_manifest_path.exists()}'
        )

    if not episodic_models and not episode_train_only and fcgat is None \
            and not pretrained_backbones and androfim_backbone is None \
            and not famcf_encoders:
        logger.error('No models found. Train some models first.')
        return

    standalone = (["fcgat"] if fcgat else []) \
               + list(pretrained_backbones) \
               + (["androfim"] if androfim_backbone is not None else []) \
               + ([f"famcf_{ft}" for ft in (famcf_encoders or {})])
    logger.info(f'\nModels to evaluate: {list(episodic_models.keys()) + standalone}')
    

    

    _, tracker = run_unified_evaluation(
        episodic_models=episodic_models,
        fcgat=fcgat,
        params=params,
        logger=logger,
        device=device,
        n_way=args.n_way,
        n_shots=args.n_shots,
        n_query=args.n_query,
        n_episodes=args.n_episodes,
        pretrained_backbones=pretrained_backbones,
        androfim_backbone=androfim_backbone,
        hash_to_path=hash_to_path,
        famcf_encoders=famcf_encoders,
        hash_to_features=hash_to_features,
        export_weights_dir=args.export_weights_dir,
    )

    # ---- Resource report ----
    report = _format_resource_report(
        tracker, episodic_models, backbone, fcgat, episode_train_only,
        params.id, params.dataset_split_type.value,
        pretrained_backbones=pretrained_backbones,
        androfim_backbone=androfim_backbone,
        famcf_encoders=famcf_encoders,
    )
    print(report)
    logger.info(report)

    report_path = os.path.join(log_dir, 'resource_report.txt' if not args.ablation_only else 'resource_report_ablation.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f'Resource report saved to {report_path}')


if __name__ == '__main__':
    main()