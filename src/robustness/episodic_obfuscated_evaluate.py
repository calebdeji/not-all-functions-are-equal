"""
episodic_obfuscated_evaluate.py

Robustness evaluation: measures few-shot accuracy on structurally-obfuscated
novel-family APKs and reports degradation vs the clean test split.

Three obfuscation methods (Obfuscapk):
  arithmetic_branch  — semantics-preserving dead-code insertion
  call_indirection   — call-graph pollution via wrapper functions
  reorder            — basic-block reordering (control-flow obfuscation)

For each method and each shot count, prints:
  clean accuracy  |  obfuscated accuracy  |  delta

Prerequisites:
  1. python src/prepare_novel_apks.py                 -- stage originals
  2. bash obfuscate_higraph.sh                        -- obfuscate staged APKs
  3. python src/prepare_novel_apks.py --collect_only  -- collect into novel folder
  4. bash run_obfuscation_extraction.sh               -- extract graphs

Usage:
  python src/episodic_obfuscated_evaluate.py --config trial_config_v3_higraph_no_overlap
  python src/episodic_obfuscated_evaluate.py --config trial_config_v3_higraph_no_overlap \\
      --methods arithmetic_branch reorder --n_episodes 200 --n_shots 1 5
"""

import argparse
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm

from torch_geometric.data import Dataset as PyGDataset
from src.core.backbone import Backbone
from src.core.experiment_configs import ExperimentConfig
from src.core.parse_dataset import HiGraphDataset, HiGraphData
from src.core.utils import Utils, ModelParameters
from src.train.episodic_train import (
    BASELINE_REGISTRY,
    create_model,
    extract_episode,
    build_episode_batches,
)
from src.eval.episodic_evaluate import OnTheFlyFewShotEvaluator
from src.train.train_famcf import MLPEncoder
from src.eval.eval_androfim import (
    run_episode   as _androfim_run_episode,
    load_backbone as _load_androfim_backbone,
    CROP_SIZE     as _ANDROFIM_CROP_SIZE,
    N_CROPS       as _ANDROFIM_N_CROPS,
)

PROJECT_ROOT = Path(Utils.get_project_root())


# ---------------------------------------------------------------------------
# Obfuscated graph dataset
# ---------------------------------------------------------------------------

class ObfuscatedHiGraphDataset(PyGDataset):
    """
    PyG Dataset for pre-extracted obfuscated HiGraph JSON files.

    Unlike HiGraphDataset, this class:
    - Discovers raw files by listing the directory (no CSV filtering)
    - Skips download() — raw files are already produced by apk_to_higraph.py
    - Reads the family label directly from each JSON file (no CSV lookup)
    """

    def __init__(self, year_dir: str, year: str, node_dim: int = 11):
        self._year_dir  = Path(year_dir)
        self.year       = year
        self.node_dim   = node_dim
        self.family_mapping = HiGraphDataset.get_family_mapping()
        super().__init__(str(self._year_dir))

    @property
    def raw_dir(self) -> str:
        return str(self._year_dir / 'raw')

    @property
    def raw_file_names(self):
        if not os.path.isdir(self.raw_dir):
            return []
        return sorted(f for f in os.listdir(self.raw_dir) if f.endswith('.json'))

    @property
    def processed_file_names(self):
        lbl_path = os.path.join(self.processed_dir, 'labels.pt')
        if os.path.exists(lbl_path):
            labels = torch.load(lbl_path, weights_only=True)
            return [f'data_{i}.pt' for i in range(len(labels))]
        return [f'data_{i}.pt' for i in range(len(self.raw_file_names))]

    def download(self):
        pass

    def get_labels(self):
        return torch.load(
            os.path.join(self.processed_dir, 'labels.pt'), weights_only=True
        )

    def get_hashes(self):
        return torch.load(
            os.path.join(self.processed_dir, 'hashes.pt'), weights_only=False
        )

    def process(self):
        idx = 0
        all_labels = []
        all_hashes = []

        for raw_path in self.raw_paths:
            with open(raw_path, 'r') as f:
                json_data = json.load(f)

            family_name = json_data['family'].lower()
            if family_name not in self.family_mapping:
                print(f'[obfuscated] Skipping {raw_path}: unknown family {family_name!r}')
                continue

            fcg_edges      = json_data['function_edges']
            function_count = json_data['function_count']
            cfe            = json_data['control_flow_edges']
            cfe_map        = cfe if isinstance(cfe, dict) else json.loads(cfe)

            in_deg, out_deg = HiGraphDataset.compute_degrees(fcg_edges, function_count)
            ldp             = HiGraphDataset.compute_fcg_ldp(fcg_edges, function_count, in_deg, out_deg)

            func_feats = []
            for i in range(function_count):
                fid = str(i)
                type_feat = torch.tensor([1.0 if fid in cfe_map else 0.0], dtype=torch.float)
                if fid in cfe_map:
                    bi = cfe_map[fid]
                    complexity = HiGraphDataset.compute_cyclomatic_complexity(
                        len(bi['block_features']), bi['block_edges']
                    )
                else:
                    complexity = torch.tensor([0.0], dtype=torch.float)
                ldp_feat = torch.log1p(torch.tensor(ldp[i].tolist(), dtype=torch.float))[1:]
                func_feats.append(torch.cat([
                    type_feat, complexity, ldp_feat,
                    torch.log1p(torch.tensor([in_deg[i]], dtype=torch.float)),
                    torch.log1p(torch.tensor([out_deg[i]], dtype=torch.float)),
                ]))

            x_list, ei_list, batch_list = [], [], []
            node_offset = 0
            for i in range(function_count):
                fid = str(i)
                if fid in cfe_map:
                    bi     = cfe_map[fid]
                    x_raw  = torch.tensor(bi['block_features'], dtype=torch.float)
                    x      = torch.log1p(x_raw[:, :self.node_dim - 1])
                    edges  = torch.tensor(bi['block_edges'], dtype=torch.long)
                    if edges.numel() > 0:
                        ei_list.append(edges + node_offset)
                else:
                    x = torch.zeros(1, self.node_dim - 1, dtype=torch.float)
                n = x.shape[0]
                x_list.append(x)
                batch_list.append(torch.full((n,), i, dtype=torch.long))
                node_offset += n

            data = HiGraphData()
            data.x               = torch.cat(x_list, dim=0)
            data.edge_index      = torch.cat(ei_list, dim=1) if ei_list else torch.empty((2, 0), dtype=torch.long)
            data.cfg_node_batch  = torch.cat(batch_list, dim=0)
            data.function_features = torch.stack(func_feats)
            data.fcg_edge_index  = torch.tensor(fcg_edges, dtype=torch.long)
            data.family          = family_name
            data.hash            = json_data['hash']
            data.year_value      = torch.tensor([int(json_data['year']) - 2010], dtype=torch.long)
            data.num_functions   = torch.tensor([function_count], dtype=torch.long)
            data.y               = self.family_mapping[family_name]

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            all_labels.append(data.y)
            all_hashes.append(data.hash)
            idx += 1

        torch.save(
            torch.tensor(all_labels, dtype=torch.long),
            os.path.join(self.processed_dir, 'labels.pt'),
        )
        torch.save(all_hashes, os.path.join(self.processed_dir, 'hashes.pt'))
        print(f'[obfuscated] Processed {idx} samples for year {self.year}')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(
            os.path.join(self.processed_dir, f'data_{idx}.pt'),
            weights_only=False,
            map_location=torch.device('cpu'),
        )

ALL_METHODS  = ["arithmetic_branch", "call_indirection", "reorder"]
METHOD_LABEL = {
    "arithmetic_branch": "Dead-code insertion (ArithmeticBranch)",
    "call_indirection":  "Call-graph pollution (CallIndirection)",
    "reorder":           "Control-flow obfuscation (Reorder)",
}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_obfuscated_dataset(graph_root: Path, logger: logging.Logger):
    """
    Build a ConcatDataset from all year subdirs under graph_root that have
    raw JSON files.  Uses ObfuscatedHiGraphDataset which reads files by
    directory listing and does not rely on the standard HiGraph CSV.

    Returns (dataset, indices_by_class, obf_hashes) or (None, None, None).
    obf_hashes is the set of APK hashes that were successfully obfuscated,
    used to build a matched clean baseline for fair degradation comparison.
    """
    year_dirs = sorted(
        d for d in graph_root.iterdir()
        if d.is_dir() and any((d / "raw").glob("*.json"))
    )
    if not year_dirs:
        return None, None, None

    year_datasets = []
    for year_dir in year_dirs:
        year = year_dir.name
        try:
            ds = ObfuscatedHiGraphDataset(year_dir=str(year_dir), year=year)
            if len(ds) > 0:
                year_datasets.append(ds)
                logger.info(f"    [{year}] {len(ds)} samples")
        except Exception as e:
            logger.warning(f"    [{year}] failed to load: {e}")

    if not year_datasets:
        return None, None, None

    combined = ConcatDataset(year_datasets)
    all_labels = torch.cat([ds.get_labels() for ds in year_datasets])
    obf_hashes = set(h for ds in year_datasets for h in ds.get_hashes())

    indices_by_class: dict = defaultdict(list)
    for i, lbl in enumerate(all_labels.tolist()):
        indices_by_class[lbl].append(i)

    return combined, indices_by_class, obf_hashes


def load_clean_test_split(params: ModelParameters):
    """
    Load the standard test split and build indices_by_class.

    For no-overlap splits the test set contains only novel families, so no
    additional filtering is needed.
    """
    _, _, test_ds = Utils.get_splits(params)

    inner = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
    if isinstance(inner, ConcatDataset):
        all_labels = torch.cat([ds.get_labels() for ds in inner.datasets])
    else:
        all_labels = inner.get_labels()

    subset_labels = all_labels[torch.tensor(test_ds.indices, dtype=torch.long)]
    indices_by_class: dict = defaultdict(list)
    for i, lbl in enumerate(subset_labels.tolist()):
        indices_by_class[lbl].append(i)

    return test_ds, indices_by_class


def load_matched_clean_dataset(test_ds, obf_hashes: set):
    """
    Filter the clean test split to only the APKs that were actually obfuscated.

    Returns:
      matched_subset       : Subset of test_ds
      indices_by_class     : {class_id: [position_in_matched_subset]}
      clean_hash_to_idx    : {hash: position_in_matched_subset}
                             Used by the paired runner to look up the clean
                             counterpart of each obfuscated APK by hash.

    Returns (None, None, None) if no overlap found.
    """
    inner = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
    datasets = inner.datasets if isinstance(inner, ConcatDataset) else [inner]

    # Build {hash: global_index_in_ConcatDataset} from raw_file_names (no .pt loading)
    hash_to_global: dict = {}
    offset = 0
    for ds in datasets:
        for local_i, fname in enumerate(ds.raw_file_names):
            h = os.path.splitext(fname)[0]
            hash_to_global[h] = offset + local_i
        offset += len(ds)

    # Map global indices → position in test_ds.indices list
    test_indices = test_ds.indices if isinstance(test_ds, Subset) else list(range(len(test_ds)))
    global_to_subset_pos = {g: j for j, g in enumerate(test_indices)}

    # Build matched_positions and clean_hash_to_idx together (deterministic order)
    clean_hash_to_idx: dict = {}
    matched_positions: list = []
    for h in sorted(obf_hashes):
        if h in hash_to_global and hash_to_global[h] in global_to_subset_pos:
            clean_hash_to_idx[h] = len(matched_positions)
            matched_positions.append(global_to_subset_pos[hash_to_global[h]])

    if not matched_positions:
        return None, None, None

    matched_subset = Subset(test_ds, matched_positions)

    # Labels for the matched subset
    all_labels = torch.cat([ds.get_labels() for ds in datasets])
    test_labels = all_labels[torch.tensor(test_indices, dtype=torch.long)]
    matched_labels = test_labels[torch.tensor(matched_positions, dtype=torch.long)]

    indices_by_class: dict = defaultdict(list)
    for i, lbl in enumerate(matched_labels.tolist()):
        indices_by_class[lbl].append(i)

    return matched_subset, indices_by_class, clean_hash_to_idx


def build_obf_idx_to_hash(obf_dataset) -> list:
    """
    Build a list where list[i] = hash of the i-th sample in the obfuscated
    ConcatDataset.  Uses the hashes.pt saved during process() — no .pt loading.
    """
    all_hashes = []
    datasets = obf_dataset.datasets if isinstance(obf_dataset, ConcatDataset) else [obf_dataset]
    for ds in datasets:
        all_hashes.extend(ds.get_hashes())
    return all_hashes


# ---------------------------------------------------------------------------
# FAMCF / AndroFIM helpers for hash-based episode evaluation
# ---------------------------------------------------------------------------

def _build_famcf_hash_to_features(features_pkl_path: str) -> dict:
    with open(features_pkl_path, "rb") as f:
        return pickle.load(f)


def _build_androfim_hash_to_path(manifest_path: str) -> dict:
    df = pd.read_csv(manifest_path)
    img_root = Path(manifest_path).parent
    result = {}
    missing = 0
    for _, row in df.iterrows():
        path = str(img_root / str(row.year) / f"{row.sha256}.png")
        if os.path.exists(path):
            result[str(row.sha256)] = path
        else:
            missing += 1
    if missing:
        print(f'[WARNING] AndroFIM manifest: {missing} entries have no image on disk — skipped')
    return result


@torch.no_grad()
def _eval_famcf_from_hashes(famcf_encoders, sup_hashes, qry_hashes,
                              sup_labels, qry_labels, hash_to_features, device):
    """
    Soft-voting ProtoNet across FAMCF encoders.
    Returns (preds, labels) or None if any hash is missing from features.
    """
    sup_entries = [hash_to_features.get(h) for h in sup_hashes]
    qry_entries = [hash_to_features.get(h) for h in qry_hashes]
    if any(e is None for e in sup_entries) or any(e is None for e in qry_entries):
        return None

    feat_keys = {"permission": "perm_vec", "api_call": "api_vec", "opcode": "opc_vec"}

    logits_sum = None
    for ft, encoder in famcf_encoders.items():
        key = feat_keys[ft]
        sup_feats = torch.tensor(
            np.stack([e[key] for e in sup_entries]), dtype=torch.float32
        ).to(device)
        qry_feats = torch.tensor(
            np.stack([e[key] for e in qry_entries]), dtype=torch.float32
        ).to(device)

        z_sup = encoder(sup_feats)
        z_qry = encoder(qry_feats)

        y_sup = sup_labels.to(device)
        n_way = sup_labels.max().item() + 1
        prototypes = torch.zeros(n_way, z_sup.shape[-1], device=device)
        for c in range(n_way):
            mask = (y_sup == c)
            prototypes[c] = z_sup[mask].mean(0)

        logits = -torch.cdist(z_qry, prototypes)
        logits_sum = logits if logits_sum is None else logits_sum + logits

    preds = logits_sum.argmax(dim=1).cpu()
    return preds, qry_labels


def _eval_androfim_from_hashes(androfim_backbone, sup_hashes, qry_hashes,
                                 sup_labels, qry_labels, hash_to_path, device):
    """
    Multi-crop ProtoNet for AndroFIM.
    Returns (preds, labels) or None if any image is missing.
    """
    sup_paths = [hash_to_path.get(h) for h in sup_hashes]
    qry_paths = [hash_to_path.get(h) for h in qry_hashes]
    if any(p is None for p in sup_paths) or any(p is None for p in qry_paths):
        return None

    preds_list, labels_list = _androfim_run_episode(
        sup_paths, sup_labels.tolist(),
        qry_paths, qry_labels.tolist(),
        androfim_backbone, device,
        b=_ANDROFIM_CROP_SIZE,
        k=_ANDROFIM_N_CROPS,
    )
    return torch.tensor(preds_list, dtype=torch.long), torch.tensor(labels_list, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(params: ModelParameters, device, logger: logging.Logger) -> dict:
    """Load backbone + all available BASELINE_REGISTRY episodic head checkpoints."""
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
        raise FileNotFoundError(f"Backbone checkpoint not found: {bb_path}")
    backbone.load_state_dict(torch.load(bb_path, map_location=device))
    backbone.eval()
    logger.info(f"Loaded backbone: {bb_path}")

    models = {}
    for variant in BASELINE_REGISTRY:
        ckpt = Utils.get_episodic_model_path(params, variant, frozen=False)
        if not os.path.exists(ckpt):
            logger.info(f"  Skipping {variant} — no checkpoint")
            continue
        model = create_model(variant, backbone, fcg_dim=params.final_channels).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models[variant] = model
        logger.info(f"  Loaded {variant}")

    return models


def load_famcf_androfim(params: ModelParameters, device, logger: logging.Logger):
    """
    Load FAMCF encoders + clean hash→features, and AndroFIM backbone + clean hash→path.
    Returns (famcf_encoders, clean_hash_to_features, androfim_backbone, clean_hash_to_path).
    Any component that fails to load is returned as None.
    """
    famcf_encoders = {}
    _short = {"permission": "per", "api_call": "api", "opcode": "opc"}
    famcf_save_dir = Utils.get_famcf_save_dir(
        params.id, params.dataset_name, params.dataset_split_type.value
    )
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
            logger.info(f'  FAMCF encoder not found: {enc_path}')

    clean_features_pkl = os.path.join(
        Utils.get_project_root(), 'dataset', params.dataset_name, 'famcf', 'features.pkl'
    )
    clean_hash_to_features = None
    if famcf_encoders and os.path.exists(clean_features_pkl):
        clean_hash_to_features = _build_famcf_hash_to_features(clean_features_pkl)
        logger.info(f'  Loaded clean FAMCF features ({len(clean_hash_to_features):,} samples)')
    elif famcf_encoders:
        logger.warning(f'  FAMCF: clean features.pkl not found at {clean_features_pkl} — skipping')
        famcf_encoders = {}
    if not famcf_encoders:
        famcf_encoders = None

    androfim_backbone = None
    clean_hash_to_path = None
    androfim_ckpt = os.path.join(
        Utils.get_androfim_save_dir(
            params.id, params.dataset_name, params.dataset_split_type.value
        ),
        'finetuned_backbone.pth',
    )
    androfim_manifest = Utils.get_androfim_manifest(params.dataset_name)
    if os.path.exists(androfim_ckpt) and os.path.exists(androfim_manifest):
        androfim_backbone = _load_androfim_backbone(androfim_ckpt, device=device)
        clean_hash_to_path = _build_androfim_hash_to_path(androfim_manifest)
        logger.info(f'  Loaded AndroFIM backbone ({len(clean_hash_to_path):,} clean images)')
    else:
        logger.info(
            f'  AndroFIM skipped — '
            f'ckpt exists={os.path.exists(androfim_ckpt)}, '
            f'manifest exists={os.path.exists(androfim_manifest)}'
        )

    return famcf_encoders, clean_hash_to_features, androfim_backbone, clean_hash_to_path


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _ci(accs: list) -> float:
    if len(accs) < 2:
        return 0.0
    return float(
        st.t.interval(0.95, df=len(accs) - 1, loc=np.mean(accs), scale=st.sem(accs))[1]
        - np.mean(accs)
    )


# ---------------------------------------------------------------------------
# Paired episode runner
# ---------------------------------------------------------------------------

def run_paired_episodes(
    evaluators: dict,
    obf_dataset,
    obf_ibc: dict,
    clean_matched_subset,
    obf_idx_to_hash: list,
    clean_hash_to_idx: dict,
    n_way: int,
    n_shots: list,
    n_query: int,
    n_episodes: int,
    device,
    label: str,
    logger: logging.Logger,
    famcf_encoders=None,
    clean_hash_to_features=None,
    obf_hash_to_features=None,
    androfim_backbone=None,
    clean_hash_to_path=None,
    obf_hash_to_path=None,
    progress_logger: logging.Logger = None,
    log_every: int = 20,
) -> dict:
    """
    Paired evaluation: each episode samples APKs from the obfuscated dataset,
    then fetches the identical APKs from the clean matched subset by hash.
    Both clean and obfuscated versions are evaluated on the exact same episode
    structure (same families, same support/query split).

    Returns {k_shot: {model_name: {'clean': (mean, ci), 'obf': (mean, ci),
                                   'delta': (mean, ci)}}}.
    Δ is computed per-episode (obf_acc - clean_acc) then averaged, giving a
    tighter estimate than the difference of two independent means.
    """
    results: dict = {}

    for k_shot in n_shots:
        eligible = {c: idxs for c, idxs in obf_ibc.items() if len(idxs) >= k_shot}
        if len(eligible) < n_way:
            logger.warning(
                f"  [{label}] {k_shot}-shot: only {len(eligible)} eligible classes, skipping"
            )
            continue

        clean_accs:  dict = defaultdict(list)
        obf_accs:    dict = defaultdict(list)
        delta_accs:  dict = defaultdict(list)
        completed = attempts = 0
        pbar = tqdm(total=n_episodes, desc=f"  [{label}] {n_way}-way {k_shot}-shot (paired)")

        while completed < n_episodes and attempts < n_episodes * 5:
            attempts += 1

            s_idx, q_idx, sampled = extract_episode(eligible, n_way, k_shot, n_query)
            if s_idx is None:
                continue

            # Map every obfuscated index → clean index by hash
            all_obf_idx = s_idx + q_idx
            clean_map = {}
            skip = False
            for oi in all_obf_idx:
                h = obf_idx_to_hash[oi]
                if h not in clean_hash_to_idx:
                    skip = True
                    break
                clean_map[oi] = clean_hash_to_idx[h]
            if skip:
                continue

            clean_s_idx = [clean_map[i] for i in s_idx]
            clean_q_idx = [clean_map[i] for i in q_idx]

            sup_obf,   qry_obf   = build_episode_batches(obf_dataset,          s_idx,       q_idx,       sampled, device)
            sup_clean, qry_clean = build_episode_batches(clean_matched_subset,  clean_s_idx, clean_q_idx, sampled, device)

            for name, ev in evaluators.items():
                preds_obf,   lbl_obf   = ev.evaluate_task(sup_obf,   qry_obf)
                preds_clean, lbl_clean = ev.evaluate_task(sup_clean, qry_clean)

                acc_obf   = (preds_obf   == lbl_obf).float().mean().item()
                acc_clean = (preds_clean == lbl_clean).float().mean().item()

                obf_accs[name].append(acc_obf)
                clean_accs[name].append(acc_clean)
                delta_accs[name].append(acc_obf - acc_clean)

            # Build hash lists once — shared by FAMCF and AndroFIM
            need_hashes = (
                (famcf_encoders and obf_hash_to_features and clean_hash_to_features) or
                (androfim_backbone and obf_hash_to_path and clean_hash_to_path)
            )
            if need_hashes:
                sup_hashes = [obf_idx_to_hash[i] for i in s_idx]
                qry_hashes = [obf_idx_to_hash[i] for i in q_idx]
                sup_labels = torch.tensor(
                    [c for c in range(n_way) for _ in range(k_shot)], dtype=torch.long
                )
                qry_labels = torch.tensor(
                    [c for c in range(n_way) for _ in range(n_query)], dtype=torch.long
                )

            # FAMCF paired evaluation
            if famcf_encoders and obf_hash_to_features and clean_hash_to_features:
                res_obf   = _eval_famcf_from_hashes(famcf_encoders, sup_hashes, qry_hashes,
                                                     sup_labels, qry_labels,
                                                     obf_hash_to_features, device)
                res_clean = _eval_famcf_from_hashes(famcf_encoders, sup_hashes, qry_hashes,
                                                     sup_labels, qry_labels,
                                                     clean_hash_to_features, device)
                if res_obf is not None and res_clean is not None:
                    acc_obf   = (res_obf[0]   == res_obf[1]).float().mean().item()
                    acc_clean = (res_clean[0]  == res_clean[1]).float().mean().item()
                    obf_accs['famcf'].append(acc_obf)
                    clean_accs['famcf'].append(acc_clean)
                    delta_accs['famcf'].append(acc_obf - acc_clean)

            # AndroFIM paired evaluation
            if androfim_backbone and obf_hash_to_path and clean_hash_to_path:
                res_obf   = _eval_androfim_from_hashes(androfim_backbone, sup_hashes, qry_hashes,
                                                        sup_labels, qry_labels,
                                                        obf_hash_to_path, device)
                res_clean = _eval_androfim_from_hashes(androfim_backbone, sup_hashes, qry_hashes,
                                                        sup_labels, qry_labels,
                                                        clean_hash_to_path, device)
                if res_obf is not None and res_clean is not None:
                    acc_obf   = (res_obf[0]   == res_obf[1]).float().mean().item()
                    acc_clean = (res_clean[0]  == res_clean[1]).float().mean().item()
                    obf_accs['androfim'].append(acc_obf)
                    clean_accs['androfim'].append(acc_clean)
                    delta_accs['androfim'].append(acc_obf - acc_clean)

            completed += 1
            pbar.update(1)

            if progress_logger and completed % log_every == 0:
                progress_logger.info(
                    f"[{label}] {k_shot}-shot  ep {completed}/{n_episodes}"
                )
                for name in sorted(obf_accs):
                    c_mu = np.mean(clean_accs[name]) * 100
                    o_mu = np.mean(obf_accs[name])   * 100
                    d_mu = np.mean(delta_accs[name]) * 100
                    progress_logger.info(
                        f"  {name:<30}  clean {c_mu:6.2f}%  "
                        f"obf {o_mu:6.2f}%  Δ {d_mu:+6.2f}%"
                    )

        pbar.close()

        results[k_shot] = {}
        for name in obf_accs:
            results[k_shot][name] = {
                'clean': (float(np.mean(clean_accs[name])) * 100, _ci(clean_accs[name]) * 100),
                'obf':   (float(np.mean(obf_accs[name]))   * 100, _ci(obf_accs[name])   * 100),
                'delta': (float(np.mean(delta_accs[name])) * 100, _ci(delta_accs[name]) * 100),
            }

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_degradation_table(
    paired_results: dict,
    n_shots: list,
    logger: logging.Logger,
):
    """
    paired_results: {method: {k_shot: {model: {'clean': (mean,ci),
                                                'obf':   (mean,ci),
                                                'delta': (mean,ci)}}}}
    Δ is the paired per-episode difference (obf - clean), averaged across episodes.
    """
    methods = list(paired_results.keys())
    if not methods:
        return

    for k_shot in n_shots:
        logger.info(f"\n{'='*90}")
        logger.info(f"  {k_shot}-shot paired degradation  ({', '.join(methods)})")
        logger.info(f"{'='*90}")

        hdr = f"{'Model':<30}"
        for m in methods:
            hdr += f"  {'Clean':>8}  {m[:12]:>12}  {'Δ (paired)':>12}"
        logger.info(hdr)
        logger.info("-" * len(hdr))

        all_models = set()
        for m in methods:
            all_models |= set(paired_results[m].get(k_shot, {}).keys())

        for name in sorted(all_models):
            row = f"{name:<30}"
            for m in methods:
                r = paired_results[m].get(k_shot, {}).get(name)
                if r is None:
                    row += f"  {'—':>8}  {'—':>12}  {'—':>12}"
                    continue
                c_mu, c_ci = r['clean']
                o_mu, o_ci = r['obf']
                d_mu, d_ci = r['delta']
                row += f"  {c_mu:>5.2f}±{c_ci:<3.2f}  {o_mu:>5.2f}±{o_ci:<3.2f}  {d_mu:>+5.2f}±{d_ci:<3.2f}"
            logger.info(row)

    logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Obfuscation robustness evaluation")
    parser.add_argument("--config",     default="trial_config_v3_higraph_no_overlap",
                        help="ExperimentConfig method suffix")
    parser.add_argument("--methods",    nargs="+", default=ALL_METHODS, choices=ALL_METHODS,
                        help="Obfuscation methods to evaluate (default: all three)")
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--n_shots",    type=int, nargs="+", default=[1, 5])
    parser.add_argument("--n_way",      type=int, default=5)
    parser.add_argument("--n_query",    type=int, default=15)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params: ModelParameters = getattr(ExperimentConfig, f"get_{args.config}")()
    device = Utils.get_device()

    log_dir = os.path.join(
        Utils.get_project_root(), "trained_models",
        params.id, params.dataset_name,
        params.dataset_split_type.value,
        "evaluation_logs", "unfrozen",
    )
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f"evaluate_obfuscated_{ts}.log")
    progress_log_path = os.path.join(log_dir, f"evaluate_obfuscated_progress_{ts}.log")
    logger = Utils.setup_logger("obfuscated_eval", log_path)
    progress_logger = Utils.setup_logger("obfuscated_progress", progress_log_path)

    logger.info(f"Config:    {params.id}")
    logger.info(f"Device:    {device}")
    logger.info(f"Seed:      {args.seed}")
    logger.info(f"Methods:   {args.methods}")
    logger.info(f"Episodes:  {args.n_episodes}  shots: {args.n_shots}  n_way: {args.n_way}")

    # ---- Load models ---------------------------------------------------------
    logger.info("\nLoading models...")
    models = load_models(params, device, logger)
    if not models:
        logger.error("No models loaded. Run training first.")
        return
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")

    evaluators = {
        name: OnTheFlyFewShotEvaluator(model=model, device=device)
        for name, model in models.items()
    }

    # ---- Load FAMCF encoders + AndroFIM backbone (shared across methods) -----
    logger.info("\nLoading FAMCF + AndroFIM...")
    famcf_encoders, clean_hash_to_features, androfim_backbone, clean_hash_to_path = \
        load_famcf_androfim(params, device, logger)

    # ---- Load full test split (used to build matched clean subsets) ----------
    _, _, full_test_ds = Utils.get_splits(params)

    # ---- Per-method: paired clean + obfuscated evaluation -------------------
    all_paired_results: dict = {}

    for method in args.methods:
        graph_root = PROJECT_ROOT / "dataset" / f"obfuscated_higraph_{method}" / "graph"
        logger.info(f"\n=== {METHOD_LABEL[method]} ===")
        logger.info(f"Graph root: {graph_root}")

        if not graph_root.exists():
            logger.warning("  Root not found — run run_obfuscation_extraction.sh first, skipping")
            continue

        logger.info("  Loading obfuscated dataset...")
        obf_ds, obf_ibc, obf_hashes = load_obfuscated_dataset(graph_root, logger)

        if obf_ds is None:
            logger.warning("  No processed samples found, skipping")
            continue

        n_families = len(obf_ibc)
        logger.info(f"  Obfuscated: {len(obf_ds)} samples, {n_families} families")

        if n_families < args.n_way:
            logger.warning(f"  Only {n_families} families (need {args.n_way}), skipping")
            continue

        # Matched clean: clean versions of the same obfuscated APKs
        logger.info(f"  Building matched clean subset ({len(obf_hashes)} obfuscated hashes)...")
        clean_ds, clean_ibc, clean_hash_to_idx = load_matched_clean_dataset(full_test_ds, obf_hashes)

        if clean_ds is None:
            logger.warning("  No clean matches found in test split, skipping")
            continue

        logger.info(f"  Matched clean: {len(clean_ds)} samples, {len(clean_ibc)} families")

        # Precompute obf dataset index → hash for fast lookup during episodes
        obf_idx_to_hash = build_obf_idx_to_hash(obf_ds)

        # Per-method obfuscated FAMCF features
        obf_hash_to_features = None
        if famcf_encoders:
            obf_feat_pkl = PROJECT_ROOT / "dataset" / f"obfuscated_famcf_{method}" / "features.pkl"
            if obf_feat_pkl.exists():
                obf_hash_to_features = _build_famcf_hash_to_features(str(obf_feat_pkl))
                logger.info(f"  Loaded obf FAMCF features ({len(obf_hash_to_features):,} samples)")
            else:
                logger.info(f"  Obf FAMCF features not found at {obf_feat_pkl} — run extraction first")

        # Per-method obfuscated AndroFIM images
        obf_hash_to_path = None
        if androfim_backbone:
            obf_manifest = PROJECT_ROOT / "dataset" / f"obfuscated_androfim_{method}" / "androFIM_manifest.csv"
            if obf_manifest.exists():
                obf_hash_to_path = _build_androfim_hash_to_path(str(obf_manifest))
                logger.info(f"  Loaded obf AndroFIM paths ({len(obf_hash_to_path):,} images)")
            else:
                logger.info(f"  Obf AndroFIM manifest not found at {obf_manifest} — run extraction first")

        logger.info("  Running paired episodes (same APKs, clean vs obfuscated)...")
        paired = run_paired_episodes(
            evaluators,
            obf_ds, obf_ibc,
            clean_ds,
            obf_idx_to_hash, clean_hash_to_idx,
            args.n_way, args.n_shots, args.n_query, args.n_episodes,
            device, method, logger,
            famcf_encoders=famcf_encoders,
            clean_hash_to_features=clean_hash_to_features,
            obf_hash_to_features=obf_hash_to_features,
            androfim_backbone=androfim_backbone,
            clean_hash_to_path=clean_hash_to_path,
            obf_hash_to_path=obf_hash_to_path,
            progress_logger=progress_logger,
        )
        all_paired_results[method] = paired

        logger.info(f"\n--- {method} paired results ---")
        for k in args.n_shots:
            for name, r in sorted(paired.get(k, {}).items()):
                c_mu, c_ci = r['clean']
                o_mu, o_ci = r['obf']
                d_mu, d_ci = r['delta']
                logger.info(
                    f"  {name:<30}  {k}-shot: "
                    f"clean {c_mu:.2f}%±{c_ci:.2f}  "
                    f"obf {o_mu:.2f}%±{o_ci:.2f}  "
                    f"Δ {d_mu:+.2f}%±{d_ci:.2f}"
                )

    # ---- Degradation table ---------------------------------------------------
    if all_paired_results:
        print_degradation_table(all_paired_results, args.n_shots, logger)

    logger.info(f"Log saved: {log_path}")
    logger.info(f"Progress log: {progress_log_path}")


if __name__ == "__main__":
    main()
