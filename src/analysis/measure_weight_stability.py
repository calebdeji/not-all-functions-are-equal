"""
measure_weight_stability.py

Node-weight stability under obfuscation, measured on MATCHED functions within
the same APK (clean vs. obfuscated), not across families.

Design (mirrors the clean-support / obfuscated-query accuracy protocol):
  * Episodes are sampled over the obfuscated novel-family dataset, exactly as in
    episodic_obfuscated_evaluate.run_paired_episodes.
  * The support set is taken from the CLEAN counterparts (by hash) and defines a
    single set of node prototypes P per episode.
  * The query APK's per-function SDL-Complexity weights are computed twice
    against the SAME prototypes P: once from its clean graph (w_clean), once from
    its obfuscated graph (w_obf). Holding P fixed isolates the effect of
    query-side obfuscation on the weights.
  * Functions are matched across clean/obf by their full Dalvik signature
    (JSON `function_names`, which aligns 1:1 with function index).

Metrics (per query APK, averaged over episodes, then pooled per method):
  * spearman   : rank correlation of w over matched functions.
  * ret@k      : clean-anchored top-k retention =
                 |top-k(clean) ∩ top-k(obf)| / k.
                 "Of the functions SDL weighted most in the clean sample, how
                  many remain top-k after obfuscation." Injected functions
                 (e.g. CallIndirection wrappers) can only hurt by displacing a
                 clean-important function; they are not otherwise counted.
  * coverage   : matched functions / clean function count. Reported separately
                 so churn (obf-only functions) is never folded into ret@k.

Obfuscated graphs live under dataset/obfuscated_higraph_{method}/graph and are
produced on the machine that ran obfuscate_higraph.sh + run_obfuscation_extraction.sh.

Usage:
  python -m src.analysis.measure_weight_stability
  python -m src.analysis.measure_weight_stability --methods reorder --n_episodes 200 --k_shot 5
"""

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


from src.core.backbone import Backbone
from src.core.experiment_configs import ExperimentConfig
from src.core.episodic_models import TrainingUtils, _scatter_mean
from src.train.episodic_train import create_model, extract_episode
from src.core.utils import Utils
from src.robustness.episodic_obfuscated_evaluate import (
    load_obfuscated_dataset,
    load_matched_clean_dataset,
    load_clean_test_split,
    build_obf_idx_to_hash,
)

ALL_METHODS = ["arithmetic_branch", "call_indirection", "reorder"]


# ---------------------------------------------------------------------------
# Core SDL-Complexity weight math (refactored from extract_sdl_weights.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def node_prototypes_from_support(model, sup_data, device):
    """Phase 1: clean support batch -> SDL-weighted, L2-normalised node prototypes."""
    sup_node_embs, _ = model.get_node_embeddings(sup_data)
    sup_node_labels = torch.repeat_interleave(sup_data.family_class, sup_data.num_functions)
    sup_weights = TrainingUtils.generate_sdl_complexity_weights(
        node_embeddings=sup_node_embs,
        node_labels=sup_node_labels,
        raw_complexity=sup_data.function_features[:, 1].to(device),
        lambda_a_val=model.raw_lambda_a,
        lambda_b_val=model.raw_lambda_b,
    )
    _, labels_inv = torch.unique(sup_node_labels, return_inverse=True)
    protos = _scatter_mean(sup_node_embs * sup_weights.view(-1, 1), labels_inv, dim=0)
    return F.normalize(protos, p=2, dim=-1)


@torch.no_grad()
def function_weights(model, qry_data, node_prototypes, device):
    """Phase 2: per-function SDL-Complexity weight vector for one query graph.

    Uses the same sigmoid(5 * (la*sim_max + lb*c_norm - sim_others)) form as
    extract_sdl_weights.py. Returns a 1-D numpy array of length n_functions.
    """
    qry_node_embs, _ = model.get_node_embeddings(qry_data)
    raw_c = qry_data.function_features[:, 1].to(device).float().view(-1)
    if raw_c.max() > raw_c.min():
        c_norm = (raw_c - raw_c.min()) / (raw_c.max() - raw_c.min() + 1e-6)
    else:
        c_norm = torch.zeros_like(raw_c)

    q = F.normalize(qry_node_embs, p=2, dim=1)
    p = F.normalize(node_prototypes, p=2, dim=1)
    sim = torch.matmul(q, p.t())                       # (n_funcs, n_classes)

    sim_max, _ = sim.max(dim=1)
    n_classes = p.size(0)
    sim_others = ((sim.sum(dim=1) - sim_max) / (n_classes - 1)
                  if n_classes > 1 else torch.zeros_like(sim_max))

    la = model.raw_lambda_a.item()
    lb = model.raw_lambda_b.item()
    raw_score = (la * sim_max) + (lb * c_norm) - sim_others
    weights = torch.sigmoid(raw_score * 5.0)
    return weights.detach().cpu().numpy()


@torch.no_grad()
def weights_and_readout(model, qry_data, node_prototypes, device):
    """Per-function weights AND the weighted graph readout h_G for one query graph.

    Same weight computation as function_weights, but also returns the SDL-weighted,
    L2-normalised graph embedding (model.forward_attention applies softmax over nodes
    and the weighted sum). h_G is what the classifier compares to class prototypes,
    so cosine(h_G_clean, h_G_obf) measures stability at the level that drives accuracy.

    Returns (weights [n_funcs] np.ndarray, h_G [D] np.ndarray, already L2-normalised).
    """
    qry_node_embs, qry_fcg_batch = model.get_node_embeddings(qry_data)
    raw_c = qry_data.function_features[:, 1].to(device).float().view(-1)
    if raw_c.max() > raw_c.min():
        c_norm = (raw_c - raw_c.min()) / (raw_c.max() - raw_c.min() + 1e-6)
    else:
        c_norm = torch.zeros_like(raw_c)

    q = F.normalize(qry_node_embs, p=2, dim=1)
    p = F.normalize(node_prototypes, p=2, dim=1)
    sim = torch.matmul(q, p.t())

    sim_max, _ = sim.max(dim=1)
    n_classes = p.size(0)
    sim_others = ((sim.sum(dim=1) - sim_max) / (n_classes - 1)
                  if n_classes > 1 else torch.zeros_like(sim_max))

    la = model.raw_lambda_a.item()
    lb = model.raw_lambda_b.item()
    raw_score = (la * sim_max) + (lb * c_norm) - sim_others
    weights = torch.sigmoid(raw_score * 5.0)

    h_G = model.forward_attention(qry_node_embs, qry_fcg_batch, weights)
    h_G = F.normalize(h_G, p=2, dim=-1).view(-1)
    return weights.detach().cpu().numpy(), h_G.detach().cpu().numpy()


def build_clean_support(model, clean_matched_subset, clean_positions, sampled_classes, device):
    """Assemble a clean support batch and return its node prototypes."""
    loader = DataLoader(Subset(clean_matched_subset, clean_positions),
                        batch_size=len(clean_positions), shuffle=False)
    sup = next(iter(loader)).to(device)
    label_map = {int(c): i for i, c in enumerate(sampled_classes)}
    sup.family_class = torch.tensor(
        [label_map[int(y)] for y in sup.y], dtype=torch.long, device=device
    )
    return node_prototypes_from_support(model, sup, device)


def load_single(dataset, pos, device):
    """Load one graph from a dataset as a batch of size 1."""
    loader = DataLoader(Subset(dataset, [pos]), batch_size=1, shuffle=False)
    return next(iter(loader)).to(device)


# ---------------------------------------------------------------------------
# function_names lookup (read from raw JSON, keyed by hash)
# ---------------------------------------------------------------------------

def build_hash_to_names(raw_glob, wanted_hashes):
    """Map APK hash -> function_names list, scanning raw JSONs under raw_glob.

    Only hashes in `wanted_hashes` are retained. function_names[i] corresponds
    to function index i in the Data object built from the same JSON.
    """
    mapping = {}
    for path in glob.glob(raw_glob):
        try:
            with open(path) as f:
                j = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        h = j.get('hash')
        if h is None or h not in wanted_hashes:
            continue
        mapping[h] = j.get('function_names', [])
    return mapping


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearman(a, b):
    """Spearman rank correlation, dependency-free. Returns nan if degenerate."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return float('nan')
    ra = _rankdata(a)
    rb = _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(x):
    """Average-rank of x (ties share the mean rank)."""
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # resolve ties to their average rank
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0 + 1.0     # average rank per unique value (1-based)
    return avg[inv]


def topk_retention(names_clean, w_clean, names_obf, w_obf, k):
    """Clean-anchored top-k retention: |top-k(clean) ∩ top-k(obf)| / k.

    Ranking is over the full function set of each graph; a clean top-k function
    absent from the obf graph simply fails to be retained.
    """
    kc = min(k, len(names_clean))
    ko = min(k, len(names_obf))
    if kc == 0:
        return float('nan')
    top_clean = {names_clean[i] for i in np.argsort(w_clean)[::-1][:kc]}
    top_obf = {names_obf[i] for i in np.argsort(w_obf)[::-1][:ko]}
    return len(top_clean & top_obf) / kc


def readout_cosine(hg_clean, hg_obf):
    """Cosine between clean and obf weighted graph readouts (both L2-normalised).

    Stability of the quantity the classifier actually uses (h_G vs class prototypes),
    so this connects directly to the accuracy-degradation result. Insensitive to which
    specific functions are top-k when weight is spread across many near-tied functions.
    """
    return float(np.dot(hg_clean, hg_obf))


def weight_mass_retention(names_clean, w_clean, names_obf, w_obf, k):
    """Fraction of the clean top-k weight MASS carried by functions that stay top-k.

    Mass-weighted analogue of top-k retention: dropping a low-weight top-k member
    costs little, dropping a dominant one costs a lot. Less fragile than hard
    set-identity when the weight distribution is compressed.
    """
    kc = min(k, len(names_clean))
    ko = min(k, len(names_obf))
    if kc == 0:
        return float('nan')
    idx_clean = np.argsort(w_clean)[::-1][:kc]
    top_clean_names = [names_clean[i] for i in idx_clean]
    top_clean_w = [float(w_clean[i]) for i in idx_clean]
    total = sum(top_clean_w)
    if total <= 0:
        return float('nan')
    top_obf = {names_obf[i] for i in np.argsort(w_obf)[::-1][:ko]}
    retained = sum(wv for n, wv in zip(top_clean_names, top_clean_w) if n in top_obf)
    return retained / total


def mean_ci(values):
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan'), 0
    mean = v.mean()
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(mean), float(ci), len(v)


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def evaluate_method(model, method, params, test_ds, device, n_episodes, k_shot,
                    n_way, logger):
    graph_root = Path(Utils.get_project_root()) / f"dataset/obfuscated_higraph_{method}/graph"
    if not graph_root.is_dir():
        logger.warning(f"[{method}] graph root not found: {graph_root} — skipping")
        return None

    obf_ds, obf_ibc, obf_hashes = load_obfuscated_dataset(graph_root, logger)
    if obf_ds is None:
        logger.warning(f"[{method}] no obfuscated samples loaded — skipping")
        return None

    clean_matched, _, clean_hash_to_idx = load_matched_clean_dataset(test_ds, obf_hashes)
    if clean_matched is None:
        logger.warning(f"[{method}] no matched clean APKs — skipping")
        return None

    obf_idx_to_hash = build_obf_idx_to_hash(obf_ds)
    matched_hashes = set(clean_hash_to_idx) & set(obf_hashes)

    names_clean_by_hash = build_hash_to_names(
        str(Path(Utils.get_project_root()) / "dataset/higraph/graph/*/raw/*.json"),
        matched_hashes,
    )
    names_obf_by_hash = build_hash_to_names(str(graph_root / "*/raw/*.json"), matched_hashes)

    logger.info(f"[{method}] obf samples={len(obf_idx_to_hash)}  matched hashes={len(matched_hashes)}  "
                f"clean names={len(names_clean_by_hash)}  obf names={len(names_obf_by_hash)}")

    # per-query-sample metric accumulators
    rows = []
    eligible = {c: idxs for c, idxs in obf_ibc.items() if len(idxs) >= k_shot}
    if len(eligible) < n_way:
        logger.warning(f"[{method}] only {len(eligible)} eligible classes for {n_way}-way — skipping")
        return None

    completed = attempts = 0
    while completed < n_episodes and attempts < n_episodes * 5:
        attempts += 1
        s_idx, q_idx, sampled = extract_episode(eligible, n_way, k_shot, 1)  # 1 query per family
        if s_idx is None:
            continue

        # map support (obf indices) -> clean matched positions by hash
        try:
            clean_sup_pos = [clean_hash_to_idx[obf_idx_to_hash[i]] for i in s_idx]
        except KeyError:
            continue  # a support APK has no clean counterpart

        proto = build_clean_support(model, clean_matched, clean_sup_pos, sampled, device)

        for oi in q_idx:
            h = obf_idx_to_hash[oi]
            if h not in clean_hash_to_idx or h not in names_clean_by_hash or h not in names_obf_by_hash:
                continue
            names_c = names_clean_by_hash[h]
            names_o = names_obf_by_hash[h]
            if not names_c or not names_o:
                continue

            clean_q = load_single(clean_matched, clean_hash_to_idx[h], device)
            obf_q = load_single(obf_ds, oi, device)
            w_c, hg_c = weights_and_readout(model, clean_q, proto, device)
            w_o, hg_o = weights_and_readout(model, obf_q, proto, device)

            # guard: weight vector length must match the JSON name list
            if len(w_c) != len(names_c) or len(w_o) != len(names_o):
                logger.debug(f"[{method}] length mismatch for {h[:12]} "
                             f"(w_c={len(w_c)} names_c={len(names_c)} "
                             f"w_o={len(w_o)} names_o={len(names_o)}) — skipping")
                continue

            name_to_wo = {n: w for n, w in zip(names_o, w_o)}
            matched = [n for n in names_c if n in name_to_wo]
            if len(matched) < 2:
                continue
            wc_m = [w_c[i] for i, n in enumerate(names_c) if n in name_to_wo]
            wo_m = [name_to_wo[n] for n in matched]

            rows.append({
                'hash': h,
                'n_clean': len(names_c),
                'n_obf': len(names_o),
                'coverage': len(matched) / len(names_c),
                'readout_cos': readout_cosine(hg_c, hg_o),
                'spearman': spearman(wc_m, wo_m),
                'ret5': topk_retention(names_c, w_c, names_o, w_o, 5),
                'ret10': topk_retention(names_c, w_c, names_o, w_o, 10),
                'mass_ret5': weight_mass_retention(names_c, w_c, names_o, w_o, 5),
                'mass_ret10': weight_mass_retention(names_c, w_c, names_o, w_o, 10),
            })
        completed += 1

    if not rows:
        logger.warning(f"[{method}] produced no usable query samples")
        return None

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='trial_config_v3_higraph_no_overlap')
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument('--n_episodes', type=int, default=200)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('weight_stability')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    params = getattr(ExperimentConfig, f'get_{args.config}')()
    device = Utils.get_device()
    project_root = Utils.get_project_root()

    # ---- load SDL-Complexity (same as extract_sdl_weights.py) ----
    backbone = Backbone(
        cfg_dim=params.cfg_dim,
        func_feat_dim=params.func_feat_dim,
        cfg_hidden_channels=params.cfg_hidden_channels,
        fcg_hidden_channels=params.fcg_hidden_channels,
        final_channels=params.final_channels,
        dropout_value=params.dropout,
    ).to(device)
    model = create_model('sdl_complexity', backbone, fcg_dim=params.final_channels).to(device)
    model_path = Utils.get_episodic_model_path(params, 'sdl_complexity', False)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Checkpoint not found: {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded SDL-Complexity  λ_a={model.raw_lambda_a.item():.4f}  "
                f"λ_b={model.raw_lambda_b.item():.4f}")

    test_ds, _ = load_clean_test_split(params)

    out_dir = os.path.join(project_root, 'results', 'interpretability')
    os.makedirs(out_dir, exist_ok=True)
    summary = ["Node-weight stability under obfuscation (clean support, matched functions)",
               f"config={args.config}  n_way={args.n_way}  k_shot={args.k_shot}  "
               f"n_episodes={args.n_episodes}", ""]

    header = (f"{'method':<20}{'n_apk':>7}{'coverage':>12}{'readout_cos':>14}"
              f"{'spearman':>14}{'ret@5':>13}{'ret@10':>13}"
              f"{'mass_ret@5':>14}{'mass_ret@10':>14}")
    print(header)
    print('-' * len(header))
    summary += [header, '-' * len(header)]

    for method in args.methods:
        rows = evaluate_method(model, method, params, test_ds, device,
                               args.n_episodes, args.k_shot, args.n_way, logger)
        if not rows:
            continue

        # per-sample CSV
        csv_path = os.path.join(out_dir, f'weight_stability_{method}.csv')
        with open(csv_path, 'w') as f:
            f.write('hash,n_clean,n_obf,coverage,readout_cos,spearman,'
                    'ret5,ret10,mass_ret5,mass_ret10\n')
            for r in rows:
                f.write(f"{r['hash']},{r['n_clean']},{r['n_obf']},"
                        f"{r['coverage']:.6f},{r['readout_cos']:.6f},{r['spearman']:.6f},"
                        f"{r['ret5']:.6f},{r['ret10']:.6f},"
                        f"{r['mass_ret5']:.6f},{r['mass_ret10']:.6f}\n")

        cov_m, cov_c, n = mean_ci([r['coverage'] for r in rows])
        rc_m, rc_c, _ = mean_ci([r['readout_cos'] for r in rows])
        sp_m, sp_c, _ = mean_ci([r['spearman'] for r in rows])
        r5_m, r5_c, _ = mean_ci([r['ret5'] for r in rows])
        r10_m, r10_c, _ = mean_ci([r['ret10'] for r in rows])
        m5_m, m5_c, _ = mean_ci([r['mass_ret5'] for r in rows])
        m10_m, m10_c, _ = mean_ci([r['mass_ret10'] for r in rows])
        line = (f"{method:<20}{n:>7}{cov_m:>8.3f}±{cov_c:<3.3f}"
                f"{rc_m:>9.3f}±{rc_c:<4.3f}{sp_m:>9.3f}±{sp_c:<4.3f}"
                f"{r5_m:>8.3f}±{r5_c:<4.3f}{r10_m:>8.3f}±{r10_c:<4.3f}"
                f"{m5_m:>9.3f}±{m5_c:<4.3f}{m10_m:>9.3f}±{m10_c:<4.3f}")
        print(line)
        summary.append(line)
        summary.append(f"    (per-sample table: {csv_path})")

    summary_path = os.path.join(out_dir, 'weight_stability_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))
    print(f"\nSummary saved → {summary_path}")


if __name__ == '__main__':
    main()
