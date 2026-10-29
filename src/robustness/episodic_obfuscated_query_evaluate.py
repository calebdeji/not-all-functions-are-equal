"""
episodic_obfuscated_query_evaluate.py

Query-side (evasion) obfuscation robustness: CLEAN support, OBFUSCATED query.

Motivation
----------
episodic_obfuscated_evaluate.py evaluates the obfuscated condition with the
whole episode obfuscated (obf support + obf query). Because obfuscation shifts
support prototypes and the query embedding in a correlated direction, that shift
partially cancels at the nearest-prototype comparison, which can mask
degradation. It measures robustness under *homogeneous* obfuscation.

This script measures the complementary, more conservative case that matches the
deployment/evasion threat model: the defender holds a curated CLEAN reference
set (support), and an adversary obfuscates the incoming sample (query) to evade
classification. Both conditions share the SAME clean support, so Δ isolates the
query-side effect:

    clean condition : evaluate_task(clean support, clean query)
    obf-query cond. : evaluate_task(clean support, OBF   query)
    Δ (per episode) : acc_obf_query − acc_clean

Everything else (models, FAMCF/AndroFIM encoders, matched-clean subset by hash,
episode sampling, reporting) is reused from episodic_obfuscated_evaluate so the
output table is directly comparable to the homogeneous-obfuscation Table XIII.

FAMCF/AndroFIM: support features/images are taken from the clean maps and query
features/images from the obfuscated maps via a per-episode merged lookup (support
and query hashes are disjoint within an episode).

Usage:
  python src/episodic_obfuscated_query_evaluate.py --config trial_config_v3_higraph_no_overlap
  python src/episodic_obfuscated_query_evaluate.py --methods reorder --n_episodes 500 --n_shots 1 5
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from src.core.experiment_configs import ExperimentConfig
from src.core.utils import Utils, ModelParameters
from src.train.episodic_train import extract_episode, build_episode_batches
from src.eval.episodic_evaluate import OnTheFlyFewShotEvaluator
from src.core.parse_dataset import HiGraphData  # noqa: F401  (needed to unpickle dataset .pt files)

from src.robustness.episodic_obfuscated_evaluate import (
    ALL_METHODS,
    METHOD_LABEL,
    PROJECT_ROOT,
    load_models,
    load_famcf_androfim,
    load_obfuscated_dataset,
    load_matched_clean_dataset,
    build_obf_idx_to_hash,
    _build_famcf_hash_to_features,
    _build_androfim_hash_to_path,
    _eval_famcf_from_hashes,
    _eval_androfim_from_hashes,
    _ci,
    print_degradation_table,
)


def run_query_side_episodes(
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
    logger,
    famcf_encoders=None,
    clean_hash_to_features=None,
    obf_hash_to_features=None,
    androfim_backbone=None,
    clean_hash_to_path=None,
    obf_hash_to_path=None,
    progress_logger=None,
    log_every: int = 20,
) -> dict:
    """Clean-support / obfuscated-query paired evaluation.

    Returns {k_shot: {model_name: {'clean': (mean, ci), 'obf': (mean, ci),
                                   'delta': (mean, ci)}}}, where 'obf' is the
    obf-QUERY condition (clean support). Shape matches run_paired_episodes so
    print_degradation_table can render it unchanged.
    """
    results: dict = {}

    for k_shot in n_shots:
        eligible = {c: idxs for c, idxs in obf_ibc.items() if len(idxs) >= k_shot}
        if len(eligible) < n_way:
            logger.warning(
                f"  [{label}] {k_shot}-shot: only {len(eligible)} eligible classes, skipping"
            )
            continue

        clean_accs: dict = defaultdict(list)
        obf_accs:   dict = defaultdict(list)
        delta_accs: dict = defaultdict(list)
        completed = attempts = 0
        pbar = tqdm(total=n_episodes,
                    desc=f"  [{label}] {n_way}-way {k_shot}-shot (clean-sup/obf-qry)")

        while completed < n_episodes and attempts < n_episodes * 5:
            attempts += 1

            s_idx, q_idx, sampled = extract_episode(eligible, n_way, k_shot, n_query)
            if s_idx is None:
                continue

            # Map every obfuscated index → clean matched index by hash
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

            # Clean support + clean query, and obfuscated query (support unused here)
            sup_clean, qry_clean = build_episode_batches(
                clean_matched_subset, clean_s_idx, clean_q_idx, sampled, device)
            _, qry_obf = build_episode_batches(
                obf_dataset, s_idx, q_idx, sampled, device)

            # ---- graph models: both conditions share the clean support -------
            for name, ev in evaluators.items():
                preds_clean, lbl_clean = ev.evaluate_task(sup_clean, qry_clean)
                preds_obf,   lbl_obf   = ev.evaluate_task(sup_clean, qry_obf)

                acc_clean = (preds_clean == lbl_clean).float().mean().item()
                acc_obf   = (preds_obf   == lbl_obf).float().mean().item()

                clean_accs[name].append(acc_clean)
                obf_accs[name].append(acc_obf)
                delta_accs[name].append(acc_obf - acc_clean)

            # ---- shared hash lists for flat-feature baselines ----------------
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

            # ---- FAMCF: clean support features, obf query features -----------
            if famcf_encoders and obf_hash_to_features and clean_hash_to_features:
                # clean baseline: support+query from clean
                res_clean = _eval_famcf_from_hashes(
                    famcf_encoders, sup_hashes, qry_hashes, sup_labels, qry_labels,
                    clean_hash_to_features, device)
                # obf-query: support from clean, query from obf (disjoint hashes → merge)
                merged = {h: clean_hash_to_features[h] for h in sup_hashes
                          if h in clean_hash_to_features}
                merged.update({h: obf_hash_to_features[h] for h in qry_hashes
                               if h in obf_hash_to_features})
                res_obf = _eval_famcf_from_hashes(
                    famcf_encoders, sup_hashes, qry_hashes, sup_labels, qry_labels,
                    merged, device)
                if res_clean is not None and res_obf is not None:
                    acc_clean = (res_clean[0] == res_clean[1]).float().mean().item()
                    acc_obf   = (res_obf[0]   == res_obf[1]).float().mean().item()
                    clean_accs['famcf'].append(acc_clean)
                    obf_accs['famcf'].append(acc_obf)
                    delta_accs['famcf'].append(acc_obf - acc_clean)

            # ---- AndroFIM: clean support images, obf query images ------------
            if androfim_backbone and obf_hash_to_path and clean_hash_to_path:
                res_clean = _eval_androfim_from_hashes(
                    androfim_backbone, sup_hashes, qry_hashes, sup_labels, qry_labels,
                    clean_hash_to_path, device)
                merged_p = {h: clean_hash_to_path[h] for h in sup_hashes
                            if h in clean_hash_to_path}
                merged_p.update({h: obf_hash_to_path[h] for h in qry_hashes
                                 if h in obf_hash_to_path})
                res_obf = _eval_androfim_from_hashes(
                    androfim_backbone, sup_hashes, qry_hashes, sup_labels, qry_labels,
                    merged_p, device)
                if res_clean is not None and res_obf is not None:
                    acc_clean = (res_clean[0] == res_clean[1]).float().mean().item()
                    acc_obf   = (res_obf[0]   == res_obf[1]).float().mean().item()
                    clean_accs['androfim'].append(acc_clean)
                    obf_accs['androfim'].append(acc_obf)
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
                        f"obf-qry {o_mu:6.2f}%  Δ {d_mu:+6.2f}%"
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


def main():
    parser = argparse.ArgumentParser(
        description="Query-side (clean support, obfuscated query) robustness evaluation")
    parser.add_argument("--config",     default="trial_config_v3_higraph_no_overlap")
    parser.add_argument("--methods",    nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
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
    log_path = os.path.join(log_dir, f"evaluate_obfuscated_query_{ts}.log")
    progress_log_path = os.path.join(log_dir, f"evaluate_obfuscated_query_progress_{ts}.log")
    logger = Utils.setup_logger("obfuscated_query_eval", log_path)
    progress_logger = Utils.setup_logger("obfuscated_query_progress", progress_log_path)

    logger.info("QUERY-SIDE OBFUSCATION EVAL — clean support, obfuscated query")
    logger.info(f"Config:    {params.id}")
    logger.info(f"Device:    {device}")
    logger.info(f"Seed:      {args.seed}")
    logger.info(f"Methods:   {args.methods}")
    logger.info(f"Episodes:  {args.n_episodes}  shots: {args.n_shots}  n_way: {args.n_way}")

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

    logger.info("\nLoading FAMCF + AndroFIM...")
    famcf_encoders, clean_hash_to_features, androfim_backbone, clean_hash_to_path = \
        load_famcf_androfim(params, device, logger)

    _, _, full_test_ds = Utils.get_splits(params)

    all_results: dict = {}

    for method in args.methods:
        graph_root = PROJECT_ROOT / "dataset" / f"obfuscated_higraph_{method}" / "graph"
        logger.info(f"\n=== {METHOD_LABEL[method]} (query-side) ===")
        logger.info(f"Graph root: {graph_root}")

        if not graph_root.exists():
            logger.warning("  Root not found — run run_obfuscation_extraction.sh first, skipping")
            continue

        obf_ds, obf_ibc, obf_hashes = load_obfuscated_dataset(graph_root, logger)
        if obf_ds is None:
            logger.warning("  No processed samples found, skipping")
            continue

        if len(obf_ibc) < args.n_way:
            logger.warning(f"  Only {len(obf_ibc)} families (need {args.n_way}), skipping")
            continue

        clean_ds, clean_ibc, clean_hash_to_idx = load_matched_clean_dataset(full_test_ds, obf_hashes)
        if clean_ds is None:
            logger.warning("  No clean matches found in test split, skipping")
            continue
        logger.info(f"  Matched clean: {len(clean_ds)} samples, {len(clean_ibc)} families")

        obf_idx_to_hash = build_obf_idx_to_hash(obf_ds)

        obf_hash_to_features = None
        if famcf_encoders:
            obf_feat_pkl = PROJECT_ROOT / "dataset" / f"obfuscated_famcf_{method}" / "features.pkl"
            if obf_feat_pkl.exists():
                obf_hash_to_features = _build_famcf_hash_to_features(str(obf_feat_pkl))
                logger.info(f"  Loaded obf FAMCF features ({len(obf_hash_to_features):,} samples)")
            else:
                logger.info(f"  Obf FAMCF features not found at {obf_feat_pkl} — skipping FAMCF")

        obf_hash_to_path = None
        if androfim_backbone:
            obf_manifest = PROJECT_ROOT / "dataset" / f"obfuscated_androfim_{method}" / "androFIM_manifest.csv"
            if obf_manifest.exists():
                obf_hash_to_path = _build_androfim_hash_to_path(str(obf_manifest))
                logger.info(f"  Loaded obf AndroFIM paths ({len(obf_hash_to_path):,} images)")
            else:
                logger.info(f"  Obf AndroFIM manifest not found at {obf_manifest} — skipping AndroFIM")

        logger.info("  Running query-side episodes (clean support, obfuscated query)...")
        res = run_query_side_episodes(
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
        all_results[method] = res

        logger.info(f"\n--- {method} query-side results (Δ = obf-query − clean) ---")
        for k in args.n_shots:
            for name, r in sorted(res.get(k, {}).items()):
                c_mu, c_ci = r['clean']
                o_mu, o_ci = r['obf']
                d_mu, d_ci = r['delta']
                logger.info(
                    f"  {name:<30}  {k}-shot: "
                    f"clean {c_mu:.2f}%±{c_ci:.2f}  "
                    f"obf-qry {o_mu:.2f}%±{o_ci:.2f}  "
                    f"Δ {d_mu:+.2f}%±{d_ci:.2f}"
                )

    if all_results:
        logger.info("\n=== QUERY-SIDE DEGRADATION (clean support, obfuscated query) ===")
        print_degradation_table(all_results, args.n_shots, logger)

    logger.info(f"Log saved: {log_path}")
    logger.info(f"Progress log: {progress_log_path}")


if __name__ == "__main__":
    main()
