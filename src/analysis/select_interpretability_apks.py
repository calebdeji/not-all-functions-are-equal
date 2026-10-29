"""
select_interpretability_apks.py

Mine the HiGraph no-overlap test split to identify APKs that are consistently
easy or hard for SDL-Complexity, then save them with a representative episode
so the weight-extraction script can replay the exact context.

Finds:
  - Two APKs from DIFFERENT families with 100% episode accuracy (always correct)
  - One APK with the lowest episode accuracy (consistently misclassified)

For each, saves the "most interesting" episode:
  - Best APK  → hardest episode in which it was still correctly classified
                (lowest overall episode accuracy)
  - Worst APK → hardest episode it appeared in (always wrong anyway)

Output:
  results/interpretability/selected_apks_{timestamp}.json

Usage:
  cd <project_root>
  python -m src.analysis.select_interpretability_apks
  python -m src.analysis.select_interpretability_apks --n_episodes 300 --min_appearances 3
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset


from src.core.backbone import Backbone
from src.core.experiment_configs import ExperimentConfig
from src.train.episodic_train import create_model, extract_episode, build_episode_batches
from src.eval.episodic_evaluate import OnTheFlyFewShotEvaluator
from src.core.parse_dataset import HiGraphDataset, get_dataset_root, HiGraphData
from src.core.utils import Utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_json_path(hash_str, graph_root):
    """Search all year subdirectories for {hash_str}.json."""
    for year_dir in sorted(os.listdir(graph_root)):
        raw_dir = os.path.join(graph_root, year_dir, 'raw')
        if os.path.isdir(raw_dir):
            candidate = os.path.join(raw_dir, f'{hash_str}.json')
            if os.path.exists(candidate):
                return candidate
    return None


# ---------------------------------------------------------------------------
# Mining loop
# ---------------------------------------------------------------------------

def mine_episodes(model, test_ds, indices_by_class,
                  n_way, k_shot, n_query, n_episodes, device):
    """
    Run n_episodes and collect per-APK accuracy stats.

    Returns:
        {hash: {family_id, appearances, correct,
                best_correct_episode,   # lowest ep_acc where this APK was correct
                hardest_episode}}       # lowest ep_acc overall (used for worst APK)
    """
    evaluator = OnTheFlyFewShotEvaluator(model=model, device=device)
    valid_classes = {c: idxs for c, idxs in indices_by_class.items()
                     if len(idxs) >= k_shot}

    apk_stats = {}
    completed = attempts = 0

    print(f'Mining {n_episodes} episodes  '
          f'({n_way}-way {k_shot}-shot, {n_query} queries/class) ...')

    while completed < n_episodes and attempts < n_episodes * 3:
        attempts += 1
        s_idx, q_idx, sampled = extract_episode(valid_classes, n_way, k_shot, n_query)
        if s_idx is None:
            continue

        sup_data, qry_data = build_episode_batches(
            test_ds, s_idx, q_idx, sampled, device
        )
        preds, labels = evaluator.evaluate_task(sup_data, qry_data)

        ep_acc      = (preds == labels).float().mean().item()
        sup_hashes  = list(sup_data.hash)
        qry_hashes  = list(qry_data.hash)
        sampled_int = [int(c) for c in sampled]

        ep_base = {
            'sampled_classes': sampled_int,   # N global family IDs
            's_idx':           s_idx,
            'q_idx':           q_idx,
            'sup_hashes':      sup_hashes,
            'qry_hashes':      qry_hashes,
            'episode_accuracy': ep_acc,
        }

        for j, h in enumerate(qry_hashes):
            true_local  = labels[j].item()
            pred_local  = preds[j].item()
            family_id   = sampled_int[true_local]
            correct     = (pred_local == true_local)

            if h not in apk_stats:
                apk_stats[h] = {
                    'family_id':            family_id,
                    'appearances':          0,
                    'correct':              0,
                    'best_correct_episode': None,
                    'hardest_episode':      None,
                }

            apk_stats[h]['appearances'] += 1
            if correct:
                apk_stats[h]['correct'] += 1

            ep_record = {**ep_base,
                         'query_position': j,
                         'prediction':     pred_local,
                         'true_label':     true_local}

            if correct:
                prev = apk_stats[h]['best_correct_episode']
                if prev is None or ep_acc < prev['episode_accuracy']:
                    apk_stats[h]['best_correct_episode'] = ep_record

            prev_hard = apk_stats[h]['hardest_episode']
            if prev_hard is None or ep_acc < prev_hard['episode_accuracy']:
                apk_stats[h]['hardest_episode'] = ep_record

        completed += 1
        if completed % 50 == 0:
            print(f'  Episode {completed}/{n_episodes}  '
                  f'(attempts={attempts}, unique APKs seen={len(apk_stats)})')

    print(f'Done. {completed} episodes, {len(apk_stats)} unique query APKs seen.')
    return apk_stats


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def select_candidates(apk_stats, idx_to_family, min_appearances,
                      best_families=None, worst_families=None):
    """
    Returns (best_list, worst_list).

    best_list : two APKs with 100% accuracy from different families,
                each with their best_correct_episode.
    worst_list: one APK with the lowest accuracy (different family
                from both best), with its hardest_episode.

    best_families / worst_families : optional sets of family names to
        restrict selection to. If None, all families are eligible.
    """
    def _eligible(h, s, allowed_families):
        if s['appearances'] < min_appearances:
            return False
        if allowed_families is not None:
            return idx_to_family.get(s['family_id'], '') in allowed_families
        return True

    scored = [
        (s['correct'] / s['appearances'], h, s)
        for h, s in apk_stats.items()
        if _eligible(h, s, None)   # appearance filter only; family filter applied per-pool below
    ]
    if not scored:
        print(f'[WARNING] No APKs with ≥{min_appearances} appearances.')
        scored = [(s['correct'] / s['appearances'], h, s) for h, s in apk_stats.items()]

    def _filter_families(pool, allowed):
        if allowed is None:
            return pool
        return [(acc, h, s) for acc, h, s in pool
                if idx_to_family.get(s['family_id'], '') in allowed]

    scored_best_pool = _filter_families(scored, best_families)
    scored_worst_pool = _filter_families(scored, worst_families)

    if best_families and not scored_best_pool:
        print(f'[WARNING] No APKs found in best_families={best_families}. '
              f'Falling back to all families.')
        scored_best_pool = scored
    if worst_families and not scored_worst_pool:
        print(f'[WARNING] No APKs found in worst_families={worst_families}. '
              f'Falling back to all families.')
        scored_worst_pool = scored

    scored_desc = sorted(scored_best_pool,  key=lambda x: (-x[0], -x[2]['appearances']))
    scored_asc  = sorted(scored_worst_pool, key=lambda x: (x[0],  x[2]['appearances']))

    def _build_entry(rank, h, s, episode_key):
        acc = s['correct'] / s['appearances']
        fid = s['family_id']
        return {
            'rank':        rank,
            'hash':        h,
            'family_id':   fid,
            'family_name': idx_to_family.get(fid, str(fid)),
            'appearances': s['appearances'],
            'correct':     s['correct'],
            'accuracy':    round(acc, 4),
            'episode':     s[episode_key],
        }

    # Best: two APKs from different families with 100% accuracy
    best = []
    seen_fam = set()
    for acc, h, s in scored_desc:
        if acc < 1.0:
            break
        if s['family_id'] in seen_fam:
            continue
        if s['best_correct_episode'] is None:
            continue
        best.append(_build_entry(len(best) + 1, h, s, 'best_correct_episode'))
        seen_fam.add(s['family_id'])
        if len(best) == 2:
            break

    if len(best) < 2:
        print(f'[NOTE] Only {len(best)} APK(s) with 100% accuracy — '
              f'relaxing to top-accuracy APKs from additional families.')
        seen_fam_fb = {e['family_id'] for e in best}
        for acc, h, s in scored_desc:
            if s['family_id'] in seen_fam_fb:
                continue
            if s['best_correct_episode'] is None:
                continue
            best.append(_build_entry(len(best) + 1, h, s, 'best_correct_episode'))
            seen_fam_fb.add(s['family_id'])
            if len(best) == 2:
                break

    # Worst: one APK with lowest accuracy, different family from both best
    best_family_ids = {e['family_id'] for e in best}
    worst = []
    for acc, h, s in scored_asc:
        if s['family_id'] in best_family_ids:
            continue
        if s['hardest_episode'] is None:
            continue
        worst.append(_build_entry(1, h, s, 'hardest_episode'))
        break

    if not worst:
        print('[NOTE] Relaxing family constraint for worst APK.')
        for acc, h, s in scored_asc:
            if s['hardest_episode'] is None:
                continue
            worst.append(_build_entry(1, h, s, 'hardest_episode'))
            break

    return best, worst


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Mine best/worst APKs by episode accuracy for interpretability.'
    )
    parser.add_argument('--config',          default='trial_config_v3_higraph_no_overlap',)
    parser.add_argument('--n_episodes',      type=int, default=500)
    parser.add_argument('--k_shot',          type=int, default=1)
    parser.add_argument('--n_way',           type=int, default=5)
    parser.add_argument('--n_query',         type=int, default=15)
    parser.add_argument('--min_appearances', type=int, default=5,
                        help='Min times an APK must appear as a query to be eligible')
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--best_families',   nargs='+', default=None,
                        help='Restrict best-APK pool to these family names '
                             '(e.g. --best_families syringe hqwar hypay)')
    parser.add_argument('--worst_families',  nargs='+', default=None,
                        help='Restrict worst-APK pool to these family names '
                             '(e.g. --worst_families wapron ewind)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = getattr(ExperimentConfig, f'get_{args.config}')()
    device = Utils.get_device()
    project_root = Utils.get_project_root()

    print(f'Config: {params.id}  Dataset: {params.dataset_name}  '
          f'Split: {params.dataset_split_type.value}  Device: {device}')

    # --- Test split ---
    _, _, test_ds = Utils.get_splits(params)
    inner = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
    if isinstance(inner, ConcatDataset):
        all_labels = torch.cat([ds.get_labels() for ds in inner.datasets])
    else:
        all_labels = inner.get_labels()
    subset_labels = all_labels[torch.tensor(test_ds.indices, dtype=torch.long)]

    indices_by_class = defaultdict(list)
    for i, lbl in enumerate(subset_labels.tolist()):
        indices_by_class[lbl].append(i)

    print(f'Test split — families: {len(indices_by_class)}  '
          f'samples: {len(subset_labels)}')

    # --- Family mapping ---
    family_to_idx = HiGraphDataset.get_family_mapping()
    idx_to_family = {v: k for k, v in family_to_idx.items()}

    # --- Load SDL-Complexity ---
    backbone = Backbone(
        cfg_dim=params.cfg_dim,
        func_feat_dim=params.func_feat_dim,
        cfg_hidden_channels=params.cfg_hidden_channels,
        fcg_hidden_channels=params.fcg_hidden_channels,
        final_channels=params.final_channels,
        dropout_value=params.dropout,
    ).to(device)

    model = create_model('sdl_complexity', backbone,
                         fcg_dim=params.final_channels).to(device)
    model_path = Utils.get_episodic_model_path(
        params, 'sdl_complexity', False
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'SDL-Complexity checkpoint not found: {model_path}\n'
            f'Train the model first with python -m src.train.episodic_train --variant sdl_complexity'
        )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'Loaded SDL-Complexity from {model_path}')

    # --- Mine ---
    apk_stats = mine_episodes(
        model, test_ds, indices_by_class,
        args.n_way, args.k_shot, args.n_query, args.n_episodes, device,
    )

    # --- Select ---
    best_fam_set  = set(args.best_families)  if args.best_families  else None
    worst_fam_set = set(args.worst_families) if args.worst_families else None
    if best_fam_set:
        print(f'Best-APK family filter:  {sorted(best_fam_set)}')
    if worst_fam_set:
        print(f'Worst-APK family filter: {sorted(worst_fam_set)}')

    best, worst = select_candidates(
        apk_stats, idx_to_family, args.min_appearances,
        best_families=best_fam_set, worst_families=worst_fam_set,
    )

    # --- Enrich: add family names to episode + find JSON path ---
    graph_root = os.path.join(project_root, 'dataset', params.dataset_name, 'graph')
    for entry in best + worst:
        entry['json_path'] = find_json_path(entry['hash'], graph_root)
        ep = entry['episode']
        if ep is not None:
            ep['sampled_family_names'] = [
                idx_to_family.get(fid, str(fid))
                for fid in ep['sampled_classes']
            ]

    # --- Print summary ---
    print('\n=== SELECTED APKs ===')
    print('\nBest (highest accuracy):')
    for e in best:
        ep = e['episode']
        ep_acc = ep['episode_accuracy'] if ep else float('nan')
        print(f"  [{e['rank']}] {e['hash'][:20]}...  "
              f"family={e['family_name']!r}  "
              f"acc={e['correct']}/{e['appearances']}={e['accuracy']:.3f}  "
              f"representative_ep_acc={ep_acc:.3f}")
        print(f"       json: {e['json_path']}")

    print('\nWorst (lowest accuracy):')
    for e in worst:
        ep = e['episode']
        ep_acc = ep['episode_accuracy'] if ep else float('nan')
        print(f"  [{e['rank']}] {e['hash'][:20]}...  "
              f"family={e['family_name']!r}  "
              f"acc={e['correct']}/{e['appearances']}={e['accuracy']:.3f}  "
              f"representative_ep_acc={ep_acc:.3f}")
        print(f"       json: {e['json_path']}")

    # --- Save ---
    out_dir = os.path.join(project_root, 'results', 'interpretability')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path  = os.path.join(out_dir, f'selected_apks_{timestamp}.json')

    output = {
        'metadata': {
            'model':           'sdl_complexity',
            'config':          args.config,
            'dataset':         params.dataset_name,
            'split':           params.dataset_split_type.value,
            'n_way':           args.n_way,
            'k_shot':          args.k_shot,
            'n_query':         args.n_query,
            'n_episodes':      args.n_episodes,
            'seed':            args.seed,
            'min_appearances': args.min_appearances,
        },
        'best': best,
        'worst': worst,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved → {out_path}')
    print('\nNext: run python -m src.analysis.extract_sdl_weights --input', out_path)


if __name__ == '__main__':
    main()
