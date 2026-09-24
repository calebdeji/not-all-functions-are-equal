"""
extract_sdl_weights.py

Replays the episodes saved by select_interpretability_apks.py and extracts
per-function SDL-Complexity weights for each selected APK.

For each APK the output table contains:
  rank | func_idx | weight | sim_max | sim_others | norm_complexity | raw_complexity | function_name

  weight          = sigmoid(5 * (λ_a·sim_max + λ_b·norm_complexity − sim_others))
  sim_max         = cosine similarity to nearest node prototype  (semantic intra-class signal)
  sim_others      = mean cosine similarity to all other prototypes (semantic inter-class signal)
  norm_complexity = min-max normalised cyclomatic complexity     (structural signal)

Outputs:
  results/interpretability/weights_{family}_{hash[:12]}.csv  — full ranked table
  results/interpretability/weights_summary.txt               — top-10 per APK to stdout + file

Usage:
  python -m src.analysis.extract_sdl_weights --input results/interpretability/selected_apks_YYYYMMDD_HHMMSS.json
  python -m src.analysis.extract_sdl_weights          # picks the most recent selected_apks_*.json automatically
"""

import argparse
import csv
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Subset as TSubset, ConcatDataset
from torch_geometric.loader import DataLoader


from src.core.backbone import Backbone
from src.core.experiment_configs import ExperimentConfig
from src.core.episodic_models import TrainingUtils, _scatter_mean
from src.train.episodic_train import create_model, build_episode_batches
from src.core.parse_dataset import HiGraphDataset, HiGraphData
from src.core.utils import Utils


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_weights(model, test_ds, entry, device):
    """
    Replay the saved episode and extract per-function weights for the target
    query APK.

    Returns list of dicts, one per function (unsorted).
    """
    ep      = entry['episode']
    s_idx   = ep['s_idx']
    q_idx   = ep['q_idx']
    qpos    = ep['query_position']
    sampled = ep['sampled_classes']

    # Build full support batch (needed for node prototypes)
    label_map   = {int(c): i for i, c in enumerate(sampled)}
    sup_loader  = DataLoader(TSubset(test_ds, s_idx), batch_size=len(s_idx), shuffle=False)
    sup_data    = next(iter(sup_loader)).to(device)
    sup_data.family_class = torch.tensor(
        [label_map[int(y)] for y in sup_data.y], dtype=torch.long, device=device
    )

    # Load only the single target query APK
    target_idx  = [q_idx[qpos]]
    qry_loader  = DataLoader(TSubset(test_ds, target_idx), batch_size=1, shuffle=False)
    qry_data    = next(iter(qry_loader)).to(device)
    qry_data.family_class = torch.tensor(
        [label_map[int(y)] for y in qry_data.y], dtype=torch.long, device=device
    )

    # --- Phase 1: support → node prototypes ---
    sup_node_embs, sup_fcg_batch = model.get_node_embeddings(sup_data)
    sup_node_labels = torch.repeat_interleave(
        sup_data.family_class, sup_data.num_functions
    )
    sup_weights = TrainingUtils.generate_sdl_complexity_weights(
        node_embeddings=sup_node_embs,
        node_labels=sup_node_labels,
        raw_complexity=sup_data.function_features[:, 1].to(device),
        lambda_a_val=model.raw_lambda_a,
        lambda_b_val=model.raw_lambda_b,
    )
    _, labels_inv   = torch.unique(sup_node_labels, return_inverse=True)
    node_prototypes = _scatter_mean(sup_node_embs * sup_weights.view(-1, 1), labels_inv, dim=0)
    node_prototypes = F.normalize(node_prototypes, p=2, dim=-1)

    # --- Phase 2: query APK → per-function decomposition ---
    qry_node_embs, qry_fcg_batch = model.get_node_embeddings(qry_data)
    raw_c = qry_data.function_features[:, 1].to(device).float().view(-1)
    c_norm = ((raw_c - raw_c.min()) / (raw_c.max() - raw_c.min() + 1e-6)
              if raw_c.max() > raw_c.min() else torch.zeros_like(raw_c))

    q   = F.normalize(qry_node_embs, p=2, dim=1)
    p   = F.normalize(node_prototypes, p=2, dim=1)
    sim = torch.matmul(q, p.t())                    # (n_funcs, n_classes)

    sim_max, _  = sim.max(dim=1)
    n_classes   = p.size(0)
    sim_others  = (sim.sum(dim=1) - sim_max) / (n_classes - 1) if n_classes > 1 \
                  else torch.zeros_like(sim_max)

    la = model.raw_lambda_a.item()
    lb = model.raw_lambda_b.item()
    raw_score = (la * sim_max) + (lb * c_norm) - sim_others
    weights   = torch.sigmoid(raw_score * 5.0)

    # Classify (for verification) — use proper fcg_batch from get_node_embeddings
    graph_embs     = model.forward_attention(qry_node_embs, qry_fcg_batch, weights)
    sup_graph_embs = model.forward_attention(sup_node_embs, sup_fcg_batch, sup_weights)
    graph_protos   = _scatter_mean(sup_graph_embs, sup_data.family_class, dim=0)
    graph_protos   = F.normalize(graph_protos, p=2, dim=-1)
    pred_local     = (F.normalize(graph_embs, p=2, dim=-1) @ graph_protos.t()).argmax(dim=1).item()

    n_funcs = weights.size(0)
    records = []
    for i in range(n_funcs):
        records.append({
            'func_idx':        i,
            'weight':          round(weights[i].item(), 6),
            'sim_max':         round(sim_max[i].item(), 6),
            'sim_others':      round(sim_others[i].item(), 6),
            'norm_complexity': round(c_norm[i].item(), 6),
            'raw_complexity':  int(raw_c[i].item()),
            'raw_score':       round(raw_score[i].item(), 6),
        })

    return records, pred_local, la, lb


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default=None,
                        help='Path to selected_apks_*.json. '
                             'Defaults to the most recent one in results/interpretability/.')
    parser.add_argument('--top_n',   type=int, default=10,
                        help='Number of top-weighted functions to print per APK.')
    parser.add_argument('--config',  default='trial_config_v3_higraph_no_overlap')
    args = parser.parse_args()

    params       = getattr(ExperimentConfig, f'get_{args.config}')()
    device       = Utils.get_device()
    project_root = Utils.get_project_root()

    # Locate input JSON
    if args.input is None:
        pattern = os.path.join(project_root, 'results', 'interpretability', 'selected_apks_*.json')
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f'No selected_apks_*.json found in results/interpretability/. '
                                    f'Run select_interpretability_apks.py first.')
        args.input = candidates[-1]
    print(f'Loading: {args.input}')
    with open(args.input) as f:
        data = json.load(f)

    # Load test split
    _, _, test_ds = Utils.get_splits(params)
    print(f'Test split loaded.')

    # Family mapping
    family_to_idx = HiGraphDataset.get_family_mapping()
    idx_to_family = {v: k for k, v in family_to_idx.items()}

    # Load SDL-Complexity
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
    print(f'Loaded SDL-Complexity  λ_a={model.raw_lambda_a.item():.4f}  '
          f'λ_b={model.raw_lambda_b.item():.4f}\n')

    # Output directory
    out_dir = os.path.join(project_root, 'results', 'interpretability')
    os.makedirs(out_dir, exist_ok=True)
    summary_lines = []

    all_entries = (
        [('best',  e) for e in data.get('best',  [])] +
        [('worst', e) for e in data.get('worst', [])]
    )

    for role, entry in all_entries:
        hash_str    = entry['hash']
        family_name = entry['family_name']
        json_path   = entry['json_path']
        ep          = entry['episode']
        true_local  = ep['true_label']
        sampled     = ep['sampled_classes']

        print(f'=== [{role.upper()}] {family_name}  {hash_str[:20]}... ===')
        print(f'    Appearances: {entry["appearances"]}  '
              f'Correct: {entry["correct"]}  '
              f'Accuracy: {entry["accuracy"]:.3f}')
        print(f'    Episode acc: {ep["episode_accuracy"]:.3f}  '
              f'True label: {idx_to_family.get(sampled[true_local], "?")}')

        # Load function names from raw JSON
        if json_path is None or not os.path.exists(json_path):
            print(f'    [SKIP] JSON not found: {json_path}\n')
            continue
        with open(json_path) as jf:
            raw = json.load(jf)
        function_names = raw.get('function_names', [])
        print(f'    Functions in graph: {len(function_names)}')

        # Extract weights
        records, pred_local, la, lb = extract_weights(model, test_ds, entry, device)
        pred_family = idx_to_family.get(sampled[pred_local], str(pred_local))

        print(f'    Replayed prediction: {pred_family}  '
              f'({"CORRECT" if pred_local == true_local else "WRONG"})')
        print(f'    λ_a={la:.4f}  λ_b={lb:.4f}')

        # Attach function names
        for r in records:
            i = r['func_idx']
            r['function_name'] = function_names[i] if i < len(function_names) else f'<func_{i}>'

        # Sort by weight descending → rank
        records.sort(key=lambda r: -r['weight'])
        for rank, r in enumerate(records, 1):
            r['rank'] = rank

        # Save full CSV
        slug = f"{family_name}_{hash_str[:12]}"
        csv_path = os.path.join(out_dir, f'weights_{slug}.csv')
        fieldnames = ['rank', 'func_idx', 'weight', 'sim_max', 'sim_others',
                      'norm_complexity', 'raw_complexity', 'raw_score', 'function_name']
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f'    Saved: {csv_path}')

        # Print top-N
        header = (f"\n    {'Rank':>4}  {'Weight':>7}  {'sim_max':>7}  "
                  f"{'sim_out':>7}  {'c_norm':>6}  {'c_raw':>5}  Function")
        divider = '    ' + '-' * 90
        print(header)
        print(divider)
        block_lines = [header, divider]
        for r in records[:args.top_n]:
            line = (f"    {r['rank']:>4}  {r['weight']:>7.4f}  {r['sim_max']:>7.4f}  "
                    f"{r['sim_others']:>7.4f}  {r['norm_complexity']:>6.3f}  "
                    f"{r['raw_complexity']:>5}  {r['function_name']}")
            print(line)
            block_lines.append(line)

        print(f'\n    --- Bottom 5 (lowest weight) ---')
        bottom_lines = ['\n    --- Bottom 5 (lowest weight) ---']
        for r in records[-5:]:
            line = (f"    {r['rank']:>4}  {r['weight']:>7.4f}  {r['sim_max']:>7.4f}  "
                    f"{r['sim_others']:>7.4f}  {r['norm_complexity']:>6.3f}  "
                    f"{r['raw_complexity']:>5}  {r['function_name']}")
            print(line)
            bottom_lines.append(line)
        print()

        summary_lines += [
            f'\n[{role.upper()}] {family_name}  hash={hash_str}',
            f'  appearances={entry["appearances"]}  accuracy={entry["accuracy"]:.3f}',
            f'  episode_acc={ep["episode_accuracy"]:.3f}  '
            f'prediction={pred_family} ({"CORRECT" if pred_local == true_local else "WRONG"})',
            f'  λ_a={la:.4f}  λ_b={lb:.4f}',
        ] + block_lines + bottom_lines

    # Save summary text
    summary_path = os.path.join(out_dir, 'weights_summary.txt')
    with open(summary_path, 'w') as sf:
        sf.write('\n'.join(summary_lines))
    print(f'Summary saved → {summary_path}')


if __name__ == '__main__':
    main()
