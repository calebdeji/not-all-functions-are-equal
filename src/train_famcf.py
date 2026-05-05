"""
train_famcf.py

Trains a FAMCF-style MLP encoder (permission, api_call, or opcode) using
TripletNet contrastive learning.

Two dataset backends are supported:
  - pkl backend  (MalDroid, HiGraph, any dataset processed by run_extract_features.sh)
                 Detected when dataset/{name}/famcf/features.pkl exists.
                 --data_dir should point to dataset/{name} (e.g. dataset/maldroid).
                 Paths derived automatically:
                   dataset/{name}/famcf/features.pkl
                   dataset/{name}/famcf/vocab.pkl
                   dataset_refined_truth/{name}/selected_samples.csv
                   dataset_refined_truth/{name}/family_to_idx.json

Usage:
  # pkl-based (MalDroid):
  python src/train_famcf.py --model_id maldroid --data_dir dataset/maldroid --feature_type permission

"""

import argparse
import json
import logging
import os
import pickle
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from utils import Utils
from experiment_configs import ExperimentConfig
from famcf_dataset import (
    FAMCFDataset,
    get_family_splits,
)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Triplet sampling
# ---------------------------------------------------------------------------

def sample_triplets(labels, n_triplets: int = 1024):
    """
    Sample (anchor_idx, positive_idx, negative_idx) tuples from label list.
    Requires at least 2 samples per class and at least 2 classes.
    """
    label_to_indices: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        label_to_indices.setdefault(l, []).append(i)

    eligible = [f for f, idxs in label_to_indices.items() if len(idxs) >= 2]
    if len(eligible) < 2:
        return []

    triplets = []
    for _ in range(n_triplets):
        anchor_fam = random.choice(eligible)
        a_idx, p_idx = random.sample(label_to_indices[anchor_fam], 2)
        neg_fam = random.choice([f for f in eligible if f != anchor_fam])
        n_idx = random.choice(label_to_indices[neg_fam])
        triplets.append((a_idx, p_idx, n_idx))

    return triplets


# ---------------------------------------------------------------------------
# Episodic validation (prototypical network, 5-way 5-shot)
# ---------------------------------------------------------------------------

def episodic_val_accuracy(
    encoder: nn.Module,
    dataset: Dataset,
    feature_type: str,
    n_way: int = 5,
    n_shot: int = 5,
    n_query: int = 15,
    n_episodes: int = 200,
    device: torch.device = torch.device("cpu"),
) -> float:
    encoder.eval()
    all_features = dataset.get_all_features(feature_type).to(device)
    labels = dataset.get_labels()

    label_to_indices: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        label_to_indices.setdefault(l, []).append(i)

    families = list(label_to_indices.keys())
    if len(families) < n_way:
        return 0.0

    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(n_episodes):
            episode_fams = random.sample(families, n_way)
            prototypes = []
            query_feats = []
            query_targets = []

            valid = True
            for cls_idx, fam in enumerate(episode_fams):
                idxs = label_to_indices[fam]
                need = n_shot + n_query
                if len(idxs) < need:
                    valid = False
                    break
                chosen = random.sample(idxs, need)
                support = all_features[chosen[:n_shot]]
                queries = all_features[chosen[n_shot:]]

                prototypes.append(encoder(support).mean(0))
                query_feats.append(encoder(queries))
                query_targets.extend([cls_idx] * n_query)

            if not valid:
                continue

            proto = torch.stack(prototypes)                        # (n_way, emb_dim)
            qfeat = torch.cat(query_feats, dim=0)                  # (n_way*n_query, emb_dim)
            targets = torch.tensor(query_targets, device=device)

            dists = torch.cdist(qfeat, proto)
            preds = dists.argmin(dim=1)

            correct += (preds == targets).sum().item()
            total += len(query_targets)

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _load_pkl_backend(data_dir: str, feature_type: str, logger,
                      split_json: str = None, years: list = None):
    """Load FAMCFDataset splits from a processed data_dir (dataset/{name}).

    split_json : path to the split JSON produced by Utils.get_splits().
                 When provided, base/val families are read from it so FAMCF
                 trains on the exact same partition as the backbone and
                 episodic heads.
    years      : for HiGraph only — restricts the generated samples CSV to
                 these years.
    """
    dataset_name       = Path(data_dir).name
    truth_dir          = Utils.get_refined_truth_root(dataset_name)
    features_pkl       = os.path.join(data_dir, "famcf", "features.pkl")
    vocab_pkl          = os.path.join(data_dir, "famcf", "vocab.pkl")
    family_to_idx_path = os.path.join(truth_dir, "family_to_idx.json")

    with open(family_to_idx_path) as f:
        family_to_idx = json.load(f)
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)

    dim_map = {
        "permission": len(vocab["perms"]),
        "api_call":   len(vocab["apis"]),
        "opcode":     len(vocab["opcodes"]),
    }
    if feature_type not in dim_map:
        raise ValueError(f"Unknown feature_type '{feature_type}'. "
                         f"pkl backend supports: {list(dim_map)}")
    input_dim = dim_map[feature_type]

    logger.info(f"Vocab dims — perms: {len(vocab['perms'])}, "
                f"apis: {len(vocab['apis'])}, opcodes: {len(vocab['opcodes'])}")

    # ── Resolve samples CSV ───────────────────────────────────────────────────
    if dataset_name == "higraph":
        samples_csv = os.path.join(truth_dir, "famcf_samples.csv")
        if not os.path.exists(samples_csv):
            year_csvs = [
                os.path.join(truth_dir, f"labelled_samples_with_family_{y}.csv")
                for y in (years or [])
            ]
            year_csvs = [p for p in year_csvs if os.path.exists(p)]
            if not year_csvs:
                raise FileNotFoundError(
                    f"No per-year CSVs found in {truth_dir} for years={years}"
                )
            dfs = []
            for ycsv in year_csvs:
                df = pd.read_csv(ycsv)
                if "hash" in df.columns:
                    df = df.rename(columns={"hash": "sha256"})
                dfs.append(df[["sha256", "family"]])
            pd.concat(dfs, ignore_index=True).to_csv(samples_csv, index=False)
            logger.info(f"Generated {samples_csv} from {len(year_csvs)} year CSVs")
    else:
        samples_csv = os.path.join(truth_dir, "selected_samples.csv")

    # ── Load family split ─────────────────────────────────────────────────────
    if split_json and os.path.exists(split_json):
        with open(split_json) as f:
            split = json.load(f)
        train_fams = set(split["base"])
        val_fams   = set(split["val"])
        logger.info(f"Splits from JSON — base: {len(train_fams)}, val: {len(val_fams)}")
    else:
        train_fams, val_fams, _ = get_family_splits(samples_csv)
        logger.info(f"Splits (random 60/20/20) — train: {len(train_fams)}, val: {len(val_fams)}")

    train_ds = FAMCFDataset(samples_csv, features_pkl, train_fams, family_to_idx)
    val_ds   = FAMCFDataset(samples_csv, features_pkl, val_fams,   family_to_idx)
    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    return train_ds, val_ds, input_dim





def train(args):
    dataset_name = Path(args.data_dir).name
    getter = getattr(ExperimentConfig, f'get_{args.config}')
    config = getter()
    save_dir = Utils.get_famcf_save_dir(args.config, dataset_name, config.dataset_split_type.value)

    logger = setup_logger(os.path.join(save_dir, f"train_{args.feature_type}.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft = args.feature_type
    logger.info(f"Device: {device} | Config: {args.config} | Dataset: {dataset_name} | Feature type: {ft}")
    logger.info(f"Epochs: {args.epochs} | LR: {args.lr} | "
                f"Batch: {args.batch_size} | Triplets/epoch: {args.n_triplets}")
    logger.info(f"Architecture: input → {args.hidden_dim} → {args.output_dim}")

    # ── Detect backend and load datasets ──────────────────────────────────────
    logger.info(f"Backend: {'pkl (FAMCFDataset)'}")

    from utils import  _FLAT_DATASET_NAMES
    years = config.years if hasattr(config, "years") else None

    split_filename = "split.json"

    split_json = os.path.join(
        Utils.get_project_root(), "dataset_config_split",
        dataset_name, args.config, split_filename
    )
    if not os.path.exists(split_json):
        split_json = os.path.join(
            Utils.get_project_root(), "models", args.config, split_filename
        )

    train_ds, val_ds, input_dim = _load_pkl_backend(
        args.data_dir, ft, logger, split_json=split_json, years=years
    )

    # ── Save model config so eval_famcf.py can reconstruct the encoder ────────
    cfg = {
        "feature_type": ft,
        "input_dim":    input_dim,
        "hidden_dim":   args.hidden_dim,
        "output_dim":   args.output_dim,
        "backend":      "pkl",
    }
    cfg_path = os.path.join(save_dir, f"model_config_{ft}.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Model config saved → {cfg_path}")

    # ── Encoder + optimiser ───────────────────────────────────────────────────
    encoder = MLPEncoder(input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)

    all_train_features = train_ds.get_all_features(ft).to(device)
    train_labels = train_ds.get_labels()

    best_val_acc = 0.0
    short = ft[:3]  # per / api / opc
    model_path = os.path.join(save_dir, f"encoder_{short}.pth")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        triplets = sample_triplets(train_labels, n_triplets=args.n_triplets)
        random.shuffle(triplets)

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(triplets), args.batch_size):
            batch = triplets[i : i + args.batch_size]
            a_idx = [t[0] for t in batch]
            p_idx = [t[1] for t in batch]
            n_idx = [t[2] for t in batch]

            a = encoder(all_train_features[a_idx])
            p = encoder(all_train_features[p_idx])
            n = encoder(all_train_features[n_idx])

            loss = triplet_loss_fn(a, p, n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        val_acc = episodic_val_accuracy(encoder, val_ds, ft, device=device)
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {epoch_loss / max(n_batches, 1):.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(encoder.state_dict(), model_path)
            logger.info(f"--> New best model saved! (Val Acc: {best_val_acc:.4f})")

    logger.info(
        f"Training complete. Best model saved to {model_path} "
        f"with Val Acc: {best_val_acc:.4f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       required=True,
                        help="ExperimentConfig preset name (e.g. trial_config_v3)")
    parser.add_argument("--data_dir",   required=True,
                        help="Dataset base dir (e.g. dataset/maldroid). "
                             "If {data_dir}/famcf/features.pkl exists, ")
    parser.add_argument("--feature_type", required=True,
                        choices=["permission", "api_call", "opcode"],
                        help="Which encoder to train")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--output_dim", type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--n_triplets", type=int,   default=1024)
    args = parser.parse_args()
    train(args)
