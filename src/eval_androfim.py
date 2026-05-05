"""
eval_androfim.py — Multi-crop ProtoNet evaluation for AndroFIM.

Implements Algorithm 1 from Zhou et al. (2025):
  - For each query q, generate k random crop positions from q's shape.
  - Crop both q and every support sample s at the same positions.
  - Query feature = mean of k backbone(crop) vectors.
  - Support feature = weighted average of k backbone(crop) vectors,
    weights = softmax(-euclidean_distance) to corresponding query crops.
  - Apply per-query centering + L2 normalisation to all vectors.
  - ProtoNet: prototype = mean of normalised support vectors per class,
    classify by argmin Euclidean distance.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from resnet12 import ResNet12


# ─── Constants ────────────────────────────────────────────────────────────────

IMAGE_SIZE = 200   # images resized to this square before cropping
CROP_SIZE  = 180   # b  (b² pixels per crop, reshaped to b×b)
N_CROPS    = 20    # k

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Backbone loading ─────────────────────────────────────────────────────────

def load_backbone(ckpt_path: str, dropout: float = 0.1,
                  device: torch.device = None) -> ResNet12:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = ResNet12(dropout=dropout)
    if "backbone" in ckpt:
        state = {k[len("backbone."):]: v
                 for k, v in ckpt["backbone"].items()
                 if k.startswith("backbone.")}
        backbone.load_state_dict(state)
    else:
        backbone.load_state_dict(ckpt)
    return backbone.to(device).eval()


# ─── Dataset helpers ──────────────────────────────────────────────────────────

def load_family_paths(manifest_path: Path, families: List[str],
                      img_root: Path) -> dict:
    """Return {family_name: [abs_image_path, ...]} for the requested families."""
    df = pd.read_csv(manifest_path)
    df = df[df["family"].isin(set(families))].reset_index(drop=True)
    family_to_paths: dict = {}
    for _, row in df.iterrows():
        path = str(img_root / str(row.year) / Path(row.image_path).name)
        family_to_paths.setdefault(row.family, []).append(path)
    return family_to_paths


# ─── Crop helpers ─────────────────────────────────────────────────────────────

def generate_positions(H: int, W: int, b: int, k: int) -> List[int]:
    """
    Randomly select k starting row positions for b×b crops.
    Each crop takes (b²//W) consecutive rows, which flatten to exactly b² pixels
    and reshape to (b, b).  With IMAGE_SIZE=200, b=180: 162 rows per crop,
    38 valid starting positions.
    """
    n_rows  = (b * b) // W
    max_pos = H - n_rows      # inclusive upper bound
    return random.sample(range(max_pos + 1), k)


def make_crops(img_path: str, positions: List[int],
               b: int = CROP_SIZE) -> torch.Tensor:
    """
    Load image, take k crops at given row positions.
    Returns tensor of shape (k, 3, b, b) normalised for ResNet-12.
    """
    img = Image.open(img_path).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img)          # (H, W, 3)
    H, W = arr.shape[:2]
    n_rows = (b * b) // W
    crops = []
    for pos in positions:
        patch = arr[pos: pos + n_rows, :, :].reshape(b, b, 3)
        crops.append(_transform(Image.fromarray(patch.astype(np.uint8))))
    return torch.stack(crops)   # (k, 3, b, b)


# ─── Core episode ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_episode(
    support_paths:  List[str],
    support_labels: List[int],
    query_paths:    List[str],
    query_labels:   List[int],
    backbone:       ResNet12,
    device:         torch.device,
    b: int = CROP_SIZE,
    k: int = N_CROPS,
) -> tuple:
    """
    Run one multi-crop ProtoNet episode (Algorithm 1).
    Returns (preds, labels) as lists of episode-local class indices.
    """
    n_support = len(support_paths)
    preds     = []

    for q_path, q_label in zip(query_paths, query_labels):

        # 1. Generate k crop positions from query's shape
        positions = generate_positions(IMAGE_SIZE, IMAGE_SIZE, b, k)

        # 2. Query crops → mean feature  (line 6–7 of Algorithm 1)
        q_crops = make_crops(q_path, positions, b).to(device)   # (k, 3, b, b)
        q_feats = backbone(q_crops)                              # (k, embed_dim)
        q_mean  = q_feats.mean(0)                               # (embed_dim,)

        # 3. Support crops → weighted feature  (lines 9–15)
        #    Batch all support images for efficiency.
        all_s_crops = torch.stack([
            make_crops(p, positions, b) for p in support_paths
        ]).to(device)                                            # (n_s, k, 3, b, b)

        flat_feats = backbone(
            all_s_crops.view(n_support * k, 3, b, b)
        )                                                        # (n_s*k, embed_dim)
        s_feats = flat_feats.view(n_support, k, -1)             # (n_s, k, embed_dim)

        # distances[i, j] = ||q_feats[j] - s_feats[i, j]||₂
        distances = torch.norm(
            q_feats.unsqueeze(0) - s_feats, dim=2)              # (n_s, k)
        weights   = F.softmax(-distances, dim=1)                 # (n_s, k)
        s_vecs    = (weights.unsqueeze(2) * s_feats).sum(1)     # (n_s, embed_dim)

        # 4. Centering + L2 normalisation (per-query)
        #    z̄^q = mean of { all support vectors for this query, q̄ }
        all_vecs = torch.cat([s_vecs, q_mean.unsqueeze(0)], dim=0)  # (n_s+1, d)
        center   = all_vecs.mean(0)

        s_norm = F.normalize(s_vecs  - center, dim=1)           # (n_s, embed_dim)
        q_norm = F.normalize(q_mean  - center, dim=0)           # (embed_dim,)

        # 5. Prototype = mean of normalised support vectors per class  (eq. 9)
        unique_labels = sorted(set(support_labels))
        protos = torch.stack([
            s_norm[[i for i, l in enumerate(support_labels) if l == c]].mean(0)
            for c in unique_labels
        ])                                                       # (n_way, embed_dim)

        # 6. Classify: argmin Euclidean distance  (eq. 10)
        dists = torch.norm(q_norm.unsqueeze(0) - protos, dim=1) # (n_way,)
        pred  = unique_labels[dists.argmin().item()]
        preds.append(pred)

    return preds, list(query_labels)


# ─── Episode sampler ──────────────────────────────────────────────────────────

def evaluate(
    backbone:      ResNet12,
    manifest_path: Path,
    families:      List[str],
    img_root:      Path,
    n_way:         int   = 5,
    k_shot:        int   = 5,
    n_query:       int   = 15,
    n_episodes:    int   = 600,
    b:             int   = CROP_SIZE,
    k:             int   = N_CROPS,
    device:        torch.device = None,
    seed:          int   = 42,
) -> Tuple[float, float]:
    """
    Sample few-shot episodes and return (mean_accuracy, 95%_confidence_interval).
    Families that have fewer than (k_shot + n_query) images are silently skipped.
    """
    if device is None:
        device = next(backbone.parameters()).device

    family_to_paths = load_family_paths(manifest_path, families, img_root)
    valid_families  = [
        f for f in families
        if f in family_to_paths and len(family_to_paths[f]) >= k_shot + n_query
    ]
    if len(valid_families) < n_way:
        raise ValueError(
            f"Only {len(valid_families)} families have ≥ {k_shot + n_query} images, "
            f"need at least {n_way} for {n_way}-way evaluation."
        )

    rng  = random.Random(seed)
    accs = []
    backbone.eval()

    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        ep_families = rng.sample(valid_families, n_way)
        support_paths, support_labels = [], []
        query_paths,   query_labels   = [], []

        for label_idx, family in enumerate(ep_families):
            chosen = rng.sample(family_to_paths[family], k_shot + n_query)
            for p in chosen[:k_shot]:
                support_paths.append(p)
                support_labels.append(label_idx)
            for p in chosen[k_shot:]:
                query_paths.append(p)
                query_labels.append(label_idx)

        ep_preds, ep_labels = run_episode(
            support_paths, support_labels,
            query_paths,   query_labels,
            backbone, device, b=b, k=k,
        )
        accs.append(sum(p == l for p, l in zip(ep_preds, ep_labels)) / len(ep_labels))

    mean_acc = float(np.mean(accs))
    ci       = 1.96 * float(np.std(accs)) / np.sqrt(n_episodes)
    return mean_acc, ci




