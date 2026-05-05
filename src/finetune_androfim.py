

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from resnet12 import ResNet12



# ─── Fine-tuning model wrapper ────────────────────────────────────────────────

class AndroFIMFinetuneModel(nn.Module):

    def __init__(self, backbone: ResNet12, n_families: int):
        super().__init__()
        self.backbone       = backbone
        self.classifier     = nn.Linear(backbone.embed_dim, n_families)
        self.rot_classifier = nn.Linear(backbone.embed_dim, 4)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward_mixup(self, x1: torch.Tensor, x2: torch.Tensor,
                      lam: float, mix_layer: int) -> torch.Tensor:

        stages = [
            self.backbone.layer1, self.backbone.layer2,
            self.backbone.layer3, self.backbone.layer4,
        ]
        h1, h2 = x1, x2
        for stage in stages[:mix_layer]:
            h1 = stage(h1)
            h2 = stage(h2)
        mixed = lam * h1 + (1 - lam) * h2
        for stage in stages[mix_layer:]:
            mixed = stage(mixed)
        return self.backbone.avgpool(mixed).flatten(1)

    def freeze_first_block(self):
        """Freeze layer1 as per paper."""
        for p in self.backbone.layer1.parameters():
            p.requires_grad = False


def load_model(dino_ckpt: str, n_families: int, dropout: float) -> AndroFIMFinetuneModel:

    ckpt = torch.load(dino_ckpt, map_location="cpu", weights_only=False)

    if "teacher" in ckpt:
        raw = ckpt["teacher"]
        if any(k.startswith("module.backbone.") for k in raw):
            prefix = "module.backbone."
        else:
            prefix = "backbone."
        backbone_state = {
            k[len(prefix):]: v
            for k, v in raw.items()
            if k.startswith(prefix)
        }
    elif "backbone" in ckpt:
        raw = ckpt["backbone"]
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in raw.items()
            if k.startswith("backbone.")
        }
    else:
        raise ValueError(
            f"Unrecognised checkpoint format: {dino_ckpt}. "
            "Expected a 'teacher' or 'backbone' key."
        )

    backbone = ResNet12(dropout=dropout)
    missing, _ = backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"[WARN] Missing backbone keys: {missing}")

    return AndroFIMFinetuneModel(backbone, n_families)


# ─── Split helper ─────────────────────────────────────────────────────────────

def load_split(manifest_path: Path, split_file: Path,
               seed: int, ratios: List[float]) -> dict:
    """Load the split JSON produced by generate_splits.py."""
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Run generate_splits.py first to generate the split."
        )
    with open(split_file) as f:
        return json.load(f)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AndroFIMBaseDataset(Dataset):

    def __init__(self, manifest_path: Path, base_years: List[str],
                 img_root: Path = None, flat: bool = False):
        df = pd.read_csv(manifest_path)
        df = df[df["year"].astype(str).isin(set(base_years))].reset_index(drop=True)

        families           = sorted(df["family"].unique().tolist())
        self.family_to_idx = {f: i for i, f in enumerate(families)}
        self.n_families    = len(families)

        if img_root is not None:
            if flat:
                self.paths = [str(img_root / "images" / Path(row.image_path).name) for _, row in df.iterrows()]
            else:
                self.paths = [str(img_root / str(row.year) / Path(row.image_path).name) for _, row in df.iterrows()]
        else:
            self.paths = df["image_path"].tolist()

        self.labels = [self.family_to_idx[f] for f in df["family"].tolist()]

        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def from_families(cls, manifest_path: Path, families: List[str],
                      img_root: Path = None, flat: bool = False) -> 'AndroFIMBaseDataset':
        """Construct dataset filtered by a list of families (no_overlap split)."""
        obj = cls.__new__(cls)
        df = pd.read_csv(manifest_path)
        df["family"] = df["family"].str.lower().str.strip()
        df = df[df["family"].isin(set(families))].reset_index(drop=True)

        fams = sorted(df["family"].unique().tolist())
        obj.family_to_idx = {f: i for i, f in enumerate(fams)}
        obj.n_families    = len(fams)

        if img_root is not None:
            if flat:
                obj.paths = [str(img_root / "images" / Path(row.image_path).name) for _, row in df.iterrows()]
            else:
                obj.paths = [str(img_root / str(row.year) / Path(row.image_path).name) for _, row in df.iterrows()]
        else:
            obj.paths = df["image_path"].tolist()

        obj.labels = [obj.family_to_idx[f] for f in df["family"].tolist()]
        obj.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return obj

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ─── Helpers ──────────────────────────────────────────────────────────────────

ROT_ANGLES = [0, 90, 180, 270]


def rotate_batch(x: torch.Tensor, angle: int) -> torch.Tensor:
    if angle == 0:
        return x
    return torch.rot90(x, k=angle // 90, dims=[2, 3])


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("androfim_finetune")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DINO ResNet-12 on AndroFIM base families"
    )
    parser.add_argument("--dataset",    default="higraph",
                        help="Dataset name (e.g. higraph, maldroid). Drives default ckpt and output paths.")
    parser.add_argument("--dino_ckpt",  default=None,
                        help="DINO checkpoint. Defaults to trained_models/{config}/{dataset}/{split_type}/androfim/checkpoint.pth.")
    parser.add_argument("--manifest",   default=None,
                        help="Path to androFIM_manifest.csv. Defaults to dataset/{dataset}/androfim/androFIM_manifest.csv.")
    parser.add_argument("--out_dir",    default=None,
                        help="Output directory. Defaults to trained_models/{config}/{dataset}/{split_type}/androfim/.")
    parser.add_argument("--config",     default="trial_config_v3",
                        help="HiGraph config ID whose split files to use (e.g. trial_config_v3).")
    parser.add_argument("--base_years", nargs="+",
                        default=["2012", "2013", "2014", "2015", "2016", "2017", "2018"],
                        help="Years used as D_base (default split only, fallback when split JSON absent).")
    parser.add_argument("--split_seed",   type=int,   default=42)
    parser.add_argument("--split_ratios", nargs=3,    type=float, default=[0.6, 0.2, 0.2],
                        help="Base/val/novel ratios for no_overlap split (used only if JSON absent).")
    parser.add_argument("--epochs",      type=int,   default=40)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=0.03)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=2.0)
    parser.add_argument("--workers",     type=int,   default=4)
    args = parser.parse_args()

    from experiment_configs import ExperimentConfig
    from utils import Utils
    _config  = getattr(ExperimentConfig, f'get_{args.config}')()
    split_type = _config.dataset_split_type.value

    project_root = Path(__file__).resolve().parent.parent
    androfim_dir = Utils.get_androfim_save_dir(_config.id, args.dataset, split_type)
    dino_ckpt = Path(args.dino_ckpt) if args.dino_ckpt is not None \
                else Path(androfim_dir) / "checkpoint.pth"
    manifest  = Path(args.manifest) if args.manifest is not None \
                else Path(Utils.get_androfim_manifest(args.dataset))
    out_dir   = Path(args.out_dir) if args.out_dir is not None \
                else Path(androfim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(out_dir / "finetune.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | Config: {args.config}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    from utils import _FLAT_DATASET_NAMES
    img_root   = Path(Utils.get_androfim_root(args.dataset))
    config_dir = project_root / "dataset_config_split" / args.dataset / _config.id

    if args.dataset in _FLAT_DATASET_NAMES:
        split_file = config_dir / "split.json"
    else:
        split_file = config_dir / "split.json"

    

    flat = args.dataset in _FLAT_DATASET_NAMES

    split = load_split(manifest, split_file, args.split_seed, args.split_ratios)
    logger.info(f"no_overlap split: base={len(split['base'])} families, "
                f"val={len(split['val'])}, novel={len(split['novel'])}")
    dataset      = AndroFIMBaseDataset.from_families(manifest, split["base"], img_root=img_root, flat=flat)
    
    
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True,
    )
    logger.info(f"D_base: {len(dataset)} images | {dataset.n_families} families "
                f"| {len(loader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(str(dino_ckpt), dataset.n_families, args.dropout)
    model.freeze_first_block()
    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── Optimisers (SGD / Adam switch on cosine schedule sign) ───────────────
    opt_sgd  = torch.optim.SGD(trainable, lr=args.lr, momentum=0.9,
                                weight_decay=1e-4, nesterov=True)
    opt_adam = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_sgd, T_max=args.epochs, eta_min=-args.lr * 0.01
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    ckpt_path   = out_dir / "finetune_checkpoint.pth"
    start_epoch = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt_sgd.load_state_dict(ckpt["opt_sgd"])
        opt_adam.load_state_dict(ckpt["opt_adam"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────────────────────
    mix_layers = [0, 1, 2, 3]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        step1_total = step2_total = 0.0

        current_lr = scheduler.get_last_lr()[0] if epoch > 0 else args.lr
        optimizer  = opt_sgd if current_lr > 0 else opt_adam
        for pg in optimizer.param_groups:
            pg["lr"] = abs(current_lr)

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)

            # ── Step 1: Manifold Mixup ────────────────────────────────────────
            lam       = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
            mix_layer = random.choice(mix_layers)
            idx       = torch.randperm(B, device=device)
            x2, y2    = x[idx], y[idx]

            mixed_feat    = model.forward_mixup(x, x2, lam, mix_layer)
            family_logits = model.classifier(mixed_feat)

            y_soft  = lam * F.one_hot(y,  dataset.n_families).float() + \
                      (1 - lam) * F.one_hot(y2, dataset.n_families).float()
            loss_mm = torch.mean(
                torch.sum(-y_soft * F.log_softmax(family_logits, dim=-1), dim=-1)
            )

            optimizer.zero_grad()
            loss_mm.backward()
            optimizer.step()
            step1_total += loss_mm.item()

            # ── Step 2: Family classification + rotation prediction ───────────
            loss_family = loss_rot = 0.0
            for r_idx, angle in enumerate(ROT_ANGLES):
                feat = model.features(rotate_batch(x, angle))
                loss_family += F.cross_entropy(model.classifier(feat), y)
                rot_labels   = torch.full((B,), r_idx, dtype=torch.long, device=device)
                loss_rot    += F.cross_entropy(model.rot_classifier(feat), rot_labels)

            L2 = (loss_family + loss_rot) / len(ROT_ANGLES)

            optimizer.zero_grad()
            L2.backward()
            optimizer.step()
            step2_total += L2.item()

        scheduler.step()
        n = len(loader)
        logger.info(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"L_mm={step1_total/n:.4f} | L2={step2_total/n:.4f} | "
            f"lr={abs(current_lr):.5f} | opt={'sgd' if current_lr > 0 else 'adam'}"
        )

        

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "opt_sgd":       opt_sgd.state_dict(),
                "opt_adam":      opt_adam.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "family_to_idx": dataset.family_to_idx,
            }, ckpt_path)
            logger.info(f"Checkpoint -> {ckpt_path}")

    # ── Save backbone only (no heads) for multi-crop ProtoNet evaluation ──────
    backbone_state = {
        k: v for k, v in model.state_dict().items()
        if k.startswith("backbone.")
    }
    final_path = out_dir / "finetuned_backbone.pth"
    torch.save({
        "backbone":      backbone_state,
        "family_to_idx": dataset.family_to_idx,
        "base_years":    args.base_years,
    }, final_path)
    logger.info(f"Fine-tuned backbone saved -> {final_path}")


if __name__ == "__main__":
    main()
