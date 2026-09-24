"""
train_fcgat.py

Supervised pre-training for StandaloneFCGAT with cross-entropy loss.

Strategy:
  - Uses Utils.get_splits() for train/val/test according to dataset_split_type.
  - For Default dataset type, training families are a strict subset of global labels
    (non-contiguous), so labels are remapped to contiguous 0..N-1 indices.
  - For PerFamilyGroup, all families are present in training split.
  - Validation uses KNN accuracy on graph embeddings (works for both dataset types,
    including Default where val families != train families).
  - Best model (highest KNN val accuracy) is saved to Utils.get_fcgat_model_saved_path().
"""

import os
import time
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.core.utils import ModelParameters, Utils, DatasetType
from src.core.fcgat_model import StandaloneFCGAT
from src.core.experiment_configs import ExperimentConfig
from src.core.parse_dataset import HiGraphData


def build_label_remap(dataset):
    """
    Build a mapping from original (global) family labels to contiguous 0..N-1 indices.
    Returns (remap_dict, num_classes).
    """
    all_labels = set()
    for data in dataset:
        all_labels.add(data.y)
    sorted_labels = sorted(all_labels)
    remap = {orig: new for new, orig in enumerate(sorted_labels)}
    return remap, len(sorted_labels)


def apply_remap(y: torch.Tensor, remap: dict) -> torch.Tensor:
    return torch.tensor([remap[int(v.item())] for v in y], dtype=torch.long, device=y.device)


def validate_knn(model: StandaloneFCGAT, val_dataset, params: ModelParameters,
                 logger, k: int = 5):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import train_test_split

    device = Utils.get_device()
    loader = DataLoader(val_dataset, batch_size=params.backbone_batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    all_embs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embs, _ = model.embed(batch)
            all_embs.append(embs.cpu())
            all_labels.append(batch.y.cpu())

    all_embs   = torch.cat(all_embs,   dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_embs   = normalize(all_embs, norm='l2')

    # Need at least 2 samples per class for stratified split
    unique, counts = torch.unique(torch.tensor(all_labels), return_counts=True)
    min_count = counts.min().item()
    if min_count < 2 or len(unique) < 2:
        logger.info(f"Validation (KNN k={k}) | Skipped — too few samples per class")
        model.train()
        return 0.0

    X_bank, X_query, y_bank, y_query = train_test_split(
        all_embs, all_labels, test_size=0.5, random_state=42, stratify=all_labels
    )

    clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    clf.fit(X_bank, y_bank)
    accuracy = clf.score(X_query, y_query)

    logger.info(f"Validation (KNN k={k}) | Accuracy: {accuracy:.4f} | Families: {len(unique)}")
    model.train()
    return accuracy


def train(params: ModelParameters):
    device      = Utils.get_device()
    use_cuda    = torch.cuda.is_available()
    saving_path = Utils.get_fcgat_model_saved_path(params)
    log_path    = os.path.join(os.path.dirname(saving_path), 'fcgat_training.log')
    logger      = Utils.setup_logger('fcgat_training', log_path)

    logger.info(f"Device: {device}")
    logger.info(f"Config ID: {params.id} | DatasetSplitType: {params.dataset_split_type.value}")

    # ------------------------------------------------------------------ data
    train_ds, val_ds, _ = Utils.get_splits(params)

    remap, num_classes = build_label_remap(train_ds)
    logger.info(f"Training classes: {num_classes}")

    train_loader = DataLoader(
        train_ds, batch_size=params.backbone_batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )

    # ------------------------------------------------------------------ model
    model = StandaloneFCGAT(
        cfg_dim=params.cfg_dim,
        func_feat_dim=params.func_feat_dim,
        num_classes=num_classes,
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"StandaloneFCGAT parameters: total={total_params:,}  trainable={trainable_params:,}"
    )

    # ------------------------------------------------------------------ optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.backbone_epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs       = params.backbone_epochs

    # Overall accumulators across all epochs
    all_wall_times = []   # seconds per epoch
    all_gpu_times  = []   # milliseconds per epoch (CUDA only)
    all_peak_mems  = []   # MB per epoch (CUDA only)

    for epoch in range(epochs):
        model.train()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        # ---- Per-epoch profiling setup ----
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()

        wall_start = time.perf_counter()

        progress = tqdm(train_loader, desc=f"FCGAT Epoch {epoch+1}/{epochs}", leave=True)
        for batch in progress:
            batch = batch.to(device)

            remapped_y = apply_remap(batch.y, remap)

            optimizer.zero_grad()
            logits = model(batch)
            loss   = criterion(logits, remapped_y)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item()
            preds          = logits.argmax(dim=1)
            total_correct += (preds == remapped_y).sum().item()
            total_samples += remapped_y.size(0)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        wall_elapsed = time.perf_counter() - wall_start
        all_wall_times.append(wall_elapsed)

        if use_cuda:
            end_event.record()
            torch.cuda.synchronize()
            gpu_elapsed_ms = start_event.elapsed_time(end_event)
            peak_mem_mb    = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            all_gpu_times.append(gpu_elapsed_ms)
            all_peak_mems.append(peak_mem_mb)

        scheduler.step()

        avg_loss  = total_loss / len(train_loader)
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}"
        )

        # ---- Per-epoch profiling summary ----
        perf_msg = f"  [perf] wall={wall_elapsed:.1f}s"
        if use_cuda:
            perf_msg += f"  gpu={gpu_elapsed_ms:.1f}ms  peak_mem={peak_mem_mb:.1f}MB"
        logger.info(perf_msg)

        if (epoch + 1) % 5 == 0:
            val_acc = validate_knn(model, val_ds, params, logger)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), saving_path)
                logger.info(f"--> New best FCGAT saved! (KNN Val Acc: {best_val_acc:.4f})")

    logger.info(
        f"Training complete. Best KNN val acc: {best_val_acc:.4f} | "
        f"Saved to: {saving_path}"
    )

    # ---- End-to-end profiling summary ----
    if all_wall_times:
        avg_wall = sum(all_wall_times) / len(all_wall_times)
        summary_msg = (
            f"[*] Profiling summary ({len(all_wall_times)} epochs total)\n"
            f"    avg wall time per epoch : {avg_wall:.1f}s\n"
            f"    total wall time         : {sum(all_wall_times):.1f}s"
        )
        if use_cuda and all_gpu_times:
            avg_gpu      = sum(all_gpu_times) / len(all_gpu_times)
            avg_mem      = sum(all_peak_mems) / len(all_peak_mems)
            peak_mem_max = max(all_peak_mems)
            summary_msg += (
                f"\n    avg GPU time per epoch  : {avg_gpu:.1f}ms"
                f"\n    avg peak GPU memory     : {avg_mem:.1f}MB"
                f"\n    max peak GPU memory     : {peak_mem_max:.1f}MB"
            )
        logger.info(summary_msg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='StandaloneFCGAT pre-training')
    parser.add_argument('--config', default='trial_config_v3',
                        help='ExperimentConfig preset name')
    args = parser.parse_args()

    getter = getattr(ExperimentConfig, f'get_{args.config}')
    config = getter()

    train(config)
