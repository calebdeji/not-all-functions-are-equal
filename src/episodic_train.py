"""
episodic_train.py

Episodic fine-tuning pipeline for few-shot malware family classification.

Protocol (mirrors evaluation exactly):
  Support phase  : labels known   → compute weights → build node+graph prototypes
  Query   phase  : labels unknown → pseudo-label from node prototypes → pool → classify
  Loss           : prototypical CE (queries vs graph prototypes) + SupCon (all embeddings)

Usage:
  python -m src.episodic_train --config trial_config_v3 --variant self_attention
"""

import argparse
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from backbone import Backbone
from experiment_configs import ExperimentConfig
from utils import Utils, ModelParameters, DatasetType
from parse_dataset import HiGraphData
from episodic_models import (
    TrainingUtils,
    SelfAttentionModel,
    SDLDegreeModel,
    SDLComplexityModel,
    SDLWithoutLearnableParamModel,
    SDLComplexityPriorModel,
    SDLWithIntraParamModel,
    SDLWithoutStructureModel,
    _scatter_mean,
    _scatter_add,
)

# ---------------------------------------------------------------------------
# Ablation registry
# ---------------------------------------------------------------------------


ABLATION_REGISTRY = {
    'sdl_no_learnable_param': {
        'class': SDLWithoutLearnableParamModel,
        'needs_fcg_dim': False,
    },
    'sdl_complexity_prior': {
        'class': SDLComplexityPriorModel,
        'needs_fcg_dim': False,
    },
    'sdl_with_intra_param': {
        'class': SDLWithIntraParamModel,
        'needs_fcg_dim': False,
    },
    'sdl_without_structure': {
        'class': SDLWithoutStructureModel,
        'needs_fcg_dim': False,
    },
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

SDL_REGISTRY = {
    'sdl_degree': {
        'class': SDLDegreeModel,
        'needs_fcg_dim': False,
    },
    'sdl_complexity': {
        'class': SDLComplexityModel,
        'needs_fcg_dim': False,
    },
}

MODEL_REGISTRY = {
    
    'self_attention': {
        'class': SelfAttentionModel,
        'needs_fcg_dim': True,
    },
    **SDL_REGISTRY,
    # Ablation variants
    **ABLATION_REGISTRY,
    
}


BASELINE_REGISTRY = {
    'self_attention': {
        'class': SelfAttentionModel,
        'needs_fcg_dim': True,
    },
    **SDL_REGISTRY,
}


def create_model(variant: str, backbone: Backbone, fcg_dim: int):
    """Wrap backbone with the requested pooling head."""
    info = MODEL_REGISTRY[variant]
    cls = info['class']
    if info['needs_fcg_dim']:
        return cls(backbone=backbone, fcg_dim=fcg_dim)
    return cls(backbone=backbone)


# ---------------------------------------------------------------------------
# SupCon Loss
# ---------------------------------------------------------------------------

class SupConLoss(torch.nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Applied to L2-normalised graph embeddings (support + query).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features : (N, D) L2-normalised embeddings
        labels   : (N,)  integer class ids
        """
        device = features.device
        n = features.size(0)

        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.t()) / self.temperature

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Positive mask: same label, different sample
        labels = labels.view(-1, 1)
        mask = (labels == labels.t()).float().to(device)
        mask.fill_diagonal_(0)

        # Log-softmax denominator: all except self
        exp_sim = torch.exp(sim)
        self_mask = torch.eye(n, device=device).bool()
        denom = exp_sim.masked_fill(self_mask, 0).sum(dim=1, keepdim=True)
        log_prob = sim - torch.log(denom + 1e-8)

        # Mean over positives
        n_pos = mask.sum(dim=1)
        loss = -(mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
        return loss.mean()


# ---------------------------------------------------------------------------
# Weight computation helpers
# ---------------------------------------------------------------------------

def compute_support_weights(model, node_embs, node_labels, data, fcg_batch):
    """Compute per-node weights for the support set (labels known)."""
    

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

    if isinstance(model, SDLWithIntraParamModel):
        return TrainingUtils.generate_sdl_intra_param_weights(
            node_embeddings=node_embs,
            node_labels=node_labels,
            lambda_a_val=model.raw_lambda_a,
        )

    if isinstance(model, SDLWithoutStructureModel):
        return TrainingUtils.generate_sdl_without_structure_weights(
            node_embeddings=node_embs,
            node_labels=node_labels,
            lambda_a_val=model.raw_lambda_a,
            lambda_b_val=model.raw_lambda_b,
        )

    raise ValueError(f'Unknown model type: {type(model).__name__}')


def compute_query_weights(model, query_node_embs, node_prototypes, data):
    """Compute per-node weights for the query set (pseudo-labels from prototypes)."""

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

    if isinstance(model, SDLWithIntraParamModel):
        return TrainingUtils.generate_sdl_intra_param_evaluation_weights(
            query_node_embeddings=query_node_embs,
            support_prototypes=node_prototypes,
            lambda_a_val=model.raw_lambda_a,
        )

    if isinstance(model, SDLWithoutStructureModel):
        return TrainingUtils.generate_sdl_without_structure_evaluation_weights(
            query_node_embeddings=query_node_embs,
            support_prototypes=node_prototypes,
            lambda_a_val=model.raw_lambda_a,
            lambda_b_val=model.raw_lambda_b,
        )

    raise ValueError(f'Unknown model type: {type(model).__name__}')


def pool_nodes(model, node_embs, batch, weights, data=None):
    """Pool node embeddings to graph level using model-specific pooling."""
    if isinstance(model, SelfAttentionModel):
        return model.forward_attention(node_embeddings=node_embs, batch=batch)

    # All weighted variants (mean_pooling ignores weights internally)
    return model.forward_attention(node_embeddings=node_embs, batch=batch, weights=weights)


# ---------------------------------------------------------------------------
# Episode sampling
# ---------------------------------------------------------------------------

def extract_episode(indices_by_class, n_way, k_shot, q_query):
    """
    Sample one N-way K-shot episode.
    Returns (support_indices, query_indices, sampled_global_class_ids)
    or (None, None, None) if not enough classes/samples.
    """
    classes = list(indices_by_class.keys())
    if len(classes) < n_way:
        return None, None, None

    sampled_classes = np.random.choice(classes, n_way, replace=False)
    s_indices, q_indices = [], []

    for cls in sampled_classes:
        idxs = indices_by_class[cls]
        needed = k_shot + q_query
        if len(idxs) < k_shot:
            return None, None, None
        selected = np.random.choice(idxs, needed, replace=(len(idxs) < needed))
        s_indices.extend(selected[:k_shot].tolist())
        q_indices.extend(selected[k_shot:].tolist())

    return s_indices, q_indices, sampled_classes


def build_episode_batches(dataset, s_idx, q_idx, sampled_classes, device):
    """
    Load support/query mini-batches and assign episode-local labels [0, N-1].

    dataset.y holds global family indices; we remap them to contiguous
    episode labels and store in data.family_class.
    """
    s_loader = DataLoader(Subset(dataset, s_idx), batch_size=len(s_idx), shuffle=False)
    q_loader = DataLoader(Subset(dataset, q_idx), batch_size=len(q_idx), shuffle=False)

    support_batch = next(iter(s_loader))
    query_batch = next(iter(q_loader))

    label_map = {int(c): i for i, c in enumerate(sampled_classes)}
    support_batch.family_class = torch.tensor(
        [label_map[int(y)] for y in support_batch.y], dtype=torch.long
    )
    query_batch.family_class = torch.tensor(
        [label_map[int(y)] for y in query_batch.y], dtype=torch.long
    )

    return support_batch.to(device), query_batch.to(device)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def episodic_step(model, support_data, query_data, aux_criterion,
                  proto_scale=16.0, lambda_sup=0.5):
    """
    One episodic forward pass + loss computation.

    Forward pass mirrors evaluation exactly:
      Support  : labels known    → compute weights with ground-truth labels
      Query    : labels unknown  → compute weights with pseudo-labels from node prototypes
      Classify : nearest prototype on cosine similarity (scaled by proto_scale)

    Loss = prototypical CE  +  lambda_sup * SupCon
    """
    # ---- SUPPORT PHASE (labels known) ----------------------------------------
    sup_node_embs, sup_fcg_batch = model.get_node_embeddings(support_data)
    sup_node_labels = torch.repeat_interleave(
        support_data.family_class, support_data.num_functions
    )

    sup_weights = compute_support_weights(
        model, sup_node_embs, sup_node_labels, support_data, sup_fcg_batch
    )

    # Node prototypes for query pseudo-labelling
    _, labels_inverse = torch.unique(sup_node_labels, return_inverse=True)
    weighted_sup = sup_node_embs * sup_weights.view(-1, 1)
    node_prototypes = _scatter_mean(weighted_sup, labels_inverse, dim=0)
    node_prototypes = F.normalize(node_prototypes, p=2, dim=-1)

    # Graph prototypes for classification
    sup_graph_embs = pool_nodes(model, sup_node_embs, sup_fcg_batch, sup_weights,
                                data=support_data)
    graph_prototypes = _scatter_mean(sup_graph_embs, support_data.family_class, dim=0)
    graph_prototypes = F.normalize(graph_prototypes, p=2, dim=-1)

    # ---- QUERY PHASE (pseudo-labels) -----------------------------------------
    query_node_embs, query_fcg_batch = model.get_node_embeddings(query_data)

    query_weights = compute_query_weights(
        model, query_node_embs, node_prototypes, query_data
    )

    query_graph_embs = pool_nodes(
        model, query_node_embs, query_fcg_batch, query_weights, data=query_data
    )

    # ---- LOSSES --------------------------------------------------------------
    # 1. Prototypical CE: queries classified by nearest graph prototype
    similarities = query_graph_embs @ graph_prototypes.t()
    loss_proto = F.cross_entropy(similarities * proto_scale, query_data.family_class)

    # 2. SupCon: tight family clusters across all graph embeddings
    all_embs = torch.cat([sup_graph_embs, query_graph_embs], dim=0)
    all_labels = torch.cat([support_data.family_class, query_data.family_class], dim=0)
    loss_aux = aux_criterion(all_embs, all_labels)

    loss = loss_proto + lambda_sup * loss_aux

    preds = similarities.argmax(dim=1)
    return loss, loss_proto.item(), loss_aux.item(), preds, query_data.family_class


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def episodic_finetune(
    model,
    training_dataset,
    evaluation_dataset,
    model_saving_path: str,
    logger: logging.Logger,
    num_epochs: int = 100,
    episodes_per_epoch: int = 200,
    n_way_train: int = 5,
    k_shot_train: int = 5,
    q_query_train: int = 15,
    lr: float = 1e-4,
    proto_scale: float = 16.0,
    supcon_temp: float = 0.1,
    lambda_sup: float = 0.5,
    eval_every: int = 5,
    eval_episodes: int = 200,
    patience: int = 6,
    accumulation_steps: int = 1,
):
    """
    Episodic fine-tuning loop.

    Each epoch runs `episodes_per_epoch` N-way K-shot episodes.
    Every `eval_every` epochs validates on evaluation_dataset using
    OnTheFlyFewShotEvaluator (nearest-prototype, same as test-time inference).
    Saves best model by validation accuracy; stops early when patience is exceeded.
    """
    # Import here to avoid circular imports at module level
    from episodic_evaluate import OnTheFlyFewShotEvaluator

    device = next(model.parameters()).device
    aux_criterion = SupConLoss(temperature=supcon_temp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    # Build class → sample index mapping using pre-computed labels.pt (fast)
    # training_dataset is Subset(ConcatDataset([HiGraphDataset, ...]), train_indices)
    all_labels = torch.cat([ds.get_labels() for ds in training_dataset.dataset.datasets])
    train_labels = all_labels[torch.tensor(training_dataset.indices, dtype=torch.long)]

    indices_by_class = defaultdict(list)
    for i, lbl in enumerate(train_labels.tolist()):
        indices_by_class[lbl].append(i)

    min_samples = k_shot_train + 1
    valid_classes = {c: idxs for c, idxs in indices_by_class.items()
                     if len(idxs) >= min_samples}

    use_cuda = torch.cuda.is_available()

    # ---- Parameter count ----
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f'[*] Episodic fine-tuning: {len(valid_classes)} classes, '
        f'{n_way_train}-way {k_shot_train}-shot {q_query_train}-query, '
        f'{episodes_per_epoch} eps/epoch, lr={lr}'
    )
    logger.info(
        f'[*] Model parameters: total={total_params:,}  trainable={trainable_params:,}'
    )

    evaluator = OnTheFlyFewShotEvaluator(model=model, device=device)
    best_val_acc = 0.0
    counter = 0

    # Overall accumulators across all epochs
    all_wall_times  = []   # seconds per episode
    all_gpu_times   = []   # milliseconds per episode (CUDA only)
    all_peak_mems   = []   # MB per episode (CUDA only)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = epoch_proto = epoch_aux = 0.0
        epoch_correct = epoch_total = valid_episodes = 0

        epoch_wall_times = []
        epoch_gpu_times  = []
        epoch_peak_mems  = []

        progress = tqdm(range(episodes_per_epoch),
                        desc=f'Epoch {epoch + 1}/{num_epochs}')

        optimizer.zero_grad()
        for episode_idx, _ in enumerate(progress):
            s_idx, q_idx, sampled_classes = extract_episode(
                valid_classes, n_way_train, k_shot_train, q_query_train
            )
            if s_idx is None:
                continue

            sup_data, qry_data = build_episode_batches(
                training_dataset, s_idx, q_idx, sampled_classes, device
            )

            # ---- Per-episode profiling ----
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                start_event.record()

            wall_start = time.perf_counter()

            loss, proto_l, aux_l, preds, labels = episodic_step(
                model, sup_data, qry_data,
                aux_criterion=aux_criterion,
                proto_scale=proto_scale,
                lambda_sup=lambda_sup,
            )
            (loss / accumulation_steps).backward()

            wall_elapsed = time.perf_counter() - wall_start

            if use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                gpu_elapsed_ms = start_event.elapsed_time(end_event)
                peak_mem_mb    = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                epoch_gpu_times.append(gpu_elapsed_ms)
                epoch_peak_mems.append(peak_mem_mb)

            epoch_wall_times.append(wall_elapsed)

            if (episode_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

            valid_episodes += 1
            epoch_loss += loss.item()
            epoch_proto += proto_l
            epoch_aux += aux_l
            epoch_correct += (preds == labels).sum().item()
            epoch_total += len(labels)

            progress.set_postfix({
                'L': f'{loss.item():.3f}',
                'P': f'{proto_l:.3f}',
                'S': f'{aux_l:.3f}',
            })

        all_wall_times.extend(epoch_wall_times)
        all_gpu_times.extend(epoch_gpu_times)
        all_peak_mems.extend(epoch_peak_mems)

        scheduler.step()

        n = max(valid_episodes, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        logger.info(
            f'Epoch {epoch + 1}: loss={epoch_loss / n:.4f}  '
            f'proto={epoch_proto / n:.4f}  supcon={epoch_aux / n:.4f}  '
            f'train_acc={train_acc:.4f}'
        )

        if isinstance(model, (SDLDegreeModel, SDLComplexityModel, SDLWithoutStructureModel)):
            logger.info(f'  λa={model.raw_lambda_a.item():.4f}  λb={model.raw_lambda_b.item():.4f}')
        elif isinstance(model, SDLWithIntraParamModel):
            logger.info(f'  λa={model.raw_lambda_a.item():.4f}')

        # ---- Epoch profiling summary ----
        avg_wall = sum(epoch_wall_times) / len(epoch_wall_times) if epoch_wall_times else 0.0
        perf_msg = f'  [perf] avg episode wall={avg_wall*1000:.1f}ms'
        if use_cuda and epoch_gpu_times:
            avg_gpu  = sum(epoch_gpu_times) / len(epoch_gpu_times)
            avg_mem  = sum(epoch_peak_mems) / len(epoch_peak_mems)
            perf_msg += f'  gpu={avg_gpu:.1f}ms  peak_mem={avg_mem:.1f}MB'
        logger.info(perf_msg)

        # ---- Validation ----
        if (epoch + 1) % eval_every == 0:
            val_acc = evaluator.evaluate_episodes(
                dataset=evaluation_dataset,
                n_way=n_way_train,
                k_shot=k_shot_train,
                n_query=q_query_train,
                n_episodes=eval_episodes,
            )
            logger.info(
                f'[Val] {n_way_train}-way {k_shot_train}-shot: {val_acc:.4f}'
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f'[*] New best: {best_val_acc:.4f}  → {model_saving_path}')
                torch.save(model.state_dict(), model_saving_path)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logger.info(
                        f'[*] Early stopping at epoch {epoch + 1}. '
                        f'Best val acc: {best_val_acc:.4f}'
                    )
                    break

            model.train()

    logger.info(f'[*] Finished. Best validation accuracy: {best_val_acc:.4f}')

    # ---- End-to-end profiling summary ----
    if all_wall_times:
        avg_wall_total = sum(all_wall_times) / len(all_wall_times)
        summary_msg = (
            f'[*] Profiling summary ({len(all_wall_times)} episodes total)\n'
            f'    avg wall time per episode : {avg_wall_total*1000:.1f}ms\n'
            f'    total wall time           : {sum(all_wall_times):.1f}s'
        )
        if use_cuda and all_gpu_times:
            avg_gpu_total  = sum(all_gpu_times) / len(all_gpu_times)
            avg_mem_total  = sum(all_peak_mems) / len(all_peak_mems)
            peak_mem_max   = max(all_peak_mems)
            summary_msg += (
                f'\n    avg GPU time per episode  : {avg_gpu_total:.1f}ms'
                f'\n    avg peak GPU memory       : {avg_mem_total:.1f}MB'
                f'\n    max peak GPU memory       : {peak_mem_max:.1f}MB'
            )
        logger.info(summary_msg)

    return best_val_acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Episodic fine-tuning')
    parser.add_argument('--config', default='trial_config_v3',
                        help='ExperimentConfig preset name')
    parser.add_argument('--variant', default='self_attention',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Pooling head variant')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone weights; only train head')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible episode sampling')
    args = parser.parse_args()

    # ---- Seed ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Setup ----
    getter = getattr(ExperimentConfig, f'get_{args.config}')
    params: ModelParameters = getter()
    device = Utils.get_device()

    frozen_str = 'frozen' if args.freeze_backbone else 'unfrozen'
    model_path = Utils.get_episodic_model_path(params, args.variant, args.freeze_backbone)
    save_dir   = os.path.dirname(model_path)
    log_path   = os.path.join(save_dir, 'train.log')
    logger = Utils.setup_logger(f'episodic_train_{args.variant}_{frozen_str}_{params.dataset_split_type.value}', log_path)

    logger.info(f'Config: {params.id}  Variant: {args.variant} Device: {device}')
    logger.info(f'Seed: {args.seed}  Training years: {params.training_years}')
    logger.info(f'Freeze backbone: {args.freeze_backbone}')
    logger.info(f'Dataset split type: {params.dataset_split_type.value}')

    # ---- Load pre-trained backbone ----
    backbone = Backbone(
        cfg_dim=params.cfg_dim,
        func_feat_dim=params.func_feat_dim,
        cfg_hidden_channels=params.cfg_hidden_channels,
        fcg_hidden_channels=params.fcg_hidden_channels,
        final_channels=params.final_channels,
        dropout_value=params.dropout,
    ).to(device)

    backbone_path = Utils.get_backbone_model_saved_path(params)
    state = torch.load(backbone_path, map_location=device)
    backbone.load_state_dict(state)
    logger.info(f'Loaded backbone from {backbone_path}')

    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)
        logger.info('[*] Backbone frozen')

    # ---- Create model ----
    model = create_model(args.variant, backbone, fcg_dim=params.final_channels).to(device)

    # ---- Load datasets ----
    logger.info('Loading datasets...')
    train_ds, val_ds, _ = Utils.get_splits(params)

    logger.info(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    # ---- Fine-tune ----
    accumulation_steps = (
        10 if args.variant in ('sdl_degree', 'sdl_complexity') and args.freeze_backbone
        else 1
    )
    episodic_finetune(
        model=model,
        training_dataset=train_ds,
        evaluation_dataset=val_ds,
        model_saving_path=model_path,
        logger=logger,
        num_epochs=params.episodic_epochs,
        episodes_per_epoch=params.episodic_episodes_per_epoch,
        n_way_train=params.episodic_n_way,
        k_shot_train=params.episodic_k_shot,
        q_query_train=params.episodic_n_query,
        lr=params.episodic_lr,
        proto_scale=params.episodic_proto_scale,
        supcon_temp=params.episodic_supcon_temp,
        lambda_sup=params.episodic_lambda_sup,
        eval_every=params.episodic_eval_every,
        eval_episodes=params.episodic_eval_episodes,
        patience=params.episodic_patience,
        accumulation_steps=accumulation_steps,
    )


if __name__ == '__main__':
    main()
