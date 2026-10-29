import os
import logging
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader
from tqdm import tqdm

from src.core.backbone import BackboneProjectionHead, Backbone
from src.core.experiment_configs import ExperimentConfig
from src.core.utils import ModelParameters, Utils
from src.core.parse_dataset import HiGraphData

def _make_weighted_sampler(training_dataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that equalises class frequencies."""
    all_labels = torch.cat([ds.get_labels() for ds in training_dataset.dataset.datasets])
    labels = all_labels[torch.tensor(training_dataset.indices, dtype=torch.long)]
    class_counts = torch.bincount(labels)
    class_weights = torch.where(
        class_counts > 0,
        1.0 / class_counts.float(),
        torch.zeros_like(class_counts, dtype=torch.float),
    )
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def validate_knn(model: torch.nn.Module, model_params: ModelParameters, validation_dataset, logger: logging.Logger, k: int = 5):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import train_test_split

    device = Utils.get_device()

    loader = DataLoader(validation_dataset, batch_size=model_params.backbone_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            graph_rep, _ = model(batch)
            all_embeddings.append(graph_rep.cpu())
            all_labels.append(batch.y.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    all_embeddings = normalize(all_embeddings, norm='l2')

    X_bank, X_query, y_bank, y_query = train_test_split(
        all_embeddings, all_labels, test_size=0.5, random_state=42, stratify=all_labels
    )

    clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')

    # Fit ONLY on the bank (The Support Set)
    clf.fit(X_bank, y_bank)

    # Score ONLY on the unseen query graphs
    accuracy = clf.score(X_query, y_query)

    logger.info(f"Validation (KNN k={k}) | Accuracy: {accuracy:.4f} | Families: {len(set(all_labels))}")

    model.train()

    return accuracy


def train(model: torch.nn.Module, model_params: ModelParameters, model_saving_path:str, training_dataset, validation_dataset, balanced: bool = False):

    device = Utils.get_device()

    log_file_path = os.path.join(os.path.dirname(model_saving_path), 'backbone_training.log')
    logger = Utils.setup_logger('whole_graph_backbone_training', log_file_path)

    logger.info(f"Device: {device}")
    logger.info(f"Model ID: {model_params.id}")
    logger.info(
        f"Epochs: {model_params.backbone_epochs} | Batch size: {model_params.backbone_batch_size} | LR: {model_params.backbone_lr}")



    if balanced:
        sampler = _make_weighted_sampler(training_dataset)
        loader = DataLoader(training_dataset, batch_size=model_params.backbone_batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True)
        logger.info("Sampler: WeightedRandomSampler (balanced classes)")
    else:
        loader = DataLoader(training_dataset, batch_size=model_params.backbone_batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
        logger.info("Sampler: shuffle=True (unweighted)")

    model = model.to(device)

    projection_head = BackboneProjectionHead(
        in_channels= model_params.final_channels, hidden_channels=model_params.backbone_projection_head_hidden_channels, out_channels=model_params.backbone_projection_head_output_channels,
    ).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(projection_head.parameters()), lr=model_params.backbone_lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=model_params.backbone_epochs, eta_min=model_params.backbone_eta_lr_min
    )

    best_val_accuracy = 0.0

    backbone_epochs = model_params.backbone_epochs

    for epoch in range(backbone_epochs):
        model.train()

        projection_head.train()

        train_loss = 0.0
        progress = tqdm(loader, desc=f"Training: Epoch {epoch + 1}/ {backbone_epochs}", leave=True)

        for batch in progress:

            batch = batch.to(device)
            optimizer.zero_grad()

            g1 = Utils.augment_graph(batch, edge_perturb=0.3, node_perturb=0.3)
            g2 = Utils.augment_graph(batch, edge_perturb=0.3, node_perturb=0.3)

            h1, _ = model(g1, return_node_embeddings=False)
            h2, _ = model(g2, return_node_embeddings=False)

            l = Utils.memory_intense_nt_xent_loss(projection_head(h1), projection_head(h2), temperature=model_params.backbone_pretrain_instance_level_projection_temperature)

            l.backward()
            optimizer.step()

            train_loss += l.item()

        if scheduler:
            scheduler.step()

        avg_train_loss = train_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]

        print(f'Average training loss: {avg_train_loss:.2f} | LR: {current_lr}')

        if (epoch + 1) % 5 == 0:
            print("\n[*] Running Validation...")
            val_acc = validate_knn(model, model_params, validation_dataset, logger)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), model_saving_path)
                logger.info(f"--> New best model saved! (Accuracy: {best_val_accuracy:.4f})")

    logger.info(f"Training complete. Best model saved to {model_saving_path} with Val Acc: {best_val_accuracy:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Backbone pre-training')
    parser.add_argument('--config', default='trial_config_v3',
                        help='ExperimentConfig preset name')
    parser.add_argument('--balanced', action='store_true',
                        help='Use WeightedRandomSampler to equalise class frequencies during pretraining')
    args = parser.parse_args()

    getter = getattr(ExperimentConfig, f'get_{args.config}')
    config = getter()

    training_ds, evaluation_ds, _ = Utils.get_splits(config)

    backbone = Backbone(
        cfg_dim=config.cfg_dim,
        cfg_hidden_channels=config.cfg_hidden_channels,
        fcg_hidden_channels=config.fcg_hidden_channels,
        final_channels=config.final_channels,
        func_feat_dim=config.func_feat_dim,
        dropout_value=config.dropout,
    )

    saving_path = Utils.get_whole_graph_backbone_model_saved_path(parameters=config)

    print(f"[*] Model saving path for this run: {saving_path}")

    train(model=backbone, model_params=config, model_saving_path=saving_path, training_dataset=training_ds,
          validation_dataset=evaluation_ds, balanced=args.balanced)