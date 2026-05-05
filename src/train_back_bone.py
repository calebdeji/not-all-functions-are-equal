import torch
import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils import ModelParameters, Utils
from backbone import Backbone, BackboneProjectionHead
from experiment_configs import ExperimentConfig

from cross_view_alignment import validate_cross_view_alignment

from parse_dataset import HiGraphData





def train(model: torch.nn.Module, model_params: ModelParameters, model_saving_path: str, training_dataset,
          validation_dataset):
    device = Utils.get_device()

    log_file_path = os.path.join(os.path.dirname(model_saving_path), 'backbone_training.log')
    logger = Utils.setup_logger('backbone_training', log_file_path)

    logger.info(f"Device: {device}")
    logger.info(f"Model ID: {model_params.id}")
    logger.info(
        f"Epochs: {model_params.backbone_epochs} | Batch size: {model_params.backbone_batch_size} | LR: {model_params.backbone_lr}")

    loader = DataLoader(training_dataset, batch_size=model_params.backbone_batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
        

    model = model.to(device)

    instance_level_projection_head = BackboneProjectionHead(
        model_params.final_channels, model_params.backbone_projection_head_hidden_channels,
        model_params.backbone_projection_head_output_channels
    ).to(device)

    set_level_projection_head = BackboneProjectionHead(
        model_params.final_channels, model_params.backbone_projection_head_hidden_channels,
        model_params.backbone_projection_head_output_channels
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(instance_level_projection_head.parameters()) + list(
            set_level_projection_head.parameters()),
        lr=model_params.backbone_lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=model_params.backbone_epochs, eta_min=model_params.backbone_eta_lr_min
    )

    best_val_accuracy = 0.0  # Track this to save the best model
    top_k = model_params.backbone_set_level_features_top_k
    backbone_epochs = model_params.backbone_epochs

    for epoch in range(backbone_epochs):
        model.train()
        instance_level_projection_head.train()
        set_level_projection_head.train()

        if epoch == 0:
            validate_cross_view_alignment(model, validation_dataset, logger)

        train_loss = 0.0
        progress = tqdm(loader, desc=f"Training: Epoch {epoch + 1}/{backbone_epochs}", leave=True)

        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad()

            g1 = Utils.augment_graph(batch, edge_perturb=0.3, node_perturb=0.3)
            g2 = Utils.augment_graph(batch, edge_perturb=0.3, node_perturb=0.3)

            # 2. FORWARD PASS
            h1, fcg_batch1 = model(g1, return_node_embeddings=True)
            h2, _ = model(g2, return_node_embeddings=True)


            l_ins = Utils.memory_intense_nt_xent_loss(
                instance_level_projection_head(h1),
                instance_level_projection_head(h2),
                temperature=model_params.backbone_pretrain_instance_level_projection_temperature
            )

            s1_collected = []
            s2_collected = []


            for i in range(batch.num_graphs):
                mask = (fcg_batch1 == i)
                h1_i = h1[mask]
                h2_i = h2[mask]

                k_i = min(top_k, h1_i.size(0))

                if k_i < 2:
                    continue


                sim_i = torch.matmul(
                    torch.nn.functional.normalize(h1_i, dim=1),
                    torch.nn.functional.normalize(h2_i, dim=1).T
                )
                _, top_k_idx = torch.topk(sim_i, k=k_i, dim=1)
                retrieved = h2_i[top_k_idx]

                omega_i_half, omega_j_half = torch.chunk(retrieved, 2, dim=1)

                # Project the summed halves
                s1_i = set_level_projection_head(torch.sum(omega_i_half, dim=1))
                s2_i = set_level_projection_head(torch.sum(omega_j_half, dim=1))

                s1_collected.append(s1_i)
                s2_collected.append(s2_i)


            if len(s1_collected) > 0:
                s1_batch = torch.cat(s1_collected, dim=0)
                s2_batch = torch.cat(s2_collected, dim=0)
                l_set = Utils.memory_intense_nt_xent_loss(
                    s1_batch, s2_batch,
                    temperature=model_params.backbone_pretrain_set_level_projection_temperature
                )
            else:
                l_set = 0.0

            # 6. BACKPROPAGATION
            loss = l_ins + l_set
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if scheduler:
            scheduler.step()

        avg_loss = train_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]

        print(f'Average training loss: {avg_loss:.2f} | LR: {current_lr}')

        if (epoch + 1) % 5 == 0:
            print("\n[*] Running Validation...")
            val_acc = validate_cross_view_alignment(model, validation_dataset, logger)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), model_saving_path)
                set_head_path = model_saving_path.replace('backbone.pth', 'set_level_head.pth')
                torch.save(set_level_projection_head.state_dict(), set_head_path)
                logger.info(f"--> New best model saved! (Accuracy: {best_val_accuracy:.4f})")

    logger.info(f"Training complete. Best model saved to {model_saving_path} with Val Acc: {best_val_accuracy:.4f}")

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Backbone pre-training')
    parser.add_argument('--config', default='trial_config_v3',
                        help='ExperimentConfig preset name')
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

    saving_path = Utils.get_backbone_model_saved_path(parameters=config)
    
    print(f"[*] Model saving path for this run: {saving_path}")

    train(model=backbone, model_params=config, model_saving_path=saving_path, training_dataset=training_ds,
          validation_dataset=evaluation_ds)