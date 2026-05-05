import torch
import torch.nn.functional as F
import logging
from torch_geometric.loader import DataLoader

from utils import Utils



def validate_cross_view_alignment(model: torch.nn.Module, validation_dataset, logger: logging.Logger):
    device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(validation_dataset, batch_size=16, shuffle=True)

    total_nodes = 0
    correct_matches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # 1. Create two heavily corrupted views
            g1 = Utils.augment_graph(batch, edge_perturb=0.3, node_perturb=0.3)
            g2 = Utils.augment_graph(batch, edge_perturb=0.3, node_perturb=0.3)

            # 2. Get embeddings
            h1, _ = model(g1, return_node_embeddings=True)
            h2, _ = model(g2, return_node_embeddings=True)

            # 3. Normalize for Cosine Similarity
            h1 = F.normalize(h1, p=2, dim=1)
            h2 = F.normalize(h2, p=2, dim=1)

            # 4. Calculate similarity between ALL nodes in View 1 and ALL nodes in View 2
            # Shape: (N, N) where N is total nodes in the batch
            sim_matrix = torch.matmul(h1, h2.T)

            # 5. For every node in h1, find the index of the most similar node in h2
            predictions = torch.argmax(sim_matrix, dim=1)

            
            ground_truth = torch.arange(h1.size(0), device=device)

            correct_matches += (predictions == ground_truth).sum().item()
            total_nodes += h1.size(0)

            # Stop early after a few batches to keep validation fast
            if total_nodes > 10000:
                break

    accuracy = correct_matches / total_nodes
    logger.info(f"Self-Supervised Validation (Cross-View Alignment) | Accuracy: {accuracy:.4f}")

    model.train()
    return accuracy