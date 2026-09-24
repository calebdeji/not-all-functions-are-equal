"""
episodic_models.py

Head wrappers for episodic few-shot fine-tuning. Variants:
  - self_attention            : learnable MLP attention (Linear → LeakyReLU → Linear → 1)
  - sdl_degree                : sigmoid(temp * (λa*sim_in + λb*norm_in_degree - sim_out))
  - sdl_complexity            : sigmoid(temp * (λa*sim_in + λb*norm_complexity - sim_out))
  - sdl_pagerank              : sigmoid(temp * (λa*sim_in + λb*norm_pagerank - sim_out))

Graph-pooling baselines (shared pretrained backbone, learned readout):
  - gmt                       : Graph Multiset Transformer (GMPool_G → SelfAtt → GMPool_1)
  - gmpool                    : Graph Multiset Pooling only (GMPool_G → mean over seeds)
  - topk_pool                 : TopKPooling (learned node scoring, gated mean readout)
  - diff_pool                 : DiffPool soft cluster assignment (+ L_LP and L_E aux losses)

Ablation variants (SDL decomposition):
  - sdl_no_learnable_param    : sigmoid(temp * (sim_in - sim_out))           0 params
  - sdl_complexity_prior      : sigmoid(temp * norm_complexity)              0 params
  - sdl_without_structure     : sigmoid(temp * (λa*sim_in - λb*sim_out))    2 params

All models expose:
  backbone       : underlying Backbone
  forward(data)  : end-to-end graph embedding

TrainingUtils provides the weight-computation methods shared by both
episodic_train.py (training) and episodic_evaluate.py (inference).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch_geometric.nn import GATConv, GCNConv, TopKPooling
from torch_geometric.utils import scatter, to_dense_batch, to_dense_adj
from torch_geometric.utils import softmax as _pyg_softmax  # noqa: E402


# ---------------------------------------------------------------------------
# Set-level fusion (STAR-style, Liu et al. KDD 2025)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class BaseEpisodicModel(nn.Module):

    def get_node_embeddings(self, data):
        return self.backbone(data, return_node_embeddings=True)


# ---------------------------------------------------------------------------
# Scatter helpers  (torch_scatter is NOT installed — use PyG equivalents)
# ---------------------------------------------------------------------------

def _scatter_mean(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


def _scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')


def _scatter_softmax(src, index):
    """Softmax of src, grouped by index (one distribution per unique index)."""
    return _pyg_softmax(src, index)


# ---------------------------------------------------------------------------
# TrainingUtils
# ---------------------------------------------------------------------------

def batched_pagerank(edge_index, batch, num_nodes, damping=0.85, num_iters=50):
    """
    Power-iteration PageRank for a batch of disjoint FCG graphs.
    Since FCG edges only connect nodes within the same APK, propagation is
    naturally per-graph. The teleportation term uses per-graph node counts
    via the batch tensor, ensuring correctness across the batched episode.
    Cost: O(num_iters * E) where E = FCG edges in the batch.
    """
    device = edge_index.device
    if num_nodes == 0 or edge_index.size(1) == 0:
        return torch.zeros(num_nodes, device=device)

    num_graphs = int(batch.max().item()) + 1

    out_deg = torch.zeros(num_nodes, device=device)
    out_deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))
    out_deg = out_deg.clamp(min=1e-6)

    n_per_graph = torch.zeros(num_graphs, device=device)
    n_per_graph.scatter_add_(0, batch, torch.ones(num_nodes, device=device))
    teleport = (1.0 - damping) / n_per_graph[batch]

    pr = torch.ones(num_nodes, device=device) / n_per_graph[batch]

    for _ in range(num_iters):
        msg = pr[edge_index[0]] / out_deg[edge_index[0]]
        agg = torch.zeros(num_nodes, device=device)
        agg.scatter_add_(0, edge_index[1], msg)
        pr = damping * agg + teleport

    return pr


class TrainingUtils:
    """
    Static weight-computation methods for all four head variants.

    Support methods  : labels are known  (training batch / evaluation support set)
    Evaluation methods : labels are unknown (evaluation query set, pseudo-labels)
    """

    # ---- Base uniqueness (used by SelfAttention and as fallback) -----------

    @staticmethod
    def generate_task_specific_weights(node_embeddings, node_labels, fcg_batch, temp=5.0):
        """
        Uniqueness weights when labels are known.
        sigmoid(temp * (sim_in - sim_out))
        """
        x = F.normalize(node_embeddings, p=2, dim=1)
        _, labels_inverse = torch.unique(node_labels, return_inverse=True)
        prototypes = _scatter_mean(x, labels_inverse, dim=0)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        sim = torch.matmul(x, prototypes.t())
        n = x.size(0)
        sim_in = sim[torch.arange(n, device=x.device), labels_inverse]
        nf = prototypes.size(0)
        sim_out = (sim.sum(dim=1) - sim_in) / (nf - 1) if nf > 1 else torch.zeros_like(sim_in)

        return torch.sigmoid((sim_in - sim_out) * temp)

    @staticmethod
    def generate_evaluation_weights(query_node_embeddings, support_prototypes, temp=5.0):
        """
        Uniqueness weights when labels are unknown.
        sigmoid(temp * (sim_max - sim_others))
        """
        q = F.normalize(query_node_embeddings, p=2, dim=1)
        p = F.normalize(support_prototypes, p=2, dim=1)

        sim = torch.matmul(q, p.t())
        sim_max, _ = sim.max(dim=1)
        nf = p.size(0)
        sim_others = (sim.sum(dim=1) - sim_max) / (nf - 1) if nf > 1 else torch.zeros_like(sim_max)

        return torch.sigmoid((sim_max - sim_others) * temp)

    # ---- SDL-Degree --------------------------------------------------------

    @staticmethod
    def generate_sdl_degree_weights(node_embeddings, node_labels, edge_index,
                                    lambda_a_val=0.0, lambda_b_val=0.0, temp=5.0):
        """
        SDL-Degree support weights (labels known).
        Formula: sigmoid(temp * (λa*sim_in + λb*norm_in_degree - sim_out))
        """
        num_nodes = node_embeddings.size(0)
        device = node_embeddings.device

        deg = torch.zeros(num_nodes, device=device)
        deg = deg.scatter_add(0, edge_index[1],
                              torch.ones(edge_index.size(1), device=device))
        if deg.max() > deg.min():
            deg = (deg - deg.min()) / (deg.max() - deg.min() + 1e-6)

        x = F.normalize(node_embeddings, p=2, dim=1)
        _, labels_inverse = torch.unique(node_labels, return_inverse=True)
        proto = _scatter_mean(x, labels_inverse, dim=0)
        proto = F.normalize(proto, p=2, dim=1)
        sim = torch.matmul(x, proto.t())
        sim_in = sim[torch.arange(num_nodes, device=device), labels_inverse]
        nf = proto.size(0)
        sim_out = (sim.sum(dim=1) - sim_in) / (nf - 1) if nf > 1 else torch.zeros_like(sim_in)

        raw = (lambda_a_val * sim_in) + (lambda_b_val * deg) - sim_out
        return torch.sigmoid(raw * temp)

    @staticmethod
    def generate_sdl_degree_evaluation_weights(query_node_embeddings, support_prototypes,
                                               edge_index, lambda_a_val=0.0,
                                               lambda_b_val=0.0, temp=5.0):
        """SDL-Degree query weights (labels unknown)."""
        num_nodes = query_node_embeddings.size(0)
        device = query_node_embeddings.device

        deg = torch.zeros(num_nodes, device=device)
        deg = deg.scatter_add(0, edge_index[1],
                              torch.ones(edge_index.size(1), device=device))
        if deg.max() > deg.min():
            deg = (deg - deg.min()) / (deg.max() - deg.min() + 1e-6)

        q = F.normalize(query_node_embeddings, p=2, dim=1)
        p = F.normalize(support_prototypes, p=2, dim=1)
        sim = torch.matmul(q, p.t())
        sim_max, _ = sim.max(dim=1)
        nf = p.size(0)
        sim_others = (sim.sum(dim=1) - sim_max) / (nf - 1) if nf > 1 else torch.zeros_like(sim_max)

        raw = (lambda_a_val * sim_max) + (lambda_b_val * deg) - sim_others
        return torch.sigmoid(raw * temp)

    # ---- SDL-Dim-Degree (per-dimension gates) ------------------------------

    @staticmethod
    def generate_sdl_dim_degree_weights(node_embeddings, node_labels, edge_index,
                                        lambda_a, lambda_b, temp=5.0):
        """
        SDL-Degree with per-dimension learnable gates (support set, labels known).

        lambda_a (D,): gates which embedding dims drive in-class similarity.
          Applied to node embeddings AND prototypes before cosine similarity.
        lambda_b (D,): gates which embedding dims encode degree information.
          (node_embs * gate_b).mean(dim=1) * norm_degree → structural score.
        """
        num_nodes = node_embeddings.size(0)
        device = node_embeddings.device

        deg = torch.zeros(num_nodes, device=device)
        deg = deg.scatter_add(0, edge_index[1],
                              torch.ones(edge_index.size(1), device=device))
        if deg.max() > deg.min():
            deg = (deg - deg.min()) / (deg.max() - deg.min() + 1e-6)

        gate_a = torch.sigmoid(lambda_a)  # (D,)
        gate_b = torch.sigmoid(lambda_b)  # (D,)

        embs_a = F.normalize(node_embeddings * gate_a, p=2, dim=1)
        _, labels_inverse = torch.unique(node_labels, return_inverse=True)
        proto_a = _scatter_mean(embs_a, labels_inverse, dim=0)
        proto_a = F.normalize(proto_a, p=2, dim=1)
        sim = torch.matmul(embs_a, proto_a.t())
        sim_in = sim[torch.arange(num_nodes, device=device), labels_inverse]
        nf = proto_a.size(0)
        sim_out = (sim.sum(dim=1) - sim_in) / (nf - 1) if nf > 1 else torch.zeros_like(sim_in)

        structural = (node_embeddings * gate_b).mean(dim=1) * deg
        raw = sim_in + structural - sim_out
        return torch.sigmoid(raw * temp)

    @staticmethod
    def generate_sdl_dim_degree_evaluation_weights(query_node_embeddings, support_prototypes,
                                                    edge_index, lambda_a, lambda_b, temp=5.0):
        """SDL-Dim-Degree query weights (labels unknown)."""
        num_nodes = query_node_embeddings.size(0)
        device = query_node_embeddings.device

        deg = torch.zeros(num_nodes, device=device)
        deg = deg.scatter_add(0, edge_index[1],
                              torch.ones(edge_index.size(1), device=device))
        if deg.max() > deg.min():
            deg = (deg - deg.min()) / (deg.max() - deg.min() + 1e-6)

        gate_a = torch.sigmoid(lambda_a)
        gate_b = torch.sigmoid(lambda_b)

        q = F.normalize(query_node_embeddings * gate_a, p=2, dim=1)
        p = F.normalize(support_prototypes * gate_a, p=2, dim=1)
        sim = torch.matmul(q, p.t())
        sim_max, _ = sim.max(dim=1)
        nf = p.size(0)
        sim_others = (sim.sum(dim=1) - sim_max) / (nf - 1) if nf > 1 else torch.zeros_like(sim_max)

        structural = (query_node_embeddings * gate_b).mean(dim=1) * deg
        raw = sim_max + structural - sim_others
        return torch.sigmoid(raw * temp)

    # ---- SDL-Dim-Complexity (per-dimension gates) --------------------------

    @staticmethod
    def generate_sdl_dim_complexity_weights(node_embeddings, node_labels, raw_complexity,
                                             lambda_a, lambda_b, temp=5.0):
        """
        SDL-Complexity with per-dimension learnable gates (support set, labels known).
        """
        c = raw_complexity.view(-1).float().to(node_embeddings.device)
        c = (c - c.min()) / (c.max() - c.min() + 1e-6) if c.max() > c.min() else torch.zeros_like(c)

        num_nodes = node_embeddings.size(0)
        device = node_embeddings.device

        gate_a = torch.sigmoid(lambda_a)
        gate_b = torch.sigmoid(lambda_b)

        embs_a = F.normalize(node_embeddings * gate_a, p=2, dim=1)
        _, labels_inverse = torch.unique(node_labels, return_inverse=True)
        proto_a = _scatter_mean(embs_a, labels_inverse, dim=0)
        proto_a = F.normalize(proto_a, p=2, dim=1)
        sim = torch.matmul(embs_a, proto_a.t())
        sim_in = sim[torch.arange(num_nodes, device=device), labels_inverse]
        nf = proto_a.size(0)
        sim_out = (sim.sum(dim=1) - sim_in) / (nf - 1) if nf > 1 else torch.zeros_like(sim_in)

        structural = (node_embeddings * gate_b).mean(dim=1) * c
        raw = sim_in + structural - sim_out
        return torch.sigmoid(raw * temp)

    @staticmethod
    def generate_sdl_dim_complexity_evaluation_weights(query_node_embeddings, support_prototypes,
                                                        raw_complexity, lambda_a, lambda_b, temp=5.0):
        """SDL-Dim-Complexity query weights (labels unknown)."""
        c = raw_complexity.view(-1).float().to(query_node_embeddings.device)
        c = (c - c.min()) / (c.max() - c.min() + 1e-6) if c.max() > c.min() else torch.zeros_like(c)

        gate_a = torch.sigmoid(lambda_a)
        gate_b = torch.sigmoid(lambda_b)

        q = F.normalize(query_node_embeddings * gate_a, p=2, dim=1)
        p = F.normalize(support_prototypes * gate_a, p=2, dim=1)
        sim = torch.matmul(q, p.t())
        sim_max, _ = sim.max(dim=1)
        nf = p.size(0)
        sim_others = (sim.sum(dim=1) - sim_max) / (nf - 1) if nf > 1 else torch.zeros_like(sim_max)

        structural = (query_node_embeddings * gate_b).mean(dim=1) * c
        raw = sim_max + structural - sim_others
        return torch.sigmoid(raw * temp)

    # ---- SDL-PageRank ------------------------------------------------------

    @staticmethod
    def generate_sdl_pagerank_weights(node_embeddings, node_labels, edge_index, fcg_batch,
                                      lambda_a_val=0.0, lambda_b_val=0.0, temp=5.0):
        """
        SDL-PageRank support weights (labels known).
        Formula: sigmoid(temp * (λa*sim_in + λb*norm_pagerank - sim_out))
        """
        num_nodes = node_embeddings.size(0)
        device = node_embeddings.device

        pr = batched_pagerank(edge_index, fcg_batch, num_nodes)
        if pr.max() > pr.min():
            pr = (pr - pr.min()) / (pr.max() - pr.min() + 1e-6)

        x = F.normalize(node_embeddings, p=2, dim=1)
        _, labels_inverse = torch.unique(node_labels, return_inverse=True)
        proto = _scatter_mean(x, labels_inverse, dim=0)
        proto = F.normalize(proto, p=2, dim=1)
        sim = torch.matmul(x, proto.t())
        sim_in = sim[torch.arange(num_nodes, device=device), labels_inverse]
        nf = proto.size(0)
        sim_out = (sim.sum(dim=1) - sim_in) / (nf - 1) if nf > 1 else torch.zeros_like(sim_in)

        raw = (lambda_a_val * sim_in) + (lambda_b_val * pr) - sim_out
        return torch.sigmoid(raw * temp)

    @staticmethod
    def generate_sdl_pagerank_evaluation_weights(query_node_embeddings, support_prototypes,
                                                  edge_index, fcg_batch,
                                                  lambda_a_val=0.0, lambda_b_val=0.0, temp=5.0):
        """SDL-PageRank query weights (labels unknown)."""
        num_nodes = query_node_embeddings.size(0)
        device = query_node_embeddings.device

        pr = batched_pagerank(edge_index, fcg_batch, num_nodes)
        if pr.max() > pr.min():
            pr = (pr - pr.min()) / (pr.max() - pr.min() + 1e-6)

        q = F.normalize(query_node_embeddings, p=2, dim=1)
        p = F.normalize(support_prototypes, p=2, dim=1)
        sim = torch.matmul(q, p.t())
        sim_max, _ = sim.max(dim=1)
        nf = p.size(0)
        sim_others = (sim.sum(dim=1) - sim_max) / (nf - 1) if nf > 1 else torch.zeros_like(sim_max)

        raw = (lambda_a_val * sim_max) + (lambda_b_val * pr) - sim_others
        return torch.sigmoid(raw * temp)

    # ---- SDL-Complexity ----------------------------------------------------

    @staticmethod
    def generate_sdl_complexity_weights(node_embeddings, node_labels, raw_complexity,
                                        lambda_a_val=0.0, lambda_b_val=0.0, temp=5.0):
        """
        SDL-Complexity support weights (labels known).
        Formula: sigmoid(temp * (λa*sim_in + λb*norm_complexity - sim_out))
        raw_complexity should be data.function_features[:, 1] (cyclomatic complexity).
        """
        c = raw_complexity.view(-1).float().to(node_embeddings.device)
        c = (c - c.min()) / (c.max() - c.min() + 1e-6) if c.max() > c.min() else torch.zeros_like(c)

        num_nodes = node_embeddings.size(0)
        x = F.normalize(node_embeddings, p=2, dim=1)
        _, labels_inverse = torch.unique(node_labels, return_inverse=True)
        proto = _scatter_mean(x, labels_inverse, dim=0)
        proto = F.normalize(proto, p=2, dim=1)
        sim = torch.matmul(x, proto.t())
        sim_in = sim[torch.arange(num_nodes, device=x.device), labels_inverse]
        nf = proto.size(0)
        sim_out = (sim.sum(dim=1) - sim_in) / (nf - 1) if nf > 1 else torch.zeros_like(sim_in)

        raw = (lambda_a_val * sim_in) + (lambda_b_val * c) - sim_out
        return torch.sigmoid(raw * temp)

    @staticmethod
    def generate_sdl_complexity_evaluation_weights(query_node_embeddings, support_prototypes,
                                                   raw_complexity, lambda_a_val=0.0,
                                                   lambda_b_val=0.0, temp=5.0):
        """SDL-Complexity query weights (labels unknown)."""
        c = raw_complexity.view(-1).float().to(query_node_embeddings.device)
        c = (c - c.min()) / (c.max() - c.min() + 1e-6) if c.max() > c.min() else torch.zeros_like(c)

        q = F.normalize(query_node_embeddings, p=2, dim=1)
        p = F.normalize(support_prototypes, p=2, dim=1)
        sim = torch.matmul(q, p.t())
        sim_max, _ = sim.max(dim=1)
        nf = p.size(0)
        sim_others = (sim.sum(dim=1) - sim_max) / (nf - 1) if nf > 1 else torch.zeros_like(sim_max)

        raw = (lambda_a_val * sim_max) + (lambda_b_val * c) - sim_others
        return torch.sigmoid(raw * temp)

    # ---- Ablation: complexity prior only (no semantic signal) ---------------

    @staticmethod
    def generate_sdl_complexity_prior_weights(raw_complexity, temp=5.0):
        """
        Pure structural prior: sigmoid(temp * norm_complexity).
        Used identically for support and query — no labels or prototypes needed.
        """
        c = raw_complexity.view(-1).float()
        c = (c - c.min()) / (c.max() - c.min() + 1e-6) if c.max() > c.min() else torch.zeros_like(c)
        return torch.sigmoid(c * temp)

    # ---- Ablation: learnable intra-class scale only (λa·sim_in − sim_out) --

    @staticmethod
    def generate_sdl_intra_param_weights(node_embeddings, node_labels,
                                         lambda_a_val, temp=5.0):
        """
        SDL with only intra-class scale learned (support set, labels known).
        Formula: sigmoid(temp * (λa*sim_in - sim_out))
        """
        x = F.normalize(node_embeddings, p=2, dim=1)
        _, labels_inverse = torch.unique(node_labels, return_inverse=True)
        proto = _scatter_mean(x, labels_inverse, dim=0)
        proto = F.normalize(proto, p=2, dim=1)
        sim = torch.matmul(x, proto.t())
        n = x.size(0)
        sim_in = sim[torch.arange(n, device=x.device), labels_inverse]
        nf = proto.size(0)
        sim_out = (sim.sum(dim=1) - sim_in) / (nf - 1) if nf > 1 else torch.zeros_like(sim_in)
        return torch.sigmoid(((lambda_a_val * sim_in) - sim_out) * temp)

    @staticmethod
    def generate_sdl_intra_param_evaluation_weights(query_node_embeddings, support_prototypes,
                                                     lambda_a_val, temp=5.0):
        """SDL-IntraParam query weights (labels unknown)."""
        q = F.normalize(query_node_embeddings, p=2, dim=1)
        p = F.normalize(support_prototypes, p=2, dim=1)
        sim = torch.matmul(q, p.t())
        sim_max, _ = sim.max(dim=1)
        nf = p.size(0)
        sim_others = (sim.sum(dim=1) - sim_max) / (nf - 1) if nf > 1 else torch.zeros_like(sim_max)
        return torch.sigmoid(((lambda_a_val * sim_max) - sim_others) * temp)

    # ---- Ablation: learnable scales on both semantic terms, no structural ---

    @staticmethod
    def generate_sdl_without_structure_weights(node_embeddings, node_labels,
                                               lambda_a_val, lambda_b_val, temp=5.0):
        """
        SDL without structural prior — learnable scale on sim_out (support, labels known).
        Formula: sigmoid(temp * (λa*sim_in - λb*sim_out))
        Note: sim_out coefficient is now learned (unlike full SDL where it is fixed at 1).
        """
        x = F.normalize(node_embeddings, p=2, dim=1)
        _, labels_inverse = torch.unique(node_labels, return_inverse=True)
        proto = _scatter_mean(x, labels_inverse, dim=0)
        proto = F.normalize(proto, p=2, dim=1)
        sim = torch.matmul(x, proto.t())
        n = x.size(0)
        sim_in = sim[torch.arange(n, device=x.device), labels_inverse]
        nf = proto.size(0)
        sim_out = (sim.sum(dim=1) - sim_in) / (nf - 1) if nf > 1 else torch.zeros_like(sim_in)
        return torch.sigmoid(((lambda_a_val * sim_in) - (lambda_b_val * sim_out)) * temp)

    @staticmethod
    def generate_sdl_without_structure_evaluation_weights(query_node_embeddings, support_prototypes,
                                                          lambda_a_val, lambda_b_val, temp=5.0):
        """SDL-WithoutStructure query weights (labels unknown)."""
        q = F.normalize(query_node_embeddings, p=2, dim=1)
        p = F.normalize(support_prototypes, p=2, dim=1)
        sim = torch.matmul(q, p.t())
        sim_max, _ = sim.max(dim=1)
        nf = p.size(0)
        sim_others = (sim.sum(dim=1) - sim_max) / (nf - 1) if nf > 1 else torch.zeros_like(sim_max)
        return torch.sigmoid(((lambda_a_val * sim_max) - (lambda_b_val * sim_others)) * temp)


# ---------------------------------------------------------------------------
# Pooling module
# ---------------------------------------------------------------------------

class SelfAttentivePooling(nn.Module):
    """MLP attention: Linear → LeakyReLU → Linear(→1) → scatter_softmax → scatter_add."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, x, batch):
        logits = self.attn_mlp(x).squeeze(-1)
        weights = _scatter_softmax(logits, batch)
        return _scatter_add(x * weights.unsqueeze(-1), batch, dim=0)


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


class SelfAttentionModel(BaseEpisodicModel):
    """Learnable self-attentive pooling over function-level node embeddings."""

    def __init__(self, backbone, fcg_dim: int):
        super().__init__()
        self.backbone = backbone
        self.attentive_pooling = SelfAttentivePooling(fcg_dim)

    def forward(self, data):
        node_embs, fcg_batch = self.get_node_embeddings(data)
        x = self.attentive_pooling(node_embs, fcg_batch)
        return F.normalize(x, p=2, dim=-1)

    def forward_attention(self, node_embeddings, batch, weights=None):
        """Uses learned MLP attention — external weights are ignored."""
        x = self.attentive_pooling(node_embeddings, batch)
        return F.normalize(x, p=2, dim=-1)


class SDLDegreeModel(BaseEpisodicModel):
    """
    SDL-Degree: sigmoid(temp * (λa*sim_in + λb*norm_in_degree - sim_out))
    λa and λb are learnable scalars initialised to 0.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.raw_lambda_a = nn.Parameter(torch.tensor(0.0))
        self.raw_lambda_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        node_embs, fcg_batch = self.get_node_embeddings(data)
        node_labels = torch.repeat_interleave(data.family_class, data.num_functions)
        weights = TrainingUtils.generate_sdl_degree_weights(
            node_embeddings=node_embs,
            node_labels=node_labels,
            edge_index=data.fcg_edge_index,
            lambda_a_val=self.raw_lambda_a,
            lambda_b_val=self.raw_lambda_b,
        )
        return self.forward_attention(node_embs, fcg_batch, weights)

    @staticmethod
    def forward_attention(node_embeddings, batch, weights):
        norm_w = _scatter_softmax(weights, batch)
        x = _scatter_add(node_embeddings * norm_w.unsqueeze(-1), batch, dim=0)
        return F.normalize(x, p=2, dim=-1)


class SDLComplexityModel(BaseEpisodicModel):
    """
    SDL-Complexity: sigmoid(temp * (λa*sim_in + λb*norm_complexity - sim_out))
    Complexity = data.function_features[:, 1] (cyclomatic complexity, column index 1).
    λa and λb are learnable scalars initialised to 0.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.raw_lambda_a = nn.Parameter(torch.tensor(0.0))
        self.raw_lambda_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        node_embs, fcg_batch = self.get_node_embeddings(data)
        node_labels = torch.repeat_interleave(data.family_class, data.num_functions)
        raw_complexity = data.function_features[:, 1].to(node_embs.device)
        weights = TrainingUtils.generate_sdl_complexity_weights(
            node_embeddings=node_embs,
            node_labels=node_labels,
            raw_complexity=raw_complexity,
            lambda_a_val=self.raw_lambda_a,
            lambda_b_val=self.raw_lambda_b,
        )
        return self.forward_attention(node_embs, fcg_batch, weights)

    @staticmethod
    def forward_attention(node_embeddings, batch, weights):
        norm_w = _scatter_softmax(weights, batch)
        x = _scatter_add(node_embeddings * norm_w.unsqueeze(-1), batch, dim=0)
        return F.normalize(x, p=2, dim=-1)



class SDLPageRankModel(BaseEpisodicModel):
    """
    SDL-PageRank: sigmoid(temp * (λa*sim_in + λb*norm_pagerank - sim_out))
    PageRank is computed via power iteration on the FCG per episode.
    λa and λb are learnable scalars initialised to 0.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.raw_lambda_a = nn.Parameter(torch.tensor(0.0))
        self.raw_lambda_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        node_embs, fcg_batch = self.get_node_embeddings(data)
        node_labels = torch.repeat_interleave(data.family_class, data.num_functions)
        weights = TrainingUtils.generate_sdl_pagerank_weights(
            node_embeddings=node_embs,
            node_labels=node_labels,
            edge_index=data.fcg_edge_index,
            fcg_batch=fcg_batch,
            lambda_a_val=self.raw_lambda_a,
            lambda_b_val=self.raw_lambda_b,
        )
        return self.forward_attention(node_embs, fcg_batch, weights)

    @staticmethod
    def forward_attention(node_embeddings, batch, weights):
        norm_w = _scatter_softmax(weights, batch)
        x = _scatter_add(node_embeddings * norm_w.unsqueeze(-1), batch, dim=0)
        return F.normalize(x, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Ablation model wrappers
# ---------------------------------------------------------------------------

class SDLWithoutLearnableParamModel(BaseEpisodicModel):
    """
    Ablation: fixed uniqueness weighting, no learnable parameters.
    sigmoid(temp * (sim_in - sim_out))  — 0 params.
    Isolates whether learning λ values adds anything over the fixed formula.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, data):
        node_embs, fcg_batch = self.get_node_embeddings(data)
        node_labels = torch.repeat_interleave(data.family_class, data.num_functions)
        weights = TrainingUtils.generate_task_specific_weights(
            node_embs, node_labels, fcg_batch, temp=5.0,
        )
        return self.forward_attention(node_embs, fcg_batch, weights)

    @staticmethod
    def forward_attention(node_embeddings, batch, weights):
        norm_w = _scatter_softmax(weights, batch)
        x = _scatter_add(node_embeddings * norm_w.unsqueeze(-1), batch, dim=0)
        return F.normalize(x, p=2, dim=-1)


class SDLComplexityPriorModel(BaseEpisodicModel):
    """
    Ablation: pure structural prior, no semantic signal, no learnable parameters.
    sigmoid(temp * norm_complexity)  — 0 params.
    Isolates whether cyclomatic complexity alone has intrinsic routing value.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, data):
        node_embs, fcg_batch = self.get_node_embeddings(data)
        raw_complexity = data.function_features[:, 1].to(node_embs.device)
        weights = TrainingUtils.generate_sdl_complexity_prior_weights(raw_complexity)
        return self.forward_attention(node_embs, fcg_batch, weights)

    @staticmethod
    def forward_attention(node_embeddings, batch, weights):
        norm_w = _scatter_softmax(weights, batch)
        x = _scatter_add(node_embeddings * norm_w.unsqueeze(-1), batch, dim=0)
        return F.normalize(x, p=2, dim=-1)




class SDLWithoutStructureModel(BaseEpisodicModel):
    """
    Ablation: learnable scales on both semantic terms, structural prior removed.
    sigmoid(temp * (λa*sim_in - λb*sim_out))  — 2 params.
    Note: sim_out coefficient is now learned (unlike full SDL where it is fixed at 1).
    Tests whether the structural prior contributes beyond learned sim_out scaling.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.raw_lambda_a = nn.Parameter(torch.tensor(0.0))
        self.raw_lambda_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        node_embs, fcg_batch = self.get_node_embeddings(data)
        node_labels = torch.repeat_interleave(data.family_class, data.num_functions)
        weights = TrainingUtils.generate_sdl_without_structure_weights(
            node_embeddings=node_embs,
            node_labels=node_labels,
            lambda_a_val=self.raw_lambda_a,
            lambda_b_val=self.raw_lambda_b,
        )
        return self.forward_attention(node_embs, fcg_batch, weights)

    @staticmethod
    def forward_attention(node_embeddings, batch, weights):
        norm_w = _scatter_softmax(weights, batch)
        x = _scatter_add(node_embeddings * norm_w.unsqueeze(-1), batch, dim=0)
        return F.normalize(x, p=2, dim=-1)


# ---------------------------------------------------------------------------
# GMT (Graph Multiset Transformer) — Baek et al., ICLR 2021
# Faithful port of layers.py / nets.py from https://github.com/JinheonBaek/GMT
# ---------------------------------------------------------------------------

class _GMT_MAB(nn.Module):
    """
    Multi-head Attention Block (MAB).

    When graph=(x, edge_index, batch) is passed, keys and values are computed
    by running GCNConv on the graph-structured nodes before converting to dense
    format with to_dense_batch.  When graph=None, K and V are plain linear
    projections of the dense input.
    """

    def __init__(self, dim_Q: int, dim_K: int, dim_V: int,
                 num_heads: int, ln: bool = False, conv: Optional[str] = None):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)

        if conv == 'GCN':
            self.fc_k = GCNConv(dim_K, dim_V)
            self.fc_v = GCNConv(dim_K, dim_V)
            self._graph_conv = True
        else:
            self.fc_k = nn.Linear(dim_K, dim_V)
            self.fc_v = nn.Linear(dim_K, dim_V)
            self._graph_conv = False

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q: torch.Tensor, K: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                graph: Optional[tuple] = None) -> torch.Tensor:
        Q = self.fc_q(Q)  # (B, q, dim_V)

        if graph is not None:
            x, edge_index, batch = graph
            K_raw = self.fc_k(x, edge_index)        # (total_nodes, dim_V)
            V_raw = self.fc_v(x, edge_index)
            K, _ = to_dense_batch(K_raw, batch)     # (B, max_n, dim_V)
            V, _ = to_dense_batch(V_raw, batch)
        else:
            K = self.fc_k(K)                        # (B, n, dim_V)
            V = self.fc_v(K)

        # Multi-head split: (B, seq, dim_V) → (B*num_heads, seq, head_dim)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask] * self.num_heads, 0)
            A = torch.softmax(
                attention_mask + Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)

        # Reconstruct full-dim output
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if not hasattr(self, 'ln0') else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if not hasattr(self, 'ln1') else self.ln1(O)
        return O


class _GMT_SAB(nn.Module):
    """Self-Attention Block (SelfAtt step in full GMT)."""

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False):
        super().__init__()
        self.mab = _GMT_MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, conv=None)

    def forward(self, X: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mab(X, X, attention_mask)


class _GMT_PMA(nn.Module):
    """
    Pooling by Multi-head Attention (PMA = GMPool).
    Learned seed vectors S act as queries; node embeddings (or k reps) are K/V.
    """

    def __init__(self, dim: int, num_heads: int, num_seeds: int,
                 ln: bool = False, conv: Optional[str] = None):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = _GMT_MAB(dim, dim, dim, num_heads, ln=ln, conv=conv)

    def forward(self, X: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                graph: Optional[tuple] = None) -> torch.Tensor:
        S = self.S.repeat(X.size(0), 1, 1)          # (B, num_seeds, dim)
        return self.mab(S, X, attention_mask, graph)


class GMTModel(BaseEpisodicModel):
    """
    Graph Multiset Transformer (Baek et al., ICLR 2021) episodic head.

    Full pipeline: GMPool_G → SelfAtt → GMPool_1 → L2-normalised graph embedding.

    GMPool_G: seed vectors attend to GCN-processed node embeddings (graph-aware).
    SelfAtt:  the k representatives exchange information via self-attention.
    GMPool_1: the k reps are compressed to a single graph vector.
    """

    def __init__(self, backbone, fcg_dim: int, num_seeds: int = 16,
                 num_heads: int = 4, ln: bool = True):
        super().__init__()
        self.backbone  = backbone
        self.gmpool_g  = _GMT_PMA(fcg_dim, num_heads, num_seeds, ln=ln, conv='GCN')
        self.self_att  = _GMT_SAB(fcg_dim, fcg_dim, num_heads, ln=ln)
        self.gmpool_1  = _GMT_PMA(fcg_dim, num_heads, 1,         ln=ln, conv=None)
        self.linear    = nn.Linear(fcg_dim, fcg_dim)

    def forward(self, data) -> torch.Tensor:
        node_embs, fcg_batch = self.get_node_embeddings(data)
        return self.forward_attention(node_embs, fcg_batch, data.fcg_edge_index)

    def forward_attention(self, node_embs: torch.Tensor, batch: torch.Tensor,
                          edge_index: torch.Tensor, weights=None) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        batch_x, mask = to_dense_batch(node_embs, batch)
        attn_mask = ((1.0 - mask.float().unsqueeze(1)) * -1e9).to(node_embs.dtype)

        reps = self.gmpool_g(batch_x, attention_mask=attn_mask,
                             graph=(node_embs, edge_index, batch))
        reps = self.self_att(reps)
        graph_emb = self.gmpool_1(reps).squeeze(1)
        return F.normalize(self.linear(graph_emb), p=2, dim=-1)


class GMPoolModel(BaseEpisodicModel):
    """
    GMPool-only episodic head (simplified GMT, no SelfAtt between pooling stages).

    Pipeline: GMPool_G (k reps) → mean over k → L2-normalised graph embedding.
    Graph-aware pooling via GCN-computed keys/values in the attention block.
    """

    def __init__(self, backbone, fcg_dim: int, num_seeds: int = 16,
                 num_heads: int = 4, ln: bool = True):
        super().__init__()
        self.backbone = backbone
        self.gmpool_g = _GMT_PMA(fcg_dim, num_heads, num_seeds, ln=ln, conv='GCN')
        self.linear   = nn.Linear(fcg_dim, fcg_dim)

    def forward(self, data) -> torch.Tensor:
        node_embs, fcg_batch = self.get_node_embeddings(data)
        return self.forward_attention(node_embs, fcg_batch, data.fcg_edge_index)

    def forward_attention(self, node_embs: torch.Tensor, batch: torch.Tensor,
                          edge_index: torch.Tensor, weights=None) -> torch.Tensor:
        batch_x, mask = to_dense_batch(node_embs, batch)
        attn_mask = ((1.0 - mask.float().unsqueeze(1)) * -1e9).to(node_embs.dtype)

        reps = self.gmpool_g(batch_x, attention_mask=attn_mask,
                             graph=(node_embs, edge_index, batch))
        graph_emb = reps.mean(dim=1)
        return F.normalize(self.linear(graph_emb), p=2, dim=-1)


# ---------------------------------------------------------------------------
# TopK Pooling — Gao & Ji, ICML 2019
# ---------------------------------------------------------------------------

class TopKPoolModel(BaseEpisodicModel):
    """
    TopK graph pooling episodic head (Gao & Ji, ICML 2019).

    A learned projection vector scores each function node; the top-k scoring
    nodes are selected and gated (x * tanh(score)), then mean-pooled into a
    graph embedding.  Uses PyG's built-in TopKPooling layer.
    """

    def __init__(self, backbone, fcg_dim: int, k: int = 16, dropout_p: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.topk    = TopKPooling(fcg_dim, ratio=k)
        self.dropout = nn.Dropout(dropout_p)
        self.linear  = nn.Linear(fcg_dim, fcg_dim)

    def forward(self, data) -> torch.Tensor:
        node_embs, fcg_batch = self.get_node_embeddings(data)
        return self.forward_attention(node_embs, fcg_batch, data.fcg_edge_index)

    def forward_attention(self, node_embs: torch.Tensor, batch: torch.Tensor,
                          edge_index: torch.Tensor, weights=None) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        x, _, _, batch_out, _, _ = self.topk(node_embs, edge_index, batch=batch)
        graph_emb = _scatter_mean(x, batch_out, dim=0, dim_size=num_graphs)
        return F.normalize(self.linear(self.dropout(graph_emb)), p=2, dim=-1)


class DiffPoolModel(BaseEpisodicModel):
    """
    DiffPool (Ying et al., NeurIPS 2019) as an episodic readout head.

    The backbone already provides node embeddings (Z), so no embed_gnn is used.
    A single GCNConv (pool_gnn) generates the soft cluster assignment S from Z.

    Pooling:
      S = softmax(pool_gnn(Z, A))               (N, num_clusters), row-wise
      X_coarse = S^T Z                           (num_clusters, fcg_dim) per graph
      graph_emb = mean_pool(X_coarse)

    Auxiliary losses are computed separately via compute_aux_loss() and added
    to the episodic training loss in episodic_step. They are not computed here
    so that forward_attention stays consistent with all other model types.
    """

    def __init__(self, backbone, fcg_dim: int, num_clusters: int = 10, secondary_device=None):
        super().__init__()
        self.backbone          = backbone
        self.num_clusters      = num_clusters
        self.pool_gnn          = GCNConv(fcg_dim, num_clusters)
        self.linear            = nn.Linear(fcg_dim, fcg_dim)
        self.secondary_device  = secondary_device

    def forward(self, data) -> torch.Tensor:
        node_embs, fcg_batch = self.get_node_embeddings(data)
        return self.forward_attention(node_embs, fcg_batch, data.fcg_edge_index)

    def forward_attention(self, node_embs: torch.Tensor, batch: torch.Tensor,
                          edge_index: torch.Tensor, weights=None) -> torch.Tensor:
        S = F.softmax(self.pool_gnn(node_embs, edge_index), dim=-1)  # (N, num_clusters)
        Z_dense, _ = to_dense_batch(node_embs, batch)                # (B, max_n, fcg_dim)
        S_dense, _ = to_dense_batch(S, batch)                        # (B, max_n, num_clusters)
        X_coarse   = S_dense.transpose(1, 2) @ Z_dense               # (B, num_clusters, fcg_dim)
        graph_emb  = X_coarse.mean(dim=1)                            # (B, fcg_dim)
        return F.normalize(self.linear(graph_emb), p=2, dim=-1)

    def compute_aux_loss(self, node_embs: torch.Tensor, batch: torch.Tensor,
                         edge_index: torch.Tensor) -> torch.Tensor:
        """
        Computes L_LP + L_E from the assignment matrix S.
        Called explicitly in episodic_step after pool_nodes.

        L_LP = ||A - S S^T||_F   (link prediction — nearby nodes pool together)
        L_E  = (1/n) Σ H(S_i)   (entropy — push assignments toward one-hot)
        """
        primary = node_embs.device
        S = F.softmax(self.pool_gnn(node_embs, edge_index), dim=-1)  # (N, num_clusters)
        S_dense, mask = to_dense_batch(S, batch)                      # (B, max_n, num_clusters)

        # L_LP — dense [B, N_max, N_max] ops offloaded to secondary_device if set
        dev = self.secondary_device if self.secondary_device is not None else primary
        A_dense = to_dense_adj(edge_index, batch, max_num_nodes=S_dense.size(1)).to(dev)
        SS_t    = (S_dense @ S_dense.transpose(1, 2)).to(dev)
        mask2d  = (mask.unsqueeze(2) & mask.unsqueeze(1)).to(dev)
        L_lp    = ((A_dense - SS_t) * mask2d.float()).norm(dim=(1, 2)).mean().to(primary)

        # L_E (minimizing entropy encourages near-one-hot assignments)
        eps = 1e-8
        L_e = -(S * (S + eps).log()).sum(dim=-1).mean()

        return L_lp + L_e
