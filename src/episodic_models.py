"""
episodic_models.py

Head wrappers for episodic few-shot fine-tuning. Variants:
  - mean_pooling              : uniform scatter_mean baseline
  - self_attention            : learnable MLP attention (Linear → LeakyReLU → Linear → 1)
  - sdl_degree                : sigmoid(temp * (λa*sim_in + λb*norm_in_degree - sim_out))
  - sdl_complexity            : sigmoid(temp * (λa*sim_in + λb*norm_complexity - sim_out))
  - fcgat                     : GATConv + Set2SetWithAttention (FCGAT-style readout)

Ablation variants (SDL decomposition):
  - sdl_no_learnable_param    : sigmoid(temp * (sim_in - sim_out))           0 params
  - sdl_complexity_prior      : sigmoid(temp * norm_complexity)              0 params
  - sdl_with_intra_param      : sigmoid(temp * (λa*sim_in - sim_out))       1 param
  - sdl_without_structure     : sigmoid(temp * (λa*sim_in - λb*sim_out))    2 params

All models expose:
  backbone       : underlying Backbone
  forward(data)  : end-to-end graph embedding

FCGATModel additionally exposes:
  forward_attention(node_embs, fcg_edge_index, batch) : graph embedding from node embs

TrainingUtils provides the weight-computation methods shared by both
episodic_train.py (training) and episodic_evaluate.py (inference).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter
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


class SDLWithIntraParamModel(BaseEpisodicModel):
    """
    Ablation: learnable scale on sim_in only, sim_out coefficient fixed at 1.
    sigmoid(temp * (λa*sim_in - sim_out))  — 1 param.
    Tests whether scaling the intra-class signal alone is sufficient.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.raw_lambda_a = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        node_embs, fcg_batch = self.get_node_embeddings(data)
        node_labels = torch.repeat_interleave(data.family_class, data.num_functions)
        weights = TrainingUtils.generate_sdl_intra_param_weights(
            node_embeddings=node_embs,
            node_labels=node_labels,
            lambda_a_val=self.raw_lambda_a,
        )
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

