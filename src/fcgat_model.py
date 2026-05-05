"""
fcgat_model.py

Standalone FCGAT model for supervised pre-training as a few-shot baseline.

Architecture:
  cfg_conv1/2   : SAGEConv(cfg_dim → cfg_hidden) x2, ReLU
  cfg_pool      : global_add_pool(by cfg_node_batch) → (num_functions, cfg_hidden)
  cfg_proj      : Linear(cfg_hidden → input_dim), ReLU
  fcg_proj      : Linear(func_feat_dim → input_dim), ReLU
  fused         : cfg_proj + fcg_proj  (element-wise sum, shape: num_functions × input_dim)
  gat           : GATConv(input_dim → 64×3=192, K=3 heads, no self-loops)
  residual      : fc_init(input_dim → 192)
  act           : LeakyReLU (×2, matching Someya et al.)
  readout       : Set2Set(192 → 384, processing_steps=4)
  dropout       : p=0.5
  classifier    : Linear(384 → num_classes)  [training only]

Node features:   data.function_features  (func_feat_dim-dim, log1p-scaled)
CFG features:    data.x                  (cfg_dim-dim per basic block, log1p-scaled)
CFG structure:   data.edge_index         (intra-function CFG edges)
CFG→func map:    data.cfg_node_batch     (block → function index)
FCG structure:   data.fcg_edge_index     (function call graph)

At few-shot evaluation time, call embed(data) to obtain 384-dim L2-normalised
graph embeddings — the classifier head is dropped.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from torch_geometric.nn import GATConv, SAGEConv, global_add_pool
from torch_geometric.utils import softmax as _pyg_softmax
from torch_geometric.utils import scatter


class _Set2Set(nn.Module):
    """LSTM-based Set2Set readout (Vinyals et al., 2015)."""

    def __init__(self, in_channels: int, processing_steps: int = 4):
        super().__init__()
        self.in_channels      = in_channels
        self.out_channels     = 2 * in_channels
        self.processing_steps = processing_steps
        self.lstm = nn.LSTM(self.out_channels, in_channels)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        h = (x.new_zeros(1, batch_size, self.in_channels),
             x.new_zeros(1, batch_size, self.in_channels))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for _ in range(self.processing_steps):
            q, h    = self.lstm(q_star.unsqueeze(0), h)
            q       = q.squeeze(0)
            e       = (x * q[batch]).sum(dim=-1)
            a       = _pyg_softmax(e, batch, num_nodes=batch_size)
            r       = scatter(a.unsqueeze(-1) * x, batch, dim=0,
                              dim_size=batch_size, reduce='sum')
            q_star  = torch.cat([q, r], dim=-1)

        return q_star  # (batch_size, 2*in_channels)


class StandaloneFCGAT(nn.Module):
    """
    FCGAT pre-trained end-to-end with cross-entropy on training families.

    Extends the original FCGAT architecture (Someya et al., BAR@NDSS 2023) by
    incorporating CFG block features via a SAGEConv encoder, matching the
    mid-fusion design of the backbone:

      CFG blocks → SAGEConv x2 → global_add_pool(by cfg_node_batch)
                                          ↓
                                     cfg_proj (Linear → ReLU)
                                          ↓
      function_features → fcg_proj (Linear → ReLU) → [element-wise sum] → fused
                                                                               ↓
                                                               GATConv (3 heads)
                                                               + residual fc_init
                                                               LeakyReLU ×2
                                                               Set2Set → 384-dim
                                                               Dropout(0.5)

    Use embed(data) at few-shot evaluation time — returns (384-dim L2-normalised
    graph embeddings, None). The None preserves compatibility with the interface
    expected by embed_samples() in evaluate.py.
    """

    def __init__(
        self,
        cfg_dim: int = 10,
        cfg_hidden: int = 128,
        func_feat_dim: int = 8,
        input_dim: int = 100,
        gat_hidden_per_head: int = 64,
        gat_heads: int = 3,
        processing_steps: int = 4,
        dropout_p: float = 0.5,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        gat_out_dim = gat_hidden_per_head * gat_heads  # 192
        embed_dim   = 2 * gat_out_dim                  # 384

        # ---- CFG encoder ----
        self.cfg_conv1 = SAGEConv(cfg_dim,    cfg_hidden)
        self.cfg_conv2 = SAGEConv(cfg_hidden, cfg_hidden)
        self.cfg_proj  = nn.Linear(cfg_hidden, input_dim)

        # ---- Function-feature projection ----
        self.fcg_proj  = nn.Linear(func_feat_dim, input_dim)

        # ---- FCG GAT (paper dimensions) ----
        self.gat     = GATConv(input_dim, gat_hidden_per_head, heads=gat_heads,
                               concat=True, add_self_loops=False)
        self.fc_init = nn.Linear(input_dim, gat_out_dim, bias=False)
        self.act     = nn.LeakyReLU()
        self.readout = _Set2Set(gat_out_dim, processing_steps=processing_steps)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(embed_dim, num_classes) if num_classes else None

    def _encode(self, data) -> torch.Tensor:
        """Shared encoding: CFG blocks + function features → 384-dim graph vector."""
        device = data.function_features.device

        # ---- CFG encoder ----
        cfg_x = self.cfg_conv1(data.x, data.edge_index)
        cfg_x = F.relu(cfg_x)
        cfg_x = self.cfg_conv2(cfg_x, data.edge_index)
        cfg_x = F.relu(cfg_x)

        num_functions = data.function_features.shape[0]
        pooled_cfg = global_add_pool(cfg_x, data.cfg_node_batch, size=num_functions)

        # ---- Mid-fusion (mirrors backbone) ----
        x = F.relu(self.cfg_proj(pooled_cfg)) + F.relu(self.fcg_proj(data.function_features))
        x_init = x

        # ---- FCG GAT ----
        fcg_batch = torch.repeat_interleave(
            torch.arange(data.num_graphs, device=device),
            data.num_functions,
        )

        x = self.gat(x, data.fcg_edge_index)
        x = self.act(x)
        x = x + self.fc_init(x_init)
        x = self.act(x)

        graph_vec = self.readout(x, fcg_batch, batch_size=data.num_graphs)
        return self.dropout(graph_vec)

    def forward(self, data) -> torch.Tensor:
        """Returns class logits. Used during supervised pre-training."""
        return self.classifier(self._encode(data))

    def embed(self, data) -> Tuple[torch.Tensor, None]:
        """
        Returns (L2-normalised 384-dim graph embeddings, None).
        Compatible with backbone(data) → (embs, _) interface.
        """
        return F.normalize(self._encode(data), p=2, dim=-1), None
