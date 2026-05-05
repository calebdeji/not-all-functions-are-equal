import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool, global_mean_pool, LayerNorm


class BackboneProjectionHead(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BackboneProjectionHead, self).__init__()

        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data):
        return self.head(data)



class Backbone(torch.nn.Module):
    def __init__(self, cfg_dim, func_feat_dim, cfg_hidden_channels, fcg_hidden_channels, final_channels, dropout_value):
        super(Backbone, self).__init__()

        self.cfg_conv1 = SAGEConv(cfg_dim, cfg_hidden_channels)
        self.cfg_conv2 = SAGEConv(cfg_hidden_channels, cfg_hidden_channels)

        self.cfg_proj = torch.nn.Linear(cfg_hidden_channels, fcg_hidden_channels)
        self.fcg_proj = torch.nn.Linear(func_feat_dim, fcg_hidden_channels)

        self.fcg_conv1 = SAGEConv(fcg_hidden_channels, fcg_hidden_channels)
        self.fcg_conv2 = SAGEConv(fcg_hidden_channels, final_channels)

        self.dropout1 = torch.nn.Dropout(dropout_value)
        self.dropout2 = torch.nn.Dropout(dropout_value)

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, batch = data.x, data.edge_index, data.cfg_node_batch

        x = self.cfg_conv1(x, edge_index)
        x = F.relu(x)

        x = self.cfg_conv2(x, edge_index)
        x = F.relu(x)

        x = self.dropout1(x)

        function_features = data.function_features
        num_functions = function_features.shape[0]

        pooled_cfg_features = global_add_pool(x, batch, size=num_functions)

        fcg_edge_index = data.fcg_edge_index

        proj_cfg = F.relu(self.cfg_proj(pooled_cfg_features))
        proj_fcg = F.relu(self.fcg_proj(function_features))
        fused = proj_cfg + proj_fcg

        f_x = self.fcg_conv1(fused, fcg_edge_index)
        f_x = F.relu(f_x)
        f_x = self.fcg_conv2(f_x, fcg_edge_index)
        f_x = F.relu(f_x)
        f_x = self.dropout2(f_x)

        if hasattr(data, 'num_graphs'):
            fcg_batch = torch.repeat_interleave(
                torch.arange(data.num_graphs, device=data.x.device),
                data.num_functions,
            )
        else:
            print(f'[!] It doesn\'t have num_graphs attribute')
            fcg_batch = torch.zeros(num_functions, device=data.x.device, dtype=torch.long)

        if return_node_embeddings:
            return f_x, fcg_batch

        graph_rep = global_mean_pool(f_x, fcg_batch)

        return graph_rep, fcg_batch
