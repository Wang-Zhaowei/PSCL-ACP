

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from util import all_labels


class MultiScaleGCN(nn.Module):
    def __init__(self, in_dim, out_dim, scales=(1, 2, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, out_dim) for _ in range(len(scales))]
        )
        self.scales = scales
        self.attention = nn.Sequential(
            nn.Linear(out_dim * len(scales), len(scales)),
            nn.Softmax(dim=-1),
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x, edge_index):
        outputs = []
        current_edge_index = edge_index
        for index, scale in enumerate(self.scales):
            if scale > 1:
                current_edge_index, _ = dense_to_sparse(
                    torch.matrix_power(
                        to_dense_adj(current_edge_index).squeeze(0),
                        scale,
                    )
                )
            x_out = self.convs[index](x, current_edge_index)
            outputs.append(F.leaky_relu(x_out))

        concatenated = torch.cat(outputs, dim=-1)
        attention_weights = self.attention(concatenated)
        hidden = sum(
            outputs[index] * attention_weights[:, index].unsqueeze(-1)
            for index in range(len(self.scales))
        )

        if self.residual:
            hidden += self.residual(x)
        return hidden


class SHTLEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_dim = 256
        self.dropout = 0.3
        self.heads = 4
        self.aa_proj = nn.Sequential(
            nn.Linear(72, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.blosum_proj = nn.Sequential(
            nn.Linear(20, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(20, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(128, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.gcn = nn.ModuleList(
            [
                MultiScaleGCN(256, self.hidden_dim),
                MultiScaleGCN(self.hidden_dim, self.hidden_dim),
            ]
        )
        self.pool = TopKPooling(in_channels=self.hidden_dim, ratio=0.5)

    def forward(self, graph_data):
        edge_index = graph_data.edge_index
        batch = graph_data.batch
        features = torch.cat(
            [
                self.aa_proj(graph_data.aa_feat),
                self.blosum_proj(graph_data.blosum62_feat),
                self.pos_proj(graph_data.pos_feat),
            ],
            dim=-1,
        )
        x = self.input_proj(features)
        x = self.gcn[0](x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, batch=batch)
        x = self.gcn[1](x, edge_index)
        return x, batch


class SHTL(SHTLEmbedding):
  

    def __init__(self, data_type):
        super().__init__()
        if data_type == "binary":
            self.output = nn.Softmax(dim=1)
            output_dim = 2
        else:
            self.output = nn.Sigmoid()
            output_dim = len(all_labels)
        self.linear = nn.Linear(256, output_dim)

    def forward(self, graph_data):
        node_tokens, batch = super().forward(graph_data)
        x = global_mean_pool(node_tokens, batch)
        internal = x.clone()
        x = self.output(self.linear(x))
        return x, internal

__all__ = [
    "SHTL",
    "SHTLEmbedding",
    "MultiScaleGCN",
]
