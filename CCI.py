

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from SSCR import SEQUENCE_DIM, SSCREmbedding
from SHTL import SHTLEmbedding
from util import all_labels


def masked_mean(x, mask):
    valid = (~mask).unsqueeze(-1).type_as(x)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def masked_max(x, mask):
    values = x.masked_fill(mask.unsqueeze(-1), torch.finfo(x.dtype).min)
    output = values.max(dim=1).values
    return output.masked_fill(mask.all(dim=1).unsqueeze(-1), 0.0)


class BCA_Block(nn.Module):
    def __init__(self, dim=256, heads=4, dropout=0.2):
        super().__init__()
        self.g2s = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.s2g = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.g_gate = nn.Linear(dim * 2, dim)
        self.s_gate = nn.Linear(dim * 2, dim)
        self.g_norm = nn.LayerNorm(dim)
        self.s_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_tokens, seq_mask, graph_tokens, graph_mask):
        graph_message, _ = self.g2s(
            query=graph_tokens,
            key=seq_tokens,
            value=seq_tokens,
            key_padding_mask=seq_mask,
            need_weights=False,
        )
        seq_message, _ = self.s2g(
            query=seq_tokens,
            key=graph_tokens,
            value=graph_tokens,
            key_padding_mask=graph_mask,
            need_weights=False,
        )
        graph_gate = torch.sigmoid(
            self.g_gate(torch.cat([graph_tokens, graph_message], dim=-1))
        )
        seq_gate = torch.sigmoid(
            self.s_gate(torch.cat([seq_tokens, seq_message], dim=-1))
        )
        graph_tokens = self.g_norm(
            graph_tokens + self.dropout(graph_gate * graph_message)
        )
        seq_tokens = self.s_norm(
            seq_tokens + self.dropout(seq_gate * seq_message)
        )
        graph_tokens = graph_tokens.masked_fill(graph_mask.unsqueeze(-1), 0.0)
        seq_tokens = seq_tokens.masked_fill(seq_mask.unsqueeze(-1), 0.0)
        return seq_tokens, graph_tokens


class AttnFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, features):
        feature_stack = torch.stack(features, dim=1)
        hidden = torch.tanh(self.proj(feature_stack))
        weights = torch.softmax(self.score(hidden).squeeze(-1), dim=1)
        fused = (weights.unsqueeze(-1) * feature_stack).sum(dim=1)
        return fused, weights


class GlobalLatentCrossAttention(nn.Module):
    def __init__(
        self,
        seq_dim=320,
        graph_dim=256,
        hidden_dim=256,
        num_latents=2,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_latents = num_latents
        self.seq_to_latents = nn.Sequential(
            nn.Linear(seq_dim, num_latents * hidden_dim),
            nn.GELU(),
        )
        self.graph_to_latents = nn.Sequential(
            nn.Linear(graph_dim, num_latents * hidden_dim),
            nn.GELU(),
        )
        self.seq_latent_norm = nn.LayerNorm(hidden_dim)
        self.graph_latent_norm = nn.LayerNorm(hidden_dim)
        self.seq_from_graph = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.graph_from_seq = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.seq_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.graph_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.seq_out_norm = nn.LayerNorm(hidden_dim)
        self.graph_out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_global, graph_global):
        batch_size = seq_global.size(0)
        seq_latents = self.seq_to_latents(seq_global).view(
            batch_size,
            self.num_latents,
            self.hidden_dim,
        )
        graph_latents = self.graph_to_latents(graph_global).view(
            batch_size,
            self.num_latents,
            self.hidden_dim,
        )
        seq_latents = self.seq_latent_norm(seq_latents)
        graph_latents = self.graph_latent_norm(graph_latents)

        seq_message, seq_attention = self.seq_from_graph(
            query=seq_latents,
            key=graph_latents,
            value=graph_latents,
            need_weights=True,
            average_attn_weights=False,
        )
        graph_message, graph_attention = self.graph_from_seq(
            query=graph_latents,
            key=seq_latents,
            value=seq_latents,
            need_weights=True,
            average_attn_weights=False,
        )
        seq_gate = torch.sigmoid(
            self.seq_gate(torch.cat([seq_latents, seq_message], dim=-1))
        )
        graph_gate = torch.sigmoid(
            self.graph_gate(torch.cat([graph_latents, graph_message], dim=-1))
        )
        seq_tokens = self.seq_out_norm(
            seq_latents + self.dropout(seq_gate * seq_message)
        )
        graph_tokens = self.graph_out_norm(
            graph_latents + self.dropout(graph_gate * graph_message)
        )
        return (
            seq_tokens.mean(dim=1),
            graph_tokens.mean(dim=1),
            seq_attention,
            graph_attention,
        )


class HierarchicalGatedFusion(nn.Module):
    def __init__(self, dim=256, dropout=0.2):
        super().__init__()
        self.global_fusion = AttnFusion(dim=dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.global_alpha = nn.Parameter(torch.tensor(-2.0))
        self.disagreement_modulator = nn.Linear(dim, 1)
        nn.init.zeros_(self.disagreement_modulator.weight)
        nn.init.zeros_(self.disagreement_modulator.bias)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_fused, seq_global_ca, graph_global_ca):
        global_fused, global_weights = self.global_fusion(
            [seq_global_ca, graph_global_ca]
        )
        joint = torch.cat(
            [
                token_fused,
                global_fused,
                torch.abs(token_fused - global_fused),
                token_fused * global_fused,
            ],
            dim=-1,
        )
        gate = self.gate(joint)
        scale = torch.sigmoid(self.global_alpha)
        correction = gate * (global_fused - token_fused)
        disagreement_scale = 1.0 + 0.1 * torch.tanh(
            self.disagreement_modulator(torch.abs(token_fused - global_fused))
        )
        final_fused = self.norm(
            token_fused
            + scale * disagreement_scale * self.dropout(correction)
        )
        return final_fused, global_weights, gate, scale


class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, dim=256, projection_dim=128, temperature=0.2):
        super().__init__()
        self.seq_proj = nn.Linear(dim, projection_dim, bias=False)
        self.graph_proj = nn.Linear(dim, projection_dim, bias=False)
        self.temperature = temperature

    def forward(self, seq_global, graph_global):
        batch_size = seq_global.size(0)
        if batch_size < 2:
            return seq_global.new_zeros(())
        seq_embedding = F.normalize(self.seq_proj(seq_global), dim=-1)
        graph_embedding = F.normalize(self.graph_proj(graph_global), dim=-1)
        logits = seq_embedding @ graph_embedding.transpose(0, 1)
        logits = logits / self.temperature
        targets = torch.arange(batch_size, device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.transpose(0, 1), targets)
        )


class CCI(nn.Module):

    def __init__(self, data_type):
        super().__init__()
        self.embed1 = SSCREmbedding()
        self.embed2 = SHTLEmbedding()
        self.seq_proj = nn.Sequential(
            nn.Linear(SEQUENCE_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.bca = BCA_Block(dim=256, heads=8, dropout=0.2)
        self.fusion = AttnFusion(dim=256)
        self.global_cross_attn = GlobalLatentCrossAttention(
            seq_dim=256,
            graph_dim=256,
            hidden_dim=256,
            num_latents=4,
            heads=8,
            dropout=0.2,
        )
        self.hierarchical_fusion = HierarchicalGatedFusion(
            dim=256,
            dropout=0.2,
        )
        self.contrastive_loss = CrossModalContrastiveLoss(
            dim=256,
            projection_dim=128,
            temperature=0.2,
        )
        self.contrastive_weight = 0.01

        if data_type == "binary":
            self.output = nn.Softmax(dim=1)
            output_dim = 2
        else:
            self.output = nn.Sigmoid()
            output_dim = len(all_labels)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
        )

    def load_sscr(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.embed1.load_state_dict(state_dict, strict=False)

    def load_shtl(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.embed2.load_state_dict(state_dict, strict=False)

 
    load_model1 = load_sscr
    load_model2 = load_shtl

    def forward(self, x1, seqlen, mask, graph_data):
        del seqlen 
        seq_tokens = self.embed1(x1, mask)
        seq_tokens = self.seq_proj(seq_tokens)
        seq_tokens = seq_tokens.masked_fill(mask.unsqueeze(-1), 0.0)

        node_tokens, batch = self.embed2(graph_data)
        graph_tokens, graph_valid = to_dense_batch(node_tokens, batch)
        graph_mask = ~graph_valid
        seq_tokens, graph_tokens = self.bca(
            seq_tokens=seq_tokens,
            seq_mask=mask,
            graph_tokens=graph_tokens,
            graph_mask=graph_mask,
        )

        seq_global = masked_max(seq_tokens, mask)
        graph_global = masked_mean(graph_tokens, graph_mask)
        seq_global_ca, graph_global_ca, _, _ = self.global_cross_attn(
            seq_global=seq_global,
            graph_global=graph_global,
        )
        graph_vector = masked_mean(graph_tokens, graph_mask)
        seq_vector = masked_max(seq_tokens, mask)

        if self.training:
            auxiliary_loss = self.contrastive_weight * self.contrastive_loss(
                seq_vector,
                graph_vector,
            )
        else:
            auxiliary_loss = seq_vector.new_zeros(())

        fused, weights = self.fusion([graph_vector, seq_vector])
        self.last_fusion_weights = weights.detach()
        final_fused, _, _, _ = self.hierarchical_fusion(
            token_fused=fused,
            seq_global_ca=seq_global_ca,
            graph_global_ca=graph_global_ca,
        )
        output = self.output(self.classifier(final_fused))
        return output, fused, auxiliary_loss

__all__ = ["CCI"]
