

import torch.nn as nn

from util import all_labels


SEQUENCE_DIM = 320


class SSCREmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=SEQUENCE_DIM,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

    def forward(self, x1, mask):
        return self.transformer_encoder(x1, src_key_padding_mask=mask)


class SSCR(nn.Module):


    def __init__(self, data_type):
        super().__init__()
        self.transformer_encoder = SSCREmbedding().transformer_encoder

        if data_type == "binary":
            self.output = nn.Softmax(dim=1)
            output_dim = 2
        else:
            self.output = nn.Sigmoid()
            output_dim = len(all_labels)

        self.linear = nn.Linear(SEQUENCE_DIM, output_dim)

    def forward(self, x1, seqlen, mask):
        del seqlen 
        x = self.transformer_encoder(x1, src_key_padding_mask=mask)
        x = x[:, 0, :].squeeze(1)
        internal = x.clone()
        x = self.output(self.linear(x))
        return x, internal

__all__ = ["SSCR", "SSCREmbedding", "SEQUENCE_DIM"]
