import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from set_transformer.model import SetTransformer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class DualTransformer(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = [
            "n_features",
            "n_classes",
            "context_dim",
            "trans_dim",
            "dropout",
        ]

        self.n_features = 15
        self.n_classes = 1
        self.context_dim = 32
        self.trans_dim = 256
        self.dropout = 0.2
        #self.pretrain = False if params["pretrain"] == 0 else True

        self.passer_fc = nn.Linear(self.n_features, self.context_dim)
        self.teammate_st = SetTransformer(self.n_features, self.context_dim * 4)
        self.opponent_st = SetTransformer(self.n_features, self.context_dim * 4)
        self.ball_fc = nn.Linear(2, self.context_dim)
        self.whole_st = SetTransformer(self.n_features + 2, self.context_dim, embed_type="equivariant")

        self.team_poss_st = SetTransformer(self.n_classes, self.context_dim)
        self.event_player_st = SetTransformer(self.n_classes, self.context_dim)
        embed_output_dim = self.context_dim * 4
        self.context_fc = nn.Sequential(nn.Linear(embed_output_dim, self.trans_dim), nn.ReLU())

        self.time_pos_encoder = PositionalEncoding(self.trans_dim)
        self.time_encoder = TransformerEncoder(
            TransformerEncoderLayer(self.trans_dim, 8, self.trans_dim * 2, self.dropout),
            6,
        )

        self.player_pos_encoder = PositionalEncoding(self.trans_dim)
        self.player_encoder = TransformerEncoder(
            TransformerEncoderLayer(self.trans_dim, 8, self.trans_dim * 2, self.dropout),
            6,
        )

        self.time_encoder2 = TransformerEncoder(
            TransformerEncoderLayer(self.trans_dim, 8, self.trans_dim * 2, self.dropout),
            6,
        )

        self.player_encoder2 = TransformerEncoder(
            TransformerEncoderLayer(self.trans_dim, 8, self.trans_dim * 2, self.dropout),
            6,
        )

        self.input_fc = nn.Linear(self.n_features, self.trans_dim)

        self.output_fc = nn.Sequential(nn.Linear(self.trans_dim, 1))

        self.output_fc_xy = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256), 
            nn.ReLU(),                           
            nn.Linear(256, 128),                
            nn.ReLU(),                          
            nn.Linear(128, 2)                   
        )

    def forward(
        self,
        passer_input: torch.Tensor,
        teammate_input: torch.Tensor,
        opponent_input: torch.Tensor,
        ball_input: torch.Tensor,
        team_poss_input: torch.Tensor,
        event_player_input: torch.Tensor,
        passer_mask_list : torch.Tensor,
    ) -> torch.Tensor:

        passer_input = passer_input.transpose(0, 1)  # [seq_len, bs, n_features]
        teammate_input = teammate_input.transpose(0, 1)  # [seq_len, bs, n_features * n_teammates]
        opponent_input = opponent_input.transpose(0, 1)  # [seq_len, bs, n_features * n_opponents]
        ball_input = ball_input.transpose(0, 1)  # [seq_len, bs, n_features]

        team_poss_input = team_poss_input.transpose(0, 1)
        event_player_input = event_player_input.transpose(0, 1)

        seq_len = passer_input.size(0)
        batch_size = passer_input.size(1)
        n_teammates = teammate_input.size(2) // self.n_features

        players_with_ball = torch.cat([teammate_input, opponent_input, passer_input, ball_input], dim=2)

        # Temporal-Spatio Encoder

        time_z = self.time_pos_encoder(
            self.input_fc(players_with_ball.reshape(seq_len, batch_size * (2 * self.n_classes + 2), -1))
            # * math.sqrt(self.params["trans_dim"])
        )
        time_h = self.time_encoder(time_z)

        time_h_reshaped = (
            time_h.reshape(seq_len, batch_size, (2 * self.n_classes + 2), -1)
            .transpose(0, 2)
            .reshape(2 * self.n_classes + 2, batch_size * seq_len, -1)
        )  # Select the last time step and transpose

        # player_z = self.player_pos_encoder(
        #    time_h_reshaped  # * math.sqrt(self.params["trans_dim"])
        # )
        player_h = self.player_encoder(time_h_reshaped)  # [n_classes, batch, trans_dim]

        player_h_reshaped = (
            player_h.transpose(0, 1).reshape(batch_size, seq_len, (2 * self.n_classes + 2), -1).transpose(0, 1)
        ).reshape(seq_len, batch_size * (2 * self.n_classes + 2), -1)

        time_z = self.time_pos_encoder(player_h_reshaped)
        time_h = self.time_encoder2(time_z).reshape(seq_len, batch_size, 2 * self.n_classes + 2, -1) #[seq_len, bs, 24, trans_dim]

        out = self.output_fc(time_h[-1, :, : self.n_classes]).reshape(batch_size, self.n_classes) #[bs, 11, 256] -> [bs, 11 * 256]

        mask = torch.zeros_like(out)
        batch_indices = np.arange(batch_size)
        mask[batch_indices, passer_mask_list] = -np.inf
        out += mask

        intended_receiver = torch.argmax(out, dim = 1)

        receiver_h = time_h[-1, batch_indices, intended_receiver] #[bs, trans_dim]
        ball_h = time_h[-1, :, -1] #[bs, trans_dim]

        return self.output_fc_xy(torch.cat((receiver_h, ball_h), dim = -1)) # [bs, trans_dim * 2] -> [bs, 2]