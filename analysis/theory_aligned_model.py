"""D2Pred predictor aligned with the method figure and Chapters 3/5."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dimension: int, maximum_length: int = 64) -> None:
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32)[:, None]
        divisor = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(maximum_length, dimension)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[: values.shape[1]][None].to(values.dtype)


class TheoryAlignedD2Pred(nn.Module):
    """DWC LSTM, event Transformer, DWC-query cross-attention, three FC layers."""

    def __init__(
        self,
        token_dimension: int = 12,
        hidden_dimension: int = 64,
        lstm_layers: int = 1,
        event_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.15,
        maximum_tokens: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dimension % attention_heads:
            raise ValueError("hidden_dimension must be divisible by attention_heads")
        self.dwc_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dimension,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.daily_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.event_projection = nn.Sequential(
            nn.Linear(token_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.position = SinusoidalPositionEncoding(hidden_dimension, maximum_tokens)
        event_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=attention_heads,
            dim_feedforward=hidden_dimension * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.event_encoder = nn.TransformerEncoder(event_layer, num_layers=event_layers)
        self.query_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.key_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.value_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dimension, attention_heads, dropout=dropout, batch_first=True
        )
        self.regression = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.ReLU(),
            nn.Linear(hidden_dimension // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: dict[str, torch.Tensor], return_details: bool = False):
        _, (hidden, _) = self.dwc_lstm(inputs["dwc"].unsqueeze(-1))
        daily = self.daily_projection(hidden[-1])
        valid = inputs["token_mask"].bool()
        events = self.position(self.event_projection(inputs["tokens"]))
        events = self.event_encoder(events, src_key_padding_mask=~valid)
        query = self.query_projection(daily).unsqueeze(1)
        keys = self.key_projection(events)
        values = self.value_projection(events)
        fused, attention = self.cross_attention(
            query, keys, values, key_padding_mask=~valid,
            need_weights=True, average_attn_weights=False,
        )
        prediction = self.regression(fused[:, 0]).squeeze(-1)
        if return_details:
            return {
                "prediction": prediction,
                "daily_feature": daily,
                "event_features": events,
                "cross_attention": attention,
                "fused_feature": fused[:, 0],
            }
        return prediction

    @staticmethod
    def architecture_metadata() -> dict:
        return {
            "dwc_channel": "unidirectional_LSTM_final_state_linear_projection",
            "event_channel": "token_MLP_positional_encoding_2layer_TransformerEncoder",
            "fusion": "DWC_query_event_key_value_multihead_cross_attention",
            "regression_head": "three_linear_layers_with_ReLU_and_sigmoid_output",
            "direct_edc_recurrent_branch": False,
        }
