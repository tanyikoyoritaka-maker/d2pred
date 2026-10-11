"""Reviewer-specified D2Pred predictor used by the approved nested rerun."""

from __future__ import annotations

import torch
import torch.nn as nn

from analysis.theory_aligned_model import SinusoidalPositionEncoding
from analysis.theory_aligned_decomposition import StageAwareTrendAutoencoder


class ExpertExactD2Pred(nn.Module):
    """Gradual LSTM, 12-descriptor event Transformer, guided fusion, Softplus RUL."""

    architecture_version = "d2pred_expert_exact_v1"

    def __init__(self, token_dimension: int = 12, hidden_dimension: int = 64,
                 attention_heads: int = 4, event_layers: int = 3,
                 maximum_tokens: int = 16, dropout: float = 0.15,
                 use_age: bool = False, adaptive_gate: bool = False,
                 age_dimension: int = 2, joint_trend_fusion: bool = True,
                 use_gradual_encoder: bool = True, use_token_projection: bool = True,
                 use_event_encoder: bool = True, use_guided_fusion: bool = True) -> None:
        super().__init__()
        if event_layers != 3:
            raise ValueError("The frozen reviewer protocol requires exactly three event blocks")
        self.gradual_encoder = nn.LSTM(1, hidden_dimension, batch_first=True)
        self.gradual_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.use_age = use_age
        self.age_dimension = age_dimension
        self.adaptive_gate = adaptive_gate
        self.joint_trend_fusion = joint_trend_fusion
        self.use_gradual_encoder = use_gradual_encoder
        self.use_token_projection = use_token_projection
        self.use_event_encoder = use_event_encoder
        self.use_guided_fusion = use_guided_fusion
        self.trend_fusion = StageAwareTrendAutoencoder() if joint_trend_fusion else None
        self.age_projection = (
            nn.Sequential(nn.Linear(age_dimension, hidden_dimension // 2), nn.SiLU(),
                          nn.Linear(hidden_dimension // 2, hidden_dimension))
            if use_age else None
        )
        self.token_projection = nn.Sequential(
            nn.Linear(token_dimension, hidden_dimension), nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.position = SinusoidalPositionEncoding(hidden_dimension, maximum_tokens)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dimension, nhead=attention_heads,
            dim_feedforward=hidden_dimension * 2, dropout=dropout,
            activation="relu", batch_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(block, num_layers=3)
        self.query = nn.Linear(hidden_dimension, hidden_dimension)
        self.key = nn.Linear(hidden_dimension, hidden_dimension)
        self.value = nn.Linear(hidden_dimension, hidden_dimension)
        self.guided_fusion = nn.MultiheadAttention(
            hidden_dimension, attention_heads, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_dimension)
        self.gate = (
            nn.Sequential(nn.Linear(hidden_dimension * 2 + (hidden_dimension if use_age else 0), hidden_dimension // 2),
                          nn.SiLU(), nn.Linear(hidden_dimension // 2, 1), nn.Sigmoid())
            if adaptive_gate else None
        )
        fusion_width = hidden_dimension * (3 if use_age else 2)
        self.regression = nn.Sequential(
            nn.Linear(fusion_width, hidden_dimension), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dimension, 1), nn.Softplus(),
        )
        # Auxiliary predictions provide explicit differentiable paths for the
        # gradual and event terms in the frozen joint objective.
        self.gradual_aux = nn.Sequential(nn.Linear(hidden_dimension, 1), nn.Softplus())
        self.event_aux = nn.Sequential(nn.Linear(hidden_dimension, 1), nn.Softplus())

    def forward(self, inputs: dict[str, torch.Tensor], return_details: bool = False):
        trend_fusion_details = None
        gradual_input = inputs["dwc"]
        if self.trend_fusion is not None:
            gradual_input, trend_fusion_details = self.trend_fusion.forward_components(
                inputs["trend_candidates"], inputs["sequence_mask"],
                inputs["stable_fraction"], inputs["wiener_reliability"],
            )
        if self.use_gradual_encoder:
            _, (hidden, _) = self.gradual_encoder(gradual_input.unsqueeze(-1))
            gradual = self.gradual_projection(hidden[-1])
        else:
            gradual = gradual_input.mean(1, keepdim=True).expand(-1, self.gradual_projection.out_features)
        valid = inputs["token_mask"].bool()
        if not self.use_token_projection:
            raw = inputs["tokens"]
            events = torch.nn.functional.pad(raw, (0, gradual.shape[-1] - raw.shape[-1]))
        else:
            events = self.token_projection(inputs["tokens"])
        events = self.position(events)
        if self.use_event_encoder:
            events = self.event_encoder(events, src_key_padding_mask=~valid)
        if self.use_guided_fusion:
            attended, attention = self.guided_fusion(
                self.query(gradual).unsqueeze(1), self.key(events), self.value(events),
                key_padding_mask=~valid, need_weights=True, average_attn_weights=False,
            )
            event = attended[:, 0]
        else:
            weights = valid.float().unsqueeze(-1)
            event = (events * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            attention = None
        fused = self.fusion_norm(gradual + event)
        gate = None
        age_feature = None
        if self.age_projection is not None:
            age_feature = self.age_projection(inputs["age"])
        if self.gate is not None:
            gate_input = torch.cat((gradual, event), dim=-1)
            if age_feature is not None:
                gate_input = torch.cat((gate_input, age_feature), dim=-1)
            gate = self.gate(gate_input)
            fused = self.fusion_norm(gradual + gate * event)
        regression_inputs = [gradual, fused]
        if age_feature is not None:
            regression_inputs.append(age_feature)
        prediction = self.regression(torch.cat(regression_inputs, dim=-1)).squeeze(-1)
        details = {
            "prediction": prediction,
            "gradual_prediction": self.gradual_aux(gradual).squeeze(-1),
            "event_prediction": self.event_aux(event).squeeze(-1),
            "gradual_feature": gradual, "event_features": events,
            "event_feature": event, "cross_attention": attention,
            "age_feature": age_feature, "gate": gate if self.gate is not None else None,
            "fused_dwc": gradual_input, "trend_fusion_details": trend_fusion_details,
        }
        return details if return_details else prediction

    def architecture_metadata(self) -> dict:
        return {
            "architecture_version": self.architecture_version,
            "trend_fusion": "training-device-only causal Wiener/STL autoencoder phase",
            "gradual_encoder": "unidirectional_LSTM_64",
            "raw_residual_descriptors": 12,
            "token_projection": "Linear(12,64)-ReLU-Linear(64,64)",
            "event_encoder": "three_block_masked_TransformerEncoder_4_heads",
            "fusion": "DWC_query_event_key_value_attention_plus_DWC_residual",
            "output": "Softplus_nonnegative_train_scale_RUL",
            "joint_loss": "RUL_MSE+0.1_gradual_aux_MSE+0.1_event_aux_MSE+0.1_trend_fusion_loss",
            "use_age": self.use_age,
            "joint_trainable_trend_fusion": self.joint_trend_fusion,
            "module_ablation_flags": {"gradual_encoder": self.use_gradual_encoder,
                "token_projection": self.use_token_projection,
                "event_encoder": self.use_event_encoder,
                "guided_fusion": self.use_guided_fusion},
            "joint_objective": "RUL+0.1_gradual_aux+0.1_event_aux+0.1_trend_fusion",
        }
