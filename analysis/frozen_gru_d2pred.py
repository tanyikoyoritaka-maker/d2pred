"""Frozen GRU D2Pred architecture used by the leakage-safe AC reruns.

This is a versioned successor to ``CPMEDCIndependentFusion``.  It preserves
the experimentally successful DWC/EDC/AEP-token design while fixing the
bidirectional recurrent summary: both final directions are taken from ``h_n``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenGRUD2Pred(nn.Module):
    """Independent DWC/EDC BiGRUs plus masked statistical-token attention."""

    architecture_version = "gru_d2pred_v2_hn_summary"

    def __init__(
        self,
        token_dimension: int = 12,
        hidden_dimension: int = 64,
        use_dwc: bool = True,
        use_edc: bool = True,
        use_tokens: bool = True,
        fusion: str = "concat",
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if hidden_dimension % 2:
            raise ValueError("hidden_dimension must be even for a bidirectional GRU")
        if fusion not in {"concat", "average_prediction"}:
            raise ValueError(f"Unsupported fusion: {fusion}")
        self.use_dwc = use_dwc
        self.use_edc = use_edc
        self.use_tokens = use_tokens
        self.fusion = fusion
        direction_hidden = hidden_dimension // 2
        self.dwc_gru = (
            nn.GRU(1, direction_hidden, batch_first=True, bidirectional=True)
            if use_dwc else None
        )
        self.edc_gru = (
            nn.GRU(1, direction_hidden, batch_first=True, bidirectional=True)
            if use_edc else None
        )
        self.token_projection = (
            nn.Sequential(
                nn.Linear(token_dimension, hidden_dimension),
                nn.ReLU(),
                nn.Linear(hidden_dimension, hidden_dimension),
            )
            if use_tokens else None
        )
        self.token_score = nn.Linear(hidden_dimension, 1) if use_tokens else None
        branch_count = int(use_dwc) + int(use_edc) + int(use_tokens)
        if branch_count == 0:
            raise ValueError("At least one branch must be enabled")
        self.branch_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dimension, hidden_dimension // 2),
                nn.ReLU(),
                nn.Linear(hidden_dimension // 2, 1),
                nn.Sigmoid(),
            )
            for _ in range(branch_count)
        ])
        self.regression = nn.Sequential(
            nn.Linear(hidden_dimension * branch_count, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _bigru_summary(gru: nn.GRU, values: torch.Tensor) -> torch.Tensor:
        _, hidden = gru(values.unsqueeze(-1))
        # One layer, two directions: [forward final, reverse final].
        return torch.cat((hidden[-2], hidden[-1]), dim=-1)

    def forward(self, inputs: dict[str, torch.Tensor], return_details: bool = False):
        features: list[torch.Tensor] = []
        names: list[str] = []
        token_attention = None
        if self.dwc_gru is not None:
            features.append(self._bigru_summary(self.dwc_gru, inputs["dwc"]))
            names.append("dwc")
        if self.edc_gru is not None:
            features.append(self._bigru_summary(self.edc_gru, inputs["edc"]))
            names.append("edc")
        if self.token_projection is not None:
            tokens = self.token_projection(inputs["tokens"])
            valid = inputs["token_mask"].bool()
            scores = self.token_score(tokens).squeeze(-1).masked_fill(~valid, -1e4)
            token_attention = torch.softmax(scores, dim=1) * valid.to(tokens.dtype)
            token_attention = token_attention / token_attention.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            features.append((tokens * token_attention.unsqueeze(-1)).sum(dim=1))
            names.append("tokens")
        if self.fusion == "average_prediction":
            branch_predictions = [
                head(feature).squeeze(-1)
                for head, feature in zip(self.branch_heads, features)
            ]
            prediction = torch.stack(branch_predictions, dim=0).mean(dim=0)
        else:
            branch_predictions = []
            prediction = self.regression(torch.cat(features, dim=-1)).squeeze(-1)
        if not return_details:
            return prediction
        return {
            "prediction": prediction,
            "branch_names": names,
            "branch_features": features,
            "branch_predictions": branch_predictions,
            "token_attention": token_attention,
        }

    def architecture_metadata(self) -> dict:
        return {
            "architecture_version": self.architecture_version,
            "dwc_encoder": "one_layer_bidirectional_GRU_hn_summary" if self.use_dwc else None,
            "edc_encoder": "one_layer_bidirectional_GRU_hn_summary" if self.use_edc else None,
            "event_encoder": (
                "Linear(12,64)-ReLU-Linear(64,64)-masked_scalar_attention"
                if self.use_tokens else None
            ),
            "fusion": self.fusion,
            "output": "sigmoid_fraction_of_train_only_target_scale",
        }
