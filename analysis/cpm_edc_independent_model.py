"""Canonical implementation and aliases for the independent-EDC CPM model."""

from __future__ import annotations

import torch
import torch.nn as nn


CANONICAL_MODEL_NAME = "cpm_stats_tokens_edc_independent_fusion"
ROBUST_TREND_MODEL_NAME = "cpm_stats_tokens_edc_independent_robust_trend"
ANCHORED_DWC_CPM_MODEL_NAME = "cpm_stats_tokens_dwc_cpm_anchored_aefusion"
MODEL_ALIASES = (
    "cpm_edc_indepent",
    "cpm_edc_independent",
    CANONICAL_MODEL_NAME,
)


def canonical_model_name(name: str) -> str:
    if name not in MODEL_ALIASES:
        raise KeyError(f"Unknown independent-EDC CPM model alias: {name}")
    return CANONICAL_MODEL_NAME


class CPMEDCIndependentFusion(nn.Module):
    """Existing independent DWC/EDC recurrent fusion implementation.

    This class is moved verbatim from the experiment runner's
    ``FlexibleIndependentFusion`` implementation so all spelling aliases resolve to
    one registry entry without inventing another architecture.
    """

    def __init__(self, token_dim=12, hidden=64, use_dwc=True, use_edc=True,
                 use_tokens=True, fusion="concat", no_cpm=False, edc_layers=1,
                 aux_head=False):
        super().__init__()
        self.use_dwc, self.use_edc, self.use_tokens = use_dwc, use_edc, use_tokens
        self.fusion, self.no_cpm, self.aux_head = fusion, no_cpm, aux_head
        self.dwc_gru = nn.GRU(1, hidden // 2, batch_first=True, bidirectional=True) if use_dwc else None
        self.edc_gru = nn.GRU(1, hidden // 2, edc_layers, batch_first=True,
                              bidirectional=True, dropout=0.1 if edc_layers > 1 else 0.0) if use_edc else None
        self.token = nn.Sequential(nn.Linear(token_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden)) if use_tokens and not no_cpm else None
        self.token_score = nn.Linear(hidden, 1) if use_tokens and not no_cpm else None
        self.edc_aux = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                     nn.Linear(hidden // 2, 1), nn.Sigmoid()) if aux_head and use_edc else None
        feature_count = int(use_dwc) + int(use_edc) + int(use_tokens and not no_cpm)
        if feature_count == 0:
            feature_count = 1
        self.branch_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                          nn.Linear(hidden // 2, 1), nn.Sigmoid())
            for _ in range(feature_count)
        ])
        self.head = nn.Sequential(nn.Linear(hidden * feature_count, hidden), nn.ReLU(),
                                  nn.Dropout(0.15), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x, return_details=False):
        feats = []
        if self.use_dwc:
            values, _ = self.dwc_gru(x["dwc"].unsqueeze(-1))
            feats.append(values[:, -1])
        edc_repr = None
        if self.use_edc:
            values, _ = self.edc_gru(x["edc"].unsqueeze(-1))
            edc_repr = values[:, -1]
            feats.append(edc_repr)
        if self.use_tokens and not self.no_cpm:
            tokens = self.token(x["tokens"])
            mask = x["token_mask"]
            scores = self.token_score(tokens).squeeze(-1).masked_fill(mask == 0, -1e4)
            weights = torch.softmax(scores, 1) * mask
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
            feats.append((tokens * weights.unsqueeze(-1)).sum(1))
        if not feats:
            feats = [torch.zeros(x["di"].size(0), 64, device=x["di"].device)]
        if self.fusion == "avg_prediction":
            predictions = [head(feature).squeeze(-1)
                           for head, feature in zip(self.branch_heads, feats)]
            prediction = torch.stack(predictions, 0).mean(0)
        else:
            prediction = self.head(torch.cat(feats, -1)).squeeze(-1)
        details = {"prediction": prediction}
        if self.edc_aux is not None and edc_repr is not None:
            details["aux_prediction"] = self.edc_aux(edc_repr).squeeze(-1)
        return details if return_details else prediction


# Compatibility name for callers that used the old runner-local class.
FlexibleIndependentFusion = CPMEDCIndependentFusion


class CPMDWCEventFusion(nn.Module):
    """Two-branch predictor: DWC sequence plus CPM tokens derived from EDC.

    There is deliberately no EDC recurrent encoder.  EDC remains an upstream
    residual used by change-point segmentation and token construction only.
    """

    def __init__(self, token_dim: int = 12, hidden: int = 64) -> None:
        super().__init__()
        self.dwc_gru = nn.GRU(
            1, hidden // 2, batch_first=True, bidirectional=True
        )
        self.token = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.token_score = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x, return_details: bool = False):
        dwc_values, _ = self.dwc_gru(x["dwc"].unsqueeze(-1))
        dwc_repr = dwc_values[:, -1]
        token_features = self.token(x["tokens"])
        mask = x["token_mask"]
        scores = self.token_score(token_features).squeeze(-1).masked_fill(
            mask == 0, -1e4
        )
        attention = torch.softmax(scores, dim=1) * mask
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-6)
        event_repr = (token_features * attention.unsqueeze(-1)).sum(1)
        prediction = self.head(torch.cat((dwc_repr, event_repr), dim=-1)).squeeze(-1)
        if return_details:
            return {
                "prediction": prediction,
                "dwc_representation": dwc_repr,
                "event_representation": event_repr,
                "token_attention": attention,
            }
        return prediction
