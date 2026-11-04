"""Strict ExpertExact-based AC ablation definitions and predictor routing.

This module defines model behavior only. Full remains the frozen comparison
prediction package and is never constructed here for training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from analysis.expert_exact_d2pred import ExpertExactD2Pred
from analysis.theory_aligned_decomposition import StageAwareFusionConfig


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    aliases: tuple[str, ...]
    trend_source: str
    trend_fusion_mode: str
    use_dwc_predictor: bool
    use_edc_predictor: bool
    event_token_mode: str
    cross_branch_fusion_mode: str
    input_mode: str
    decomposition: bool
    event_loss_active: bool
    gradual_loss_active: bool
    trend_loss_active: bool
    joint_trend_fusion: bool
    predictive_path_equivalent_to: str | None = None


VARIANT_SPECS: dict[str, VariantSpec] = {
    "no_wiener": VariantSpec(
        "no_wiener", (), "stl_only", "adaptive_stl_anchored_ae", True, True,
        "adaptive_aep", "dwc_guided_attention", "di_dwc_edc_tokens", True,
        True, True, True, True,
    ),
    "no_stl": VariantSpec(
        "no_stl", (), "wiener_only", "adaptive_wiener_anchored_ae", True, True,
        "adaptive_aep", "dwc_guided_attention", "di_dwc_edc_tokens", True,
        True, True, True, True,
    ),
    "no_autoencoder_fusion": VariantSpec(
        "no_autoencoder_fusion", ("no_adaptive_trend_fusion",), "wiener_and_stl",
        "fixed_equal_average", True, True, "adaptive_aep", "dwc_guided_attention",
        "di_dwc_edc_tokens", True, True, True, False, False,
    ),
    "no_dwc": VariantSpec(
        "no_dwc", (), "wiener_and_stl", "fold_pretrained_adaptive_ae", False,
        True, "adaptive_aep", "event_only_masked_pool", "di_edc_tokens", True,
        True, False, False, False,
    ),
    "no_edc": VariantSpec(
        "no_edc", (), "wiener_and_stl", "fold_pretrained_adaptive_ae", True,
        False, "disabled_no_edc", "dwc_only", "di_dwc", True,
        False, True, True, True,
    ),
    "no_decomposition": VariantSpec(
        "no_decomposition", (), "none", "none", True, True,
        "adaptive_aep_raw_di", "dwc_guided_attention", "raw_di_dual_channel", False,
        True, True, False, False,
    ),
    "no_fusion": VariantSpec(
        "no_fusion", ("no_cross_branch_fusion",), "wiener_and_stl",
        "fold_pretrained_adaptive_ae", True, True, "adaptive_aep",
        "fixed_mean_of_branch_auxiliary_rul", "di_dwc_edc_tokens", True,
        True, True, True, True,
    ),
    "no_aep": VariantSpec(
        "no_aep", (), "wiener_and_stl", "fold_pretrained_adaptive_ae", True,
        True, "single_global_edc_token", "dwc_guided_attention", "di_dwc_edc_tokens",
        True, True, True, True, True,
    ),
}


def resolve_variant(name: str) -> VariantSpec:
    for spec in VARIANT_SPECS.values():
        if name == spec.variant or name in spec.aliases:
            return spec
    raise ValueError(f"Unregistered strict ablation variant: {name}")


def fusion_config_for_variant(name: str) -> StageAwareFusionConfig | None:
    spec = resolve_variant(name)
    if spec.variant == "no_wiener":
        return StageAwareFusionConfig(candidate_source="stl")
    if spec.variant == "no_stl":
        return StageAwareFusionConfig(candidate_source="wiener")
    if spec.joint_trend_fusion:
        return StageAwareFusionConfig()
    return None


class ControlledExpertExactD2Pred(nn.Module):
    """Variant-specific path assembled from ExpertExactD2Pred components."""

    def __init__(self, variant: str, *, token_dimension: int = 12,
                 hidden_dimension: int = 64, attention_heads: int = 4,
                 event_layers: int = 3, maximum_tokens: int = 16,
                 dropout: float = 0.15) -> None:
        super().__init__()
        self.spec = resolve_variant(variant)
        self.variant = self.spec.variant
        spec = self.spec

        use_gradual = spec.use_dwc_predictor
        use_event = spec.use_edc_predictor
        use_attention = spec.cross_branch_fusion_mode == "dwc_guided_attention"
        self.core = ExpertExactD2Pred(
            token_dimension=token_dimension,
            hidden_dimension=hidden_dimension,
            attention_heads=attention_heads,
            event_layers=event_layers,
            maximum_tokens=maximum_tokens,
            dropout=dropout,
            joint_trend_fusion=spec.joint_trend_fusion,
            use_gradual_branch=use_gradual,
            use_event_branch=use_event,
            use_guided_fusion=use_attention,
        )
        self.core.trend_loss_active = spec.trend_loss_active
        self.core.gradual_loss_active = spec.gradual_loss_active
        self.core.event_loss_active = spec.event_loss_active
        trend_config = fusion_config_for_variant(self.variant)
        if self.core.trend_fusion is not None and trend_config is not None:
            self.core.trend_fusion.config = trend_config

        if self.variant == "no_dwc":
            self.core.gradual_encoder = None
            self.core.gradual_projection = None
            self.core.trend_fusion = None
            self.core.gradual_aux = None
            self.core.query = None
            self.core.key = None
            self.core.value = None
            self.core.guided_fusion = None
            self.core.fusion_norm = None
        elif self.variant == "no_edc":
            self._remove_event_modules()
            self.core.event_aux = None
        elif self.variant == "no_fusion":
            self.core.query = None
            self.core.key = None
            self.core.value = None
            self.core.guided_fusion = None
            self.core.fusion_norm = None
            self.core.regression = None

        self.core.architecture_version = f"d2pred_expert_exact_v1+{self.variant}"

    def _remove_event_modules(self) -> None:
        self.core.token_projection = None
        self.core.position = None
        self.core.event_encoder = None
        self.core.query = None
        self.core.key = None
        self.core.value = None
        self.core.guided_fusion = None
        self.core.fusion_norm = None

    def _trend_feature(self, inputs: dict[str, torch.Tensor]):
        if self.core.trend_fusion is None:
            return inputs["dwc"], None
        return self.core.trend_fusion.forward_components(
            inputs["trend_candidates"], inputs["sequence_mask"],
            inputs["stable_fraction"], inputs["wiener_reliability"],
        )

    def _encode_event(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.core.token_projection(inputs["tokens"])
        valid = inputs["token_mask"].bool()
        encoded = self.core.position(tokens)
        encoded = self.core.event_encoder(encoded, src_key_padding_mask=~valid)
        weights = valid.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return encoded, pooled

    def forward(self, inputs: dict[str, torch.Tensor], return_details: bool = False):
        spec = self.spec
        if spec.input_mode == "raw_di_dual_channel":
            trend_input = inputs["di"]
            trend_details = None
        else:
            trend_input, trend_details = self._trend_feature(inputs)

        gradual = None
        if spec.use_dwc_predictor:
            _, (hidden, _) = self.core.gradual_encoder(trend_input.unsqueeze(-1))
            gradual = self.core.gradual_projection(hidden[-1])

        event = None
        encoded_events = None
        attention = None
        if spec.use_edc_predictor:
            encoded_events, event = self._encode_event(inputs)
            if spec.cross_branch_fusion_mode == "dwc_guided_attention":
                attended, attention = self.core.guided_fusion(
                    self.core.query(gradual).unsqueeze(1),
                    self.core.key(encoded_events), self.core.value(encoded_events),
                    key_padding_mask=~inputs["token_mask"].bool(),
                    need_weights=True, average_attn_weights=False,
                )
                event = attended[:, 0]

        gradual_prediction = (
            self.core.gradual_aux(gradual).squeeze(-1)
            if gradual is not None and self.core.gradual_aux is not None else None
        )
        event_prediction = (
            self.core.event_aux(event).squeeze(-1)
            if event is not None and self.core.event_aux is not None else None
        )

        if spec.cross_branch_fusion_mode == "fixed_mean_of_branch_auxiliary_rul":
            prediction = 0.5 * (gradual_prediction + event_prediction)
        elif spec.use_dwc_predictor and spec.use_edc_predictor:
            fused = self.core.fusion_norm(gradual + event)
            prediction = self.core.regression(torch.cat((gradual, fused), dim=-1)).squeeze(-1)
        elif spec.use_dwc_predictor:
            prediction = self.core.regression(gradual).squeeze(-1)
        else:
            prediction = self.core.regression(event).squeeze(-1)

        if not return_details:
            return prediction
        return {
            "prediction": prediction,
            "gradual_prediction": gradual_prediction,
            "event_prediction": event_prediction,
            "gradual_feature": gradual,
            "event_feature": event,
            "event_features": encoded_events,
            "cross_attention": attention,
            "fused_dwc": trend_input,
            "trend_fusion_details": trend_details,
        }

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "architecture_version": self.core.architecture_version,
            "parent_model": "d2pred_expert_exact_v1",
            "variant": self.variant,
            "configuration": asdict(self.spec),
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "trend_fusion": (
                self.core.trend_fusion.architecture_metadata()
                if self.core.trend_fusion is not None else None
            ),
            "output": "ExpertExact train-scale nonnegative RUL path; branch control as registered",
        }
