"""Theory-aligned, prefix-causal Wiener/STL/autoencoder decomposition.

The legacy anchored fusion standardized every candidate to the standard
deviation of the complete observed prefix.  A nearly flat or wrong-direction
Wiener fit could therefore be amplified to the full DI range.  This module
keeps both candidates in their native DI units and uses the detected healthy
stage to anchor Wiener early and STL later.  A temporal autoencoder is retained
as a bounded correction, as required by the D2Pred decomposition framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.causal_decomposition import (
    CausalDecompositionConfig,
    StableStageConfig,
    TemporalTrendAutoencoder,
    rolling_stl_trend,
    stable_stage_wiener_expected_trend,
    increment_rate_wiener_expected_trend,
    online_local_drift_wiener_expected_trend,
    piecewise_linear_wiener_expected_trend,
)


@dataclass(frozen=True)
class StageAwareFusionConfig:
    transition_temperature: float = 0.08
    minimum_transition_fraction: float = 0.0
    stl_post_smoothing_fraction: float = 0.0
    maximum_gate_deviation: float = 0.10
    residual_fraction: float = 0.05
    lowpass_window: int = 21
    candidate_weight: float = 1.0
    anchor_weight: float = 0.25
    low_frequency_weight: float = 1.0
    smoothness_weight: float = 0.05
    epsilon: float = 1e-8


def wiener_reliability(
    di: np.ndarray, wiener: np.ndarray, stl: np.ndarray, metadata: dict[str, Any]
) -> float:
    """Score early Wiener reliability without using observations after the prefix."""

    scale = max(float(np.std(di)), 1e-8)
    fit_score = float(np.exp(-float(metadata["RMSE"]) / scale))
    fit_len = min(len(di), max(2, int(metadata["stable_end"])))
    if fit_len < 2:
        # A one-sample prefix contains no identifiable slope direction.
        direction_score = 1.0
    else:
        time = np.arange(fit_len, dtype=float)
        wiener_slope = float(np.polyfit(time, wiener[:fit_len], 1)[0])
        stl_slope = float(np.polyfit(time, stl[:fit_len], 1)[0])
        direction_score = 1.0 if wiener_slope * stl_slope >= 0.0 else 0.15
    return float(np.clip(fit_score * direction_score, 0.05, 1.0))


def smooth_stl_candidate(values: np.ndarray, fraction: float) -> np.ndarray:
    """Optionally remove event-scale detail from an already extracted STL trend."""

    result = np.asarray(values, dtype=np.float32)
    if fraction <= 0.0 or len(result) < 5:
        return result.copy()
    width = max(3, int(round(len(result) * fraction)))
    width = min(width, 101, len(result) if len(result) % 2 else len(result) - 1)
    width = max(3, width | 1)
    padding = width // 2
    padded = np.pad(result, (padding, padding), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


class StageAwareTrendAutoencoder(nn.Module):
    """Healthy-stage Wiener/STL gate with a bounded temporal-AE correction."""

    def __init__(
        self,
        hidden_channels: int = 32,
        latent_channels: int = 8,
        encoder_depth: int = 2,
        config: StageAwareFusionConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or StageAwareFusionConfig()
        self.gate_delta = nn.Sequential(
            nn.Conv1d(2, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, 1),
        )
        self.residual_autoencoder = TemporalTrendAutoencoder(
            input_channels=2,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            encoder_depth=encoder_depth,
        )
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.encoder_depth = encoder_depth

    def forward_components(
        self,
        trends: torch.Tensor,
        mask: torch.Tensor,
        stable_fraction: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if trends.ndim != 3 or trends.shape[1] != 2:
            raise ValueError("trends must have shape [batch, 2, length]")
        batch, _, width = trends.shape
        if mask.shape != (batch, width):
            raise ValueError("mask must have shape [batch, length]")
        lengths = mask.sum(1).clamp_min(1).to(trends.dtype)
        positions = torch.arange(width, device=trends.device, dtype=trends.dtype)[None]
        normalized_time = positions / (lengths[:, None] - 1.0).clamp_min(1.0)
        transition = stable_fraction.reshape(-1, 1).clamp(0.0, 1.0)
        transition = torch.maximum(
            transition,
            torch.full_like(transition, self.config.minimum_transition_fraction),
        )
        reliability = reliability.reshape(-1, 1).clamp(0.0, 1.0)
        prior_alpha = reliability * torch.sigmoid(
            (transition - normalized_time) / self.config.transition_temperature
        )
        float_mask = mask.to(trends.dtype)
        learned_delta = self.config.maximum_gate_deviation * torch.tanh(
            self.gate_delta(trends * float_mask[:, None]).squeeze(1)
        )
        alpha = (prior_alpha + learned_delta).clamp(0.0, 1.0) * float_mask
        wiener, stl = trends[:, 0], trends[:, 1]
        base = alpha * wiener + (1.0 - alpha) * stl
        disagreement = torch.sqrt(
            (((wiener - stl) ** 2 * float_mask).sum(1) / lengths).clamp_min(
                self.config.epsilon
            )
        )[:, None]
        raw_correction = self.residual_autoencoder(trends, mask)
        correction = (
            self.config.residual_fraction * disagreement * torch.tanh(raw_correction)
        ) * float_mask
        output = (base + correction) * float_mask
        return output, {
            "alpha": alpha,
            "prior_alpha": prior_alpha * float_mask,
            "base": base * float_mask,
            "correction": correction,
            "disagreement_scale": disagreement,
        }

    def forward(
        self,
        trends: torch.Tensor,
        mask: torch.Tensor,
        stable_fraction: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_components(
            trends, mask, stable_fraction, reliability
        )[0]

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "stage_aware_native_scale_wiener_stl_bounded_temporal_ae",
            "hidden_channels": self.hidden_channels,
            "latent_channels": self.latent_channels,
            "encoder_depth": self.encoder_depth,
            "candidate_units": "native_DI_units_no_prefix_std_forcing",
            "gate_prior": "reliable_Wiener_during_detected_healthy_stage_then_STL",
            **self.config.__dict__,
        }


class StlResidualTrendAutoencoder(nn.Module):
    """STL-only temporal AE used to isolate the marginal Wiener contribution.

    The autoencoder receives only the causal STL candidate.  Its correction is
    bounded by the observed-prefix STL scale, so removing Wiener does not also
    remove the AE while the decoder remains unable to replace the STL anchor.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        latent_channels: int = 8,
        encoder_depth: int = 2,
        config: StageAwareFusionConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or StageAwareFusionConfig()
        self.residual_autoencoder = TemporalTrendAutoencoder(
            input_channels=1,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            encoder_depth=encoder_depth,
        )
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.encoder_depth = encoder_depth

    def forward_components(
        self, stl: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if stl.ndim != 2:
            raise ValueError("stl must have shape [batch, length]")
        if mask.shape != stl.shape:
            raise ValueError("mask must have shape [batch, length]")
        weights = mask.to(stl.dtype)
        lengths = weights.sum(1).clamp_min(1.0)
        mean = (stl * weights).sum(1) / lengths
        variance = (((stl - mean[:, None]) ** 2) * weights).sum(1) / lengths
        stl_scale = torch.sqrt(variance.clamp_min(self.config.epsilon))[:, None]
        raw_correction = self.residual_autoencoder(stl[:, None], mask)
        correction = (
            self.config.residual_fraction * stl_scale * torch.tanh(raw_correction)
        ) * weights
        base = stl * weights
        output = (base + correction) * weights
        return output, {
            "base": base,
            "correction": correction,
            "stl_scale": stl_scale,
        }

    def forward(self, stl: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward_components(stl, mask)[0]

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "stl_anchored_bounded_temporal_ae",
            "input_channels": 1,
            "ae_input": "prefix_causal_STL_only",
            "hidden_channels": self.hidden_channels,
            "latent_channels": self.latent_channels,
            "encoder_depth": self.encoder_depth,
            "base": "STL",
            "residual_bound": "residual_fraction_times_prefix_STL_std",
            "raw_DI_is_AE_input": False,
            **self.config.__dict__,
        }


def _masked_lowpass(values: torch.Tensor, mask: torch.Tensor, window: int) -> torch.Tensor:
    window = max(3, int(window) | 1)
    kernel = torch.ones(1, 1, window, device=values.device, dtype=values.dtype)
    numerator = F.conv1d(
        (values * mask).unsqueeze(1), kernel, padding=window // 2
    ).squeeze(1)
    denominator = F.conv1d(
        mask.unsqueeze(1), kernel, padding=window // 2
    ).squeeze(1).clamp_min(1.0)
    return numerator / denominator * mask


def stage_aware_fusion_loss(
    output: torch.Tensor,
    di: torch.Tensor,
    trends: torch.Tensor,
    mask: torch.Tensor,
    details: dict[str, torch.Tensor],
    config: StageAwareFusionConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paper candidate loss plus physical residual and DWC-shape constraints."""

    weights = mask.to(output.dtype)
    count = weights.sum(1).clamp_min(1.0)
    mean = (di * weights).sum(1) / count
    variance = (((di - mean[:, None]) ** 2) * weights).sum(1) / count
    scale = variance.clamp_min(config.epsilon)
    alpha = details["prior_alpha"].detach()
    wiener, stl = trends[:, 0], trends[:, 1]
    candidate_per_sample = (
        (alpha * (output - wiener) ** 2 + (1.0 - alpha) * (output - stl) ** 2)
        * weights
    ).sum(1) / count / scale
    anchor_per_sample = (((output - details["base"]) ** 2) * weights).sum(1) / count / scale
    residual = (di - output) * weights
    low_frequency = _masked_lowpass(residual, weights, config.lowpass_window)
    low_frequency_per_sample = (low_frequency**2).sum(1) / count / scale
    if output.shape[1] >= 3:
        second = output[:, 2:] - 2.0 * output[:, 1:-1] + output[:, :-2]
        second_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        second_count = second_mask.sum(1).clamp_min(1).to(output.dtype)
        smooth_per_sample = (
            (second**2 * second_mask.to(output.dtype)).sum(1) / second_count / scale
        )
    else:
        smooth_per_sample = torch.zeros_like(scale)
    candidate = candidate_per_sample.mean()
    anchor = anchor_per_sample.mean()
    low_frequency_loss = low_frequency_per_sample.mean()
    smoothness = smooth_per_sample.mean()
    loss = (
        config.candidate_weight * candidate
        + config.anchor_weight * anchor
        + config.low_frequency_weight * low_frequency_loss
        + config.smoothness_weight * smoothness
    )
    return loss, {
        "candidate": candidate,
        "anchor": anchor,
        "low_frequency": low_frequency_loss,
        "smoothness": smoothness,
    }


def stl_residual_fusion_loss(
    output: torch.Tensor,
    di: torch.Tensor,
    stl: torch.Tensor,
    mask: torch.Tensor,
    details: dict[str, torch.Tensor],
    config: StageAwareFusionConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match the selected fusion objective with Wiener removed only."""

    weights = mask.to(output.dtype)
    count = weights.sum(1).clamp_min(1.0)
    mean = (di * weights).sum(1) / count
    variance = (((di - mean[:, None]) ** 2) * weights).sum(1) / count
    scale = variance.clamp_min(config.epsilon)
    candidate_per_sample = (((output - stl) ** 2) * weights).sum(1) / count / scale
    anchor_per_sample = (
        ((output - details["base"]) ** 2) * weights
    ).sum(1) / count / scale
    residual = (di - output) * weights
    low_frequency = _masked_lowpass(residual, weights, config.lowpass_window)
    low_frequency_per_sample = (low_frequency**2).sum(1) / count / scale
    if output.shape[1] >= 3:
        second = output[:, 2:] - 2.0 * output[:, 1:-1] + output[:, :-2]
        second_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        second_count = second_mask.sum(1).clamp_min(1).to(output.dtype)
        smooth_per_sample = (
            (second**2 * second_mask.to(output.dtype)).sum(1) / second_count / scale
        )
    else:
        smooth_per_sample = torch.zeros_like(scale)
    candidate = candidate_per_sample.mean()
    anchor = anchor_per_sample.mean()
    low_frequency_loss = low_frequency_per_sample.mean()
    smoothness = smooth_per_sample.mean()
    loss = (
        config.candidate_weight * candidate
        + config.anchor_weight * anchor
        + config.low_frequency_weight * low_frequency_loss
        + config.smoothness_weight * smoothness
    )
    return loss, {
        "candidate": candidate,
        "anchor": anchor,
        "low_frequency": low_frequency_loss,
        "smoothness": smoothness,
    }


def stage_aware_causal_decompose(
    di_prefix: Sequence[float] | np.ndarray,
    fusion_model: StageAwareTrendAutoencoder,
    stable_config: StableStageConfig,
    decomposition_config: CausalDecompositionConfig,
    device: str | torch.device = "cpu",
    return_candidates: bool = False,
    healthy_method: str = "current",
    healthy_params: dict[str, Any] | None = None,
    wiener_mode: str = "standard",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build DWC/EDC from exactly one observed prefix."""

    di = np.asarray(di_prefix, dtype=np.float32).reshape(-1)
    if not len(di) or np.any(~np.isfinite(di)):
        raise ValueError("DI prefix must be non-empty and finite")
    if wiener_mode == "online_local_drift":
        wiener, wiener_meta = online_local_drift_wiener_expected_trend(di, stable_config, decomposition_config, **(healthy_params or {}))
    elif wiener_mode == "piecewise_linear":
        wiener, wiener_meta = piecewise_linear_wiener_expected_trend(di, stable_config, decomposition_config)
    elif healthy_method == "increment_rate":
        wiener, wiener_meta = increment_rate_wiener_expected_trend(di, stable_config, decomposition_config, **(healthy_params or {}))
    else:
        wiener, wiener_meta = stable_stage_wiener_expected_trend(di, stable_config, decomposition_config)
    stl, stl_meta = rolling_stl_trend(di, decomposition_config)
    stl = smooth_stl_candidate(stl, fusion_model.config.stl_post_smoothing_fraction)
    stl_meta = {
        **stl_meta,
        "post_smoothing_fraction": fusion_model.config.stl_post_smoothing_fraction,
    }
    if wiener_mode == "reliability_weighted":
        w_rmse = float(np.sqrt(np.mean((di - wiener) ** 2)))
        s_rmse = float(np.sqrt(np.mean((di - stl) ** 2)))
        weight = float(np.exp(-w_rmse) / (np.exp(-w_rmse) + np.exp(-s_rmse) + 1e-8))
        wiener = (weight * wiener + (1.0 - weight) * stl).astype(np.float32)
        wiener_meta = {**wiener_meta, "detector": "reliability_weighted_wiener_stl", "wiener_weight": weight,
                       "wiener_rmse_raw": w_rmse, "stl_rmse": s_rmse}
    reliability = wiener_reliability(di, wiener, stl, wiener_meta)
    stable_fraction = float(wiener_meta["stable_end"] / len(di))
    target_device = torch.device(device)
    with torch.inference_mode():
        trends = torch.from_numpy(np.stack((wiener, stl)))[None].to(target_device)
        mask = torch.ones((1, len(di)), dtype=torch.bool, device=target_device)
        dwc, details = fusion_model.forward_components(
            trends,
            mask,
            torch.tensor([stable_fraction], dtype=torch.float32, device=target_device),
            torch.tensor([reliability], dtype=torch.float32, device=target_device),
        )
    dwc_np = dwc[0].cpu().numpy().astype(np.float32)
    edc = (di - dwc_np).astype(np.float32)
    metadata = {
        "prefix_length": len(di),
        "causal_prefix_only": True,
        "candidate_units": "native_DI_units",
        "stable_fraction": stable_fraction,
        "wiener_reliability": reliability,
        "wiener": wiener_meta,
        "stl": stl_meta,
        "fusion": fusion_model.architecture_metadata(),
        "alpha_mean": float(details["alpha"].mean().cpu()),
        "correction_rms": float(torch.sqrt(torch.mean(details["correction"] ** 2)).cpu()),
        "edc_identity_max_abs_error": float(np.max(np.abs(edc - (di - dwc_np)))),
    }
    if return_candidates:
        metadata["wiener_candidate"] = np.asarray(wiener, dtype=np.float32)
        metadata["stl_candidate"] = np.asarray(stl, dtype=np.float32)
    return dwc_np, edc, metadata
