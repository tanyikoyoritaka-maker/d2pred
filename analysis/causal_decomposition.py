"""Leakage-free prefix decomposition for D2Pred experiments.

This module is the canonical implementation of health-state selection, Wiener
expected trends, rolling STL, and temporal encoder-decoder fusion.  Every public
entry point accepts one *observed prefix* and returns an output of the same length.
It deliberately has no RUL or failure-cycle argument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


class MissingHealthStateError(RuntimeError):
    """Raised when a dataset cannot support the specified physical SoH rule."""

    def __init__(self, dataset: str, missing_fields: Sequence[str], detail: str) -> None:
        self.dataset = dataset
        self.missing_fields = tuple(missing_fields)
        self.detail = detail
        super().__init__(
            f"{dataset}: physically valid SoH unavailable; missing fields: "
            f"{', '.join(self.missing_fields)}. {detail}"
        )


@dataclass(frozen=True)
class HealthStateSeries:
    """Observed SoH prefix in percentage points (new state is 100)."""

    soh_percent: np.ndarray
    source: str
    definition: str
    threshold_percent: float = 80.0


@dataclass(frozen=True)
class HealthStateAdapter:
    """Explicit schema adapter; it never infers health from DI, HI, RUL, or EoL."""

    dataset: str
    soh_column: str | None = None
    soh_scale: str = "percent"
    actual_capacity_column: str | None = None
    nominal_capacity_column: str | None = None
    documented_semantics: str | None = None

    def extract(self, table: Any) -> HealthStateSeries:
        columns = set(getattr(table, "columns", ()))
        if self.soh_column and self.soh_column in columns and self.documented_semantics:
            values = np.asarray(table[self.soh_column], dtype=float)
            if self.soh_scale == "fraction":
                values = values * 100.0
            elif self.soh_scale != "percent":
                raise ValueError(f"Unsupported SoH scale: {self.soh_scale}")
            return self._validated(values, f"existing_project_field:{self.soh_column}")

        capacity_fields = (self.actual_capacity_column, self.nominal_capacity_column)
        if all(capacity_fields) and all(field in columns for field in capacity_fields):
            actual = np.asarray(table[self.actual_capacity_column], dtype=float)
            nominal = np.asarray(table[self.nominal_capacity_column], dtype=float)
            if np.any(~np.isfinite(nominal)) or np.any(nominal <= 0):
                raise ValueError(f"{self.dataset}: nominal capacity must be finite and positive")
            return self._validated(
                actual / nominal * 100.0,
                f"capacity_ratio:{self.actual_capacity_column}/{self.nominal_capacity_column}",
            )

        expected = []
        if self.soh_column:
            expected.append(self.soh_column)
        if self.actual_capacity_column:
            expected.append(self.actual_capacity_column)
        else:
            expected.append("Q_actual (actual capacity at current state)")
        if self.nominal_capacity_column:
            expected.append(self.nominal_capacity_column)
        else:
            expected.append("Q_nominal (new-unit nominal capacity)")
        raise MissingHealthStateError(
            self.dataset,
            expected,
            "No documented project SoH field, capacity ratio, or manufacturer/training-only "
            "resistance calibration is present. DI, HI, RUL, failure cycle, and final values "
            "are explicitly disallowed substitutes.",
        )

    def _validated(self, values: np.ndarray, source: str) -> HealthStateSeries:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError(f"{self.dataset}: SoH prefix must be non-empty and finite")
        if np.any(values < 0):
            raise ValueError(f"{self.dataset}: SoH cannot be negative")
        return HealthStateSeries(
            soh_percent=values,
            source=source,
            definition=self.documented_semantics
            or "SoH_t = Q_actual,t / Q_nominal * 100%",
        )


@dataclass(frozen=True)
class CausalDecompositionConfig:
    health_threshold_percent: float = 80.0
    stl_period: int | None = None
    stl_min_period: int = 2
    stl_max_period: int = 30
    stl_min_cycles: int = 2
    short_prefix_window: int = 5
    clip_trends_to_prefix_range: bool = False
    normalization_epsilon: float = 1e-8

    def hash(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StableStageConfig:
    """Online Adaptive Initial Stable Degradation Stage detector settings."""

    k_min: int = 20
    gamma_r: float = 4.0
    gamma_s: float = 4.0
    rolling_window: int = 10
    consecutive_q: int = 3
    huber_delta: float = 1.345
    irls_iterations: int = 30
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.k_min < 2 or self.rolling_window < 2 or self.consecutive_q < 1:
            raise ValueError("k_min/window must be >=2 and consecutive_q must be positive")
        if self.gamma_r <= 0 or self.gamma_s <= 0 or self.huber_delta <= 0:
            raise ValueError("Stable-stage multipliers and Huber delta must be positive")

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


STABLE_STAGE_GRID = (
    StableStageConfig(k_min=12, gamma_r=3.0, gamma_s=3.0, rolling_window=8, consecutive_q=2),
    StableStageConfig(k_min=12, gamma_r=4.0, gamma_s=4.0, rolling_window=8, consecutive_q=3),
    StableStageConfig(k_min=20, gamma_r=3.0, gamma_s=3.0, rolling_window=10, consecutive_q=2),
    StableStageConfig(k_min=20, gamma_r=4.0, gamma_s=4.0, rolling_window=12, consecutive_q=3),
)


def _mad(values: np.ndarray, epsilon: float) -> float:
    if not len(values):
        return epsilon
    median = float(np.median(values))
    return max(1.4826 * float(np.median(np.abs(values - median))), epsilon)


def huber_irls_line(
    values: Sequence[float] | np.ndarray,
    delta: float = 1.345,
    iterations: int = 30,
    epsilon: float = 1e-8,
) -> tuple[float, float, np.ndarray]:
    """Small deterministic Huber IRLS line fit with no external estimator state."""

    y = _clean_prefix(values, "stable-stage fit values")
    x = np.arange(len(y), dtype=float)
    design = np.column_stack((np.ones(len(y)), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(iterations):
        residual = y - design @ beta
        scale = _mad(residual, epsilon)
        normalized = np.abs(residual) / (delta * scale)
        weights = np.ones_like(normalized)
        large = normalized > 1.0
        weights[large] = 1.0 / normalized[large]
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        updated = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        if np.max(np.abs(updated - beta)) <= epsilon:
            beta = updated
            break
        beta = updated
    return float(beta[0]), float(beta[1]), y - design @ beta


def detect_initial_stable_stage(
    di_prefix: Sequence[float] | np.ndarray,
    config: StableStageConfig,
) -> tuple[int, dict[str, Any]]:
    """Detect the first persistent departure using only the supplied DI prefix."""

    di = _clean_prefix(di_prefix, "DI prefix")
    fit_len = min(len(di), config.k_min)
    intercept, slope, initial_residual = huber_irls_line(
        di[:fit_len], config.huber_delta, config.irls_iterations, config.epsilon
    )
    base_mad = _mad(initial_residual, config.epsilon)
    insufficient = len(di) < config.k_min
    stable_end = len(di)
    boundary_detected = False
    residual_threshold = config.gamma_r * base_mad
    slope_scale = max(base_mad / max(config.rolling_window, 2),
                      abs(slope) * 0.05, config.epsilon)
    slope_threshold = config.gamma_s * slope_scale
    consecutive = 0
    first_exceedance = None
    if not insufficient:
        all_time = np.arange(len(di), dtype=float)
        residuals = di - (intercept + slope * all_time)
        for index in range(config.k_min, len(di)):
            start = max(0, index - config.rolling_window + 1)
            residual_window = residuals[start : index + 1]
            rolling_mad = max(_mad(residual_window, config.epsilon), base_mad)
            residual_threshold = config.gamma_r * rolling_mad
            local = di[start : index + 1]
            _, local_slope, _ = huber_irls_line(
                local, config.huber_delta, config.irls_iterations, config.epsilon
            )
            slope_threshold = config.gamma_s * max(
                rolling_mad / max(len(local), 2), slope_scale
            )
            exceeds = (
                abs(float(residuals[index])) > residual_threshold
                or abs(local_slope - slope) > slope_threshold
            )
            if exceeds:
                consecutive += 1
                if first_exceedance is None:
                    first_exceedance = index
            else:
                consecutive = 0
                first_exceedance = None
            if consecutive >= config.consecutive_q:
                stable_end = int(first_exceedance)
                boundary_detected = True
                break
    fallback_status = (
        "insufficient_initial_points" if insufficient else
        "boundary_not_yet_detected_use_observed_prefix" if not boundary_detected else "none"
    )
    diagnostic = {
        "stable_end": stable_end,
        "fit_len": stable_end,
        "baseline_intercept": intercept,
        "baseline_slope": slope,
        "residual_mad": base_mad,
        "slope_threshold": float(slope_threshold),
        "residual_threshold": float(residual_threshold),
        "boundary_detected": boundary_detected,
        "fallback_status": fallback_status,
        "insufficient_initial_points": insufficient,
        "prefix_length": len(di),
        "detector_config_hash": config.hash(),
    }
    return stable_end, diagnostic


def stable_stage_wiener_expected_trend(
    di_prefix: Sequence[float] | np.ndarray,
    stable_config: StableStageConfig,
    decomposition_config: CausalDecompositionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic Wiener expectation fitted to the detected stable stage."""

    di = _clean_prefix(di_prefix, "DI prefix")
    stable_end, detector = detect_initial_stable_stage(di, stable_config)
    fit_len = stable_end
    if fit_len >= 2:
        beta_0, mu, residual = huber_irls_line(
            di[:fit_len], stable_config.huber_delta,
            stable_config.irls_iterations, stable_config.epsilon
        )
        sigma = float(np.std(residual, ddof=0))
        rmse = float(np.sqrt(np.mean(residual**2)))
        fallback = detector["fallback_status"]
    else:
        beta_0, mu, sigma, rmse = float(di[0]), 0.0, 0.0, 0.0
        fallback = "constant_trend_fewer_than_two_stable_points"
    trend = beta_0 + mu * np.arange(len(di), dtype=float)
    if decomposition_config.clip_trends_to_prefix_range:
        trend = np.clip(trend, np.min(di), np.max(di))
    return trend.astype(np.float32), {
        **detector, "fit_len": fit_len, "beta_0": beta_0, "mu": mu,
        "sigma": sigma, "RMSE": rmse, "fallback_status": fallback,
    }


def _clean_prefix(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must contain at least one observed sample")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite observed samples")
    return result


def increment_rate_wiener_expected_trend(
    di_prefix: Sequence[float] | np.ndarray,
    stable_config: StableStageConfig,
    decomposition_config: CausalDecompositionConfig,
    initial_window: int = 20,
    rate_mad_multiplier: float = 3.0,
    consecutive_violations: int = 2,
    minimum_rate_sign_consistency: float = 0.8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Causal healthy-stage detector based on increment/rate consistency."""
    di = _clean_prefix(di_prefix, "DI prefix")
    fit_len = min(len(di), max(2, int(initial_window)))
    increments = np.diff(di)
    base = increments[:max(1, fit_len - 1)]
    med = float(np.median(base)) if len(base) else 0.0
    mad = max(_mad(base, decomposition_config.normalization_epsilon), decomposition_config.normalization_epsilon)
    direction = 1.0 if med >= 0 else -1.0
    violations = 0; first = None; stable_end = len(di)
    for idx in range(fit_len, len(di)):
        delta = float(increments[idx - 1])
        window = increments[max(0, idx - fit_len):idx]
        local_mad = max(_mad(window, decomposition_config.normalization_epsilon), mad)
        rate_ok = abs(delta - med) <= rate_mad_multiplier * local_mad
        sign_ratio = float(np.mean(direction * window >= 0.0)) if len(window) else 1.0
        bad = (not rate_ok) or sign_ratio < minimum_rate_sign_consistency
        if bad:
            if first is None: first = idx
            violations += 1
        else:
            first = None; violations = 0
        if violations >= consecutive_violations:
            stable_end = int(first); break
    beta_0, mu, residual = huber_irls_line(
        di[:max(2, stable_end)], stable_config.huber_delta,
        stable_config.irls_iterations, stable_config.epsilon
    )
    trend = beta_0 + mu * np.arange(len(di), dtype=float)
    if decomposition_config.clip_trends_to_prefix_range:
        trend = np.clip(trend, np.min(di), np.max(di))
    return trend.astype(np.float32), {
        "stable_end": stable_end, "fit_len": max(2, stable_end), "beta_0": beta_0,
        "mu": mu, "sigma": float(np.std(residual)),
        "RMSE": float(np.sqrt(np.mean(residual**2))), "fallback_status": "none",
        "detector": "increment_rate_consistency", "initial_window": initial_window,
        "rate_mad_multiplier": rate_mad_multiplier,
        "consecutive_violations": consecutive_violations,
        "minimum_rate_sign_consistency": minimum_rate_sign_consistency,
    }


def online_local_drift_wiener_expected_trend(di_prefix, stable_config, decomposition_config,
                                             local_window: int = 12, **_ignored):
    """Prefix-causal Wiener trend with a robust rolling local drift estimate."""
    di = _clean_prefix(di_prefix, "DI prefix")
    base_end, detector = detect_initial_stable_stage(di, stable_config)
    trend = np.empty(len(di), dtype=float)
    for end in range(1, len(di) + 1):
        start = max(0, end - max(2, int(local_window)))
        y = di[start:end]
        if len(y) >= 2:
            b0, mu, _ = huber_irls_line(y, stable_config.huber_delta, stable_config.irls_iterations, stable_config.epsilon)
            trend[end - 1] = b0 + mu * (len(y) - 1)
        else:
            trend[end - 1] = y[-1]
    if decomposition_config.clip_trends_to_prefix_range:
        trend = np.clip(trend, np.min(di), np.max(di))
    return trend.astype(np.float32), {**detector, "stable_end": base_end, "fit_len": base_end,
                                      "detector": "online_local_drift", "local_window": local_window,
                                      "beta_0": float(trend[0]), "mu": float(np.polyfit(np.arange(len(di)), trend, 1)[0]) if len(di)>1 else 0.0,
                                      "sigma": float(np.std(di-trend)), "RMSE": float(np.sqrt(np.mean((di-trend)**2)))}


def piecewise_linear_wiener_expected_trend(di_prefix, stable_config, decomposition_config):
    """Prefix-causal two-segment robust linear Wiener trend."""
    di = _clean_prefix(di_prefix, "DI prefix")
    stable_end, detector = detect_initial_stable_stage(di, stable_config)
    split = max(2, min(len(di)-1, stable_end)) if len(di) > 2 else len(di)
    b0, mu0, _ = huber_irls_line(di[:split], stable_config.huber_delta, stable_config.irls_iterations, stable_config.epsilon)
    if len(di) - split >= 2:
        b1, mu1, _ = huber_irls_line(di[split:], stable_config.huber_delta, stable_config.irls_iterations, stable_config.epsilon)
    else:
        b1, mu1 = b0 + mu0 * split, mu0
    trend = np.empty(len(di), dtype=float)
    trend[:split] = b0 + mu0 * np.arange(split)
    trend[split:] = (b0 + mu0 * split) + mu1 * np.arange(len(di)-split)
    if decomposition_config.clip_trends_to_prefix_range:
        trend = np.clip(trend, np.min(di), np.max(di))
    residual = di-trend
    return trend.astype(np.float32), {**detector, "stable_end": split, "fit_len": len(di), "detector": "piecewise_linear_wiener",
                                      "beta_0": b0, "mu": mu1, "mu_initial": mu0, "sigma": float(np.std(residual)),
                                      "RMSE": float(np.sqrt(np.mean(residual**2)))}


def initial_healthy_interval(
    soh_percent: Sequence[float] | np.ndarray, threshold_percent: float = 80.0
) -> tuple[int, bool]:
    """Return the exclusive end of the initial contiguous healthy interval."""

    soh = _clean_prefix(soh_percent, "SoH prefix")
    below = np.flatnonzero(soh < threshold_percent)
    healthy_end = int(below[0]) if below.size else len(soh)
    return healthy_end, healthy_end < 2


def wiener_expected_trend(
    di_prefix: Sequence[float] | np.ndarray,
    health_prefix: HealthStateSeries,
    config: CausalDecompositionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a deterministic Wiener expected path from observed healthy samples only."""

    di = _clean_prefix(di_prefix, "DI prefix")
    soh = _clean_prefix(health_prefix.soh_percent, "SoH prefix")
    if len(di) != len(soh):
        raise ValueError("DI and SoH prefixes must have identical lengths")
    healthy_end, fallback = initial_healthy_interval(
        soh, config.health_threshold_percent
    )
    fit_len = healthy_end
    if fit_len >= 2:
        time = np.arange(fit_len, dtype=float)
        design = np.column_stack((np.ones(fit_len), time))
        beta_0, mu = np.linalg.lstsq(design, di[:fit_len], rcond=None)[0]
        residual = di[:fit_len] - (beta_0 + mu * time)
        sigma = float(np.std(residual, ddof=0))
        rmse = float(np.sqrt(np.mean(residual**2)))
        fallback_status = "none"
    else:
        # Required deterministic fallback: never extend past the SoH threshold.
        beta_0 = float(di[0])
        mu = 0.0
        residual = di[:fit_len] - beta_0
        sigma = float(np.std(residual, ddof=0)) if fit_len else 0.0
        rmse = float(np.sqrt(np.mean(residual**2))) if fit_len else 0.0
        fallback_status = "constant_trend_fewer_than_two_healthy_observations"
    trend = beta_0 + mu * np.arange(len(di), dtype=float)
    if config.clip_trends_to_prefix_range:
        trend = np.clip(trend, np.min(di), np.max(di))
    metadata = {
        "healthy_end": healthy_end,
        "fit_len": fit_len,
        "beta_0": float(beta_0),
        "mu": float(mu),
        "sigma": sigma,
        "RMSE": rmse,
        "fallback": bool(fallback),
        "fallback_status": fallback_status,
        "health_source": health_prefix.source,
        "health_definition": health_prefix.definition,
        "health_threshold_percent": config.health_threshold_percent,
        "prefix_length": len(di),
    }
    return trend.astype(np.float32), metadata


def _causal_short_prefix_trend(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing-mean fallback using observations available at each sample."""

    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - max(1, window) + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


def rolling_stl_trend(
    di_prefix: Sequence[float] | np.ndarray,
    config: CausalDecompositionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run robust STL on exactly one observed prefix, with a causal short fallback."""

    di = _clean_prefix(di_prefix, "DI prefix")
    if config.stl_period is None:
        period = max(config.stl_min_period, len(di) // 4)
        period = min(config.stl_max_period, period)
    else:
        period = int(config.stl_period)
    sufficiently_long = len(di) >= max(2 * period + 1, config.stl_min_cycles * period)
    fallback_status = "none"
    method = "statsmodels_robust_stl_prefix_only"
    if sufficiently_long:
        try:
            from statsmodels.tsa.seasonal import STL

            trend = np.asarray(STL(di, period=period, robust=True).fit().trend)
        except (ImportError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            trend = _causal_short_prefix_trend(di, config.short_prefix_window)
            method = "causal_trailing_mean_fallback"
            fallback_status = f"stl_failure:{type(exc).__name__}"
    else:
        trend = _causal_short_prefix_trend(di, config.short_prefix_window)
        method = "causal_trailing_mean_fallback"
        fallback_status = "prefix_too_short_for_two_stl_periods"
    if config.clip_trends_to_prefix_range:
        trend = np.clip(trend, np.min(di), np.max(di))
    return trend.astype(np.float32), {
        "period": period,
        "method": method,
        "prefix_length": len(di),
        "fallback": fallback_status != "none",
        "fallback_status": fallback_status,
    }


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values + self.block(values)) * mask


class TemporalTrendAutoencoder(nn.Module):
    """Masked temporal encoder-decoder with a channel-compressed latent sequence."""

    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: int = 32,
        latent_channels: int = 8,
        encoder_depth: int = 2,
    ) -> None:
        super().__init__()
        if min(input_channels, hidden_channels, latent_channels, encoder_depth) < 1:
            raise ValueError("Autoencoder dimensions and depth must be positive")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.encoder_depth = encoder_depth
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, 3, padding=1)
        self.encoder = nn.ModuleList(
            [_TemporalBlock(hidden_channels, 2**index) for index in range(encoder_depth)]
        )
        self.to_latent = nn.Conv1d(hidden_channels, latent_channels, 3, padding=1)
        self.from_latent = nn.Conv1d(latent_channels, hidden_channels, 3, padding=1)
        self.decoder = nn.ModuleList(
            [_TemporalBlock(hidden_channels, 2**index) for index in reversed(range(encoder_depth))]
        )
        self.output_projection = nn.Conv1d(hidden_channels, 1, 3, padding=1)

    def forward(self, trends: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if trends.ndim != 3 or trends.shape[1] != self.input_channels:
            raise ValueError("Expected trends with shape [batch, 2, length]")
        if mask is None:
            mask = torch.ones(
                trends.shape[0], trends.shape[-1], dtype=torch.bool, device=trends.device
            )
        if mask.shape != (trends.shape[0], trends.shape[-1]):
            raise ValueError("Mask must have shape [batch, length]")
        float_mask = mask.unsqueeze(1).to(dtype=trends.dtype)
        encoded = torch.relu(self.input_projection(trends * float_mask)) * float_mask
        for block in self.encoder:
            encoded = block(encoded, float_mask)
        latent = torch.relu(self.to_latent(encoded)) * float_mask
        decoded = torch.relu(self.from_latent(latent)) * float_mask
        for block in self.decoder:
            decoded = block(decoded, float_mask)
        return self.output_projection(decoded).squeeze(1) * mask.to(dtype=trends.dtype)

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "masked_temporal_conv_encoder_decoder",
            "input_channels": self.input_channels,
            "hidden_channels": self.hidden_channels,
            "latent_channels": self.latent_channels,
            "encoder_depth": self.encoder_depth,
            "kernel_size": 3,
            "latent_representation": "length-preserving channel-compressed temporal sequence",
            "output_channels": 1,
            "variable_length_support": "boolean padding mask applied after every block",
        }


class AnchoredTemporalTrendAutoencoder(nn.Module):
    """Wiener/STL fusion with a mandatory convex anchor and bounded AE residual.

    The temporal autoencoder is retained, but it can only correct a convex
    Wiener/STL base by a bounded fraction of the candidate disagreement scale.
    Consequently the decoder cannot freely replace both trends with a zero line.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        latent_channels: int = 8,
        encoder_depth: int = 2,
        residual_fraction: float = 0.1,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if not 0.0 <= residual_fraction <= 1.0:
            raise ValueError("residual_fraction must be in [0, 1]")
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.encoder_depth = encoder_depth
        self.residual_fraction = residual_fraction
        self.epsilon = epsilon
        self.gate = nn.Sequential(
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

    def forward_components(
        self, trends: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if trends.ndim != 3 or trends.shape[1] != 2:
            raise ValueError("Expected Wiener/STL trends with shape [batch, 2, length]")
        if mask is None:
            mask = torch.ones(
                trends.shape[0], trends.shape[-1], dtype=torch.bool, device=trends.device
            )
        if mask.shape != (trends.shape[0], trends.shape[-1]):
            raise ValueError("Mask must have shape [batch, length]")
        weights = mask.to(dtype=trends.dtype)
        masked = trends * weights.unsqueeze(1)
        alpha = torch.sigmoid(self.gate(masked).squeeze(1))
        wiener, stl = trends[:, 0], trends[:, 1]
        base = alpha * wiener + (1.0 - alpha) * stl
        denominator = weights.sum(1, keepdim=True).clamp_min(1.0)
        disagreement_scale = torch.sqrt(
            torch.sum((wiener - stl) ** 2 * weights, dim=1, keepdim=True)
            / denominator
            + self.epsilon
        )
        raw_residual = self.residual_autoencoder(trends, mask)
        correction = (
            self.residual_fraction * disagreement_scale * torch.tanh(raw_residual)
        )
        output = (base + correction) * weights
        return output, {
            "alpha": alpha * weights,
            "base": base * weights,
            "correction": correction * weights,
            "disagreement_scale": disagreement_scale,
        }

    def forward(
        self, trends: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_components(trends, mask)[0]

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "anchored_convex_wiener_stl_temporal_ae_residual",
            "input_channels": 2,
            "hidden_channels": self.hidden_channels,
            "latent_channels": self.latent_channels,
            "encoder_depth": self.encoder_depth,
            "residual_fraction": self.residual_fraction,
            "gate": "time_varying_sigmoid_convex_weight",
            "base": "alpha*Wiener+(1-alpha)*STL",
            "residual": "bounded_temporal_autoencoder_correction",
            "output_channels": 1,
        }


def masked_fusion_loss(
    output: torch.Tensor,
    wiener: torch.Tensor,
    stl: torch.Tensor,
    mask: torch.Tensor,
    lambda_wiener: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Manuscript L_DWC with padding excluded from both MSE terms."""

    if not 0.0 <= lambda_wiener <= 1.0:
        raise ValueError("lambda_wiener must be in [0, 1]")
    weights = mask.to(dtype=output.dtype)
    denominator = weights.sum().clamp_min(1.0)
    wiener_mse = torch.sum((output - wiener) ** 2 * weights) / denominator
    stl_mse = torch.sum((output - stl) ** 2 * weights) / denominator
    loss = lambda_wiener * wiener_mse + (1.0 - lambda_wiener) * stl_mse
    return loss, {"wiener_mse": wiener_mse, "stl_mse": stl_mse}


def anchored_fusion_loss(
    output: torch.Tensor,
    wiener: torch.Tensor,
    stl: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
    lambda_wiener: float = 0.5,
    anchor_weight: float = 0.1,
    dynamic_weight: float = 5.0,
    smoothness_weight: float = 0.01,
    minimum_dynamic_ratio: float = 0.35,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Anchored AEFusion objective with an explicit flat-line penalty."""

    if not 0.0 <= lambda_wiener <= 1.0:
        raise ValueError("lambda_wiener must be in [0, 1]")
    if min(anchor_weight, dynamic_weight, smoothness_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if not 0.0 <= minimum_dynamic_ratio <= 1.0:
        raise ValueError("minimum_dynamic_ratio must be in [0, 1]")
    weights = mask.to(dtype=output.dtype)
    denominator = weights.sum().clamp_min(1.0)
    wiener_mse = torch.sum((output - wiener) ** 2 * weights) / denominator
    stl_mse = torch.sum((output - stl) ** 2 * weights) / denominator
    candidate_loss = (
        lambda_wiener * wiener_mse + (1.0 - lambda_wiener) * stl_mse
    )
    anchor_mse = torch.sum((output - base) ** 2 * weights) / denominator

    per_sample_count = weights.sum(1, keepdim=True).clamp_min(1.0)
    def masked_std(values: torch.Tensor) -> torch.Tensor:
        mean = torch.sum(values * weights, dim=1, keepdim=True) / per_sample_count
        variance = torch.sum((values - mean) ** 2 * weights, dim=1) / per_sample_count.squeeze(1)
        return torch.sqrt(variance.clamp_min(1e-12))

    output_std = masked_std(output)
    candidate_std = torch.maximum(masked_std(wiener), masked_std(stl))
    required_std = minimum_dynamic_ratio * candidate_std.detach()
    dynamic_floor = torch.mean(torch.relu(required_std - output_std) ** 2)

    pair_mask = (mask[:, 1:] & mask[:, :-1]).to(dtype=output.dtype)
    pair_count = pair_mask.sum().clamp_min(1.0)
    first_difference = output[:, 1:] - output[:, :-1]
    smoothness = torch.sum(first_difference**2 * pair_mask) / pair_count
    loss = (
        candidate_loss
        + anchor_weight * anchor_mse
        + dynamic_weight * dynamic_floor
        + smoothness_weight * smoothness
    )
    return loss, {
        "wiener_mse": wiener_mse,
        "stl_mse": stl_mse,
        "candidate_loss": candidate_loss,
        "anchor_mse": anchor_mse,
        "dynamic_floor": dynamic_floor,
        "smoothness": smoothness,
        "output_std_mean": output_std.mean(),
        "required_std_mean": required_std.mean(),
    }


def freeze_fusion_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def prefix_scale(trend: np.ndarray, di_prefix: np.ndarray, epsilon: float) -> np.ndarray:
    """Map a trend to DI units using statistics of the observed prefix only."""

    trend_std = float(np.std(trend))
    if trend_std <= epsilon:
        return np.full(len(trend), float(np.mean(di_prefix)), dtype=np.float32)
    standardized = (trend - float(np.mean(trend))) / (trend_std + epsilon)
    return (standardized * float(np.std(di_prefix)) + float(np.mean(di_prefix))).astype(
        np.float32
    )


def causal_decompose(
    di_prefix: Sequence[float] | np.ndarray,
    health_prefix: HealthStateSeries,
    fusion_model: TemporalTrendAutoencoder,
    config: CausalDecompositionConfig,
    variant: str = "full",
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute prefix-only DWC and EDC; output lengths always equal prefix length."""

    di = _clean_prefix(di_prefix, "DI prefix").astype(np.float32)
    if len(health_prefix.soh_percent) != len(di):
        raise ValueError("Health information must be sliced to the same observed endpoint")
    wiener, wiener_meta = wiener_expected_trend(di, health_prefix, config)
    stl, stl_meta = rolling_stl_trend(di, config)
    wiener_scaled = prefix_scale(wiener, di, config.normalization_epsilon)
    stl_scaled = prefix_scale(stl, di, config.normalization_epsilon)
    if variant == "no_wiener":
        dwc = stl_scaled
        method = "stl_only"
    elif variant == "no_stl":
        dwc = wiener_scaled
        method = "wiener_only"
    elif variant == "no_autoencoder_fusion":
        dwc = 0.5 * (wiener_scaled + stl_scaled)
        method = "fixed_equal_average"
    elif variant == "no_decomposition":
        dwc = di.copy()
        method = "no_decomposition"
    elif variant == "full":
        if fusion_model.training or any(p.requires_grad for p in fusion_model.parameters()):
            raise RuntimeError("Fusion model must be frozen before validation/test decomposition")
        model_device = torch.device(device)
        with torch.inference_mode():
            inputs = torch.from_numpy(np.stack((wiener_scaled, stl_scaled)))[None].to(
                model_device
            )
            mask = torch.ones((1, len(di)), dtype=torch.bool, device=model_device)
            dwc = fusion_model(inputs, mask).squeeze(0).cpu().numpy()
        method = "masked_temporal_autoencoder_fusion"
    else:
        raise ValueError(f"Unknown decomposition variant: {variant}")
    dwc = np.asarray(dwc, dtype=np.float32).reshape(-1)
    if len(dwc) != len(di):
        raise AssertionError("Fusion output length differs from observed prefix length")
    edc = (di - dwc).astype(np.float32)
    metadata: dict[str, Any] = {
        "prefix_length": len(di),
        "variant": variant,
        "method": method,
        "causal_prefix_only": True,
        "configuration_hash": config.hash(),
        "wiener": wiener_meta,
        "stl": stl_meta,
        "fusion": fusion_model.architecture_metadata(),
        "normalization": "observed_prefix_only_mean_and_standard_deviation",
        "edc_identity_max_abs_error": float(np.max(np.abs(edc - (di - dwc)))),
    }
    return dwc, edc, metadata


def stable_stage_causal_decompose(
    di_prefix: Sequence[float] | np.ndarray,
    fusion_model: TemporalTrendAutoencoder,
    stable_config: StableStageConfig,
    decomposition_config: CausalDecompositionConfig,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Canonical adaptive-stable-stage Wiener/STL/fusion decomposition."""

    di = _clean_prefix(di_prefix, "DI prefix").astype(np.float32)
    wiener, wiener_meta = stable_stage_wiener_expected_trend(
        di, stable_config, decomposition_config
    )
    stl, stl_meta = rolling_stl_trend(di, decomposition_config)
    wiener_scaled = prefix_scale(wiener, di, decomposition_config.normalization_epsilon)
    stl_scaled = prefix_scale(stl, di, decomposition_config.normalization_epsilon)
    if fusion_model.training or any(p.requires_grad for p in fusion_model.parameters()):
        raise RuntimeError("Fusion model must be frozen for decomposition inference")
    model_device = torch.device(device)
    with torch.inference_mode():
        inputs = torch.from_numpy(np.stack((wiener_scaled, stl_scaled)))[None].to(
            model_device
        )
        mask = torch.ones((1, len(di)), dtype=torch.bool, device=model_device)
        dwc = fusion_model(inputs, mask).squeeze(0).cpu().numpy().astype(np.float32)
    if len(dwc) != len(di):
        raise AssertionError("Fusion output length differs from prefix length")
    edc = (di - dwc).astype(np.float32)
    metadata = {
        "prefix_length": len(di), "causal_prefix_only": True,
        "decomposition_config_hash": decomposition_config.hash(),
        "stable_config_hash": stable_config.hash(), "stable_stage": wiener_meta,
        "wiener": {key: wiener_meta[key] for key in
                   ("stable_end", "fit_len", "beta_0", "mu", "sigma", "RMSE",
                    "fallback_status")},
        "stl": stl_meta, "fusion": fusion_model.architecture_metadata(),
        "normalization": "observed_prefix_only_mean_and_standard_deviation",
        "edc_identity_max_abs_error": float(np.max(np.abs(edc - (di - dwc)))),
    }
    return dwc, edc, metadata


def causal_cache_key(
    dataset: str, unit: str, tau: int, variant: str, config: CausalDecompositionConfig
) -> str:
    payload: Mapping[str, Any] = {
        "dataset": dataset,
        "unit": unit,
        "tau": int(tau),
        "variant": variant,
        "configuration_hash": config.hash(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
