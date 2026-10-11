"""Non-collapsing DWC/EDC decomposition for the independent-EDC model.

The historical Wiener/STL/autoencoder path can map two centred candidates to an
almost constant zero output.  This module deliberately has no unconstrained
fusion network.  DWC is a robust Whittaker trend and EDC is its exact residual.

For one observed prefix ``x`` the trend solves

    min_z sum_t w_t (x_t - z_t)^2 + lambda * sum_t (Delta^2 z_t)^2.

Constant and linear sequences are in the null space of ``Delta^2`` and are
therefore reproduced exactly.  Huber IRLS downweights event-like residuals.
Every entry point consumes only the supplied prefix; callers must recompute it
at each prediction endpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np
from scipy.linalg import solveh_banded


@dataclass(frozen=True)
class RobustTrendConfig:
    """Settings for robust second-order trend filtering."""

    smoothness: float = 10_000.0
    reference_length: int = 128
    huber_delta: float = 1.5
    irls_iterations: int = 4
    convergence_tolerance: float = 1e-5
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.smoothness < 0:
            raise ValueError("smoothness must be non-negative")
        if self.reference_length < 3:
            raise ValueError("reference_length must be at least three")
        if self.huber_delta <= 0 or self.irls_iterations < 1:
            raise ValueError("huber_delta and irls_iterations must be positive")
        if self.convergence_tolerance <= 0 or self.epsilon <= 0:
            raise ValueError("tolerances must be positive")

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


# All candidates preserve affine degradation exactly.  Selection must use only
# training/validation devices, never test curves.
ROBUST_TREND_GRID = tuple(
    RobustTrendConfig(smoothness=value)
    for value in (1_000.0, 10_000.0, 100_000.0, 1_000_000.0)
)


def _clean_prefix(values: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size == 0:
        raise ValueError("DI prefix must contain at least one sample")
    if np.any(~np.isfinite(result)):
        raise ValueError("DI prefix must contain only finite samples")
    return result


def _difference_penalty_bands(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the main, first and second upper bands of D2.T @ D2."""

    main = np.zeros(length, dtype=float)
    first = np.zeros(max(0, length - 1), dtype=float)
    second = np.zeros(max(0, length - 2), dtype=float)
    if length >= 3:
        main[:] = 6.0
        main[:2] = (1.0, 5.0)
        main[-2:] = (5.0, 1.0)
        first[:] = -4.0
        first[0] = first[-1] = -2.0
        second[:] = 1.0
    return main, first, second


def _solve_weighted(
    values: np.ndarray, weights: np.ndarray, smoothness: float, epsilon: float
) -> np.ndarray:
    length = len(values)
    if length < 3 or smoothness == 0:
        return values.copy()
    main, first, second = _difference_penalty_bands(length)
    # scipy.linalg.solveh_banded upper-band layout: ab[u + i - j, j] = a[i, j].
    bands = np.zeros((3, length), dtype=float)
    bands[2] = weights + smoothness * main + epsilon
    bands[1, 1:] = smoothness * first
    bands[0, 2:] = smoothness * second
    return solveh_banded(
        bands, weights * values, lower=False, overwrite_ab=False,
        overwrite_b=False, check_finite=False,
    )


def robust_whittaker_trend(
    di_prefix: Sequence[float] | np.ndarray,
    config: RobustTrendConfig = RobustTrendConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a robust low-frequency DWC from exactly one observed DI prefix."""

    values = _clean_prefix(di_prefix)
    # A second-difference penalty scales with the fourth power of the sampling
    # density.  This keeps the effective cutoff comparable when prefixes are
    # later resized to the model's fixed 128-point input.
    effective_smoothness = config.smoothness * (
        len(values) / float(config.reference_length)
    ) ** 4
    if len(values) < 3:
        trend = values.copy()
        iterations = 0
        robust_scale = 0.0
    else:
        weights = np.ones(len(values), dtype=float)
        trend = _solve_weighted(values, weights, effective_smoothness, config.epsilon)
        iterations = 0
        robust_scale = 0.0
        for iteration in range(config.irls_iterations):
            residual = values - trend
            centre = float(np.median(residual))
            robust_scale = max(
                1.4826 * float(np.median(np.abs(residual - centre))), config.epsilon
            )
            threshold = config.huber_delta * robust_scale
            absolute = np.abs(residual - centre)
            updated_weights = np.ones_like(absolute)
            outliers = absolute > threshold
            updated_weights[outliers] = threshold / absolute[outliers]
            updated = _solve_weighted(
                values, updated_weights, effective_smoothness, config.epsilon
            )
            relative_change = float(
                np.linalg.norm(updated - trend)
                / max(np.linalg.norm(trend), config.epsilon)
            )
            trend = updated
            weights = updated_weights
            iterations = iteration + 1
            if relative_change <= config.convergence_tolerance:
                break
    return trend.astype(np.float32), {
        "method": "robust_whittaker_second_difference",
        "prefix_length": len(values),
        "smoothness": config.smoothness,
        "reference_length": config.reference_length,
        "effective_smoothness": effective_smoothness,
        "huber_delta": config.huber_delta,
        "irls_iterations_used": iterations,
        "robust_residual_scale": robust_scale,
        "causal_prefix_only": True,
        "configuration_hash": config.hash(),
        "affine_trend_preservation": "exact_up_to_numerical_precision",
    }


def decomposition_diagnostics(
    di: np.ndarray, dwc: np.ndarray, edc: np.ndarray, epsilon: float = 1e-8
) -> dict[str, float | bool]:
    """Scale-free diagnostics; a mean near zero is not itself a collapse."""

    di = np.asarray(di, dtype=float)
    dwc = np.asarray(dwc, dtype=float)
    edc = np.asarray(edc, dtype=float)
    di_std = float(np.std(di))
    di_range = float(np.ptp(di))
    dwc_std_ratio = float(np.std(dwc) / max(di_std, epsilon))
    dwc_range_ratio = float(np.ptp(dwc) / max(di_range, epsilon))
    residual_variance_ratio = float(np.var(edc) / max(np.var(di), epsilon))
    identity_error = float(np.max(np.abs(edc - (di - dwc))))
    dynamic_input = di_std > 10.0 * epsilon and di_range > 10.0 * epsilon
    # This flags genuine flat-line collapse.  It does not reject a legitimate
    # strongly smoothed trend merely because its mean is close to zero.
    collapsed = bool(
        dynamic_input and dwc_std_ratio < 1e-3 and dwc_range_ratio < 1e-3
    )
    return {
        "dwc_mean": float(np.mean(dwc)),
        "dwc_std_ratio": dwc_std_ratio,
        "dwc_range_ratio": dwc_range_ratio,
        "edc_variance_ratio": residual_variance_ratio,
        "edc_identity_max_abs_error": identity_error,
        "dwc_flatline_collapse": collapsed,
    }


def robust_trend_decompose(
    di_prefix: Sequence[float] | np.ndarray,
    config: RobustTrendConfig = RobustTrendConfig(),
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return non-collapsing DWC and an exact independent EDC residual."""

    di = _clean_prefix(di_prefix).astype(np.float32)
    dwc, trend_metadata = robust_whittaker_trend(di, config)
    edc = (di - dwc).astype(np.float32)
    diagnostics = decomposition_diagnostics(di, dwc, edc, config.epsilon)
    return dwc, edc, {
        "method": "robust_trend_plus_exact_residual",
        "prefix_length": len(di),
        "causal_prefix_only": True,
        "uses_wiener": False,
        "uses_autoencoder_fusion": False,
        "trend": trend_metadata,
        **diagnostics,
    }
