"""Core diagnostics and causal change-point tokenization for rebuilt AC runs.

This module is deliberately independent of the legacy AGGS preprocessing path.
"""

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


STAT_NAMES = (
    "start_rel", "end_rel", "length_rel", "mean", "std", "min", "max",
    "slope", "energy", "area", "abs_residual_energy", "pre_post_jump",
)


def robust_scale(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    med = np.nanmedian(x)
    x = np.where(finite, x, med)
    lo, hi = np.percentile(x, [5, 95])
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), -1.0, 2.0)


def moving_average(values: Sequence[float], window: int = 11) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2 or window <= 1:
        return x.copy()
    window = min(int(window), len(x))
    left = window // 2
    padded = np.pad(x, (left, window - left - 1), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def orient_degradation(values: Sequence[float], rul: Sequence[float]) -> np.ndarray:
    """Orient an indicator so larger values correspond to more degradation."""
    x = robust_scale(values)
    r = np.asarray(rul, dtype=np.float64)
    if len(x) > 2 and np.corrcoef(x, r)[0, 1] > 0:
        x = 1.0 - x
    return x


def candidate_series(values: Sequence[float], rul: Sequence[float], smooth_window: int = 11):
    raw = orient_degradation(values, rul)
    smooth = moving_average(raw, smooth_window)
    diff = np.r_[0.0, np.diff(smooth)]
    enhanced = robust_scale(smooth + 0.25 * moving_average(np.maximum(diff, 0.0), 5))
    trend_residual = robust_scale(smooth + 0.15 * np.abs(raw - smooth))
    return {
        "raw": raw,
        "smoothed": smooth,
        "normalized": robust_scale(raw),
        "diff_enhanced": enhanced,
        "trend_residual": trend_residual,
    }


def monotonicity(x: Sequence[float]) -> float:
    d = np.diff(np.asarray(x, dtype=float))
    if not len(d):
        return 0.0
    return float(abs(np.sum(d > 0) - np.sum(d < 0)) / len(d))


def smoothness(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 1.0
    scale = np.ptp(x) + 1e-12
    return float(np.clip(1.0 - np.std(np.diff(x)) / scale, 0.0, 1.0))


def _segment_cost(prefix_sum, prefix_sq, start: int, end: int) -> float:
    n = end - start
    if n <= 0:
        return 0.0
    total = prefix_sum[end] - prefix_sum[start]
    sq = prefix_sq[end] - prefix_sq[start]
    return max(0.0, float(sq - total * total / n))


def causal_change_points(
    values: Sequence[float], sensitivity: float = 2.5, min_size: int = 8,
    max_points: int = 12,
) -> List[int]:
    """Binary SSE segmentation using only the supplied (causal) prefix.

    A split is accepted when its variance reduction exceeds
    ``sensitivity * robust_noise^2 * log(n)``. Lower sensitivity yields more
    events. Returned points exclude 0 and ``len(values)``.
    """
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if n < 2 * min_size:
        return []
    ps = np.r_[0.0, np.cumsum(x)]
    pq = np.r_[0.0, np.cumsum(x * x)]
    noise = np.median(np.abs(np.diff(x) - np.median(np.diff(x)))) * 1.4826
    threshold = sensitivity * max(noise * noise, 1e-8) * math.log(max(n, 3))
    pending: List[Tuple[int, int]] = [(0, n)]
    accepted: List[Tuple[float, int]] = []
    while pending and len(accepted) < max_points:
        start, end = pending.pop()
        if end - start < 2 * min_size:
            continue
        base = _segment_cost(ps, pq, start, end)
        choices = []
        candidate_count = end - start - 2 * min_size + 1
        step = max(1, candidate_count // 128)
        for point in range(start + min_size, end - min_size + 1, step):
            gain = base - _segment_cost(ps, pq, start, point) - _segment_cost(ps, pq, point, end)
            choices.append((gain, point))
        gain, point = max(choices)
        if gain > threshold:
            accepted.append((gain, point))
            pending.extend(((start, point), (point, end)))
    return sorted(point for _, point in accepted)


def segments(length: int, change_points: Iterable[int]) -> List[Tuple[int, int]]:
    bounds = [0] + sorted(set(int(p) for p in change_points if 0 < p < length)) + [length]
    return list(zip(bounds[:-1], bounds[1:]))


def _resize(x: np.ndarray, length: int) -> np.ndarray:
    if not len(x):
        return np.zeros(length, dtype=np.float32)
    if len(x) == 1:
        return np.full(length, x[0], dtype=np.float32)
    return np.interp(np.linspace(0, len(x) - 1, length), np.arange(len(x)), x).astype(np.float32)


def segment_tokens(
    values: Sequence[float], change_points: Sequence[int], mode: str = "stats",
    raw_length: int = 30,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    rows = []
    spans = segments(len(x), change_points)
    for index, (start, end) in enumerate(spans):
        part = x[start:end]
        t = np.arange(len(part), dtype=float)
        slope = float(np.polyfit(t, part, 1)[0]) if len(part) > 1 else 0.0
        fitted = part.mean() + slope * (t - t.mean()) if len(part) else part
        before = x[max(0, start - min(5, start)):start]
        jump = float(part[:min(5, len(part))].mean() - before.mean()) if len(before) else 0.0
        stats = np.asarray([
            start / max(len(x), 1), end / max(len(x), 1), len(part) / max(len(x), 1),
            part.mean(), part.std(), part.min(), part.max(), slope,
            np.mean(part * part), np.trapz(part) / max(len(part), 1),
            np.mean(np.abs(part - fitted)), jump,
        ], dtype=np.float32)
        raw = _resize(part, raw_length)
        rows.append(stats if mode == "stats" else raw if mode == "raw" else np.r_[stats, raw])
    return np.stack(rows).astype(np.float32)


def pad_tokens(tokens: np.ndarray, max_tokens: int) -> Tuple[np.ndarray, np.ndarray]:
    tokens = tokens[-max_tokens:]
    out = np.zeros((max_tokens, tokens.shape[1]), dtype=np.float32)
    mask = np.zeros(max_tokens, dtype=np.float32)
    out[:len(tokens)] = tokens
    mask[:len(tokens)] = 1.0
    return out, mask
