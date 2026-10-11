"""Prefix-causal event segmenters used in the controlled D2Pred comparison."""

from __future__ import annotations

import heapq
import math
from typing import Callable, Sequence

import numpy as np

from analysis.ac_cpm_core import causal_change_points


def _prefix_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.r_[0.0, np.cumsum(values)], np.r_[0.0, np.cumsum(values * values)]


def _sse(prefix: np.ndarray, squared: np.ndarray, start: int, end: int) -> float:
    length = end - start
    if length <= 0:
        return 0.0
    total = prefix[end] - prefix[start]
    return max(0.0, float(squared[end] - squared[start] - total * total / length))


def _best_sse_split(
    prefix: np.ndarray, squared: np.ndarray, start: int, end: int, minimum: int
) -> tuple[float, int] | None:
    if end - start < 2 * minimum:
        return None
    base = _sse(prefix, squared, start, end)
    best = (-float("inf"), -1)
    candidate_count = end - start - 2 * minimum + 1
    step = max(1, candidate_count // 128)
    for point in range(start + minimum, end - minimum + 1, step):
        gain = base - _sse(prefix, squared, start, point) - _sse(prefix, squared, point, end)
        if gain > best[0]:
            best = (gain, point)
    return best


def priority_aep_change_points(
    values: Sequence[float],
    sensitivity: float = 2.5,
    min_size: int = 8,
    max_points: int = 15,
) -> list[int]:
    """AEP with global best-gain priority instead of right-child DFS."""

    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2 * min_size:
        return []
    prefix, squared = _prefix_statistics(x)
    noise = np.median(np.abs(np.diff(x) - np.median(np.diff(x)))) * 1.4826
    threshold = sensitivity * max(noise * noise, 1e-8) * math.log(max(len(x), 3))
    heap: list[tuple[float, int, int, int]] = []

    def propose(start: int, end: int) -> None:
        result = _best_sse_split(prefix, squared, start, end, min_size)
        if result is not None:
            gain, point = result
            heapq.heappush(heap, (-gain, start, end, point))

    propose(0, len(x))
    accepted = []
    while heap and len(accepted) < max_points:
        negative_gain, start, end, point = heapq.heappop(heap)
        gain = -negative_gain
        if gain <= threshold:
            break
        accepted.append(point)
        propose(start, point)
        propose(point, end)
    return sorted(accepted)


def _gaussian_score(
    prefix: np.ndarray,
    squared: np.ndarray,
    start: int,
    end: int,
    regularization: float,
) -> float:
    length = end - start
    if length <= 0:
        return -float("inf")
    variance = _sse(prefix, squared, start, end) / length
    adjusted = max(variance + regularization / length, 1e-10)
    return -0.5 * (length * math.log(adjusted) + regularization / adjusted)


def _best_gaussian_split(
    prefix: np.ndarray,
    squared: np.ndarray,
    start: int,
    end: int,
    minimum: int,
    regularization: float,
) -> tuple[float, int] | None:
    if end - start < 2 * minimum:
        return None
    base = _gaussian_score(prefix, squared, start, end, regularization)
    best = (-float("inf"), -1)
    candidate_count = end - start - 2 * minimum + 1
    step = max(1, candidate_count // 128)
    for point in range(start + minimum, end - minimum + 1, step):
        gain = (
            _gaussian_score(prefix, squared, start, point, regularization)
            + _gaussian_score(prefix, squared, point, end, regularization)
            - base
        )
        if gain > best[0]:
            best = (gain, point)
    return best


def causal_ggs_change_points(
    values: Sequence[float],
    min_size: int = 8,
    max_points: int = 15,
    regularization: float = 0.1,
    bic_multiplier: float = 1.0,
) -> list[int]:
    """Paper-derived adaptive GGS with BIC-defined stopping and relocation.

    The thesis leaves ``max_error`` undefined.  We operationalize its stopping
    condition as a Gaussian likelihood gain exceeding ``bic_multiplier*log(n)``.
    """

    raw = np.asarray(values, dtype=np.float64)
    if len(raw) < 2 * min_size:
        return []
    center = float(np.median(raw))
    scale = float(np.median(np.abs(raw - center)) * 1.4826)
    if scale < 1e-8:
        scale = max(float(np.std(raw)), 1.0)
    x = (raw - center) / scale
    prefix, squared = _prefix_statistics(x)
    points = [0, len(x)]
    stopping_gain = bic_multiplier * math.log(max(len(x), 3))
    while len(points) - 2 < max_points:
        best = (-float("inf"), -1)
        for start, end in zip(points[:-1], points[1:]):
            proposal = _best_gaussian_split(
                prefix, squared, start, end, min_size, regularization
            )
            if proposal is not None and proposal[0] > best[0]:
                best = proposal
        if best[1] < 0 or best[0] <= stopping_gain:
            break
        points.append(best[1])
        points.sort()
        # GGS-style local relocation of every existing internal breakpoint.
        changed = True
        passes = 0
        while changed and passes < 4:
            changed = False
            passes += 1
            for index in range(1, len(points) - 1):
                left, old, right = points[index - 1], points[index], points[index + 1]
                proposal = _best_gaussian_split(
                    prefix, squared, left, right, min_size, regularization
                )
                if proposal is None:
                    continue
                new = proposal[1]
                old_score = (
                    _gaussian_score(prefix, squared, left, old, regularization)
                    + _gaussian_score(prefix, squared, old, right, regularization)
                )
                new_score = (
                    _gaussian_score(prefix, squared, left, new, regularization)
                    + _gaussian_score(prefix, squared, new, right, regularization)
                )
                if new != old and new_score > old_score + 1e-9:
                    points[index] = new
                    changed = True
            points.sort()
    return points[1:-1]


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, width)
    prefix = np.r_[0.0, np.cumsum(values)]
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - width + 1)
        result[index] = (prefix[index + 1] - prefix[start]) / (index + 1 - start)
    return result


def _cpm_features(values: np.ndarray) -> np.ndarray:
    scale = max(float(np.std(values)), 1e-8)
    normalized = (values - float(np.mean(values))) / scale
    gradient = np.r_[0.0, np.diff(normalized)]
    return np.column_stack((
        normalized,
        _moving_average(normalized, 3),
        _moving_average(normalized, 7),
        _moving_average(normalized, 15),
        gradient,
        _moving_average(np.abs(gradient), 7),
    ))


def _candidate_grid(length: int, minimum: int, maximum: int = 128) -> np.ndarray:
    if length <= maximum:
        return np.arange(length + 1, dtype=int)
    interior = np.linspace(minimum, length - minimum, maximum - 1).round().astype(int)
    return np.unique(np.r_[0, interior, length])


def _dp_partition(values: np.ndarray, segments: int, minimum: int) -> tuple[np.ndarray, float]:
    length = len(values)
    grid = _candidate_grid(length, minimum)
    prefix, squared = _prefix_statistics(values)
    count = len(grid)
    cost = np.full((segments + 1, count), np.inf)
    back = np.full((segments + 1, count), -1, dtype=int)
    for end_index in range(1, count):
        if grid[end_index] >= minimum:
            cost[1, end_index] = _sse(prefix, squared, 0, int(grid[end_index]))
    for number in range(2, segments + 1):
        for end_index in range(1, count):
            end = int(grid[end_index])
            split_indices = np.arange(number - 1, end_index, dtype=int)
            if not len(split_indices):
                continue
            split_values = grid[split_indices]
            lengths = end - split_values
            valid = (lengths >= minimum) & np.isfinite(cost[number - 1, split_indices])
            if not valid.any():
                continue
            split_indices = split_indices[valid]
            split_values = split_values[valid]
            lengths = lengths[valid]
            totals = prefix[end] - prefix[split_values]
            segment_costs = np.maximum(
                0.0,
                squared[end] - squared[split_values] - totals * totals / lengths,
            )
            values = cost[number - 1, split_indices] + segment_costs
            local_index = int(np.argmin(values))
            cost[number, end_index] = float(values[local_index])
            back[number, end_index] = int(split_indices[local_index])
    index = count - 1
    points = []
    for number in range(segments, 1, -1):
        index = int(back[number, index])
        if index < 0:
            return np.asarray([], dtype=int), float("inf")
        points.append(int(grid[index]))
    return np.asarray(sorted(points), dtype=int), float(cost[segments, -1])


def _dtw_distance(left: np.ndarray, right: np.ndarray, maximum: int = 64) -> float:
    if len(left) > maximum:
        indices = np.linspace(0, len(left) - 1, maximum).round().astype(int)
        left, right = left[indices], right[indices]
    previous = np.full(len(right) + 1, np.inf)
    previous[0] = 0.0
    for value in left:
        current = np.full(len(right) + 1, np.inf)
        for index, other in enumerate(right, 1):
            current[index] = abs(value - other) + min(
                current[index - 1], previous[index], previous[index - 1]
            )
        previous = current
    return float(previous[-1] / max(len(left) + len(right), 1))


def _repair_points(points: np.ndarray, length: int, minimum: int) -> list[int]:
    repaired = []
    for point in np.sort(points.astype(int)):
        lower = repaired[-1] + minimum if repaired else minimum
        remaining = len(points) - len(repaired) - 1
        upper = length - minimum * (remaining + 1)
        repaired.append(int(np.clip(point, lower, upper)))
    return repaired


def causal_cpm_dp_change_points(
    values: Sequence[float],
    segments: int = 8,
    min_size: int = 8,
    graph_weight: float = 0.5,
) -> list[int]:
    """Causal adaptation of Sun Ch.4 DP + DTW graph-aligned CPM boundaries."""

    x = np.asarray(values, dtype=np.float64)
    segments = min(int(segments), max(1, len(x) // min_size))
    if segments <= 1:
        return []
    features = _cpm_features(x)
    local_points, errors = [], []
    for channel in range(features.shape[1]):
        points, error = _dp_partition(features[:, channel], segments, min_size)
        if len(points) != segments - 1:
            return []
        local_points.append(points / max(len(x), 1))
        errors.append(error / max(len(x), 1))
    local = np.asarray(local_points, dtype=float)
    confidence = 1.0 / (np.asarray(errors) + 1e-6)
    confidence /= max(float(np.mean(confidence)), 1e-8)
    channels = features.shape[1]
    distances = np.zeros((channels, channels), dtype=float)
    for left in range(channels):
        for right in range(left + 1, channels):
            value = _dtw_distance(features[:, left], features[:, right])
            distances[left, right] = distances[right, left] = value
    nonzero = distances[distances > 0]
    sigma = max(float(np.median(nonzero)) if len(nonzero) else 1.0, 1e-6)
    adjacency = np.exp(-(distances**2) / (2.0 * sigma * sigma))
    np.fill_diagonal(adjacency, 0.0)
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    fidelity = np.diag(confidence)
    aligned = np.linalg.solve(
        fidelity + graph_weight * laplacian + np.eye(channels) * 1e-8,
        fidelity @ local,
    )
    weights = confidence / confidence.sum()
    consensus = weights @ aligned
    return _repair_points(np.rint(consensus * len(x)), len(x), min_size)


SEGMENTERS: dict[str, Callable[..., list[int]]] = {
    "aep_original": causal_change_points,
    "aep_priority_fixed": priority_aep_change_points,
    "ggs_causal": causal_ggs_change_points,
    "cpm_dp_causal": causal_cpm_dp_change_points,
}


def event_change_points(values: Sequence[float], method: str) -> list[int]:
    if method == "aep_original":
        return causal_change_points(values, sensitivity=2.5, min_size=8, max_points=15)
    if method == "aep_priority_fixed":
        return priority_aep_change_points(values, sensitivity=2.5, min_size=8, max_points=15)
    if method == "ggs_causal":
        return causal_ggs_change_points(values, min_size=8, max_points=15)
    if method == "cpm_dp_causal":
        return causal_cpm_dp_change_points(values, segments=8, min_size=8)
    raise KeyError(f"Unknown event segmentation method: {method}")
