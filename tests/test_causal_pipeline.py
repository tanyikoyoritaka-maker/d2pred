from __future__ import annotations

import numpy as np
import torch
import unittest
from pathlib import Path
import re

from analysis.ac_cpm_core import causal_change_points, pad_tokens, segment_tokens
from analysis.causal_decomposition import (
    AnchoredTemporalTrendAutoencoder,
    CausalDecompositionConfig,
    HealthStateSeries,
    TemporalTrendAutoencoder,
    StableStageConfig,
    causal_decompose,
    freeze_fusion_model,
    initial_healthy_interval,
    rolling_stl_trend,
    detect_initial_stable_stage,
    stable_stage_causal_decompose,
    stable_stage_wiener_expected_trend,
    wiener_expected_trend,
    anchored_fusion_loss,
)
from analysis.cpm_edc_independent_model import (
    CPMDWCEventFusion,
    CPMEDCIndependentFusion,
)
from analysis.robust_trend_decomposition import (
    RobustTrendConfig,
    robust_trend_decompose,
    robust_whittaker_trend,
)


def health(values):
    return HealthStateSeries(np.asarray(values, dtype=float), "test_soh",
                             "SoH=Q_actual/Q_nominal*100%")


def frozen_fusion(seed=7):
    torch.manual_seed(seed)
    return freeze_fusion_model(TemporalTrendAutoencoder(hidden_channels=8,
                                                        latent_channels=3,
                                                        encoder_depth=2))


def tokens(edc):
    changes = causal_change_points(edc, sensitivity=2.5, min_size=4, max_points=6)
    return pad_tokens(segment_tokens(edc, changes, "stats", 12), 8)


def test_initial_contiguous_health_does_not_reenter_after_recovery():
    assert initial_healthy_interval([100, 91, 80, 79.9, 95, 99]) == (3, False)
    assert initial_healthy_interval([79, 100, 100]) == (0, True)
    assert initial_healthy_interval([80, 79, 100]) == (1, True)


def test_wiener_uses_only_healthy_observations_and_constant_fallback():
    config = CausalDecompositionConfig()
    di = np.array([1.0, 2.0, 200.0, -900.0])
    trend, metadata = wiener_expected_trend(di, health([100, 85, 79, 99]), config)
    np.testing.assert_allclose(trend, [1, 2, 3, 4], atol=1e-6)
    assert metadata["healthy_end"] == metadata["fit_len"] == 2
    fallback, fallback_metadata = wiener_expected_trend(
        di, health([79, 99, 99, 99]), config
    )
    np.testing.assert_allclose(fallback, np.ones(4))
    assert fallback_metadata["fallback"]
    assert fallback_metadata["fit_len"] == 0


def test_all_decomposition_lengths_and_edc_identity():
    tau = 37
    di = np.sin(np.arange(tau) / 5) + np.arange(tau) * 0.02
    soh = np.linspace(100, 75, tau)
    config = CausalDecompositionConfig()
    wiener, _ = wiener_expected_trend(di, health(soh), config)
    stl, _ = rolling_stl_trend(di, config)
    dwc, edc, _ = causal_decompose(di, health(soh), frozen_fusion(), config)
    assert len(wiener) == len(stl) == len(dwc) == len(edc) == tau
    np.testing.assert_allclose(edc, di - dwc, atol=1e-6)


def test_fusion_must_be_frozen_for_inference():
    di = np.linspace(0, 1, 20)
    model = TemporalTrendAutoencoder(hidden_channels=8, latent_channels=2)
    try:
        causal_decompose(di, health(np.full(20, 100.0)), model,
                         CausalDecompositionConfig())
    except RuntimeError as exc:
        assert "must be frozen" in str(exc)
    else:
        raise AssertionError("Unfrozen fusion model was accepted for inference")
    freeze_fusion_model(model)
    causal_decompose(di, health(np.full(20, 100.0)), model,
                     CausalDecompositionConfig())
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_suffix_perturbation_preserves_dwc_edc_tokens_and_prediction():
    torch.manual_seed(11)
    tau, total = 31, 60
    original = np.sin(np.arange(total) / 6) + np.arange(total) * 0.01
    perturbed = original.copy(); perturbed[tau:] = 1e6 * np.arange(1, total - tau + 1)
    soh = np.r_[np.linspace(100, 78, tau), np.full(total - tau, 1.0)]
    perturbed_soh = soh.copy(); perturbed_soh[tau:] = 100.0
    fusion = frozen_fusion(13); config = CausalDecompositionConfig()
    first = causal_decompose(original[:tau], health(soh[:tau]), fusion, config)
    second = causal_decompose(perturbed[:tau], health(perturbed_soh[:tau]), fusion, config)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    first_tokens = tokens(first[1]); second_tokens = tokens(second[1])
    np.testing.assert_array_equal(first_tokens[0], second_tokens[0])
    np.testing.assert_array_equal(first_tokens[1], second_tokens[1])

    model = CPMEDCIndependentFusion(token_dim=12, hidden=16).eval()
    def prediction(dwc, edc, event_tokens, event_mask):
        values = {
            "di": torch.tensor(original[:tau], dtype=torch.float32)[None],
            "dwc": torch.tensor(dwc, dtype=torch.float32)[None],
            "edc": torch.tensor(edc, dtype=torch.float32)[None],
            "tokens": torch.tensor(event_tokens, dtype=torch.float32)[None],
            "token_mask": torch.tensor(event_mask, dtype=torch.float32)[None],
        }
        with torch.inference_mode():
            return model(values).item()
    assert prediction(first[0], first[1], *first_tokens) == prediction(
        second[0], second[1], *second_tokens
    )


def test_validation_and_test_suffixes_cannot_affect_prefix():
    tau = 25
    prefix = np.linspace(-1, 1, tau)
    suffix_a = np.zeros(20); suffix_b = np.full(20, -9e8)
    soh_prefix = np.linspace(100, 70, tau)
    fusion = frozen_fusion(); config = CausalDecompositionConfig()
    for _role in ("validation", "test"):
        a = causal_decompose(np.r_[prefix, suffix_a][:tau],
                             health(np.r_[soh_prefix, np.zeros(20)][:tau]), fusion, config)
        b = causal_decompose(np.r_[prefix, suffix_b][:tau],
                             health(np.r_[soh_prefix, np.full(20, 100.0)][:tau]), fusion, config)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def test_autoencoder_is_temporal_encoder_decoder_not_pointwise_mlp():
    model = TemporalTrendAutoencoder(hidden_channels=8, latent_channels=3,
                                     encoder_depth=2)
    kernels = [module.kernel_size[0] for module in model.modules()
               if isinstance(module, torch.nn.Conv1d)]
    assert kernels and all(kernel > 1 for kernel in kernels)
    assert model.latent_channels < model.hidden_channels
    x = torch.randn(2, 2, 19); mask = torch.ones(2, 19, dtype=torch.bool)
    assert model(x, mask).shape == (2, 19)


def test_robust_trend_preserves_affine_degradation_and_cannot_flatline():
    values = 3.5 - 0.07 * np.arange(80, dtype=float)
    trend, metadata = robust_whittaker_trend(
        values, RobustTrendConfig(smoothness=1_000_000.0)
    )
    np.testing.assert_allclose(trend, values, atol=2e-5)
    assert metadata["affine_trend_preservation"].startswith("exact")


def test_robust_decomposition_rejects_event_spikes_and_keeps_identity():
    time = np.arange(100, dtype=float)
    baseline = 0.02 * time + 0.0005 * time**2
    observed = baseline.copy()
    observed[[25, 62, 63]] += [4.0, -5.0, 3.5]
    dwc, edc, metadata = robust_trend_decompose(
        observed, RobustTrendConfig(smoothness=100_000.0)
    )
    np.testing.assert_allclose(edc, observed - dwc, atol=1e-6)
    assert np.max(np.abs(edc[[25, 62, 63]])) > 2.0
    assert metadata["dwc_std_ratio"] > 0.05
    assert not metadata["dwc_flatline_collapse"]
    assert not metadata["uses_wiener"]


def test_robust_decomposition_is_observed_prefix_only():
    tau = 45
    prefix = np.sin(np.arange(tau) / 11.0) + np.arange(tau) * 0.03
    full_a = np.r_[prefix, np.zeros(30)]
    full_b = np.r_[prefix, np.full(30, 1e12)]
    config = RobustTrendConfig(smoothness=10_000.0)
    a = robust_trend_decompose(full_a[:tau], config)
    b = robust_trend_decompose(full_b[:tau], config)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_anchored_aefusion_residual_is_bounded_and_keeps_convex_base():
    torch.manual_seed(123)
    model = AnchoredTemporalTrendAutoencoder(
        hidden_channels=8, latent_channels=3, residual_fraction=0.1
    )
    trends = torch.randn(3, 2, 41)
    mask = torch.ones(3, 41, dtype=torch.bool)
    output, details = model.forward_components(trends, mask)
    bound = 0.1 * details["disagreement_scale"] + 1e-7
    assert output.shape == (3, 41)
    assert torch.all(torch.abs(details["correction"]) <= bound)
    torch.testing.assert_close(output, details["base"] + details["correction"])
    assert torch.all((details["alpha"] >= 0) & (details["alpha"] <= 1))


def test_anchored_loss_prefers_dynamic_trend_over_zero_cancellation():
    time = torch.linspace(-1, 1, 61)[None]
    wiener, stl = time, -time
    mask = torch.ones_like(time, dtype=torch.bool)
    base = torch.zeros_like(time)
    zero_loss, _ = anchored_fusion_loss(
        torch.zeros_like(time), wiener, stl, base, mask
    )
    dynamic_loss, _ = anchored_fusion_loss(
        0.3 * wiener, wiener, stl, base, mask
    )
    assert dynamic_loss < zero_loss


def test_dwc_cpm_model_has_no_edc_gru_and_ignores_raw_edc_sequence():
    torch.manual_seed(321)
    model = CPMDWCEventFusion(token_dim=12, hidden=16).eval()
    assert not hasattr(model, "edc_gru")
    values = {
        "di": torch.randn(2, 31),
        "dwc": torch.randn(2, 31),
        "edc": torch.randn(2, 31),
        "tokens": torch.randn(2, 8, 12),
        "token_mask": torch.ones(2, 8),
    }
    changed = dict(values)
    changed["edc"] = torch.full_like(values["edc"], 1e9)
    with torch.inference_mode():
        first = model(values)
        second = model(changed)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_adaptive_stable_stage_boundary_freezes_after_persistent_departure():
    prefix = np.r_[np.arange(20) * 0.1, 8.0 + np.arange(15) * 0.5]
    config = StableStageConfig(k_min=12, gamma_r=3.0, gamma_s=3.0,
                               rolling_window=8, consecutive_q=2)
    boundary, diagnostic = detect_initial_stable_stage(prefix, config)
    extended = np.r_[prefix, -1000.0, 1000.0, 0.0]
    extended_boundary, _ = detect_initial_stable_stage(extended, config)
    assert boundary == extended_boundary
    assert diagnostic["boundary_detected"]
    assert diagnostic["stable_end"] == diagnostic["fit_len"] == boundary
    for field in ("baseline_intercept", "baseline_slope", "residual_mad",
                  "slope_threshold", "residual_threshold", "fallback_status"):
        assert field in diagnostic


def test_stable_detector_insufficient_prefix_diagnostic():
    boundary, diagnostic = detect_initial_stable_stage(
        [1.0, 1.1, 1.2], StableStageConfig(k_min=12)
    )
    assert boundary == 3
    assert diagnostic["fallback_status"] == "insufficient_initial_points"


def test_adaptive_pipeline_suffix_invariance_through_prediction():
    torch.manual_seed(91)
    tau, total = 29, 55
    observed = np.r_[np.arange(17) * 0.03, 1.5 + np.arange(tau - 17) * 0.2]
    full_a = np.r_[observed, np.zeros(total - tau)]
    full_b = np.r_[observed, np.arange(total - tau) * 1e9]
    stable = StableStageConfig(k_min=12, gamma_r=3, gamma_s=3,
                               rolling_window=8, consecutive_q=2)
    decomposition = CausalDecompositionConfig()
    fusion = frozen_fusion(23)
    boundary_a, _ = detect_initial_stable_stage(full_a[:tau], stable)
    boundary_b, _ = detect_initial_stable_stage(full_b[:tau], stable)
    w_a, _ = stable_stage_wiener_expected_trend(full_a[:tau], stable, decomposition)
    w_b, _ = stable_stage_wiener_expected_trend(full_b[:tau], stable, decomposition)
    s_a, _ = rolling_stl_trend(full_a[:tau], decomposition)
    s_b, _ = rolling_stl_trend(full_b[:tau], decomposition)
    dwc_a, edc_a, _ = stable_stage_causal_decompose(
        full_a[:tau], fusion, stable, decomposition
    )
    dwc_b, edc_b, _ = stable_stage_causal_decompose(
        full_b[:tau], fusion, stable, decomposition
    )
    assert boundary_a == boundary_b
    for left, right in ((w_a, w_b), (s_a, s_b), (dwc_a, dwc_b), (edc_a, edc_b)):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-7)
    token_a, mask_a = tokens(edc_a); token_b, mask_b = tokens(edc_b)
    np.testing.assert_allclose(token_a, token_b, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(mask_a, mask_b)
    model = CPMEDCIndependentFusion(token_dim=12, hidden=16).eval()
    def predict(dwc, edc, event, mask):
        values = {"di": torch.tensor(observed)[None].float(),
                  "dwc": torch.tensor(dwc)[None].float(),
                  "edc": torch.tensor(edc)[None].float(),
                  "tokens": torch.tensor(event)[None].float(),
                  "token_mask": torch.tensor(mask)[None].float()}
        with torch.inference_mode(): return model(values).item()
    assert predict(dwc_a, edc_a, token_a, mask_a) == predict(
        dwc_b, edc_b, token_b, mask_b
    )
    assert len(w_a) == len(s_a) == len(dwc_a) == len(edc_a) == tau
    np.testing.assert_allclose(edc_a, observed - dwc_a, atol=1e-6)


def detect_full_life_decompose_then_slice(source):
    decomposition = re.search(
        r"(?:corrected_decompose|causal_decompose)\([^\n]*(?:full|to_numpy|values)",
        source,
    )
    later_slice = re.search(r"\b(?:dwc|edc|DWC|EDC)\s*\[:\s*(?:tau|end)\b", source)
    return bool(decomposition and later_slice and decomposition.start() < later_slice.start())


def test_static_detector_finds_full_life_decomposition_then_prefix_slice():
    bad = "raw, dwc, edc = corrected_decompose(full_device_di)\nvalue = dwc[:tau]"
    assert detect_full_life_decompose_then_slice(bad)


def test_repository_has_no_full_life_decomposition_then_prefix_slice():
    root = Path(__file__).resolve().parents[1]
    excluded = {"outputs", "new_expriments", "last_results", "paper", "reports",
                "孙乐乐-毕设", ".venv", ".venv-linux", ".tmp_pip"}
    violations = []
    for path in root.rglob("*.py"):
        if path == Path(__file__).resolve():
            continue
        if any(part in excluded for part in path.parts):
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if detect_full_life_decompose_then_slice(source):
            violations.append(str(path.relative_to(root)))
    assert not violations, "Full-life decomposition followed by slicing: " + ", ".join(violations)


def test_all_retraining_dataset_device_splits_are_isolated():
    import json
    root = Path(__file__).resolve().parents[1]
    folds = json.loads((root / "manifests/folds.json").read_text())
    for fold in folds:
        partitions = {key: set(map(str, fold[key + "_units"]))
                      for key in ("train", "validation", "test")}
        assert all(partitions.values())
        assert not (partitions["train"] & partitions["validation"])
        assert not (partitions["train"] & partitions["test"])
        assert not (partitions["validation"] & partitions["test"])
    for name in ("xjtu_sy", "pronostia", "ims"):
        frame = __import__("pandas").read_csv(root / f"public_processing/{name}_split.csv")
        partitions = {
            split: set(map(str, frame.loc[frame.split.eq(split), "unit_id"].unique()))
            for split in ("train", "val", "test")
        }
        assert all(partitions.values()), f"{name} has an empty device partition"
        assert not (partitions["train"] & partitions["val"])
        assert not (partitions["train"] & partitions["test"])
        assert not (partitions["val"] & partitions["test"])


def load_tests(loader, tests, pattern):
    del loader, tests, pattern
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
