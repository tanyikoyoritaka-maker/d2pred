#!/usr/bin/env python3
"""Generate the strict-ablation candidate protocol from active code specs."""

from __future__ import annotations

import csv
import ast
from dataclasses import asdict, field, make_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ablation_final/protocol"
DATASETS = ("ACSim", "ACCom", "ACAll")
VARIANT_ORDER = (
    "full", "no_wiener", "no_stl", "no_autoencoder_fusion", "no_dwc",
    "no_edc", "no_decomposition", "no_fusion", "no_aep",
)
SOURCE_FILES = (
    "analysis/expert_exact_d2pred.py",
    "analysis/strict_ablation.py",
    "analysis/theory_aligned_decomposition.py",
    "analysis/causal_decomposition.py",
    "ablation_final/strict_pipeline.py",
    "ablation_final/train_strict_ablation.py",
    "run_ac_nested_gru_rerun.py",
    "run_stage_aware_aefusion_ac.py",
)


def load_variant_specs_from_source():
    """Read literal variant definitions without importing the training stack."""
    source = ast.parse((ROOT / "analysis/strict_ablation.py").read_text(encoding="utf-8"))
    variant_fields = None
    variant_defaults = {}
    variant_assignment = None
    for node in source.body:
        if isinstance(node, ast.ClassDef) and node.name == "VariantSpec":
            variant_fields = []
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    variant_fields.append(statement.target.id)
                    if statement.value is not None:
                        variant_defaults[statement.target.id] = ast.literal_eval(statement.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "VARIANT_SPECS":
            variant_assignment = node.value
    if not variant_fields or not isinstance(variant_assignment, ast.Dict):
        raise ValueError("Could not locate VariantSpec and VARIANT_SPECS in strict_ablation.py")
    dataclass_fields = []
    for name in variant_fields:
        if name in variant_defaults:
            dataclass_fields.append((name, object, field(default=variant_defaults[name])))
        else:
            dataclass_fields.append((name, object))
    variant_type = make_dataclass("VariantSpec", dataclass_fields, frozen=True)
    variants = {}
    for key_node, value_node in zip(variant_assignment.keys, variant_assignment.values):
        if not isinstance(value_node, ast.Call) or not isinstance(value_node.func, ast.Name) or value_node.func.id != "VariantSpec":
            raise ValueError("Variant registry values must be literal VariantSpec calls")
        values = [ast.literal_eval(argument) for argument in value_node.args]
        values.extend(ast.literal_eval(keyword.value) for keyword in value_node.keywords)
        variants[ast.literal_eval(key_node)] = variant_type(*values)
    return variants


VARIANT_SPECS = load_variant_specs_from_source()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def component_rows():
    rows = {
        "Wiener trend estimator": {},
        "STL trend estimator": {},
        "Adaptive trend fusion / bounded AE": {},
        "DWC predictor branch": {},
        "EDC residual used for event input": {},
        "Event-token construction": {},
        "Adaptive Event Partitioning (AEP)": {},
        "Event Transformer encoder": {},
        "DWC-guided cross-branch attention": {},
        "Trend-residual decomposition": {},
    }
    for variant in VARIANT_ORDER:
        if variant == "full":
            values = {
                "Wiener trend estimator": "✓ (frozen Full)",
                "STL trend estimator": "✓ (frozen Full)",
                "Adaptive trend fusion / bounded AE": "✓ (frozen Full)",
                "DWC predictor branch": "✓ (frozen Full)",
                "EDC residual used for event input": "✓ (frozen Full)",
                "Event-token construction": "✓ (frozen Full)",
                "Adaptive Event Partitioning (AEP)": "✓ (frozen Full)",
                "Event Transformer encoder": "✓ (frozen Full)",
                "DWC-guided cross-branch attention": "✓ (frozen Full)",
                "Trend-residual decomposition": "✓ (frozen Full)",
            }
        else:
            spec = VARIANT_SPECS[variant]
            values = {
                "Wiener trend estimator": "✓" if spec.trend_source in {"wiener_only", "wiener_and_stl"} else "×",
                "STL trend estimator": "✓" if spec.trend_source in {"stl_only", "wiener_and_stl"} else "×",
                "Adaptive trend fusion / bounded AE": "✓" if spec.trend_fusion_mode in {
                    "adaptive_stl_anchored_ae", "adaptive_wiener_anchored_ae", "fold_pretrained_adaptive_ae"
                } else ("× (fixed mean)" if spec.trend_fusion_mode == "fixed_equal_average" else "N/A"),
                "DWC predictor branch": (
                    "✓ (raw DI)" if variant == "no_decomposition" else
                    "✓" if spec.use_dwc_predictor else "×"
                ),
                "EDC residual used for event input": (
                    "Raw DI (no EDC)" if variant == "no_decomposition" else
                    "✓" if spec.use_edc_predictor else "×"
                ),
                "Event-token construction": "×" if spec.event_token_mode.startswith("disabled") else "✓",
                "Adaptive Event Partitioning (AEP)": (
                    "✓ (raw DI)" if variant == "no_decomposition" else
                    "N/A" if variant == "no_edc" else
                    "×" if variant == "no_aep" else "✓"
                ),
                "Event Transformer encoder": "✓" if spec.use_edc_predictor else "×",
                "DWC-guided cross-branch attention": "✓" if spec.cross_branch_fusion_mode == "dwc_guided_attention" else "×",
                "Trend-residual decomposition": "✓" if spec.decomposition else "×",
            }
        for row, value in values.items():
            rows[row][variant] = value
    return rows


def trainable_modules(spec):
    modules = []
    if spec.use_dwc_predictor or spec.input_mode == "raw_causal_di":
        modules.extend(("gradual_encoder", "gradual_projection"))
        if spec.gradual_loss_active:
            modules.append("gradual_aux")
    if spec.use_edc_predictor:
        modules.extend(("token_projection", "position_encoding", "event_encoder", "event_aux"))
        if spec.cross_branch_fusion_mode == "dwc_guided_attention":
            modules.extend(("query", "key", "value", "guided_fusion", "fusion_norm"))
    if spec.cross_branch_fusion_mode != "fixed_mean_of_branch_auxiliary_rul":
        modules.append("regression_head")
    if spec.joint_trend_fusion:
        modules.append("trend_fusion")
    return modules


def main():
    data_path = ROOT / "data/industrial/DI_df.csv"
    group_manifest = ROOT / "manifests/group_manifest.csv"
    endpoint_manifest = ROOT / "manifests/failure_endpoint_manifest.csv"
    folds = {}
    for dataset in DATASETS:
        path = ROOT / "results/frozen_protocol_final/comparison" / dataset / "folds.json"
        fold_data = read_json(path)
        folds[dataset] = {
            "path": str(path),
            "sha256": digest(path),
            "outer_folds": len(fold_data),
            "test_units": sorted({int(unit) for fold in fold_data for unit in fold["test_units"]}),
        }

    hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    rows = component_rows()
    module_matrix = []
    for component, values in rows.items():
        module_matrix.append({"component": component, **values})
    matrix_csv = OUT / "ablation_config_matrix_v2.csv"
    matrix_csv.parent.mkdir(parents=True, exist_ok=True)
    with matrix_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["component", *VARIANT_ORDER], lineterminator="\n")
        writer.writeheader()
        writer.writerows(module_matrix)

    variants = {
        "full": {
            "source": "results/frozen_protocol_final/comparison/{dataset}/test_predictions.csv",
            "action": "Reuse frozen ExpertExactD2Pred predictions; do not construct or retrain Full.",
            "comparison_architecture": "d2pred_expert_exact_v1",
        }
    }
    for name, spec in VARIANT_SPECS.items():
        variants[name] = {
            **asdict(spec),
            "implementation": "analysis.strict_ablation.ControlledExpertExactD2Pred",
            "parent_model": "analysis.expert_exact_d2pred.ExpertExactD2Pred",
            "predictor_architecture": "d2pred_expert_exact_v1 with registered path removal/replacement",
            "preprocessing_entry": "ablation_final.strict_pipeline.build_variant_fold_rows",
            "training_entry": "ablation_final.train_strict_ablation.train_one_variant",
            "cache_key": "dataset + variant + outer_fold + preprocessing_version + input/fold hashes",
            "trainable_modules": trainable_modules(spec),
            "initialization": (
                "Source-isolated trend-fusion pretraining for single-source variants; predictor weights initialized under the run seed"
                if spec.trend_fusion_mode in {"adaptive_stl_anchored_ae", "adaptive_wiener_anchored_ae"} else
                "Matching Full fold trend checkpoint used only to construct EDC; no trend module or RUL checkpoint is loaded into the predictor"
                if spec.variant == "no_dwc" else
                "Matching Full fold trend-fusion checkpoint initializes a trainable predictor trend module; predictor weights use the run seed"
                if spec.joint_trend_fusion else
                "No trend-fusion checkpoint; predictor weights initialized under the run seed"
            ),
            "rul_checkpoint_selection": "Validation-only minimum objective; no Full RUL checkpoint loaded",
            "input_semantics": (
                "The same causal raw DI prefix enters both ExpertExact branches: directly as the gradual-channel sequence and, without residual subtraction, as the signal for AEP event-token descriptors. No Wiener/STL estimator or DWC/EDC decomposition is used."
                if spec.variant == "no_decomposition" else
                "Retain DWC, EDC, AEP-derived descriptors, event Transformer, and cross-branch attention; replace adaptive change-point segmentation with one statistics token computed over the full observed EDC prefix."
                if spec.variant == "no_aep" else None
            ),
        }

    manifest = {
        "protocol_id": "d2pred_strict_ablation_v2",
        "status": "IMPLEMENTED_NOT_VALIDATED",
        "execution_enabled": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "Generated from analysis.strict_ablation.VARIANT_SPECS and the active code hashes below; supersedes prior draft descriptions for this candidate only.",
        "full_policy": "Reuse the frozen main-comparison ExpertExactD2Pred predictions; no Full training, inference, checkpoint, or split changes.",
        "data": {
            "dataset_id": "industrial_ac_di_v1",
            "path": str(data_path),
            "sha256": digest(data_path),
            "group_manifest": {"path": str(group_manifest), "sha256": digest(group_manifest)},
            "failure_endpoint_manifest": {"path": str(endpoint_manifest), "sha256": digest(endpoint_manifest)},
            "datasets": list(DATASETS),
            "device_counts": {name: len(value["test_units"]) for name, value in folds.items()},
            "fold_manifests": folds,
        },
        "architecture": {
            "full_model_class": "analysis.expert_exact_d2pred.ExpertExactD2Pred",
            "ablated_model_class": "analysis.strict_ablation.ControlledExpertExactD2Pred",
            "trend_fusion": "fold-pretrained StageAwareTrendAutoencoder; jointly updated in the RUL fit only when retained by the variant",
            "gradual_encoder": "ExpertExact unidirectional LSTM-64 and projection",
            "event_encoder": "ExpertExact 12-to-64 token projection and three-block, four-head masked Transformer",
            "model_settings": {"hidden_dimension": 64, "attention_heads": 4, "event_layers": 3, "token_dimension": 12, "maximum_tokens": 16, "dropout": 0.15},
            "cross_branch_fusion": "ExpertExact DWC-query event attention when retained; no_fusion uses a fixed mean of the existing gradual_aux and event_aux RUL predictions",
            "output": "ExpertExact Softplus nonnegative RUL scaled by inner-training target scale",
        },
        "preprocessing": {
            "version": "strict_expert_exact_causal_preprocess_v4",
            "prefix_generation": {"start": 30, "stride": 20, "maximum": 20, "include_final_endpoint": True, "sequence_length": 128},
            "trend_estimators": "Causal Wiener and rolling-prefix STL; a single-source variant invokes only its retained estimator.",
            "decomposition": "For decomposition-enabled variants, EDC = observed causal DI prefix - variant DWC; both are resampled to 128 points after prefix-only computation. no_decomposition bypasses Wiener/STL and supplies the raw causal DI prefix to both predictor branches.",
            "no_decomposition_definition": "Two-channel raw-DI control: the gradual branch receives the causal DI sequence; the event branch computes AEP statistical tokens directly from the same causal DI sequence, with no trend estimator, DWC/EDC subtraction, or decomposition cache.",
            "aep": {"sensitivity": 2.5, "minimum_segment_length": 8, "maximum_change_points": 15, "maximum_tokens": 16, "descriptor_dimension": 12},
            "normalization": "Fit on inner-training device rows only; tokens use valid training tokens only.",
            "no_aep_definition": "AEP-only control: retain DWC and EDC, event-token statistics, token encoder, and cross-branch fusion; bypass adaptive change-point detection and encode the full causal EDC prefix as one statistics token. No uniform segmentation is used.",
            "cache_policy": "Variant/dataset/fold-specific cache path; cache signature includes preprocessing version, raw DI hash, and fold manifest hash. No legacy cache discovery or fallback.",
        },
        "training": {
            "datasets": list(DATASETS),
            "outer_folds": 3,
            "seeds": [17, 42, 73],
            "inner_validation_fraction": 0.22,
            "optimizer": "Adam",
            "learning_rate": {"epochs_1_25": 0.001, "epochs_26_plus": 0.0005},
            "batch_size": 8,
            "maximum_epochs": 50,
            "early_stopping": {"starts_after_epoch": 25, "patience": 8},
            "checkpoint_selection": "Minimum inner-validation objective only; one final outer-test evaluation.",
            "trend_pretraining": "Reuse the matching frozen Full fold checkpoint except no_wiener/no_stl, which fit a source-isolated train/validation fusion model; no_autoencoder_fusion uses no learned trend checkpoint; no_decomposition has no trend pretraining.",
            "initialization": "ExpertExact predictor parameters use the specified run seed; no RUL-trained Full checkpoint is loaded. Retained fold trend-fusion checkpoint is loaded from the comparison package.",
            "loss": "Normalized RUL MSE + 0.1 gradual auxiliary MSE when retained + 0.1 event auxiliary MSE when retained + 0.1 trend-fusion objective when retained.",
            "output_root": str(ROOT / "ablation_final/outputs/strict_ablation_v2"),
        },
        "evaluation": {
            "metrics": ["RMSE", "MAE", "P90AE"],
            "aggregation": "Compute each metric over prefixes per device per seed; average seeds within device; then take the equal-device mean.",
            "full_reference": "Frozen main comparison predictions on the same predefined folds/endpoints.",
        },
        "variants": variants,
        "module_matrix": module_matrix,
        "source_hashes": hashes,
        "validation_state": {
            "static_implementation": "present",
            "runtime_validation": "Pending Runtime Validation",
            "training_performed": False,
            "inference_performed": False,
            "gate_tests_performed": False,
            "combined_historical_ablation_table_updated": True,
        },
    }
    manifest_path = OUT / "ablation_manifest_v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    md = [
        "# D2Pred Strict Ablation Protocol v2",
        "",
        "Status: `IMPLEMENTED_NOT_VALIDATED`. Execution is disabled pending code review and separate approval. This document is generated from the variant registry and source hashes in `ablation_manifest_v2.json`.",
        "",
        "## Full Reference",
        "",
        "Full reuses frozen ExpertExactD2Pred predictions from the primary comparison package. It is not retrained or reevaluated.",
        "",
        "## Module Matrix",
        "",
        "`✓` retained, `×` removed, and `N/A` not applicable. The no_decomposition control routes raw causal DI to both predictor branches without decomposition. The AEP row is the only row marked `×` for no_aep: it retains EDC event statistics, the Transformer, and cross-branch attention while representing the full causal EDC prefix with one statistics token. The AEP entry is `N/A` for no_edc because that variant removes the event branch.",
        "",
        "| Component | " + " | ".join(VARIANT_ORDER) + " |",
        "|---|" + "|".join(["---"] * len(VARIANT_ORDER)) + "|",
    ]
    for item in module_matrix:
        md.append("| " + item["component"] + " | " + " | ".join(item[key] for key in VARIANT_ORDER) + " |")
    md += [
        "",
        "## Run Protocol",
        "",
        "Use the comparison package's dataset-specific fold manifests, seeds 17/42/73, prefix schedule 30:20 with the final endpoint included and at most 20 prefixes, Adam at 0.001 with 0.0005 from epoch 26, batch size 8, maximum 50 epochs, and validation-only checkpoint selection with patience 8 after epoch 25. Target scaling and feature normalization are fit on inner-training devices only.",
        "",
        "Each ablation cache is isolated by dataset, variant, fold, preprocessing version, DI checksum, and fold-manifest checksum. The runner rejects execution unless this protocol is explicitly enabled after separate approval.",
        "",
        "## Validation State",
        "",
        "No training, inference, gate, or performance test was run while generating this protocol. Runtime and information-path validation remain pending.",
    ]
    (OUT / "ablation_protocol_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"generated {manifest_path}")
    print(f"generated {OUT / 'ablation_protocol_v2.md'}")
    print(f"generated {matrix_csv}")


if __name__ == "__main__":
    main()
