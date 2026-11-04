# D2Pred Strict Ablation Protocol v2

Status: `VERIFIED_COMPLETED`.  This document is generated from the variant registry and source hashes in `ablation_manifest_v2.json`.



## Module Matrix

`✓` retained, `×` removed, and `N/A` not applicable. The no_decomposition control routes raw causal DI to both predictor branches without decomposition. The AEP row is the only row marked `×` for no_aep: it retains EDC event statistics, the Transformer, and cross-branch attention while representing the full causal EDC prefix with one statistics token. The AEP entry is `N/A` for no_edc because that variant removes the event branch.

| Component | full | no_wiener | no_stl | no_autoencoder_fusion | no_dwc | no_edc | no_decomposition | no_fusion | no_aep |
|---|---|---|---|---|---|---|---|---|---|
| Wiener trend estimator | ✓ (frozen Full) | × | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ |
| STL trend estimator | ✓ (frozen Full) | ✓ | × | ✓ | ✓ | ✓ | × | ✓ | ✓ |
| Adaptive trend fusion / bounded AE | ✓ (frozen Full) | ✓ | ✓ | × (fixed mean) | ✓ | ✓ | N/A | ✓ | ✓ |
| DWC predictor branch | ✓ (frozen Full) | ✓ | ✓ | ✓ | × | ✓ | ✓ (raw DI) | ✓ | ✓ |
| EDC residual used for event input | ✓ (frozen Full) | ✓ | ✓ | ✓ | ✓ | × | Raw DI (no EDC) | ✓ | ✓ |
| Event-token construction | ✓ (frozen Full) | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ |
| Adaptive Event Partitioning (AEP) | ✓ (frozen Full) | ✓ | ✓ | ✓ | ✓ | N/A | ✓ (raw DI) | ✓ | × |
| Event Transformer encoder | ✓ (frozen Full) | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ |
| DWC-guided cross-branch attention | ✓ (frozen Full) | ✓ | ✓ | ✓ | × | × | ✓ | × | ✓ |
| Trend-residual decomposition | ✓ (frozen Full) | ✓ | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ |

## Run Protocol

Use the comparison package's dataset-specific fold manifests, seeds 17/42/73, prefix schedule 30:20 with the final endpoint included and at most 20 prefixes, Adam at 0.001 with 0.0005 from epoch 26, batch size 8, maximum 50 epochs, and validation-only checkpoint selection with patience 8 after epoch 25. Target scaling and feature normalization are fit on inner-training devices only.

Each ablation cache is isolated by dataset, variant, fold, preprocessing version, DI checksum, and fold-manifest checksum. The runner rejects execution unless this protocol is explicitly enabled after separate approval.

## Validation State

VERIFIED_COMPLETED
