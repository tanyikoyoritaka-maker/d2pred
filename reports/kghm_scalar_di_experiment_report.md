# KGHM scalar-DI experiment

This report records the replacement experiment required by the baseline-provenance review.

## Input and implementation

- External input: one scalar DI prefix.
- Internal adapter: causal DI level plus causal first difference as two relation nodes.
- No external DWC or EDC tensor is supplied to KGHM.
- Training protocol: three outer folds, seeds 17/42/73, batch size 8, maximum 50 epochs, inner-validation checkpoint selection, and patience 8.
- The implementation is a local scalar-DI surrogate reimplementation; it is not official authors' code.

## AC results

| Scope | Devices | RMSE (cycles) | MAE (cycles) | P90AE (cycles) | RMSE 95% CI |
|---|---:|---:|---:|---:|---|
| ACSim | 17 | 270.0273 | 238.1559 | 405.7708 | 220.7277--322.8573 |
| ACCom | 16 | 373.6881 | 300.4407 | 616.8156 | 302.9957--447.5397 |
| ACAll | 33 | 292.4046 | 245.0434 | 467.0069 | 249.4467--340.2364 |

For comparison, the manuscript's D²Pred RMSE values are 236.07 cycles (ACSim), 315.76 cycles (ACCom), and 276.98 cycles (ACAll). The scalar-DI KGHM RMSE is therefore higher by 33.96, 57.93, and 15.42 cycles, respectively.

## Artifact locations

Complete AC outputs are retained under:

- `PythonProject2/outputs/kghm_scalar_di_nested_experiment/`
- `PythonProject2/outputs/kghm_scalar_di_acall_experiment/`

The former DWC/EDC-input KGHM checkpoints and logs were moved to `d2pred_rubbish/kghm_old_dwc_edc/` outside the Git repository and are not used for compliant comparisons.

The public-bearing KGHM results are retained as compliant existing results. The replacement experiment in this report concerns the AC KGHM input path only.
