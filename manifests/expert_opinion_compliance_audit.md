# Expert-opinion compliance audit

## Overall result

The repository passes the industrial cohort/fold audit and the completed AC comparison rerun audit, but it is **not yet compliant for the requested final freeze**. The principal implementation blocker is a mismatch in the descriptor definition.

## Sections 2 and 3 implementation

The code contains the trend-fusion network, gradual LSTM encoder, token projection, three-layer Transformer encoder with padding masking, attention-guided fusion, Softplus output and joint loss. The metadata now records the actual joint loss:

`RUL_MSE + 0.1 gradual_aux_MSE + 0.1 event_aux_MSE + 0.1 trend_fusion_loss`.

However, `event_tokens()` calls `segment_tokens(..., mode="stats")`; those twelve values are statistical descriptors (`start_rel`, `end_rel`, length, mean, standard deviation, extrema, slope, energy, area, residual energy and jump). They are not demonstrably the manuscript's “twelve raw residual descriptors”. Therefore exact compliance with Sections 2 and 3 cannot be claimed until the descriptor definition is reconciled with the manuscript.

The three-block encoder uses a padding key mask. Whether this is the exact Section 3 masked self-attention mask must be checked against the manuscript equation; the local code alone is insufficient to certify that detail.

## Industrial data

The frozen industrial audit passes: initial cohort 38; excluded units 1, 3, 11, 35 and 38; 33 eligible devices; ACSim 17 and ACCom 16. The endpoint rule is deterministic (`RUL=0` final observed cycle), and the approved cycles-based grouping with Unit 12/16 in ACCom and Unit 23 in ACSim is recorded in the manifests.

## Experiments

The formal ACSim/ACCom rerun is complete and verified: 3 folds × 3 seeds × 11 models, 21,714 predictions, no NaNs. ACAll, public benchmark, ablation, sensitivity, generalisation, semi-synthetic and robustness experiments are not yet executed. Consequently their tables and figures cannot be final.

## Reproducibility and submission materials

Retained baseline registry and labels are consistent. Figures 3 and 4, the complete supplementary archive, final all-suite environment/checksum records and owner authorisation artifact are not yet complete. The Data Availability draft permits anonymised DI arrays and unit names but not detailed DI construction; the exact owner-authorised artefact list still needs to be recorded.

The machine-readable audit is `expert_opinion_compliance_audit.json`. No new experiments were started by this audit.
