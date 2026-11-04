# D2Pred experiment archive

The active sources and statuses are recorded in
`manifests/active_experiment_registry.json`. The primary AC comparisons and
sensitivity results remain in `results/frozen_protocol_final/`; public results
and computational measurements retain their separate verified packages.

The proposed controlled-ablation package is invalidated and has no active
runner. Historical recovery is documented in
`ablation_recovery/reports/historical_ablation_recovery_report.md`; its actual
code, run records, data lineage, and saved predictions are the review evidence.
Earlier protocols are retrieval clues only. No new protocol or training run is
authorized until the recovered implementation and results receive human review.

Superseded ablation results, protocols, and invalid v3 outputs are preserved
with SHA-256 manifests under `../d2pred_rubbish/`. They are historical or
rejected materials, not current paper evidence. The cleanup status is recorded
in `ablation_recovery/reports/ablation_contamination_cleanup_report.md`.
Large data, checkpoints and runtime outputs remain outside Git and are retained
in the working archive. The proprietary raw-waveform-to-DI mapping is not released.
