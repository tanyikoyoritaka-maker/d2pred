# Ablation Evidence and Submission Status

The primary comparison, public benchmark, sensitivity, and computational results
remain registered independently of the ablation work.

The recovered 2026-10-30 ablation run and its source snapshot are documented in
`ablation_recovery/reports/historical_ablation_recovery_report.md`. Its metrics
were recomputed from saved predictions. The run is only partially verified because
the executed source was not cryptographically bound to the run, and several model
routes do not match the ExpertExactD2Pred reference architecture.

The recent v3 controlled-ablation implementation and outputs are marked
`INVALID_LEGACY_ABLATION`. Their evidence is preserved under
`../../d2pred_rubbish/ablation_contamination_cleanup_20261103/`; the candidate
figure sources and outputs have a separate supplemental quarantine manifest.
The former pre-execution `PASS` audit is retained as historical evidence and is
not execution authorization.

No formal protocol has been generated or activated. The historical recovery
report is awaiting human review. No ablation values have been approved as paper
sources, and the manuscript table and figure have not been changed in this cleanup.

The manuscript figure `eaaipaper/fig/ACAll_ablation_extreme_cases.pdf` is retained
unchanged. Its hash is recorded in `manifests/paper_checksum_manifest.csv`, but
its generation provenance has not yet been established against the historical
run or the quarantined v3 output.
