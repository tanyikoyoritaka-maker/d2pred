# D2Pred final reproducibility rebuild — pre-experiment gate

This directory is a clean target for the final frozen repository. No model fitting,
checkpoint selection, result replacement, or manuscript-table update has been run.
The existing `PythonProject2` tree and all historical results remain untouched.

The core implementation and authorised anonymised industrial DI have now been
migrated. Frozen cohort/fold and model pre-execution audits pass, and the migrated
causal pipeline passes 19 unit tests. No final model fitting has started. Historical
checkpoints, predictions, provisional tables, and figures were not migrated.

Current gate: `MIGRATION_COMPLETE_FINAL_TRAINING_NOT_STARTED`.
