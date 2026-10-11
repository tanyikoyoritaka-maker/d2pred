"""Compatibility import for public-data preprocessing.

The frozen public runner no longer trains the historical ExpertAlignedD2Pred.
This name is retained only because the legacy public DI helper imports it while
building decomposition rows; all actual Expert training uses ExpertExactD2Pred.
"""
from .expert_exact_d2pred import ExpertExactD2Pred


class ExpertAlignedD2Pred(ExpertExactD2Pred):
    """Compatibility alias; never selected as a benchmark model."""

    architecture_version = "compatibility_alias_not_used_for_training"
