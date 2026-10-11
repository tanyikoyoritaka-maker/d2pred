"""Minimal compatibility stub for legacy public preprocessing imports.

The frozen public queue does not select the historical CPM model. Keeping a
non-training stub prevents importing that provisional model into the runner.
"""
import torch.nn as nn


class CPMRULModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("Historical CPMRULModel is excluded from the frozen public protocol")
