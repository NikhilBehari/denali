"""1D-CNN architectures used by the supplement generalization checkpoints.

Re-exports the classifier and regressor from `main_table.models` so the
supplement experiments use the same implementation as the main benchmark.
The class names match the saved state-dict keys.
"""
from main_table.models import MediumConv1DNet, MediumConv1DNetReg

__all__ = [
    "MediumConv1DNet",
    "MediumConv1DNetReg",
]
