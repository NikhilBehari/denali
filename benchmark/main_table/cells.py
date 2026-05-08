"""Registry of the 12 entries in the main benchmark table (Sec. 4).

Each :class:`Cell` records the architecture, training config, and checkpoint
path for one entry. Hyperparameters were chosen per entry to bring out the
best of each architecture on its task.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from torch import nn

from .models import (
    SmallMLP, SmallMLPReg,
    MediumConv1DNet, MediumConv1DNetReg,
    MediumConv3DOverCNet, MediumConv3DOverCRegNet,
    DeepConv3DNet,
    TransformerNet, TransformerNetReg,
)


__all__ = [
    "CELLS",
    "Cell",
    "TrainConfig",
    "Task",
    "PACKAGE_DIR",
    "CHECKPOINTS_DIR",
    "cells_in_table_order",
]


PACKAGE_DIR     = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PACKAGE_DIR / "checkpoints"

Task = Literal["regression", "object_classification", "size_classification"]


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters used to train one entry's checkpoint."""
    lr:            float
    num_epochs:    int
    batch_size:    int
    log_transform: bool  = False
    normalize:     bool  = False
    random_state:  int   = 42
    test_size:     float = 0.3


@dataclass(frozen=True)
class Cell:
    """One main-table entry: architecture, training config, checkpoint path."""
    name:         str
    task:         Task
    y_key:        str
    n_outputs:    int
    arch_factory: Callable[[], nn.Module]
    arch_name:    str
    ckpt:         Path
    config:       TrainConfig

    def short(self) -> str:
        return f"{self.name:<18} [{self.arch_name}, lr={self.config.lr}, ep={self.config.num_epochs}]"


# Architecture factories: zero-arg closures so each Cell is fully described by
# `Cell.arch_factory()`. Classification factories curry in `n_outputs`.

def _reg_smallmlp():    return SmallMLPReg()
def _reg_1dcnn():       return MediumConv1DNetReg()
def _reg_3dcnn():       return MediumConv3DOverCRegNet()
def _reg_transformer(): return TransformerNetReg()

def _cls_smallmlp(n: int):     return lambda: SmallMLP(num_classes=n)
def _cls_1dcnn(n: int):        return lambda: MediumConv1DNet(num_classes=n)
def _cls_3dcnn_single(n: int): return lambda: MediumConv3DOverCNet(num_classes=n)
def _cls_3dcnn_deep(n: int):   return lambda: DeepConv3DNet(num_outputs=n)
def _cls_transformer(n: int):  return lambda: TransformerNet(num_classes=n)


CELLS: dict[str, Cell] = {

    # Location regression (planar (x, y))
    "reg_mlp": Cell(
        name="reg_mlp",
        task="regression",
        y_key="locations", n_outputs=2,
        arch_factory=_reg_smallmlp, arch_name="SmallMLP",
        ckpt=CHECKPOINTS_DIR / "reg_mlp.pth",
        config=TrainConfig(lr=1e-3, num_epochs=60, batch_size=64),
    ),
    "reg_1dcnn": Cell(
        name="reg_1dcnn",
        task="regression",
        y_key="locations", n_outputs=2,
        arch_factory=_reg_1dcnn, arch_name="MediumConv1DNetReg",
        ckpt=CHECKPOINTS_DIR / "reg_1dcnn.pth",
        config=TrainConfig(lr=1e-3, num_epochs=60, batch_size=64),
    ),
    "reg_3dcnn": Cell(
        name="reg_3dcnn",
        task="regression",
        y_key="locations", n_outputs=2,
        arch_factory=_reg_3dcnn, arch_name="MediumConv3DOverCRegNet",
        ckpt=CHECKPOINTS_DIR / "reg_3dcnn.pth",
        config=TrainConfig(lr=1e-3, num_epochs=75, batch_size=64),
    ),
    "reg_transformer": Cell(
        name="reg_transformer",
        task="regression",
        y_key="locations", n_outputs=2,
        arch_factory=_reg_transformer, arch_name="TransformerNet (cls-token)",
        ckpt=CHECKPOINTS_DIR / "reg_transformer.pth",
        config=TrainConfig(lr=1e-4, num_epochs=75, batch_size=64),
    ),

    # Object classification (30 classes)
    "obj_mlp": Cell(
        name="obj_mlp",
        task="object_classification",
        y_key="objects", n_outputs=30,
        arch_factory=_cls_smallmlp(30), arch_name="SmallMLP",
        ckpt=CHECKPOINTS_DIR / "obj_mlp.pth",
        config=TrainConfig(lr=1e-5, num_epochs=100, batch_size=256,
                           log_transform=True, normalize=True),
    ),
    "obj_1dcnn": Cell(
        name="obj_1dcnn",
        task="object_classification",
        y_key="objects", n_outputs=30,
        arch_factory=_cls_1dcnn(30), arch_name="MediumConv1DNet",
        ckpt=CHECKPOINTS_DIR / "obj_1dcnn.pth",
        config=TrainConfig(lr=1e-3, num_epochs=100, batch_size=64),
    ),
    "obj_3dcnn": Cell(
        name="obj_3dcnn",
        task="object_classification",
        y_key="objects", n_outputs=30,
        arch_factory=_cls_3dcnn_single(30), arch_name="MediumConv3DOverCNet",
        ckpt=CHECKPOINTS_DIR / "obj_3dcnn.pth",
        config=TrainConfig(lr=1e-3, num_epochs=200, batch_size=256,
                           log_transform=True, normalize=True),
    ),
    "obj_transformer": Cell(
        name="obj_transformer",
        task="object_classification",
        y_key="objects", n_outputs=30,
        arch_factory=_cls_transformer(30), arch_name="TransformerNet (cls-token)",
        ckpt=CHECKPOINTS_DIR / "obj_transformer.pth",
        config=TrainConfig(lr=1e-5, num_epochs=50, batch_size=256,
                           log_transform=True, normalize=True),
    ),

    # Size classification (2 classes)
    "size_mlp": Cell(
        name="size_mlp",
        task="size_classification",
        y_key="sizes", n_outputs=2,
        arch_factory=_cls_smallmlp(2), arch_name="SmallMLP",
        ckpt=CHECKPOINTS_DIR / "size_mlp.pth",
        config=TrainConfig(lr=1e-3, num_epochs=75, batch_size=64),
    ),
    "size_1dcnn": Cell(
        name="size_1dcnn",
        task="size_classification",
        y_key="sizes", n_outputs=2,
        arch_factory=_cls_1dcnn(2), arch_name="MediumConv1DNet",
        ckpt=CHECKPOINTS_DIR / "size_1dcnn.pth",
        config=TrainConfig(lr=1e-3, num_epochs=100, batch_size=64),
    ),
    "size_3dcnn": Cell(
        name="size_3dcnn",
        task="size_classification",
        y_key="sizes", n_outputs=2,
        arch_factory=_cls_3dcnn_deep(2), arch_name="DeepConv3DNet",
        ckpt=CHECKPOINTS_DIR / "size_3dcnn.pth",
        config=TrainConfig(lr=1e-3, num_epochs=75, batch_size=64),
    ),
    "size_transformer": Cell(
        name="size_transformer",
        task="size_classification",
        y_key="sizes", n_outputs=2,
        arch_factory=_cls_transformer(2), arch_name="TransformerNet (cls-token)",
        ckpt=CHECKPOINTS_DIR / "size_transformer.pth",
        config=TrainConfig(lr=1e-4, num_epochs=10, batch_size=256,
                           log_transform=True, normalize=True),
    ),
}


def cells_in_table_order() -> list[Cell]:
    """Return the 12 entries in the order they appear in the main table."""
    return list(CELLS.values())
