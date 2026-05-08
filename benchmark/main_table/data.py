"""Dataset, preprocessing, and split utilities for the main benchmark table.

The 3x3 saved dataset is a flat list of `(samples, H, W, 128)` SPAD histograms
along with parallel metadata lists (object, size, lighting, capture number,
capture resolution). `DataSet.load_from_directory` reads the bundled joblib
artifacts; `query_data` filters and explodes per-capture repeats; the array
datasets handle the `log1p` + z-score preprocessing applied during training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# The 30 canonical base objects evaluated in Sec. 4. Held-out variants and
# no-object backgrounds are not part of the main-table training or
# evaluation set and are filtered out at query time.
BASE_OBJECTS: tuple[str, ...] = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "I", "O", "S", "U", "V",
    "circle", "downarrow", "halfcircle", "hexagon", "minus",
    "plus", "square", "triangle", "uparrow", "widerectangle",
)

# Kept for backwards-compatible callers; ignored when ``included_objects``
# is set on :meth:`DataSet.query_data`.
DEFAULT_EXCLUDED_OBJECTS: tuple[str, ...] = ("NOOBJECT", "NOOBJECTMOUNT")

QueryResult = dict[str, list[Any]]


@dataclass
class DataSet:
    """In-memory container for a saved 3x3 capture set, one entry per capture-folder/capture-number."""
    names:               list[str]        = field(default_factory=list)
    objects:             list[str]        = field(default_factory=list)
    sizes:               list[str]        = field(default_factory=list)
    capture_resolutions: list[str]        = field(default_factory=list)
    lightings:           list[str]        = field(default_factory=list)
    capture_numbers:     list[int]        = field(default_factory=list)
    spad_histograms:     list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.names)

    @classmethod
    def load_from_directory(cls, directory: str | Path) -> "DataSet":
        d = Path(directory)
        spad_dir = d / "spad_histograms"
        return cls(
            names               = joblib.load(d / "names.joblib"),
            objects             = joblib.load(d / "objects.joblib"),
            sizes               = joblib.load(d / "sizes.joblib"),
            capture_resolutions = joblib.load(d / "capture_resolutions.joblib"),
            lightings           = joblib.load(d / "lightings.joblib"),
            capture_numbers     = joblib.load(d / "capture_numbers.joblib"),
            spad_histograms     = [np.load(p) for p in sorted(spad_dir.glob("*.npy"))],
        )

    def save_to_directory(self, directory: str | Path) -> None:
        """Write this dataset to ``directory`` in the joblib + .npy layout
        that ``load_from_directory`` reads."""
        d = Path(directory)
        spad_dir = d / "spad_histograms"
        spad_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.names,               d / "names.joblib")
        joblib.dump(self.objects,             d / "objects.joblib")
        joblib.dump(self.sizes,               d / "sizes.joblib")
        joblib.dump(self.capture_resolutions, d / "capture_resolutions.joblib")
        joblib.dump(self.lightings,           d / "lightings.joblib")
        joblib.dump(self.capture_numbers,     d / "capture_numbers.joblib")
        width = max(3, len(str(len(self.spad_histograms) - 1)))
        for i, h in enumerate(self.spad_histograms):
            np.save(spad_dir / f"{i:0{width}d}.npy", np.asarray(h))

    @classmethod
    def from_raw_captures(
        cls,
        data_dir:                str | Path,
        *,
        exclude_name_prefixes:   Sequence[str] = (),
    ) -> "DataSet":
        """Build a :class:`DataSet` by walking the raw capture folders in ``data_dir``.

        Each subfolder is expected to be named
        ``<object>_<size>_<resolution>_<lighting>_NLOSdata`` and to contain a
        ``spad_histogram/`` directory with one ``.npy`` file per location
        (one entry's `spad_histogram` is the stack of three repeats).

        Pass ``exclude_name_prefixes`` to skip folders whose name starts
        with any of the given strings.
        """
        root = Path(data_dir)
        excluded = tuple(exclude_name_prefixes)
        names, objects, sizes, capture_resolutions = [], [], [], []
        lightings, capture_numbers, spad_histograms = [], [], []

        for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_NLOSdata")):
            if excluded and any(folder.name.startswith(prefix) for prefix in excluded):
                continue
            stem = folder.name[:-len("_NLOSdata")]
            parts = stem.split("_")
            if len(parts) < 4:
                continue
            object_type, size, capture_resolution, lighting = parts[0], parts[1], parts[2], parts[3]
            spad_dir = folder / "spad_histogram"
            if not spad_dir.exists():
                continue
            for npy_path in sorted(spad_dir.glob("*.npy")):
                try:
                    capture_number = int(npy_path.stem)
                except ValueError:
                    continue
                names.append(folder.name)
                objects.append(object_type)
                sizes.append(size)
                capture_resolutions.append(capture_resolution)
                lightings.append(lighting)
                capture_numbers.append(capture_number)
                spad_histograms.append(np.load(npy_path))

        return cls(
            names=names, objects=objects, sizes=sizes,
            capture_resolutions=capture_resolutions, lightings=lightings,
            capture_numbers=capture_numbers, spad_histograms=spad_histograms,
        )

    def query_data(
        self,
        *,
        capture_resolutions:        Sequence[str] | None     = None,
        average_spad_histograms:    bool                     = True,
        included_objects:           Sequence[str] | None     = None,
        excluded_objects:           Sequence[str]            = DEFAULT_EXCLUDED_OBJECTS,
        location_mapping:           Mapping[int, Sequence[float]] | None = None,
    ) -> QueryResult:
        """Filter the dataset to a tabular query result of equal-length lists.

        With `average_spad_histograms=False` the three repeated captures per
        scene are emitted as separate samples (giving 3x more rows). When
        `included_objects` is set, only those object labels are kept (the
        more permissive ``excluded_objects`` is ignored). When
        `location_mapping` is provided, each row also carries its `(x, y)`
        from the AprilTag-derived gantry-position table.
        """
        keep = set(included_objects) if included_objects is not None else None
        out: QueryResult = {k: [] for k in (
            "names", "capture_numbers", "objects", "sizes",
            "capture_resolutions", "lightings", "spad_histograms", "locations",
        )}
        for i, name in enumerate(self.names):
            obj = self.objects[i]
            if keep is not None:
                if obj not in keep:
                    continue
            elif obj in excluded_objects:
                continue
            if capture_resolutions is not None and self.capture_resolutions[i] not in capture_resolutions:
                continue

            stack = np.asarray(self.spad_histograms[i])
            histograms = ([stack.mean(axis=0)] if average_spad_histograms
                          else [np.asarray(h) for h in stack])

            location = (
                None if location_mapping is None
                else [float(location_mapping[self.capture_numbers[i]][0]),
                      float(location_mapping[self.capture_numbers[i]][1])]
            )

            for h in histograms:
                out["names"].append(name)
                out["capture_numbers"].append(int(self.capture_numbers[i]))
                out["objects"].append(self.objects[i])
                out["sizes"].append(self.sizes[i])
                out["capture_resolutions"].append(self.capture_resolutions[i])
                out["lightings"].append(self.lightings[i])
                out["spad_histograms"].append(np.asarray(h))
                out["locations"].append(location)

        if location_mapping is None:
            out.pop("locations")
        return out


def load_capture_number_to_location_mapping(json_path: str | Path) -> dict[int, tuple[float, float]]:
    """Read AprilTag gantry positions; key on integer capture-number, value `(x, y)`.

    Keys in the JSON are formatted ``frame{N}_tag1`` (e.g. `frame17_tag1`); we
    keep only the leading frame index.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    mapping: dict[int, tuple[float, float]] = {}
    for key, value in raw.items():
        if not key.startswith("frame"):
            continue
        frame = int(key.split("_", 1)[0].removeprefix("frame"))
        mapping[frame] = (float(value[0]), float(value[1]))
    return mapping


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std:  float


def apply_histogram_transform(
    X: np.ndarray,
    *,
    log_transform: bool = True,
    normalize:     bool = True,
    stats:         NormalizationStats | None = None,
) -> tuple[np.ndarray, NormalizationStats]:
    """Preprocess a histogram batch.

    Steps:
      1. (optional) `log1p(clip(X, 0, ∞))` to compress the photon-count range.
      2. (optional) z-score with `stats.mean` / `stats.std`. If `stats` is
         `None`, statistics are computed from `X` itself.
    """
    X = np.asarray(X, dtype=np.float32)
    if log_transform:
        X = np.log1p(np.clip(X, 0.0, None))
    if normalize:
        if stats is None:
            stats = NormalizationStats(mean=float(X.mean()), std=float(X.std() + 1e-6))
        X = (X - stats.mean) / stats.std
    else:
        stats = NormalizationStats(mean=0.0, std=1.0)
    return X.astype(np.float32, copy=False), stats


class _ArrayDatasetBase(Dataset):
    """Common preprocessing path for the classification and regression datasets."""

    def __init__(
        self,
        X_list:        Sequence[np.ndarray],
        meta:          Mapping[str, Sequence[Any]],
        *,
        log_transform: bool                         = True,
        normalize:     bool                         = True,
        stats:         NormalizationStats | None    = None,
    ):
        X = np.stack(X_list, axis=0).astype(np.float32)
        self.X, self.normalization_stats = apply_histogram_transform(
            X, log_transform=log_transform, normalize=normalize, stats=stats,
        )
        self.meta = {k: list(v) for k, v in meta.items()}

    def __len__(self) -> int:
        return self.X.shape[0]


class ClassificationArrayDataset(_ArrayDatasetBase):
    """X = (N, H, W, 128) float, y = int class index."""

    def __init__(self, X_list, y_idx, meta, **kwargs):
        super().__init__(X_list, meta, **kwargs)
        self.y = np.asarray(y_idx, dtype=np.int64)

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.X[i]).float()
        y = torch.tensor(self.y[i]).long()
        return x, y, {k: v[i] for k, v in self.meta.items()}


class RegressionArrayDataset(_ArrayDatasetBase):
    """X = (N, H, W, 128) float, y = (x, y) location."""

    def __init__(self, X_list, y_pairs, meta, **kwargs):
        super().__init__(X_list, meta, **kwargs)
        y_pairs = np.asarray(y_pairs, dtype=np.float32)
        if y_pairs.ndim != 2 or y_pairs.shape[1] != 2:
            raise ValueError(f"Regression targets must be shape [N, 2]; got {y_pairs.shape}.")
        self.y = y_pairs

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.X[i]).float()
        y = torch.from_numpy(self.y[i]).float()
        return x, y, {k: v[i] for k, v in self.meta.items()}


def collate_with_meta(batch):
    """Stack tensors and aggregate per-sample meta dicts into a batch dict-of-lists."""
    xs, ys, metas = zip(*batch)
    X = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    out: dict[str, list[Any]] = {}
    for m in metas:
        for k, v in m.items():
            out.setdefault(k, []).append(v)
    return X, y, out


# Train/test splits: random_state=42, test_size=0.3.

@dataclass(frozen=True)
class Split:
    train_indices: list[int]
    eval_indices:  list[int]


def stratified_split(
    y: Sequence[Any], *, test_size: float = 0.3, random_state: int = 42,
) -> Split:
    """Stratified random split on label `y` (used for classification cells)."""
    n = len(y)
    train_idx, eval_idx = train_test_split(
        np.arange(n), test_size=test_size, stratify=np.asarray(y),
        random_state=random_state,
    )
    return Split(train_indices=train_idx.tolist(), eval_indices=eval_idx.tolist())


def random_split(n: int, *, test_size: float = 0.3, random_state: int = 42) -> Split:
    """Plain random split (used for regression cells)."""
    train_idx, eval_idx = train_test_split(
        np.arange(n), test_size=test_size, shuffle=True, random_state=random_state,
    )
    return Split(train_indices=train_idx.tolist(), eval_indices=eval_idx.tolist())


__all__ = [
    "DEFAULT_EXCLUDED_OBJECTS",
    "DataSet",
    "NormalizationStats",
    "Split",
    "ClassificationArrayDataset",
    "RegressionArrayDataset",
    "apply_histogram_transform",
    "collate_with_meta",
    "load_capture_number_to_location_mapping",
    "random_split",
    "stratified_split",
]
