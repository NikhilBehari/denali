# generalization

Sec. 7 of the DENALI paper: 15 generalization experiments grouped into two
families.

**Held-out object variants** (5 experiments, supplement Figs `non_rr` and
`italics`): train on the base 30-object dataset, evaluate on unseen object
variants.

| Experiment    | Train data        | Held-out test slice                              |
|---------------|-------------------|--------------------------------------------------|
| `italics_loc` | Base 8in objects  | Italic 8in variants of A, 1, circle (loc reg)    |
| `italics_obj` | Base 8in objects  | Italic 8in variants of A, 1, circle (object cls) |
| `nonrr_loc`   | Retroreflective   | Cardstock + white object 1 variants (loc reg)    |
| `nonrr_obj`   | Retroreflective   | Cardstock + white object 1 variants (object cls) |
| `nonrr_size`  | Retroreflective   | Cardstock + white object 1 variants (size cls)   |

**Train/test split variants** (10 experiments, supplement Table
`identity_splits`): train the 1D CNN under different stratifications of
the same base dataset.

| Experiment                   | Strategy        | Train/test partition                    |
|------------------------------|-----------------|------------------------------------------|
| `splits_loc_random_grouped`  | random-grouped  | 70/30 random with 3x repeats grouped     |
| `splits_loc_by_location`     | by_location     | 70 train locations / 30 test locations   |
| `splits_loc_by_object`       | by_object       | 21 train shapes / 9 test shapes          |
| `splits_loc_by_size`         | by_size         | Train on 4in / test on 8in               |
| `splits_obj_random_grouped`  | random-grouped  | "                                        |
| `splits_obj_by_location`     | by_location     | "                                        |
| `splits_obj_by_size`         | by_size         | "                                        |
| `splits_size_random_grouped` | random-grouped  | "                                        |
| `splits_size_by_location`    | by_location     | "                                        |
| `splits_size_by_object`      | by_object       | "                                        |

`obj_by_object` and `size_by_size` are ill-posed (the test labels are
unseen at training time) and are reported as `n/a` in the supplement; no
checkpoint is bundled.

All 15 experiments use the 1D CNN classifier / regressor from
[`main_table`](../main_table/) (`MediumConv1DNet` / `MediumConv1DNetReg`).
Each bundled checkpoint carries its own training-set normalization
statistics (`train_mean`, `train_std`); evaluation applies `log1p`
followed by z-score with those statistics so test inputs match the
distribution the model trained on.

## Install

```bash
pip install -r requirements.txt
```

Both families read from the shared `../saved_dataset/` built by
`main_table.scripts.build_dataset` (the held-out variants are bundled
alongside the base 30 objects). Override the parent directory with
`$DENALI_HELDOUT_DATASETS` or pass `held_out_root=` to the API.

## Commands

```bash
python -m generalization.scripts.eval                  # all 15 experiments
python -m generalization.scripts.eval --only held_out  # 5 held-out object variants
python -m generalization.scripts.eval --only splits    # 10 split variants
```

## Library

```python
from generalization import (
    EXPERIMENTS, SPLIT_EXPERIMENTS, evaluate, evaluate_split, format_metrics,
)
from main_table import build_query

# Held-out experiment:
exp     = EXPERIMENTS["italics_obj"]
metrics = evaluate(exp, device="cuda")
print(format_metrics(exp, metrics))

# Split experiment:
sexp    = SPLIT_EXPERIMENTS["splits_loc_by_location"]
query   = build_query()
metrics = evaluate_split(sexp, query, device="cuda")
print(format_metrics(sexp, metrics))
```
