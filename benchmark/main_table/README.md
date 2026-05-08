# main_table

Sec. 4 of the DENALI paper: the 12 main benchmark entries — three tasks
(location regression, object classification, size classification) × four
architectures (MLP, 1D CNN, 3D CNN, Transformer). Configurations are
listed in [`cells.py`](cells.py).

## Install

```bash
pip install -r requirements.txt
```

The 3x3 SPAD dataset is read from `../saved_dataset/` by default. Override
with `$DENALI_SAVED_DATASET=/path/to/saved_dataset` or with the
`saved_dataset_dir=` kwarg to `build_query()`. To build it from the raw
captures in `denali-data/data/` (run from inside `benchmark/`):

```bash
python -m main_table.scripts.build_dataset \
    --data-dir   ../denali-data/data \
    --output-dir saved_dataset
```

`build_query()` filters out held-out variants (italic, cardstock,
white) and no-object background captures (`NOOBJECT`, `NOOBJECTMOUNT`)
so `main_table` only sees the 30 base objects from Sec. 3.

## Commands

```bash
python -m main_table.scripts.list_cells                       # print the registry
python -m main_table.scripts.eval                             # evaluate all 12 entries
python -m main_table.scripts.train_cell --cell <name> --overwrite
```

## Library

```python
from main_table import CELLS, build_query, load_cell, evaluate, format_metrics

query   = build_query()
cell    = CELLS["obj_1dcnn"]
model   = load_cell(cell, device="cuda")
metrics = evaluate(cell, model, query, device="cuda")
print(format_metrics(cell, metrics))
```
