# DENALI benchmark

Code for the experiments in the DENALI paper, split into two folders:

- [**`main_table/`**](main_table/) — the 12 entries of the main benchmark
  table (Sec. 4): three tasks (location regression, object classification,
  size classification) crossed with four architectures (MLP, 1D CNN,
  3D CNN, Transformer).

- [**`generalization/`**](generalization/) — the 15 entries of the
  supplement generalization analyses (Sec. 7): 10 train/test split
  variants (Table `identity_splits`) and 5 held-out object-variant
  evaluations (italics, cardstock, white).

## Layout

```
benchmark/
├── README.md
├── main_table/                       Sec. 4 main benchmark table
│   ├── __init__.py / cells.py / models.py / data.py / loaders.py
│   ├── evaluate.py / training.py / train.py / metrics.py
│   ├── checkpoints/                  12 .pth, bundled
│   ├── requirements.txt / README.md
│   └── scripts/                      list_cells, eval, train_cell, build_dataset
├── generalization/                   Sec. 7 supplement experiments
│   ├── __init__.py / experiments.py / loaders.py / splits.py / evaluate.py
│   ├── models.py
│   ├── checkpoints/                  5 held-out .pth, bundled
│   ├── split_checkpoints/            10 split .pth + preprocessing manifest
│   ├── requirements.txt / README.md
│   └── scripts/                      eval
└── saved_dataset/                    base 3x3 dataset (joblib + .npy)
```

`tag1_positions.json` lives in the shared [`assets/`](../assets/) folder
at the repo root and is read from both `main_table` and `generalization`.

## Build the dataset

Both `main_table` and `generalization` read from `saved_dataset/`. Build
it once from the raw captures:

```bash
python -m main_table.scripts.build_dataset \
    --data-dir   ../denali-data/data \
    --output-dir saved_dataset
```

`main_table` filters out the held-out object variants at query time, so
the same dataset serves both folders.

## Quickstart

```bash
pip install -r main_table/requirements.txt

# Main benchmark table
python -m main_table.scripts.list_cells            # show registry
python -m main_table.scripts.eval                  # evaluate all 12 entries
python -m main_table.scripts.train_cell --cell <name> --overwrite

# Supplement generalization
python -m generalization.scripts.eval
```

## Citation

```bibtex
@inproceedings{behari2026denali,
  title     = {{DENALI}: A Dataset Enabling Non-Line-of-Sight Spatial Reasoning with Low-Cost LiDARs},
  author    = {Behari, Nikhil and Rivero, Diego and Apostolides, Luke and Ghosh, Suman and Liang, Paul Pu and Raskar, Ramesh},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```
