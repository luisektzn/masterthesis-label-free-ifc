# Classification of HeLa and NIH3T3 cells

Code for the label-free classification experiments of the master thesis *Towards High-Throughput Label-Free Imaging
Flow Cytometry: Hydrodynamic Focusing, Cell Dynamics and ONN-Compatible Classification* (Heidelberg University, 2026).

Two networks are compared on single-cell images of HeLa and NIH3T3 cells:

* **Chip network:** fully connected network with an 8 × 8 pixel input, two hidden layers with 64 neurons each, ReLU,
  two outputs (8450 parameters). It is designed for the optoelectronic architecture of Song et al. (2024) and is
  trained and evaluated here in software only.
* **CNN:** digital reference with a 64 × 64 pixel input and four convolutional blocks (about 420 000 parameters).

A logistic regression on four simple image properties (crop height and width, fill fraction, raw contrast) serves as
a feature control.

## Notebooks

Run the notebooks in this order. Notebooks 02 to 07 are independent of each other. Notebook 08 only reads their
result files.

| Notebook | Content | Thesis |
|---|---|---|
| `00_cellpose_crops` | Cellpose segmentation, morphological filter and square single-cell crops (z-stack, Andor Zyla flow, static Phantom) | Methods, Cell segmentation with Cellpose |
| `01_datasets` | Loading, edge check, size filter, image counts, preprocessing figures | Methods, Datasets and Preprocessing; Results, Datasets; appendix tables |
| `02_zstack` | Z-stack dataset, split by stack and by acquisition day, result per held-out day | Results, Andor Zyla z-stack dataset |
| `03_zstack_to_zyla_flow` | Transfer from the z-stack to the Andor Zyla flow dataset, raw contrast | Results, Andor Zyla flow dataset (transfer) |
| `04_within_flow` | Cross-validation within the Andor Zyla flow and the Phantom flow dataset | Results, both flow datasets (within flow) |
| `05_static_phantom` | Static Phantom dataset (cross-validation, per imaging mode, held-out day) and transfer to the Phantom flow dataset | Results, static Phantom dataset and Phantom flow dataset (transfer) |
| `06_input_size` | Input size from 8 × 8 to 64 × 64 on the static Phantom Ph2 images | Results, static Phantom dataset (input size) |
| `07_apd` | APD dataset: cross-validation over videos, stained and unstained cells, held-out day, transfer from static Ph2 images | Results, APD dataset |
| `08_figures` | All classification figures of the thesis and an overview table, from the result files | Results and appendix figures |

Shared modules:

* `thesis_config.py`: data paths, dataset loading, preprocessing, augmentation, training settings and splits.
* `thesis_ml.py`: data scanning, preprocessing, grouped splits, models, training and evaluation.
* `thesis_style.py`: figure style (fonts, sizes, colours).
* `thesis_figures.py`: classification figures (confusion matrices, summary plots, input size).

## Methods in short

* **Preprocessing:** background margin trimmed to 1 µm, size filter on the shorter crop side (1st to 99th percentile
  per dataset), background subtraction, resampling to 0.65 µm/px (Andor Zyla datasets) or 0.925 µm/px (Phantom
  datasets), padding to a square, resizing to 64 × 64, circular background mask, percentile normalization. Smaller
  inputs are computed from the 64 × 64 image.
* **Training:** AdamW (learning rate 1e-3, weight decay 1e-4), cosine decay to 1e-5, 50 epochs, batch size 64,
  class-weighted cross-entropy, augmentation at 64 × 64 (shift, rotation ±5°, flips, Gaussian blur, brightness,
  contrast, gamma, noise). The epoch with the highest validation macro-F1 is kept. Seeds 0, 1 and 2.
* **Splits:** grouped cross-validation, so that images of one stack, recording, raw image or video are never split
  between training and test. Held-out acquisition days where both cell lines were recorded on each day.

## Data

The image data are not part of this repository. `thesis_config.py` expects the following folders below
`DATA_ROOT` (set with the environment variable `THESIS_DATA_ROOT` or directly in the file):

```
cellpose/z-stack/<HeLa_new|NIH3T3_new>/<date>/<stack>/*.tif           z-stack crops
cellpose/microchannel_bf/<HeLa_new|NIH3T3_new>/<recording>/*.tif      Andor Zyla flow crops (with summary.csv)
cellpose/snapshot_phantom/data/<HeLa|NIH3T3>/<BF|Ph2>/<date>/*.tif    static Phantom crops
Code/PC/cell_analysis/.../run_*/                                      Phantom flow (tables/ and crop_curation/)
cellpose/mixed_cells/<date>/<video>/<HeLa|NIH3T3>/crops/*.tif         APD dataset
```

The edge check of the Andor Zyla flow crops needs the `summary.csv` files written by `00_cellpose_crops` and the raw
images. If the raw images were moved, set `RAW_PATH_REPLACEMENTS` or `RAW_SEARCH_ROOTS` in `thesis_config.py`.

## Installation

```
pip install -r requirements.txt
```

Training on a GPU (CUDA) or on Apple silicon (MPS) is selected automatically. The thesis results were computed with
the MPS backend. A full run of notebooks 02 to 07 takes several days on a laptop, mainly because of the CNN.

## Outputs

All outputs are written to `results/` (environment variable `THESIS_RESULTS`):

* `<notebook>_metrics.csv`: one row per training run (fold and seed) with accuracy, balanced accuracy, macro-F1,
  recall per class and ROC-AUC.
* `<notebook>_predictions.csv.gz`: one row per test image and run.
* `<notebook>_history.csv`: learning curves (notebook 02).
* `figures/`: figures as PDF and PNG with the caption as a text file.
* `cache/`: preprocessed inputs. The cache is rebuilt automatically when a setting changes.

## Quick test

`THESIS_FAST=1` uses at most 300 images per class, three epochs, one seed and three folds. It only checks that the
notebooks run. The numbers are not meaningful.
