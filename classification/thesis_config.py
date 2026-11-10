"""Shared settings and data loading for all classification notebooks.

Datasets
--------
Andor Zyla setup (resampled to 0.65 µm/px):
  * z-stack: static cells, used for training and for cross-validation
  * Andor Zyla flow: cells in flow, used for testing (transfer) and for cross-validation within the dataset
Phantom TMX setup (native sampling 0.925 µm/px):
  * static Phantom: static cells, both cell lines on every acquisition day, BF and Ph2
  * Phantom flow: cells in flow, one representative crop per track, BF and Ph2
  * APD: both cell lines in the same videos (Ph2), labels assigned from the APD signal

All paths, the preprocessing, the augmentation, the training settings and the splits are defined only here, so
that every notebook uses the same settings. The data paths can also be set with environment variables
(THESIS_DATA_ROOT, THESIS_RESULTS, see below).
"""
from dataclasses import dataclass, replace
from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd

from thesis_ml import AUGMENT_PRESETS, PreprocConfig, TrainConfig

# ------------------------------------------------------------------ data root
# Folder that contains the Cellpose crops and the output of the tracking analysis.
DATA_ROOT = Path(os.environ.get('THESIS_DATA_ROOT', '/Volumes/Extreme SSD/Masterarbeit'))

# ------------------------------------------------------------------ classes
CLASSES = ['HeLa', 'NIH3T3']

# ------------------------------------------------------------------ z-stack (Andor Zyla, static cells)
# <ZSTACK_ROOT>/<class folder>/<date>/<stack folder>/...tif
ZSTACK_ROOT = DATA_ROOT / 'cellpose/z-stack'
ZSTACK_CLASS_FOLDERS = {'HeLa_new': 'HeLa', 'NIH3T3_new': 'NIH3T3'}
# Andor Zyla 4.2P, 6.5 µm pixels at 20x: 0.325 µm/px without binning, 0.65 µm/px with 2 x 2 binning.
# Determined from the size of the raw images:
#   2048 x 2048 px, no binning : 130426, 220726 (HeLa), 130526 (NIH3T3)
#   1024 x 1024 px, 2 x 2 binning: 180326, 250326 (HeLa), 110526 (NIH3T3)
ZSTACK_PX_UM = {'default': 0.325, '110526': 0.65, '180326': 0.65, '250326': 0.65}

# ------------------------------------------------------------------ Andor Zyla flow
# <ZYLA_FLOW_ROOT>/<class folder>/<subfolder>/...tif. Every subfolder is one recording.
ZYLA_FLOW_ROOT = DATA_ROOT / 'cellpose/microchannel_bf'
ZYLA_FLOW_CLASS_FOLDERS = {'HeLa_new': 'HeLa', 'NIH3T3_new': 'NIH3T3'}
ZYLA_FLOW_PX_UM = 0.325   # raw images 100 or 120 x 2048 px (ROI over the full width), no binning
# The Zyla flow ROI is only 100 or 120 px high. Crops whose square window reached beyond the ROI were padded by the
# Cellpose notebook and show a straight edge. True removes these crops. The check needs the summary.csv files of
# the Cellpose output and the raw images (thesis_ml.edge_crop_flags).
ZYLA_EXCLUDE_EDGE_CROPS = True
# If the raw images were moved after Cellpose ran: (old prefix, new prefix) pairs, or folders that are searched
# for a raw image with the same file name.
RAW_PATH_REPLACEMENTS = []
RAW_SEARCH_ROOTS = []

# ------------------------------------------------------------------ Phantom flow
# run_* folder of the tracking analysis (contains tables/ and crop_curation/)
PHANTOM_RUN_DIRS = [DATA_ROOT / 'Code/PC/cell_analysis/cell_thesis_results_v2_2/run_20261006_153418_713106']
PHANTOM_PX_UM = 0.925     # Phantom TMX at 20x
PHANTOM_SOURCE = 'representatives'   # one crop per track
PHANTOM_PATH_REPLACEMENTS = []       # (old prefix, new prefix) if the tables point to another computer
PHANTOM_EXCLUDED_RATES = [600.0]     # chip failure at this flow rate

# ------------------------------------------------------------------ static Phantom
# <PS_ROOTS>/<class folder>/<mode>/<day>/<raw image>_cellNNNN.tif (Cellpose crops)
PS_ROOTS = [DATA_ROOT / 'cellpose/snapshot_phantom/data']
PS_CLASS_FOLDERS = {'HeLa': 'HeLa', 'NIH3T3': 'NIH3T3'}
PS_DAY_PATTERN = r'(141026|201026)'  # acquisition day in the folder names
PS_GROUP_COL = 'group_image'         # crops of one raw image stay together (also across BF and Ph2)

# ------------------------------------------------------------------ APD dataset
# <APD_ROOT>/<date>/<video>/<HeLa|NIH3T3>/crops/*.tif, one crop per track
APD_ROOT = DATA_ROOT / 'cellpose/mixed_cells'
APD_CLASS_FOLDERS = {'HeLa': 'HeLa', 'NIH3T3': 'NIH3T3'}
APD_DATES = None                     # None: every date folder under APD_ROOT
APD_CROP_FOLDER = 'crops'            # crops with background, as in all other datasets
APD_SKIP_FOLDERS = ('masks', 'uneindeutig')   # 'uneindeutig': tracks without a clear assignment
APD_MODE = 'Ph2'                     # all APD videos were recorded in Ph2
STAINED = {'161026': 'NIH3T3', '201026': 'HeLa'}   # cell line with the APD peak, per date folder

# ------------------------------------------------------------------ results
RESULTS_DIR = Path(os.environ.get('THESIS_RESULTS', 'results'))
CACHE_DIR = RESULTS_DIR / 'cache'            # preprocessed inputs, rebuilt automatically when a setting changes
FIGURE_DIR = RESULTS_DIR / 'figures'

# ------------------------------------------------------------------ size filter
# Shorter crop side in µm, outliers removed per dataset at the 1st and 99th percentile.
SIZE_FILTER_PERCENTILES = (1, 99)
SIZE_FILTER_UM = {}                  # optional fixed limits per dataset, e.g. {'zstack': (17.0, 32.0)}

# ------------------------------------------------------------------ preprocessing
# Background margin that the Cellpose notebooks leave around each cell, in pixels per side. It is trimmed to
# MARGIN_UM in all datasets (3 px are 0.98 µm on the Zyla and 2.8 µm on the Phantom).
CROP_MARGIN_PX = {'zstack': 3, 'flow_zyla': 3, 'flow_phantom': 3, 'phantom_static': 3, 'flow_apd': 3}
MARGIN_UM = 1.0
# Common pixel size per setup: crops with a finer sampling are resampled to the coarsest sampling of their setup
# (0.65 µm/px for the Zyla datasets because of the binned z-stack days, 0.925 µm/px for the Phantom datasets).
COMMON_PX_UM_BY_SET = {'zstack': 0.65, 'flow_zyla': 0.65, 'flow_phantom': 0.925, 'phantom_static': 0.925,
                       'flow_apd': 0.925}
# Base image: whole crop padded to a square ('letterbox'), resized to 64 x 64, circular background mask,
# percentile normalization. Every smaller input, including 8 x 8, is computed from this image.
# The values below are part of the cache key, therefore fov_um is kept although 'letterbox' does not use it.
PREPROC = PreprocConfig(
    base_size=64,
    geometry='letterbox',
    fov_um=30.0,
    common_px_um=0.925,     # replaced per dataset by COMMON_PX_UM_BY_SET
    mask='circle',
    mask_radius=1.0,
    trim_frac=0.0,
    border_px=2,
    p_low=1.0, p_high=99.0,
    pad_noise=True,
)

# ------------------------------------------------------------------ augmentation and training
# 'strong': shift ±10 %, rotation ±5°, flips, Gaussian blur (p = 0.5, sigma 0.1 to 1.5 px), brightness and
# contrast factors 0.5 to 1.5, gamma 0.6 to 1.6, Gaussian noise 0.05
AUGMENT = 'strong'
TRAIN = TrainConfig(
    epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-4, lr_min=1e-5,
    class_weight_exp=1.0,   # balanced class weights
    motion_blur_max=0.0,
    select_metric='macro_f1',
    **AUGMENT_PRESETS[AUGMENT],
)

# ------------------------------------------------------------------ splits and repetitions
N_FOLDS = 5
VAL_FRAC = 0.2
SEEDS = [0, 1, 2]
SEEDS_BY_MODEL = {'CNN': SEEDS, 'ChipNet': SEEDS, 'Dense': SEEDS}
MODELS = [('ChipNet', 8), ('CNN', 64)]   # chip network (8 x 8) and CNN (64 x 64)

# ------------------------------------------------------------------ quick test
# THESIS_FAST=1 reduces data and epochs. Only to check that a notebook runs, the numbers are meaningless.
FAST = os.environ.get('THESIS_FAST', '0') == '1'
MAX_PER_CLASS = 300 if FAST else None
if FAST:
    TRAIN = replace(TRAIN, epochs=3)
    SEEDS = [0]
    SEEDS_BY_MODEL = {'CNN': [0], 'ChipNet': [0], 'Dense': [0]}
    N_FOLDS = 3


# ------------------------------------------------------------------ data loading
@dataclass
class Dataset:
    name: str
    meta: pd.DataFrame      # one row per image
    X: np.ndarray           # base images (n, 64, 64), values 0 to 1
    feat: pd.DataFrame      # simple raw features per image (crop size, fill fraction, raw contrast)

    def __len__(self):
        return len(self.meta)


SIZE_LIMITS = None      # limits of the size filter per dataset (last call of load_data)
EDGE_INFO = None        # result of the ROI edge check for the Zyla flow crops


def preproc_for(name: str) -> PreprocConfig:
    """Preprocessing of one dataset: PREPROC with the common pixel size of its setup."""
    return replace(PREPROC, common_px_um=COMMON_PX_UM_BY_SET.get(name, PREPROC.common_px_um))


def scan(name: str, ps_modes=None) -> pd.DataFrame:
    """One row per crop of a dataset, before the size filter.

    ``ps_modes`` selects imaging modes of the static Phantom dataset before the size filter (e.g. ['Ph2']).
    """
    global EDGE_INFO
    import thesis_ml as M
    if name == 'zstack':
        return M.scan_zstack(ZSTACK_ROOT, ZSTACK_CLASS_FOLDERS, ZSTACK_PX_UM, CLASSES)
    if name == 'flow_zyla':
        zy = M.scan_zyla_flow(ZYLA_FLOW_ROOT, ZYLA_FLOW_CLASS_FOLDERS, ZYLA_FLOW_PX_UM, CLASSES)
        EDGE_INFO = pd.concat([zy[['uid', 'label', 'group', 'path']],
                               M.edge_crop_flags(zy, ZYLA_FLOW_ROOT, CROP_MARGIN_PX['flow_zyla'],
                                                 RAW_PATH_REPLACEMENTS, True, RAW_SEARCH_ROOTS)], axis=1)
        if ZYLA_EXCLUDE_EDGE_CROPS:
            if EDGE_INFO.edge.isna().all():
                warnings.warn('Edge check not possible (no summary.csv or raw images found). '
                              'All Zyla flow crops are kept.')
            zy = zy[~EDGE_INFO.edge.eq(1).to_numpy()].reset_index(drop=True)
        return zy
    if name == 'flow_phantom':
        return M.load_flow_runs(PHANTOM_RUN_DIRS, PHANTOM_PX_UM, CLASSES, PHANTOM_SOURCE,
                                PHANTOM_PATH_REPLACEMENTS, PHANTOM_EXCLUDED_RATES)
    if name == 'phantom_static':
        ps = M.scan_static_phantom(PS_ROOTS, PS_CLASS_FOLDERS, PHANTOM_PX_UM, CLASSES, PS_DAY_PATTERN)
        if ps_modes is not None:
            ps = ps[ps['mode'].isin(ps_modes)].reset_index(drop=True)
        return ps
    if name == 'flow_apd':
        return M.scan_apd(APD_ROOT, APD_CLASS_FOLDERS, PHANTOM_PX_UM, CLASSES, STAINED, APD_DATES,
                          APD_CROP_FOLDER, APD_SKIP_FOLDERS, APD_MODE)
    raise ValueError(f'Unknown dataset: {name}')


def load_data(sets=('zstack', 'flow_zyla', 'flow_phantom'), ps_modes=None, verbose: bool = True) -> dict:
    """Load datasets, apply the size filter (per dataset) and compute the base images (cached).

    ``sets``: any of 'zstack', 'flow_zyla', 'flow_phantom', 'phantom_static', 'flow_apd'.
    ``ps_modes``: imaging modes of the static Phantom dataset that are kept before the size filter. None keeps
    BF and Ph2 (as in the thesis everywhere except the transfer to the APD dataset, which uses ['Ph2']).
    Returns {name: Dataset}.
    """
    global SIZE_LIMITS
    import thesis_ml as M
    metas = {k: scan(k, ps_modes) for k in sets}
    sizes = pd.concat([M.scan_sizes(m, CACHE_DIR) for m in metas.values()], ignore_index=True)
    keep, SIZE_LIMITS = M.size_filter(sizes, SIZE_FILTER_PERCENTILES, SIZE_FILTER_UM)
    data, start = {}, 0
    for k, meta in metas.items():
        n = len(meta)
        meta = M.subsample(meta[keep[start:start + n]].reset_index(drop=True), MAX_PER_CLASS)
        start += n
        if meta.empty:
            raise RuntimeError(f'No images left in {k} after loading and filtering. Check the paths above.')
        # equal background margin in µm: remove (margin in px - MARGIN_UM in px) per side
        trim = np.maximum(0, CROP_MARGIN_PX[k] - np.round(MARGIN_UM / meta.px_um))
        meta = meta.assign(trim_px=np.asarray(trim, dtype=int))
        X, feat = M.build_inputs(meta, preproc_for(k), CACHE_DIR, verbose=verbose)
        data[k] = Dataset(k, meta, X, feat)
    if verbose:
        print(SIZE_LIMITS.round(2).to_string(index=False))
        for k, d in data.items():
            print(f'{k:15s} {len(d):6d} images, {d.meta.group.nunique():4d} groups, '
                  f'{preproc_for(k).common_px_um} µm/px, {d.meta.label.value_counts().to_dict()}')
        print(f"Augmentation '{AUGMENT}', training device:", M.get_device(TRAIN.device))
    return data
