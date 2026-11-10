"""Shared pipeline for the label-free classification of HeLa and NIH3T3 cells.

The same functions are used by every notebook, so that data loading, preprocessing,
splits, training and evaluation are identical for all models and all test sets.

Contents
--------
1. Data loading with group identifiers (stack, day, recording, track, clip)
2. One deterministic preprocessing for all datasets and models
3. Grouped, stratified splits without leakage between groups
4. Models: CNN reference and the chip network after Song et al. (64-64-64-2, ReLU, no BatchNorm)
5. Training with image augmentation, identical for all models
6. Evaluation and the feature control (logistic regression on simple image properties)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path, PureWindowsPath
import copy
import hashlib
import json
import re
import time
import warnings
import zlib

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError('PyTorch is required: pip install torch') from exc

import tifffile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Part of every cache key and every result file. Changing it forces a new preprocessing cache.
PIPELINE_VERSION = '2.0.0'
IMAGE_SUFFIXES = {'.tif', '.tiff'}


# ============================================================================ 1. Data loading
def _mode_from_name(name: str) -> str:
    """Imaging mode from a folder name: phase contrast (Ph2) or bright field (BF)."""
    return 'Ph2' if re.search(r'ph\s*2|phase', name, re.I) else 'BF'


def _resolve_class_folder(root: Path, folder: str, label: str) -> Path | None:
    """Find the folder of one class. First the configured name, otherwise the only
    subfolder whose name starts with the class name."""
    base = root / folder
    if base.is_dir():
        return base
    cands = [d for d in root.iterdir() if d.is_dir() and d.name.lower().startswith(label.lower())] \
        if root.is_dir() else []
    if len(cands) == 1:
        warnings.warn(f'Folder {base} not found, using {cands[0]} instead.')
        return cands[0]
    warnings.warn(f'Folder not found: {base}')
    return None


def _flow_from_path(parts) -> float:
    """Total flow rate from folder names such as 100_50 or 100_50_BF (sheath + sample, µL/min)."""
    for p in parts:
        m = re.search(r'(?<!\d)(\d{1,4})_(\d{1,4})(?!\d)', p)
        if m:
            return float(m.group(1)) + float(m.group(2))
    return np.nan


def scan_class_folders(root: Path, class_folders: dict, px_um, classes: list[str],
                       domain: str) -> pd.DataFrame:
    """Read <root>/<class folder>/<subfolders ...>/<image>.tif into a table with one row per image.

    The group of an image is the folder that contains it, relative to the class folder. For the
    z-stack this is one stack, for the Zyla flow data one recording. Images of the same group are
    never split between training and test.
    ``px_um`` is a number or a dict {date folder: µm/px, 'default': µm/px} for days that were
    recorded with a different binning.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Folder not found: {root}. Check the path in thesis_config.py.')
    rows = []
    for folder, label in class_folders.items():
        base = _resolve_class_folder(root, folder, label)
        if base is None:
            continue
        for path in sorted(base.rglob('*')):
            if path.suffix.lower() not in IMAGE_SUFFIXES or path.name.startswith('.'):
                continue
            rel = path.relative_to(base).parts
            parts = rel[:-1] if len(rel) > 1 else ('root',)
            folder_rel = '/'.join(parts)
            date = parts[0] if len(parts) > 1 else ''
            px = px_um.get(date, px_um['default']) if isinstance(px_um, dict) else px_um
            rows.append(dict(
                path=str(path), uid=f'{domain}|{label}|{"/".join(rel)}', label=label,
                domain=domain, mode=_mode_from_name(folder_rel), px_um=px, date=date,
                group=f'{label}|{folder_rel}',
                # The z-stack has no flow. Folder names there can contain numbers that look like rates.
                flow_rate=np.nan if domain == 'zstack' else _flow_from_path(parts),
                deformation=np.nan, speed_m_s=np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f'No TIFF files found under {root}')
    df['y'] = df.label.map({c: i for i, c in enumerate(classes)}).astype(int)
    return df


def scan_zstack(root: Path, class_folders: dict, px_um, classes: list[str]) -> pd.DataFrame:
    """Z-stack. Two grouping levels:
    group_stack : one stack folder, so all focal planes of the same cells stay together
    group_date  : one measurement day (stricter, every day contains only one cell line)
    """
    df = scan_class_folders(root, class_folders, px_um, classes, 'zstack')
    df['group_stack'] = df['group']
    df['group_date'] = df['label'] + '|' + df['date']
    return df


def scan_zyla_flow(root: Path, class_folders: dict, px_um, classes: list[str]) -> pd.DataFrame:
    """Zyla flow crops (microchannel_bf). Group = recording folder."""
    df = scan_class_folders(root, class_folders, px_um, classes, 'flow_zyla')
    df['group_recording'] = df['group']
    return df


def scan_static_phantom(roots, class_folders: dict, px_um: float, classes: list[str],
                        day_pattern: str) -> pd.DataFrame:
    """Static Phantom images: <root>/<class folder>/<any subfolders>/<raw image>_cellNNNN.tif.

    The acquisition day is read from the folder names with ``day_pattern``. The imaging mode (BF or Ph2) is read
    from the folder names as well. ``group_image`` joins all crops with the same raw-image file name, cell line and
    day, also across BF and Ph2. These groups are never split between training and test.
    """
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    parts = []
    for i, root in enumerate(roots):
        d = scan_class_folders(root, class_folders, px_um, classes, 'phantom_static')
        rel = d.path.map(lambda p, r=root: str(Path(p).parent.relative_to(r)))   # folders only
        found = rel.map(lambda r: (re.search(day_pattern, r) or [None])[0])
        d['date'] = found.fillna(f'root{i + 1}' if len(roots) > 1 else 'day1').astype(str)
        d['mode'] = rel.map(_mode_from_name)
        d['uid'] = f'root{i}|' + d.uid
        d['group'] = d.date + '|' + d.group
        parts.append(d)
    ps = pd.concat(parts, ignore_index=True)
    raw_image = ps.path.map(lambda p: Path(p).stem.rsplit('_cell', 1)[0])
    ps['group_image'] = ps.label + '|' + ps.date + '|' + raw_image
    return ps


def _norm_name(name) -> str:
    """Folder name without case, spaces, hyphens or underscores, e.g. 'masked crops' = 'masked_crops'."""
    return re.sub(r'[\s_-]+', '_', str(name).strip().lower())


def scan_apd(root: Path, class_folders: dict, px_um: float, classes: list[str], stained: dict,
             dates=None, crop_folder: str = 'crops', skip_folders=('masks', 'uneindeutig'),
             mode: str = 'Ph2') -> pd.DataFrame:
    """APD dataset: <root>/<date>/<video>/<class folder>/<crop_folder>/*.tif, one crop per track.

    The labels were assigned beforehand from the association between tracks and APD peaks. Folders in
    ``skip_folders`` are never read ('uneindeutig' holds the tracks without a clear assignment).
    ``stained`` gives the stained cell line per date folder (or per 'date/video'). All videos were recorded in
    Ph2, so the imaging mode is set to ``mode``. Groups: ``group_video`` (one video), ``group_cell`` (one crop)
    and ``group_date``.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'APD folder not found: {root}')
    kind_folders = {'crops', 'masks', 'masked_crops'}
    skip = {_norm_name(f) for f in skip_folders}

    def keep_crop(rel_folders):
        names = [_norm_name(p) for p in rel_folders]
        if set(names) & skip:
            return False
        kinds = [p for p in names if p in kind_folders]
        return not kinds or kinds[-1] == _norm_name(crop_folder)

    rows = []
    for date_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        if dates is not None and date_dir.name not in dates:
            continue
        for video_dir in sorted(p for p in date_dir.iterdir() if p.is_dir() and not p.name.startswith('.')):
            for folder, label in class_folders.items():
                base = _resolve_class_folder(video_dir, folder, label)
                if base is None:
                    continue
                for path in sorted(base.rglob('*')):
                    rel = path.relative_to(base).parts
                    if (path.suffix.lower() not in IMAGE_SUFFIXES or path.name.startswith('.')
                            or not keep_crop(rel[:-1])):
                        continue
                    where = f'{date_dir.name}/{video_dir.name}'
                    rows.append(dict(path=str(path), uid=f'flow_apd|{where}|{label}|{"/".join(rel)}', label=label,
                                     domain='flow_apd', mode=mode, px_um=px_um, date=date_dir.name,
                                     video=video_dir.name, group=where,
                                     flow_rate=_flow_from_path([video_dir.name, date_dir.name]),
                                     deformation=np.nan, speed_m_s=np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f'No TIFF crops found under {root}.')
    df['y'] = df.label.map({c: i for i, c in enumerate(classes)}).astype(int)
    df['group_video'] = df.group
    df['group_cell'] = df.uid                    # one crop per track, i.e. one crop per cell
    df['group_date'] = df.date
    # stained cell line: key 'date/video' before 'date'
    line = [stained.get(f'{d}/{v}', stained.get(d)) for d, v in zip(df.date, df.video)]
    df['stained_line'] = pd.Series(line, index=df.index, dtype=object)
    df['stained'] = np.where(df.stained_line.isna(), np.nan, (df.label == df.stained_line).astype(float))
    df['stain_setup'] = df.stained_line.map(lambda s: f'{s} stained' if isinstance(s, str) else 'unknown')
    return df


def _read_table(tables: Path, name: str) -> pd.DataFrame | None:
    for suffix in ('.csv', '.csv.gz'):
        p = tables / f'{name}{suffix}'
        if p.is_file():
            return pd.read_csv(p, low_memory=False)
    return None


def _as_path(s: str):
    """Path object for a string that may come from Windows or from macOS/Linux."""
    return PureWindowsPath(s) if '\\' in s else Path(s)


def _resolve_path(original: str, replacements, fallback: Path | None) -> str | None:
    """Locate a crop file. Tries the stored path, then the path with replaced prefixes, then the fallback."""
    if isinstance(original, str) and original:
        candidates = [original]
        for old, new in replacements:
            if original.startswith(old):
                rest = original[len(old):]
                parts = [p for p in _as_path(rest).parts if p not in ('\\', '/')]
                candidates.append(str(Path(new).joinpath(*parts)))
        for c in candidates:
            if Path(c).is_file():
                return c
    if fallback is not None and fallback.is_file():
        return str(fallback)
    return None


def best_tail_match(stored: str, candidates) -> str | None:
    """Of several local files with the same name, the one whose folder names agree with the stored path
    over the longest tail. None if there is no unique best candidate."""
    sp = _as_path(stored).parts
    best, best_n, tie = None, -1, False
    for c in candidates:
        cp = Path(c).parts
        n = 0
        while n < min(len(sp), len(cp)) and sp[-1 - n] == cp[-1 - n]:
            n += 1
        if n > best_n:
            best, best_n, tie = str(c), n, False
        elif n == best_n:
            tie = True
    return None if tie else best


def _scan_representative_copies(copy_root: Path, classes) -> pd.DataFrame:
    """Fallback without tables: read only the folders <cell line>/<mode>/<detection_uid>.tif."""
    rows = []
    for path in sorted(copy_root.rglob('*.tif*')):
        rel = path.relative_to(copy_root).parts
        if len(rel) < 3 or rel[0] not in classes:
            continue
        rows.append(dict(detection_uid=path.stem, cell_uid=path.stem, cell_type=rel[0], imaging_mode=rel[1],
                         experiment_id='unknown', folder=rel[1], video='unknown', flow_rate_ul_min=np.nan,
                         recording_group=f'{rel[0]}|{rel[1]}', path=str(path)))
    return pd.DataFrame(rows)


def load_flow_runs(run_dirs, px_um: float, classes: list[str], source='representatives',
                   path_replacements=(), excluded_rates=(600.0,)) -> pd.DataFrame:
    """Load the Phantom crops from the output folders of the trajectory notebook.

    Groups:
      group_track     = experiment + track (cell_uid)
      group_clip      = experiment + clip (video file)
      group_recording = experiment + folder (e.g. 100_50_BF), the strictest separation
    Bright field and phase contrast stay one dataset. The mode is stored in ``mode``.
    """
    frames = []
    for run in map(Path, run_dirs):
        # The path to .../crop_curation/representative_copies is accepted as well.
        if run.name == 'representative_copies' and run.parent.name == 'crop_curation':
            run = run.parent.parent
        if not run.is_dir():
            raise FileNotFoundError(
                f'Flow folder not found: {run}\n'
                'Set PHANTOM_RUN_DIRS in thesis_config.py to the run_* folder of the trajectory notebook '
                '(the folder that contains tables/ and crop_curation/).')
        tables = run / 'tables'
        if source == 'representatives':
            df = _read_table(tables, 'ml_representative_candidates')
        elif source == 'all_eligible':
            df = _read_table(tables, 'crop_manifest')
            if df is not None:
                df = df.loc[df.ml_eligible.astype(bool) & df.track_retained.astype(bool)]
                if 'sha256' in df:
                    df = df.drop_duplicates('sha256')
        else:
            raise ValueError(source)
        copy_root = run / 'crop_curation' / 'representative_copies'
        if df is None and copy_root.is_dir():
            warnings.warn(f'No tables in {tables}. Only the crops in {copy_root} are read, '
                          'without flow rate, track, recording and deformation.')
            frames.append(_scan_representative_copies(copy_root, classes).assign(run=str(run)))
            continue
        if df is None:
            raise FileNotFoundError(f'No crop table in {tables}')
        cells = _read_table(tables, 'cells_primary')
        keep = [c for c in ['cell_uid', 'deformation', 'speed_m_s', 'equivalent_diameter_um']
                if cells is not None and c in cells]
        if cells is not None and 'cell_uid' in keep:
            df = df.merge(cells[keep].drop_duplicates('cell_uid'), on='cell_uid', how='left',
                          suffixes=('_det', ''))
        resolved = []
        for r in df.itertuples():
            fb = copy_root / str(r.cell_type) / str(r.imaging_mode) / f'{r.detection_uid}.tif'
            resolved.append(_resolve_path(getattr(r, 'crop_output_path', ''), path_replacements, fb))
        frames.append(df.assign(path=resolved, run=str(run)))
    df = pd.concat(frames, ignore_index=True)
    missing = df.path.isna()
    if missing.any():
        warnings.warn(f'{int(missing.sum())} of {len(df)} crops not found and skipped. '
                      'Check PHANTOM_PATH_REPLACEMENTS.')
    df = df.loc[~missing & df.cell_type.isin(classes)]
    df = df.loc[~df.flow_rate_ul_min.isin(list(excluded_rates))]
    out = pd.DataFrame(dict(
        path=df.path, uid='flow_phantom|' + df.detection_uid.astype(str), label=df.cell_type,
        domain='flow_phantom', mode=df.imaging_mode.astype(str), px_um=px_um,
        date=df.experiment_id.astype(str), flow_rate=df.flow_rate_ul_min.astype(float),
        deformation=df.get('deformation', np.nan), speed_m_s=df.get('speed_m_s', np.nan),
        cell_uid=df.cell_uid.astype(str),
        group_track=df.experiment_id.astype(str) + '|' + df.cell_uid.astype(str),
        group_clip=df.experiment_id.astype(str) + '|' + df.video.astype(str),
        group_recording=(df.recording_group.astype(str) if 'recording_group' in df
                         else df.experiment_id.astype(str) + '|' + df.folder.astype(str)),
    )).reset_index(drop=True)
    out['group'] = out['group_recording']
    out['y'] = out.label.map({c: i for i, c in enumerate(classes)}).astype(int)
    return out


# ============================================================================ Crops at the ROI edge
def edge_crop_flags(meta: pd.DataFrame, root: Path, margin_px: int = 3, raw_path_replacements=(),
                    verbose: bool = True, raw_search_roots=()) -> pd.DataFrame:
    """Find crops whose square crop window extended beyond the raw image (ROI edge).

    The Cellpose notebook writes one ``summary.csv`` per output folder with the bounding box of every
    accepted cell, the raw image path and the crop path. The crop window is reconstructed exactly as in
    ``crop_square_from_mask`` (side = longer bounding-box side + 2 * margin, centred on the box).
    If the window leaves the raw image, the crop was padded with a constant background value or cut,
    which leaves a straight edge in the crop. In the Zyla flow recordings the ROI is only 100 or 120 px
    high, so this happens often.

    ``raw_path_replacements``: (old prefix, new prefix) pairs if the raw images were moved after Cellpose.
    ``raw_search_roots``: folders searched for a raw image with the same file name if the path is not found.
    Returns a table aligned with ``meta``: edge (1, 0 or NaN if the crop or its raw image was not found),
    raw_h, raw_w.
    """
    root = Path(root)
    by_name: dict = {}
    for i, p in enumerate(meta.path):
        by_name.setdefault(Path(p).name, []).append(i)
    parts = [Path(p).parts for p in meta.path]

    def match(out_path: str, prefix: str):
        # crop in ``meta`` that belongs to a path from summary.csv: same file name, below the folder of the
        # summary.csv, and the longest common tail of the folder names (paths may come from another computer)
        op = _as_path(out_path).parts
        best, best_len, tie = None, 0, False
        for i in by_name.get(op[-1], []):
            if not meta.path.iloc[i].startswith(prefix):
                continue
            n = 0
            while n < min(len(op), len(parts[i])) and op[-1 - n] == parts[i][-1 - n]:
                n += 1
            if n > best_len:
                best, best_len, tie = i, n, False
            elif n == best_len:
                tie = True
        return None if (best is None or tie) else best
    edge = np.full(len(meta), np.nan)
    raw_h = np.full(len(meta), np.nan)
    raw_w = np.full(len(meta), np.nan)
    raw_sizes: dict = {}
    name_index = None
    n_rows = n_missing_raw = 0
    for summary in sorted(root.rglob('summary.csv')):
        df = pd.read_csv(summary, low_memory=False)
        need = {'output_path', 'input_path', 'bbox_y0', 'bbox_y1', 'bbox_x0', 'bbox_x1'}
        if not need <= set(df.columns):
            continue
        if 'status' in df:
            df = df[df.status.astype(str).eq('accepted')]
        prefix = str(summary.parent)
        for r in df.itertuples():
            out = str(r.output_path)
            if not out or out == 'nan':
                continue
            i = match(out, prefix)
            if i is None:
                continue
            n_rows += 1
            src = str(r.input_path)
            if src not in raw_sizes:
                found = _resolve_path(src, raw_path_replacements, None)
                if found is None and raw_search_roots:
                    if name_index is None:
                        name_index = {}
                        for sr in raw_search_roots:
                            for p in Path(sr).rglob('*.tif*'):
                                name_index.setdefault(p.name, []).append(str(p))
                    found = best_tail_match(src, name_index.get(_as_path(src).name, []))
                raw_sizes[src] = image_size(found) if found else None
            hw = raw_sizes[src]
            if hw is None:
                n_missing_raw += 1
                continue
            H, W = hw
            y0, y1, x0, x1 = int(r.bbox_y0), int(r.bbox_y1), int(r.bbox_x0), int(r.bbox_x1)
            side = max(y1 - y0, x1 - x0) + 2 * margin_px
            ys = int(round((y0 + y1) / 2.0 - side / 2.0))
            xs = int(round((x0 + x1) / 2.0 - side / 2.0))
            edge[i] = float(ys < 0 or xs < 0 or ys + side > H or xs + side > W)
            raw_h[i], raw_w[i] = H, W
    out = pd.DataFrame(dict(edge=edge, raw_h=raw_h, raw_w=raw_w))
    if verbose:
        n_edge = int(np.nansum(edge))
        n_known = int(np.isfinite(edge).sum())
        print(f'Edge check: {n_known} of {len(meta)} crops found in a summary.csv, {n_edge} reach beyond '
              f'the raw image. Raw image missing for {n_missing_raw} crops.')
    return out


# ============================================================================ Image sizes
def image_size(path: str) -> tuple[int, int]:
    """Height and width of a TIFF, read from the file header only."""
    with tifffile.TiffFile(path) as tf:
        shape = tf.pages[0].shape
    if len(shape) == 3 and shape[-1] in (3, 4):
        return int(shape[0]), int(shape[1])
    return int(shape[-2]), int(shape[-1])


def scan_sizes(meta: pd.DataFrame, cache_dir: Path | None = None, workers: int = 16) -> pd.DataFrame:
    """Size of every crop in pixels and in µm. The size filter uses the shorter side."""
    key = hashlib.sha1('|'.join(meta.path).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f'sizes_{key}.csv' if cache_dir else None
    if cache is not None and cache.is_file():
        hw = pd.read_csv(cache)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            hw = pd.DataFrame(list(ex.map(image_size, meta.path)), columns=['h_px', 'w_px'])
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            hw.to_csv(cache, index=False)
    out = meta[['uid', 'domain', 'mode', 'label', 'px_um', 'date']].reset_index(drop=True).copy()
    out[['h_px', 'w_px']] = hw[['h_px', 'w_px']].to_numpy()
    out['min_side_px'] = out[['h_px', 'w_px']].min(axis=1)
    out['min_side_um'] = out.min_side_px * out.px_um
    return out


def size_filter(sizes: pd.DataFrame, percentiles=(1, 99), limits_um: dict | None = None):
    """Remove very small crops (segmentation errors) and very large crops (two cells, aggregates).

    Limits per dataset: fixed values in µm (``limits_um``) or percentiles of the shorter side.
    Returns a mask of the kept crops and a table of the limits.
    """
    keep = np.ones(len(sizes), bool)
    rows = []
    for dom, g in sizes.groupby('domain'):
        fixed = (limits_um or {}).get(dom)
        lo, hi = fixed if fixed is not None else np.percentile(g.min_side_um, percentiles)
        out = (sizes.domain == dom).to_numpy() & ((sizes.min_side_um < lo) | (sizes.min_side_um > hi)).to_numpy()
        keep &= ~out
        px = g.px_um.median()
        rows.append(dict(domain=dom, px_um=px, min_um=lo, max_um=hi, min_px=lo / px, max_px=hi / px,
                         rule='fixed' if fixed is not None else f'p{percentiles[0]:g} to p{percentiles[1]:g}',
                         n_before=len(g), n_removed=int(out.sum())))
    return keep, pd.DataFrame(rows)


def subsample(meta: pd.DataFrame, max_per_class: int | None, seed: int = 0) -> pd.DataFrame:
    """Quick test only: at most n images per class (groups are not kept complete)."""
    if not max_per_class:
        return meta
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in meta.groupby('label').indices.items():
        keep.extend(rng.choice(idx, size=min(len(idx), max_per_class), replace=False))
    return meta.iloc[np.sort(keep)].reset_index(drop=True)


# ============================================================================ 2. Preprocessing
@dataclass(frozen=True)
class PreprocConfig:
    base_size: int = 64          # base image size; every smaller input is derived from it
    # Geometry of the base image:
    # 'letterbox': whole crop, the shorter side padded with background noise to a square, then resized.
    #              No cropping, the aspect ratio is kept, every cell fills the image to a similar extent.
    # 'physical' : fixed field of view in µm, centre-cropped or padded. The absolute size is kept.
    # 'fit'      : whole crop stretched to N x N. Only meaningful if all crops are square.
    geometry: str = 'letterbox'
    fov_um: float = 32.0         # field of view for 'physical'
    # Common pixel size in µm/px. Crops with a finer sampling are resampled to this pixel size before the
    # geometry step, so that all datasets carry the same spatial information (the Phantom samples at
    # 0.925 µm/px, the Zyla at 0.325 or 0.65 µm/px). None switches the step off.
    common_px_um: float | None = 0.925
    # Background mask: 'circle' sets everything outside the inscribed circle to the background level,
    # so corners and crop edges carry no information. 'none' keeps the full square.
    mask: str = 'circle'
    mask_radius: float = 1.0     # radius as a fraction of half the image side
    trim_frac: float = 0.0       # fraction removed at the top and bottom
    border_px: int = 2           # border width for the background estimate
    p_low: float = 1.0
    p_high: float = 99.0
    pad_noise: bool = True       # pad with background noise instead of zeros

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def read_image(path: str) -> np.ndarray:
    img = np.asarray(tifffile.imread(path), dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(-1) if img.shape[-1] in (3, 4) else img[img.shape[0] // 2]
    if img.ndim != 2:
        raise ValueError(f'Unexpected image shape {img.shape} in {path}')
    return img


def _ring(img: np.ndarray, w: int) -> np.ndarray:
    """Pixels of the outer border of width w."""
    w = max(1, min(w, min(img.shape) // 4 or 1))
    return np.concatenate([img[:w].ravel(), img[-w:].ravel(), img[w:-w, :w].ravel(), img[w:-w, -w:].ravel()])


def _resize(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resampling with antialiasing (area-preserving when shrinking)."""
    t = torch.from_numpy(np.ascontiguousarray(img))[None, None]
    return F.interpolate(t, size=shape, mode='bilinear', align_corners=False, antialias=True)[0, 0].numpy()


def _noise_level(x: np.ndarray, w: int) -> float:
    """Robust standard deviation of the background (median absolute deviation of the border)."""
    r = _ring(x, w)
    return 1.4826 * float(np.median(np.abs(r - np.median(r))))


def circle_mask(n: int, radius: float = 1.0) -> np.ndarray:
    """Boolean n x n mask of the inscribed circle (radius as a fraction of n / 2)."""
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    return (yy - c) ** 2 + (xx - c) ** 2 <= (radius * n / 2.0) ** 2


def preprocess_image(img: np.ndarray, px_um: float, cfg: PreprocConfig, seed: int = 0, trim_px: int = 0):
    """Raw crop -> base image (base x base, values 0 to 1) and simple raw features.

    Order of the steps:
    1. remove ``trim_px`` pixels per side (equal background margin in µm across datasets)
    2. raw features (size in µm, mean, standard deviation)
    3. subtract the background (median of the border)
    4. resample to the common pixel size ``cfg.common_px_um``
    5. geometry (letterbox, physical or fit) to base x base
    6. background mask (inscribed circle)
    7. percentile normalisation to 0 to 1, percentiles taken inside the mask
    """
    if trim_px > 0 and min(img.shape) - 2 * trim_px >= 8:
        img = img[trim_px:-trim_px, trim_px:-trim_px]
    h0, w0 = img.shape
    feats = dict(h_um=h0 * px_um, w_um=w0 * px_um, raw_mean=float(img.mean()), raw_std=float(img.std()))
    if cfg.trim_frac > 0:
        t = int(round(cfg.trim_frac * h0))
        img = img[t:h0 - t] if h0 - 2 * t >= 4 else img
    bg = float(np.median(_ring(img, cfg.border_px)))
    feats['bg'] = bg
    x = img - bg
    # 4. common pixel size: only downsampling, never upsampling
    px_eff = px_um
    if cfg.common_px_um and px_um < cfg.common_px_um:
        f = px_um / cfg.common_px_um
        x = _resize(x, (max(4, int(round(x.shape[0] * f))), max(4, int(round(x.shape[1] * f)))))
        px_eff = cfg.common_px_um
    feats['h_px_common'] = x.shape[0]
    n = cfg.base_size
    rng = np.random.default_rng(seed)
    if cfg.geometry == 'physical':
        scale = px_eff / (cfg.fov_um / n)
        x = _resize(x, (max(2, int(round(x.shape[0] * scale))), max(2, int(round(x.shape[1] * scale)))))
        noise = _noise_level(x, cfg.border_px)
        canvas = (rng.normal(0, noise, (n, n)).astype(np.float32) if cfg.pad_noise
                  else np.zeros((n, n), np.float32))
        # centre-crop or centre-pad
        sy, sx = max(0, (x.shape[0] - n) // 2), max(0, (x.shape[1] - n) // 2)
        crop = x[sy:sy + n, sx:sx + n]
        oy, ox = (n - crop.shape[0]) // 2, (n - crop.shape[1]) // 2
        canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        feats['fov_fill'] = crop.shape[0] * crop.shape[1] / n ** 2
        x = canvas
    elif cfg.geometry == 'letterbox':
        h, w = x.shape
        side = max(h, w)
        canvas = (rng.normal(0, _noise_level(x, cfg.border_px), (side, side)).astype(np.float32)
                  if cfg.pad_noise else np.zeros((side, side), np.float32))
        oy, ox = (side - h) // 2, (side - w) // 2
        canvas[oy:oy + h, ox:ox + w] = x
        feats['fov_fill'] = h * w / side ** 2
        x = _resize(canvas, (n, n))
    elif cfg.geometry == 'fit':
        x = _resize(x, (n, n))
        feats['fov_fill'] = 1.0
    else:
        raise ValueError(cfg.geometry)
    if cfg.mask == 'circle':
        inside = circle_mask(n, cfg.mask_radius)
        x = np.where(inside, x, 0.0).astype(np.float32)     # 0 = background level after step 3
    elif cfg.mask == 'none':
        inside = np.ones((n, n), bool)
    else:
        raise ValueError(cfg.mask)
    lo, hi = np.percentile(x[inside], [cfg.p_low, cfg.p_high])
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    return x, feats


def preprocess_steps(path: str, px_um: float, cfg: PreprocConfig, trim_px: int = 0,
                     small: int = 8) -> dict:
    """Intermediate images of ``preprocess_image`` for one crop, for a figure of the pipeline.

    raw    : crop after the margin trim (display only)
    common : background subtracted and resampled to the common pixel size
    base   : final base image (geometry, mask, normalisation)
    small  : base image reduced to small x small (chip input)
    """
    img = read_image(path)
    if trim_px > 0 and min(img.shape) - 2 * trim_px >= 8:
        img = img[trim_px:-trim_px, trim_px:-trim_px]
    x = img - float(np.median(_ring(img, cfg.border_px)))
    if cfg.common_px_um and px_um < cfg.common_px_um:
        f = px_um / cfg.common_px_um
        x = _resize(x, (max(4, int(round(x.shape[0] * f))), max(4, int(round(x.shape[1] * f)))))
    base, _ = preprocess_image(read_image(path), px_um, cfg, seed=zlib.crc32(path.encode()), trim_px=trim_px)
    return dict(raw=img, common=x, base=base, small=resize_batch(base[None], small)[0])


def build_inputs(meta: pd.DataFrame, cfg: PreprocConfig, cache_dir: Path | None = None,
                 workers: int = 8, verbose: bool = True):
    """Compute all base images (n, base, base) as float16 together with the raw features.

    The result is cached. The cache key depends on the files, pixel sizes, margins, the preprocessing
    settings and the pipeline version, so every change produces a new cache automatically.
    """
    trims = meta['trim_px'] if 'trim_px' in meta else pd.Series(0, index=meta.index)
    key = hashlib.sha1(('|'.join(meta.path) + '|'.join(map(str, meta.px_um)) + '|'.join(map(str, trims))
                        + cfg.key() + PIPELINE_VERSION).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f'inputs_{key}.npz' if cache_dir else None
    if cache is not None and cache.is_file():
        z = np.load(cache, allow_pickle=False)
        feats = pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith('f_')})
        if verbose:
            print(f'Cache loaded: {cache.name} ({len(feats)} images)')
        return z['X'], feats

    def work(args):
        path, px, trim = args
        return preprocess_image(read_image(path), px, cfg, seed=zlib.crc32(path.encode()), trim_px=int(trim))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(work, zip(meta.path, meta.px_um, trims)))
    X = np.stack([o[0] for o in out]).astype(np.float16)
    feats = pd.DataFrame([o[1] for o in out])
    if verbose:
        print(f'{len(X)} images preprocessed in {time.time() - t0:.0f} s')
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, **{f'f_{c}': feats[c].to_numpy() for c in feats})
    return X, feats


def _resize_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    """(n, B, B) -> (n, size, size), bilinear with antialiasing, values clipped to 0 to 1."""
    if x.shape[-1] == size:
        return x
    return F.interpolate(x[:, None], size=(size, size), mode='bilinear', align_corners=False,
                         antialias=True)[:, 0].clamp(0, 1)


def resize_batch(X: np.ndarray, size: int) -> np.ndarray:
    """Base images (n, B, B) -> (n, size, size). Same operation as inside the training loop."""
    X = np.asarray(X)
    if X.shape[-1] == size:
        return X.astype(np.float32)
    out = [_resize_tensor(torch.from_numpy(X[i:i + 4096].astype(np.float32)), size).numpy()
           for i in range(0, len(X), 4096)]
    return np.concatenate(out)


# ============================================================================ 3. Splits
def grouped_folds(y, groups, n_splits: int, seed: int = 0):
    """Stratified k-fold split by groups. No group appears in training and test."""
    y, groups = np.asarray(y), np.asarray(groups)
    min_groups = min(len(np.unique(groups[y == c])) for c in np.unique(y))
    k = min(n_splits, min_groups)
    if k < n_splits:
        warnings.warn(f'Only {min_groups} groups in the smallest class, therefore {k} folds instead of {n_splits}.')
    if k < 2:
        raise ValueError('Too few groups for a grouped cross-validation.')
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups))
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
    return folds


def train_val_split(idx, y, groups, val_frac: float = 0.2, seed: int = 0):
    """Split training indices by groups into training and validation."""
    idx = np.asarray(idx)
    k = max(2, int(round(1 / val_frac)))
    yy, gg = np.asarray(y)[idx], np.asarray(groups)[idx]
    min_groups = min(len(np.unique(gg[yy == c])) for c in np.unique(yy))
    k = min(k, min_groups)
    if k < 2:
        raise ValueError('Too few groups for a validation set.')
    tr, va = next(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
                  .split(np.zeros(len(idx)), yy, gg))
    return idx[tr], idx[va]


def split_table(meta: pd.DataFrame, group_col: str, folds) -> pd.DataFrame:
    """Number of images, groups and images per class in every fold."""
    rows = []
    for f, (tr, te) in enumerate(folds):
        for part, ix in (('train', tr), ('test', te)):
            sub = meta.iloc[ix]
            rows.append(dict(fold=f, part=part, n_images=len(sub), n_groups=sub[group_col].nunique(),
                             **{f'n_{c}': int((sub.label == c).sum()) for c in sorted(meta.label.unique())}))
    return pd.DataFrame(rows)


# ============================================================================ 4. Models
class ChipNet(nn.Module):
    """Fully connected network as in Song et al.: input N*N, two hidden layers with 64 neurons each,
    ReLU, no BatchNorm. For N = 8 this is the chip configuration 64-64-64-2."""

    def __init__(self, input_size: int = 8, hidden=(64, 64), n_classes: int = 2, dropout: float = 0.0):
        super().__init__()
        layers, n_in = [nn.Flatten()], input_size * input_size
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            n_in = h
        layers.append(nn.Linear(n_in, n_classes))
        self.net = nn.Sequential(*layers)
        self.input_size = input_size

    def forward(self, x):
        return self.net(x)

    def linear_layers(self):
        return [m for m in self.net if isinstance(m, nn.Linear)]


class CNN(nn.Module):
    """Reference CNN: four blocks of 3x3 convolution, BatchNorm, ReLU and max pooling
    (32, 64, 128, 256 channels), global average pooling and a head 256 -> 128 -> classes."""

    def __init__(self, input_size: int = 64, channels=(32, 64, 128, 256), n_classes: int = 2,
                 dropout: float = 0.25):
        super().__init__()
        blocks, c_in = [], 1
        for c in channels:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, 128),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes))
        self.input_size = input_size

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, None]
        return self.head(self.features(x))


def make_model(kind: str, input_size: int, n_classes: int = 2) -> nn.Module:
    if kind == 'CNN':
        return CNN(input_size, n_classes=n_classes)
    if kind in ('ChipNet', 'Dense'):
        return ChipNet(input_size, n_classes=n_classes)
    raise ValueError(kind)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================================ 5. Training
@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_min: float = 1e-5
    class_weight_exp: float = 1.0
    # Augmentation, applied on the CPU to the base images (64 x 64) before the resize to the input size.
    augment: bool = True
    flips: bool = True                       # horizontal and vertical, p = 0.5 each
    rotate_deg: float = 5.0                  # rotation, uniform in ±rotate_deg
    translate_frac: float = 0.10             # shift, uniform in ±translate_frac of the image side
    blur_prob: float = 0.5                   # probability of a Gaussian blur
    blur_sigma: tuple = (0.1, 1.5)           # sigma in pixels at 64 x 64
    blur_kernel: int = 5                     # kernel size at 64 x 64
    brightness: float = 0.3                  # factor uniform in 1 ± brightness
    contrast: float = 0.3                    # factor uniform in 1 ± contrast, around the image mean
    gamma: tuple = (1.0, 1.0)                # x -> x ** gamma, log-uniform in this range
    noise_sigma: float = 0.03                # additive Gaussian noise
    # Horizontal motion blur along x (flow direction), maximum length as a fraction of the image width.
    # 0 switches it off (setting used in the thesis).
    motion_blur_max: float = 0.0
    select_metric: str = 'macro_f1'
    device: str = 'auto'


AUGMENT_PRESETS = {
    # no augmentation
    'none': dict(augment=False),
    # moderate illumination and contrast changes
    'original': dict(augment=True, flips=True, rotate_deg=5.0, translate_frac=0.10, blur_prob=0.5,
                     blur_sigma=(0.1, 1.5), blur_kernel=5, brightness=0.3, contrast=0.3, gamma=(1.0, 1.0),
                     noise_sigma=0.03),
    # original geometry and blur, wider illumination and contrast range plus a gamma change
    'strong': dict(augment=True, flips=True, rotate_deg=5.0, translate_frac=0.10, blur_prob=0.5,
                   blur_sigma=(0.1, 1.5), blur_kernel=5, brightness=0.5, contrast=0.5, gamma=(0.6, 1.6),
                   noise_sigma=0.05),
}


def get_device(name: str = 'auto') -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _uniform(n: int, lo: float, hi: float, gen: torch.Generator) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, generator=gen)


def _gaussian_blur(x: torch.Tensor, sigma: torch.Tensor, k: int) -> torch.Tensor:
    """Separable Gaussian blur with one sigma per image. x: (n, H, W), sigma: (n,)."""
    n = x.shape[0]
    r = torch.arange(k, dtype=x.dtype) - (k - 1) / 2
    w = torch.exp(-0.5 * (r[None] / sigma[:, None].clamp_min(1e-3)) ** 2)
    w = w / w.sum(1, keepdim=True)                                # (n, k)
    y = F.pad(x[None], (k // 2, k // 2, 0, 0), mode='replicate')   # (1, n, H, W + k - 1)
    y = F.conv2d(y, w[:, None, None, :], groups=n)
    y = F.pad(y, (0, 0, k // 2, k // 2), mode='replicate')
    y = F.conv2d(y, w[:, None, :, None], groups=n)
    return y[0]


def _motion_blur(x: torch.Tensor, max_frac: float, gen: torch.Generator) -> torch.Tensor:
    """Horizontal box blur over k pixels, k uniform between 1 and max_frac * image width."""
    n, w = x.shape[0], x.shape[-1]
    k_max = max(1, int(round(max_frac * w)))
    ks = torch.randint(1, k_max + 1, (n,), generator=gen)
    out = x.clone()
    for k in ks.unique().tolist():
        if k == 1:
            continue
        idx = (ks == k).nonzero().flatten()
        xi = F.pad(x[idx][:, None], (k // 2, k - 1 - k // 2, 0, 0), mode='replicate')
        out[idx] = F.conv2d(xi, torch.full((1, 1, 1, k), 1.0 / k))[:, 0]
    return out


def augment_batch(x: torch.Tensor, cfg: TrainConfig, gen: torch.Generator) -> torch.Tensor:
    """Random augmentation of a batch of base images (n, B, B) on the CPU, values 0 to 1.

    Order: affine (shift, rotation), flips, Gaussian blur,
    brightness and contrast, then gamma, motion blur (optional) and noise. Parameters given in
    pixels refer to 64 x 64 and are scaled for other base sizes.
    """
    n, size = x.shape[0], x.shape[-1]
    s = size / 64.0
    if cfg.rotate_deg or cfg.translate_frac:
        a = torch.deg2rad(_uniform(n, -cfg.rotate_deg, cfg.rotate_deg, gen))
        # affine_grid uses coordinates from -1 to 1, so a shift of t * side is 2 t
        tx = 2 * _uniform(n, -cfg.translate_frac, cfg.translate_frac, gen)
        ty = 2 * _uniform(n, -cfg.translate_frac, cfg.translate_frac, gen)
        theta = torch.stack([torch.stack([torch.cos(a), -torch.sin(a), tx], 1),
                             torch.stack([torch.sin(a), torch.cos(a), ty], 1)], 1)
        grid = F.affine_grid(theta, (n, 1, size, size), align_corners=False)
        # 'border' repeats the edge value, which is the flat background level when the mask is used
        x = F.grid_sample(x[:, None], grid, mode='bilinear', padding_mode='border', align_corners=False)[:, 0]
    if cfg.flips:
        fh = torch.rand(n, generator=gen) < 0.5
        fv = torch.rand(n, generator=gen) < 0.5
        x = torch.where(fh[:, None, None], x.flip(-1), x)
        x = torch.where(fv[:, None, None], x.flip(-2), x)
    if cfg.blur_prob > 0:
        on = torch.rand(n, generator=gen) < cfg.blur_prob
        if on.any():
            k = max(3, int(round(cfg.blur_kernel * s)) | 1)
            sig = _uniform(int(on.sum()), cfg.blur_sigma[0] * s, cfg.blur_sigma[1] * s, gen)
            x = x.clone()
            x[on] = _gaussian_blur(x[on], sig, k)
    if cfg.brightness:
        b = _uniform(n, 1 - cfg.brightness, 1 + cfg.brightness, gen)[:, None, None]
        x = (x * b).clamp(0, 1)
    if cfg.contrast:
        c = _uniform(n, 1 - cfg.contrast, 1 + cfg.contrast, gen)[:, None, None]
        m = x.mean(dim=(1, 2), keepdim=True)
        x = ((x - m) * c + m).clamp(0, 1)
    if cfg.gamma[0] != 1.0 or cfg.gamma[1] != 1.0:
        g = torch.exp(_uniform(n, np.log(cfg.gamma[0]), np.log(cfg.gamma[1]), gen))[:, None, None]
        x = x.clamp_min(1e-6) ** g
    if cfg.motion_blur_max:
        x = _motion_blur(x, cfg.motion_blur_max, gen)
    if cfg.noise_sigma:
        x = x + cfg.noise_sigma * torch.randn(x.shape, generator=gen)
    return x.clamp(0, 1)


def augment_examples(X: np.ndarray, cfg: TrainConfig, n_variants: int = 5, seed: int = 0) -> np.ndarray:
    """Augmented versions of a few base images for a figure: (n_images, n_variants, B, B)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(np.asarray(X, dtype=np.float32))
    return np.stack([augment_batch(x, cfg, gen).numpy() for _ in range(n_variants)], 1)


def class_weights(y, n_classes: int, exponent: float) -> torch.Tensor:
    """Inverse class frequency to the power ``exponent``, normalised to mean 1."""
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(float)
    w = (counts.sum() / (n_classes * np.maximum(counts, 1))) ** exponent
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict_proba(model, X: np.ndarray, device=None, batch: int = 1024) -> np.ndarray:
    """Class probabilities. X must already have the input size of the model."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32)).to(device)
        out.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def train_model(kind: str, input_size: int, Xtr, ytr, Xva, yva, cfg: TrainConfig, seed: int = 0,
                n_classes: int = 2, verbose: bool = False):
    """Train one model and return the state with the best validation score (``cfg.select_metric``).

    Xtr and Xva are base images (n, 64, 64). Every training batch is augmented at the base size and
    then resized to ``input_size``. Validation images are resized once, without augmentation.
    """
    set_seed(seed)
    device = get_device(cfg.device)
    model = make_model(kind, input_size, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes, cfg.class_weight_exp).to(device))
    Xtr_t = torch.from_numpy(np.asarray(Xtr, dtype=np.float32))
    ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.int64))
    Xva_s = resize_batch(Xva, input_size)
    yva_t = torch.from_numpy(np.asarray(yva, dtype=np.int64))
    gen = torch.Generator().manual_seed(seed)
    best, best_score, history = None, -np.inf, []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=gen)
        total = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            ix = perm[i:i + cfg.batch_size]
            xb = Xtr_t[ix]
            if cfg.augment:
                xb = augment_batch(xb, cfg, gen)
            xb = _resize_tensor(xb, input_size).to(device)
            yb = ytr_t[ix].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(ix)
        sched.step()
        p = predict_proba(model, Xva_s, device)
        m = compute_metrics(yva, p)
        val_loss = float(F.nll_loss(torch.log(torch.from_numpy(p).clamp_min(1e-8)), yva_t))
        history.append(dict(epoch=epoch + 1, train_loss=total / len(perm), val_loss=val_loss,
                            val_acc=m['accuracy'], val_macro_f1=m['macro_f1'], lr=opt.param_groups[0]['lr']))
        if m[cfg.select_metric] > best_score:
            best_score, best = m[cfg.select_metric], copy.deepcopy(model.state_dict())
            history[-1]['selected'] = True
        if verbose:
            print(f"epoch {epoch + 1:3d}  loss {history[-1]['train_loss']:.3f}  val F1 {m['macro_f1']:.3f}")
    model.load_state_dict(best)
    return model, pd.DataFrame(history)


# ============================================================================ 6. Evaluation
def compute_metrics(y, prob) -> dict:
    """Accuracy, balanced accuracy, macro-F1, recall per class and ROC-AUC (two classes)."""
    y = np.asarray(y)
    pred = prob.argmax(1)
    m = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
             macro_f1=f1_score(y, pred, average='macro', zero_division=0), n=len(y))
    rec = recall_score(y, pred, average=None, labels=list(range(prob.shape[1])), zero_division=0)
    for i, r in enumerate(rec):
        m[f'recall_{i}'] = r
    if prob.shape[1] == 2 and len(np.unique(y)) == 2:
        m['roc_auc'] = roc_auc_score(y, prob[:, 1])
    return m


def confusion(y, pred, n_classes: int = 2) -> np.ndarray:
    return confusion_matrix(y, pred, labels=list(range(n_classes)))


def prediction_frame(meta: pd.DataFrame, idx, prob, **info) -> pd.DataFrame:
    """One row per test image with label, prediction, probability and metadata."""
    cols = [c for c in ['uid', 'label', 'y', 'domain', 'mode', 'date', 'flow_rate', 'deformation',
                        'speed_m_s', 'group', 'group_stack', 'group_date', 'group_track', 'group_clip',
                        'group_recording'] if c in meta]
    df = meta.iloc[idx][cols].reset_index(drop=True).copy()
    df['p1'] = prob[:, 1]
    df['pred'] = prob.argmax(1)
    df['correct'] = df.pred.eq(df.y)
    for k, v in info.items():
        df[k] = v
    return df


def run_grouped_cv(meta, X_base, kind: str, input_size: int, group_col: str, train_cfg: TrainConfig,
                   n_folds: int | None, seeds, val_frac: float = 0.2, scenario: str = '', verbose=True,
                   val_group_col: str | None = None, folds=None, fold_names=None):
    """Grouped cross-validation within one dataset. Returns metrics, predictions and learning curves.

    ``group_col`` separates training and test, ``val_group_col`` (default: the same column) separates
    training and validation inside the training part. ``folds`` replaces the automatic folds by a list of
    (train indices, test indices), e.g. one fold per held-out day; ``fold_names`` labels them.
    The number of classes is taken from ``meta.y`` (two cell lines, or several days for the day classifier).
    """
    X_test = resize_batch(X_base, input_size)
    y, g = meta.y.to_numpy(), meta[group_col].to_numpy()
    n_classes = int(y.max()) + 1
    gv = meta[val_group_col or group_col].to_numpy()
    if folds is None:
        folds = grouped_folds(y, g, n_folds, seed=0)
    rows, preds, hists = [], [], []
    for f, (tr_all, te) in enumerate(folds):
        for seed in seeds:
            tr, va = train_val_split(tr_all, y, gv, val_frac, seed)
            t0 = time.time()
            model, hist = train_model(kind, input_size, X_base[tr], y[tr], X_base[va], y[va], train_cfg, seed,
                                      n_classes=n_classes)
            p = predict_proba(model, X_test[te])
            m = compute_metrics(y[te], p)
            info = dict(scenario=scenario, target=meta.domain.iloc[0], model=kind, input_size=input_size,
                        split=group_col, fold=f, fold_name=fold_names[f] if fold_names else str(f), seed=seed,
                        n_train=len(tr), n_val=len(va), n_params=count_parameters(model),
                        seconds=time.time() - t0)
            rows.append({**info, **m})
            preds.append(prediction_frame(meta, te, p, **info))
            hists.append(hist.assign(**info))
            if verbose:
                print(f"{scenario} {kind}{input_size} fold {info['fold_name']} seed {seed}: "
                      f"F1 {m['macro_f1']:.3f}  Acc {m['accuracy']:.3f}  ({info['seconds']:.0f} s)")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), pd.concat(hists, ignore_index=True)


def run_transfer(meta_src, X_src, targets: dict, kind: str, input_size: int, src_group: str,
                 train_cfg: TrainConfig, seeds, val_frac: float = 0.2, scenario: str = 'transfer',
                 return_models=False, verbose=True):
    """Train on the source only (z-stack), with validation stacks from the source as well.

    Every trained model is evaluated on all test sets in ``targets`` ({name: (meta, X_base)}).
    The labels of the test sets are used for evaluation only.
    """
    Xt = {name: resize_batch(X, input_size) for name, (meta, X) in targets.items()}
    Xs_val = resize_batch(X_src, input_size)
    ys, gs = meta_src.y.to_numpy(), meta_src[src_group].to_numpy()
    rows, preds, models = [], [], []
    for seed in seeds:
        tr, va = train_val_split(np.arange(len(ys)), ys, gs, val_frac, seed)
        t0 = time.time()
        model, hist = train_model(kind, input_size, X_src[tr], ys[tr], X_src[va], ys[va], train_cfg, seed,
                                  n_classes=int(ys.max()) + 1)
        m_src = {f'src_val_{k}': v for k, v in compute_metrics(ys[va], predict_proba(model, Xs_val[va])).items()}
        base = dict(scenario=scenario, model=kind, input_size=input_size, split=src_group, fold=-1, seed=seed,
                    n_train=len(tr), n_val=len(va), n_params=count_parameters(model),
                    seconds=time.time() - t0)
        for name, (meta_t, _) in targets.items():
            p = predict_proba(model, Xt[name])
            m = compute_metrics(meta_t.y.to_numpy(), p)
            info = {**base, 'target': name}
            rows.append({**info, **m, **m_src})
            preds.append(prediction_frame(meta_t, np.arange(len(meta_t)), p, **info))
            if verbose:
                print(f"{scenario} {kind}{input_size} seed {seed}: z-stack val F1 {m_src['src_val_macro_f1']:.3f} "
                      f"-> {name} F1 {m['macro_f1']:.3f}")
        models.append(model)
    out = (pd.DataFrame(rows), pd.concat(preds, ignore_index=True))
    return (*out, models) if return_models else out


def save_results(name: str, results_dir: Path, metrics: pd.DataFrame | None = None,
                 predictions: pd.DataFrame | None = None, history: pd.DataFrame | None = None):
    """Write <name>_metrics.csv, <name>_predictions.csv.gz and <name>_history.csv.
    The figure notebook reads these files."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for suffix, df in (('metrics', metrics), ('predictions', predictions), ('history', history)):
        if df is not None and len(df):
            ext = '.csv.gz' if suffix == 'predictions' else '.csv'
            df.assign(pipeline_version=PIPELINE_VERSION).to_csv(results_dir / f'{name}_{suffix}{ext}', index=False)


def summarize(metrics: pd.DataFrame, by=('scenario', 'model', 'input_size', 'split'),
              cols=('accuracy', 'balanced_accuracy', 'macro_f1')) -> pd.DataFrame:
    """Mean, standard deviation and count of the metrics over folds and seeds."""
    by = [b for b in by if b in metrics]
    agg = metrics.groupby(by)[list(cols)].agg(['mean', 'std', 'count'])
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    return agg.reset_index()


def grouped_metrics(pred: pd.DataFrame, by: str | list) -> pd.DataFrame:
    """Metrics per subgroup (e.g. flow rate, mode, deformation bin), per seed."""
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, g in pred.groupby(by + ['seed', 'model', 'input_size'], dropna=False, observed=True):
        rows.append(dict(zip(by + ['seed', 'model', 'input_size'], key if isinstance(key, tuple) else (key,)),
                         n=len(g), accuracy=g.correct.mean(),
                         balanced_accuracy=balanced_accuracy_score(g.y, g.pred) if g.y.nunique() > 1 else np.nan,
                         macro_f1=f1_score(g.y, g.pred, average='macro', zero_division=0)
                         if g.y.nunique() > 1 else np.nan,
                         **{f'recall_{c}': (g.pred[g.y == c] == c).mean() if (g.y == c).any() else np.nan
                            for c in sorted(pred.y.unique())}))
    return pd.DataFrame(rows)


# ============================================================================ Controls and domain distance
# Only features without camera units (µm, fractions, ratios), so that the control remains meaningful
# for the transfer between Zyla and Phantom.
CONTROL_FEATURES = ['h_um', 'w_um', 'fov_fill', 'contrast']


def _control_matrix(feats: pd.DataFrame) -> np.ndarray:
    f = feats.assign(contrast=feats.raw_std / feats.raw_mean.abs().clip(lower=1e-6))
    return f[CONTROL_FEATURES].to_numpy(float)


def run_control_cv(meta, feats, group_col: str, n_folds: int | None, scenario: str = '', folds=None,
                   fold_names=None) -> pd.DataFrame:
    """Logistic regression on simple raw features (crop size, fill fraction, raw contrast).
    If this control already performs well, a network can use the same shortcuts.
    ``folds`` and ``fold_names`` as in ``run_grouped_cv``."""
    Z, y, g = _control_matrix(feats), meta.y.to_numpy(), meta[group_col].to_numpy()
    rows = []
    for f, (tr, te) in enumerate(folds if folds is not None else grouped_folds(y, g, n_folds)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
        clf.fit(Z[tr], y[tr])
        rows.append(dict(scenario=scenario, target=meta.domain.iloc[0], model='Control', input_size=0,
                         split=group_col, fold=f, fold_name=fold_names[f] if fold_names else str(f), seed=0,
                         n_train=len(tr),
                         **compute_metrics(y[te], clf.predict_proba(Z[te]))))
    return pd.DataFrame(rows)


def control_transfer(meta_src, feats_src, targets: dict, src_group: str = 'group_stack',
                     scenario='transfer') -> pd.DataFrame:
    """Train the control on the z-stack and evaluate it on every test set ({name: (meta, feats)})."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
    clf.fit(_control_matrix(feats_src), meta_src.y)
    rows = []
    for name, (meta_t, feats_t) in targets.items():
        p = clf.predict_proba(_control_matrix(feats_t))
        rows.append(dict(scenario=scenario, target=name, model='Control', input_size=0, split=src_group,
                         fold=-1, seed=0, **compute_metrics(meta_t.y, p)))
    return pd.DataFrame(rows)


# ============================================================================ Chip network: export, robustness
def subdomain(meta: pd.DataFrame) -> np.ndarray:
    """Dataset name, for the Phantom additionally split by imaging mode (flow_phantom_BF, flow_phantom_Ph2)."""
    return np.where(meta.domain == 'flow_phantom', 'flow_phantom_' + meta['mode'].astype(str), meta.domain)
