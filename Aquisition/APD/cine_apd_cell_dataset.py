"""CINE + APD -> one Cellpose crop per tracked cell, with auditable APD labels.

Notebook entry points: prepare, load_model, preview, export_dataset,
reclassify_run, apply_manual_assignments, assignment_summary.
No acquisition hardware is used. CINE/NPZ inputs are read-only.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import inspect
import importlib.metadata
import tempfile
import argparse

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import tifffile


@dataclass
class Config:
    cine_path: str = ''
    apd_path: str = ''
    output_root: str = 'cine_apd_dataset'
    # 'trigger': shared NI trigger + CINE timestamps (reference notebook).
    # 'edges': optional strict F-Sync matching; 'manual': known first frame NI timestamp.
    sync_mode: str = 'trigger'
    active_low: bool = True
    first_frame_time_s: float = 0.0
    sync_correction_s: float = 0.0
    # Positive delay means the APD response occurs AFTER the optical event.
    apd_delay_s: float = 0.0
    # Temporal safety margin after clock-drift correction.
    match_padding_ms: float = 0.50
    # Fit an affine APD-to-CINE time correction from mutual nearest events.
    correct_time_drift: bool = True
    drift_search_ms: float = 2.0
    drift_min_pairs: int = 8
    drift_max_ppm: float = 500.0
    drift_residual_ms: float = 0.10
    # Fractional (x_min, x_max, y_min, y_max); None retains whole-field matching.
    detection_roi: tuple | None = None
    peak_interval_rel_height: float = 0.95
    peak_polarity: int = 1
    peak_height_sigma: float = 6.0
    peak_prominence_sigma: float = 4.0
    peak_min_distance_ms: float = 0.1
    peak_smooth_ms: float = 0.0
    # Set a baseline-relative threshold in volts instead of height_sigma if desired.
    peak_height_v: float | None = None
    background_samples: int = 101
    screen_z: float = 5.0
    screen_min_area: int = 20
    screen_enabled: bool = True
    # Original notebook: background screening, Cellpose on original grayscale image.
    cellpose_input: str = 'raw'   # 'raw' or 'difference'
    pretrained_model: str | None = None  # existing local weights file; None = built-in
    builtin_model: str = 'cyto3'  # Cellpose 3; Cellpose 4 uses its default SAM model
    use_gpu: bool = True
    diameter: float = 20.0
    min_size: int = 50
    min_area: int = 100
    cellprob_threshold: float = -1.5
    flow_threshold: float = 0.4
    invert: bool = False
    max_link_distance_px: float = 60.0
    max_area_ratio: float = 2.5
    track_memory_frames: int = 1
    min_track_detections: int = 1
    # Directed constant-velocity linking. None disables the initial speed prior.
    tracking_expected_step_px: float | None = None
    tracking_prediction_tolerance_px: float = 20.0
    tracking_reverse_tolerance_px: float = 3.0
    tracking_ambiguity_margin_px: float = 5.0
    close_tracks_at_exit: bool = True
    # Reject tracks containing touching, implausibly large, or elongated masks.
    cluster_separation_px: int = 2
    max_cell_area: int | None = 800
    max_cell_elongation: float = 1.8
    reject_cluster_tracks: bool = True
    # Automated review of nearby or potentially fragmented tracks.
    review_passage_delta_ms: float = 0.10
    review_close_distance_px: float = 30.0
    review_split_gap_frames: int = 3
    review_context_frames: int = 2
    # Inherited flow direction: right -> left. Set x/+1 for left -> right.
    flow_axis: str = 'x'
    flow_sign: int = -1
    edge_margin_px: int = 3
    passage_edge_px: int = 8
    crop_margin_px: int = 3
    start_frame: int = 0
    stop_frame: int | None = None  # exclusive; all frames in this interval are read


def validate(cfg):
    if not 0 < cfg.peak_interval_rel_height < 1:
        raise ValueError('peak_interval_rel_height must be between 0 and 1.')
    if cfg.detection_roi is not None:
        x0, x1, y0, y1 = cfg.detection_roi
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('detection_roi must contain fractional (x0, x1, y0, y1) bounds between 0 and 1.')
    for name in ('cine_path', 'apd_path'):
        if not Path(getattr(cfg, name)).is_file():
            raise FileNotFoundError(f'{name}: {getattr(cfg, name)}')
    positive = ('peak_height_sigma', 'peak_prominence_sigma', 'background_samples',
                'screen_z', 'screen_min_area', 'diameter', 'min_size', 'min_area',
                'max_link_distance_px', 'min_track_detections', 'drift_min_pairs',
                'drift_max_ppm')
    for name in positive:
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f'{name} must be positive and finite.')
    for name in ('match_padding_ms', 'peak_min_distance_ms', 'peak_smooth_ms',
                 'track_memory_frames', 'edge_margin_px', 'passage_edge_px', 'crop_margin_px',
                 'drift_search_ms', 'drift_residual_ms', 'review_passage_delta_ms',
                 'review_close_distance_px', 'review_split_gap_frames',
                'review_context_frames', 'tracking_prediction_tolerance_px',
                'tracking_reverse_tolerance_px', 'tracking_ambiguity_margin_px',
                'cluster_separation_px'):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) < 0:
            raise ValueError(f'{name} must be non-negative and finite.')
    for name in ('first_frame_time_s', 'sync_correction_s', 'apd_delay_s'):
        if not np.isfinite(getattr(cfg, name)):
            raise ValueError(f'{name} must be finite.')
    for name in ('background_samples', 'screen_min_area', 'min_size', 'min_area',
                 'track_memory_frames', 'min_track_detections', 'edge_margin_px',
                 'passage_edge_px', 'crop_margin_px', 'start_frame', 'drift_min_pairs',
                 'review_split_gap_frames', 'review_context_frames',
                 'cluster_separation_px'):
        if int(getattr(cfg, name)) != getattr(cfg, name):
            raise ValueError(f'{name} must be an integer.')
    if cfg.peak_height_v is not None and (not np.isfinite(cfg.peak_height_v) or cfg.peak_height_v <= 0):
        raise ValueError('peak_height_v must be positive or None.')
    if cfg.tracking_expected_step_px is not None and (
            not np.isfinite(cfg.tracking_expected_step_px) or cfg.tracking_expected_step_px <= 0):
        raise ValueError('tracking_expected_step_px must be positive or None.')
    if cfg.max_cell_area is not None and (
            int(cfg.max_cell_area) != cfg.max_cell_area or cfg.max_cell_area <= cfg.min_area):
        raise ValueError('max_cell_area must be an integer larger than min_area or None.')
    if not np.isfinite(cfg.max_cell_elongation) or cfg.max_cell_elongation < 1:
        raise ValueError('max_cell_elongation must be at least 1.')
    if not np.isfinite(cfg.max_area_ratio) or cfg.max_area_ratio < 1:
        raise ValueError('max_area_ratio must be at least 1.')
    if cfg.peak_polarity not in (-1, 1) or cfg.flow_sign not in (-1, 1):
        raise ValueError('peak_polarity and flow_sign must be -1 or +1.')
    if cfg.flow_axis not in ('x', 'y') or cfg.sync_mode not in ('trigger', 'edges', 'manual'):
        raise ValueError('Invalid flow_axis or sync_mode.')
    if cfg.cellpose_input not in ('raw', 'difference'):
        raise ValueError("cellpose_input must be 'raw' or 'difference'.")


class Recording:
    def __init__(self, cfg):
        from pims import Cine
        validate(cfg)
        self.cfg = Config(**asdict(cfg))
        with np.load(cfg.apd_path, allow_pickle=False) as z:
            self.t = np.asarray(z['time_s'], dtype=float)
            self.apd = np.asarray(z['photodiode_signal'], dtype=float)
            self.sync = np.asarray(z['camera_frame_signal'], dtype=float) if 'camera_frame_signal' in z else None
            self.metadata = json.loads(str(z['metadata_json'].item())) if 'metadata_json' in z else {}
        if self.t.ndim != 1 or len(self.t) < 3 or self.apd.shape != self.t.shape:
            raise ValueError('APD time and signal must be one-dimensional arrays of equal length.')
        if not np.all(np.isfinite(self.t)) or not np.all(np.isfinite(self.apd)) or np.any(np.diff(self.t) <= 0):
            raise ValueError('Invalid APD values or time axis.')
        self.dt = float(np.median(np.diff(self.t)))
        if not np.allclose(np.diff(self.t), self.dt, rtol=0.01, atol=1e-12):
            raise ValueError('APD sampling must be uniform.')
        self.cine = Cine(str(Path(cfg.cine_path)))
        try:
            self.fps = float(self.cine.frame_rate)
            self.count = len(self.cine)
            if self.count < 1 or not np.isfinite(self.fps) or self.fps <= 0:
                raise ValueError('Invalid CINE frame rate or frame count.')
            self.start = cfg.start_frame
            self.stop = self.count if cfg.stop_frame is None else cfg.stop_frame
            if not 0 <= self.start < self.stop <= self.count or int(self.stop) != self.stop:
                raise ValueError('Invalid frame range; stop_frame is exclusive.')
            self.header_fps = self.fps
            self.times = frame_clock(self, cfg)
            self.frame_period_s = float(np.median(np.diff(self.times))) if self.count > 1 else 1/self.fps
            self.fps = 1/self.frame_period_s
            self.overlap = require_overlap(self.times, self.t, self.frame_period_s)
            self.shape = self.frame(self.start).shape
            if cfg.passage_edge_px * 2 >= self.shape[1 if cfg.flow_axis == 'x' else 0]:
                raise ValueError('passage_edge_px is too large for this image.')
        except Exception:
            self.close()
            raise

    def frame(self, index):
        image = np.array(self.cine[int(index)], copy=True)
        if image.ndim != 2:
            raise ValueError('This pipeline expects grayscale CINE frames.')
        return image

    def close(self):
        self.cine.close()


def shared_trigger_times(cine, metadata, correction_s=0.0):
    """Same clock mapping as apd_phantom_cine_event_clips.ipynb."""
    trigger = metadata.get('trigger_time_from_acquisition_start_s')
    if trigger is None:
        raise ValueError('metadata_json has no trigger_time_from_acquisition_start_s. '
                         'Shared-trigger alignment is unavailable; use a known manual frame start.')
    trigger = float(trigger)
    if not np.isfinite(trigger) or not np.isfinite(correction_s):
        raise ValueError('Trigger time and time correction must be finite.')
    cine_trigger = cine.trigger_time
    trigger_datetime = cine_trigger['datetime']
    trigger_fraction = float(cine_trigger['second_fraction'])
    stamps = cine.frame_time_stamps
    relative = np.fromiter(
        ((dt-trigger_datetime).total_seconds() + float(frac)-trigger_fraction
         for dt, frac in stamps), dtype=np.float64, count=len(stamps))
    if len(relative) != len(cine) or not len(relative):
        raise ValueError('The number of CINE timestamps does not match the frame count.')
    if not np.all(np.isfinite(relative)) or np.any(np.diff(relative) <= 0):
        raise ValueError('CINE frame timestamps are invalid or not strictly increasing.')
    return trigger + relative + correction_s


def require_overlap(times, apd_times, period_s):
    start = max(apd_times[0], times[0])
    stop = min(apd_times[-1], times[-1] + period_s)
    if stop <= start:
        raise ValueError('CINE and APD recordings do not overlap in time.')
    return float(start), float(stop)


def frame_clock(rec, cfg):
    if cfg.sync_mode == 'trigger':
        return shared_trigger_times(rec.cine, rec.metadata, cfg.sync_correction_s)
    if cfg.sync_mode == 'manual':
        return cfg.first_frame_time_s + cfg.sync_correction_s + np.arange(rec.count) / rec.fps
    s = rec.sync
    if s is None or s.shape != rec.t.shape or not np.all(np.isfinite(s)):
        raise ValueError('F-Sync is missing or invalid; use a known manual frame start.')
    if 1 / rec.dt < 4 * rec.fps:
        raise ValueError('Fewer than four NI samples per frame; F-Sync edges cannot be resolved reliably.')
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        raise ValueError('F-Sync is constant.')
    for _ in range(20):
        low = s < (lo + hi) / 2
        if low.all() or not low.any():
            break
        lo, hi = float(s[low].mean()), float(s[~low].mean())
    active = s < (lo + hi) / 2 if cfg.active_low else s > (lo + hi) / 2
    edges = rec.t[np.flatnonzero(~active[:-1] & active[1:]) + 1]
    if len(edges) != rec.count:
        raise ValueError(f'Found {len(edges)} complete F-Sync edges for {rec.count} CINE frames. '
                         'Check the frame range or use sync_mode="manual" with a known first_frame_time_s.')
    if len(edges) > 1 and not np.allclose(np.diff(edges), 1 / rec.fps, rtol=0.05, atol=2 * rec.dt):
        raise ValueError('F-Sync intervals do not match the CINE frame rate.')
    return edges + cfg.sync_correction_s


def detect_peaks(rec, cfg):
    signal = cfg.peak_polarity * rec.apd
    if cfg.peak_smooth_ms > 0:
        signal = ndi.gaussian_filter1d(signal, cfg.peak_smooth_ms / 1000 / rec.dt)
    baseline = float(np.median(signal))
    sigma = max(float(1.4826 * np.median(np.abs(signal - baseline))), np.finfo(float).eps)
    height = cfg.peak_height_v if cfg.peak_height_v is not None else cfg.peak_height_sigma * sigma
    indices, properties = find_peaks(signal - baseline, height=height,
                                    prominence=cfg.peak_prominence_sigma * sigma,
                                    distance=max(1, int(np.ceil(cfg.peak_min_distance_ms / 1000 / rec.dt))))
    _, _, left_ips, right_ips = peak_widths(
        signal-baseline, indices, rel_height=cfg.peak_interval_rel_height,
        prominence_data=(properties['prominences'], properties['left_bases'], properties['right_bases']))
    onset = np.interp(left_ips, np.arange(len(rec.t)), rec.t)
    end = np.interp(right_ips, np.arange(len(rec.t)), rec.t)
    table = pd.DataFrame({'apd_onset_s': onset, 'apd_end_s': end,
                          'event_onset_s': onset-cfg.apd_delay_s,
                          'event_end_s': end-cfg.apd_delay_s, 'peak_id': np.arange(len(indices)), 'sample': indices,
                          'apd_time_s': rec.t[indices],
                          'event_time_s': rec.t[indices] - cfg.apd_delay_s,
                          'raw_v': rec.apd[indices], 'height_v': properties['peak_heights'],
                          'prominence_v': properties['prominences']})
    return table, signal, baseline + height


def plot_peaks(rec, cfg, limits=None):
    peaks, signal, threshold = detect_peaks(rec, cfg)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, constrained_layout=True)
    left, right = limits if limits is not None else (rec.t[0], rec.t[-1])
    a, b = np.searchsorted(rec.t, (left, right))
    # A min/max envelope preserves narrow peaks in long overview plots.
    ids = np.arange(max(0, a-1), min(len(rec.t), b+1))
    if len(ids) > 30000:
        blocks = np.array_split(ids, 7500)
        ids = np.sort(np.concatenate([[g[np.argmin(signal[g])], g[np.argmax(signal[g])]] for g in blocks]))
    axes[0].plot(rec.t[ids], signal[ids], lw=0.7, label='APD (polarity and smoothing applied)')
    axes[0].axhline(threshold, color='red', ls='--', label='Height threshold')
    chosen = peaks[(peaks.apd_time_s >= left) & (peaks.apd_time_s <= right)]
    ix = chosen['sample'].to_numpy(dtype=int)
    axes[0].scatter(rec.t[ix], signal[ix], c='red', s=24, label=f'Peaks ({len(peaks)} total)')
    for peak in peaks.itertuples():
        if peak.apd_end_s >= left and peak.apd_onset_s <= right:
            axes[0].axvspan(peak.apd_onset_s, peak.apd_end_s, color='orange', alpha=0.2)
    axes[0].set_ylabel('APD [V]'); axes[0].legend()
    if rec.sync is not None:
        step = max(1, (b-a)//30000)
        axes[1].plot(rec.t[a:b:step], rec.sync[a:b:step], lw=0.7)
    axes[1].set(xlabel='NI time [s]', ylabel='F-Sync [V]', xlim=(left, right))
    for ax in axes:
        ax.axvspan(rec.times[rec.start], rec.times[rec.stop-1] + 1/rec.fps, alpha=0.08, color='green')
        ax.grid(alpha=0.2)
    return fig, peaks


def normalize(image):
    arr = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
    return np.clip((arr-lo)/max(float(hi-lo), 1e-6), 0, 1)


def prepare(cfg):
    rec = Recording(cfg)
    try:
        ids = np.unique(np.linspace(rec.start, rec.stop-1, min(cfg.background_samples, rec.stop-rec.start)).astype(int))
        background = np.median(np.stack([normalize(rec.frame(i)) for i in ids]), axis=0).astype(np.float32)
        print(f'CINE: {rec.count} frames, {rec.fps:g} fps; NI: {1/rec.dt:g} samples/s')
        print(f'Time mapping: {cfg.sync_mode}; APD overlap {rec.overlap[0]:.6f} to {rec.overlap[1]:.6f} s')
        return rec, background
    except Exception:
        rec.close()
        raise


def foreground(image, background, cfg):
    difference = ndi.gaussian_filter(np.abs(normalize(image)-background), 1)
    center = np.median(difference)
    sigma = 1.4826 * np.median(np.abs(difference-center))
    mask = ndi.binary_closing(difference > center + cfg.screen_z * max(sigma, 1e-6), iterations=2)
    labels, _ = ndi.label(mask)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    keep = sizes >= cfg.screen_min_area
    return difference, keep[labels]


def load_model(cfg):
    from cellpose import models, core
    major = int(importlib.metadata.version('cellpose').split('.')[0])
    kwargs = {'gpu': bool(cfg.use_gpu and core.use_gpu())}
    if cfg.pretrained_model:
        path = Path(cfg.pretrained_model).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'Cellpose model not found: {path}')
        kwargs['pretrained_model'] = str(path)
    elif major < 4:
        kwargs['model_type'] = cfg.builtin_model
    print(f'Cellpose {importlib.metadata.version("cellpose")}; GPU={kwargs["gpu"]}; '
          f'Model={cfg.pretrained_model or (cfg.builtin_model if major < 4 else "Cellpose-SAM")}')
    print('Custom weights must match the installed Cellpose version.')
    model = models.CellposeModel(**kwargs)
    model._dataset_model_config = (cfg.pretrained_model, cfg.builtin_model, cfg.use_gpu)
    return model


def segment(image, background, model, cfg, force=False):
    expected = (cfg.pretrained_model, cfg.builtin_model, cfg.use_gpu)
    if getattr(model, '_dataset_model_config', expected) != expected:
        raise ValueError('Model configuration changed; run load_model() again.')
    difference, candidate = foreground(image, background, cfg)
    if cfg.screen_enabled and not candidate.any() and not force:
        return np.zeros(image.shape, dtype=np.int32), difference, candidate
    cp = normalize(image) if cfg.cellpose_input == 'raw' else normalize(difference)
    if cfg.invert:
        cp = 1-cp
    kwargs = dict(diameter=cfg.diameter, min_size=cfg.min_size,
                  cellprob_threshold=cfg.cellprob_threshold, flow_threshold=cfg.flow_threshold,
                  normalize=True, channel_axis=None)
    signature = inspect.signature(model.eval)
    if 'channels' in signature.parameters:
        kwargs['channels'] = [0, 0]
    masks = np.asarray(model.eval(cp, **kwargs)[0], dtype=np.int32)
    if masks.shape != image.shape:
        raise ValueError('The Cellpose mask does not match the image shape.')
    return masks, difference, candidate


def preview(rec, background, model, cfg, frame_index):
    raw = rec.frame(frame_index)
    # Force segmentation in QC even if the fast screen would skip this frame.
    masks, diff, candidate = segment(raw, background, model, cfg, force=True)
    rows = objects(masks, frame_index, rec.times[frame_index], cfg)
    rows_by_label = {row['label']: row for row in rows}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, data, title in zip(axes, (raw, diff, candidate, raw),
                               ('Original', 'Background difference', 'Candidates', 'Cellpose QC')):
        ax.imshow(data, cmap='gray'); ax.set_title(title); ax.axis('off')
    for label in np.unique(masks):
        if label:
            rejected = rows_by_label.get(int(label), {}).get('cluster_candidate', False)
            axes[3].contour(masks == label, levels=[0.5], linewidths=1.2,
                            colors='red' if rejected else 'lime')
    if cfg.detection_roi is not None:
        from matplotlib.patches import Rectangle
        x0,x1,y0,y1 = cfg.detection_roi
        h,w = raw.shape
        for ax in (axes[0], axes[3]):
            ax.add_patch(Rectangle((x0*w,y0*h),(x1-x0)*w,(y1-y0)*h,
                                   fill=False, edgecolor='orange', linewidth=2))
    fig.suptitle(f'Frame {frame_index}, NI {rec.times[frame_index]:.6f} s; '
                 f'pre-screen: {"keep" if candidate.any() else "skip"}; '
                 f'green: single cell, red: rejected cluster mask')
    return fig


def objects(masks, frame, timestamp, cfg):
    rows = []
    for label, sl in enumerate(ndi.find_objects(masks), start=1):
        if sl is None:
            continue
        yy, xx = np.where(masks[sl] == label)
        area = len(yy)
        if area < cfg.min_area:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        coordinates = np.column_stack((yy, xx)).astype(float)
        covariance = np.cov(coordinates, rowvar=False) if len(coordinates) > 1 else np.eye(2)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), np.finfo(float).eps)
        elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
        component = masks == label
        nearby_mask = False
        if cfg.cluster_separation_px > 0:
            expanded = ndi.binary_dilation(component, iterations=cfg.cluster_separation_px)
            nearby_mask = bool(np.any(expanded & (masks != 0) & (masks != label)))
        too_large = cfg.max_cell_area is not None and area > cfg.max_cell_area
        too_elongated = elongation > cfg.max_cell_elongation
        cluster_flags = []
        if nearby_mask:
            cluster_flags.append('nearby_mask')
        if too_large:
            cluster_flags.append('large_mask')
        if too_elongated:
            cluster_flags.append('elongated_mask')
        edge = min(y0, x0, masks.shape[0]-y1, masks.shape[1]-x1)
        rows.append(dict(frame=int(frame), time_s=float(timestamp), label=label, area=area,
                         y=float(yy.mean()+y0), x=float(xx.mean()+x0),
                         y0=y0, x0=x0, y1=y1, x1=x1, edge=edge,
                         mask_elongation=elongation, nearby_mask=nearby_mask,
                         cluster_candidate=bool(cluster_flags),
                         cluster_flags=';'.join(cluster_flags)))
    return rows


class Tracker:
    """Link detections using directed constant-velocity motion prediction."""
    def __init__(self, cfg, image_shape=None):
        self.cfg = cfg
        self.image_shape = image_shape
        self.tracks = {}
        self.next_id = 0
        self.active = set()

    def _has_exited(self, row):
        if not self.cfg.close_tracks_at_exit:
            return False
        axis = self.cfg.flow_axis
        if self.cfg.flow_sign < 0:
            return axis+'0' in row and row[axis+'0'] <= 0
        if self.image_shape is None:
            return False
        length = self.image_shape[1 if axis == 'x' else 0]
        return axis+'1' in row and row[axis+'1'] >= length

    def _predicted_position(self, track, gap):
        cfg = self.cfg
        rows = track['rows']
        last = rows[-1]
        velocity = np.zeros(2, dtype=float)
        has_prediction = cfg.tracking_expected_step_px is not None
        axis_index = 0 if cfg.flow_axis == 'x' else 1
        if has_prediction:
            velocity[axis_index] = cfg.flow_sign * cfg.tracking_expected_step_px
        if len(rows) >= 2:
            recent = rows[-4:]
            estimates = []
            for previous, current in zip(recent, recent[1:]):
                frame_gap = current['frame'] - previous['frame']
                if frame_gap > 0:
                    estimates.append(((current['x']-previous['x']) / frame_gap,
                                      (current['y']-previous['y']) / frame_gap))
            if estimates:
                measured = np.median(np.asarray(estimates, dtype=float), axis=0)
                if measured[axis_index] * cfg.flow_sign > 0:
                    velocity = measured
                    has_prediction = True
        predicted = np.array([last['x'], last['y']], dtype=float) + velocity * gap
        return predicted, has_prediction

    def update(self, rows, frame):
        cfg = self.cfg
        flag = lambda value: bool(value) if pd.notna(value) else False
        self.active = {
            i for i in self.active
            if frame-self.tracks[i]['rows'][-1]['frame'] <= cfg.track_memory_frames+1
            and not self._has_exited(self.tracks[i]['rows'][-1])
        }
        active = sorted(self.active)
        used = set()
        assignments = []
        if active and rows:
            cost = np.full((len(active), len(rows)), 1e12)
            for a, tid in enumerate(active):
                track = self.tracks[tid]
                last = track['rows'][-1]
                gap = frame-last['frame']
                predicted, has_prediction = self._predicted_position(track, gap)
                for b, row in enumerate(rows):
                    distance = np.hypot(row['x']-last['x'], row['y']-last['y'])
                    ratio = max(row['area'], last['area'])/min(row['area'], last['area'])
                    axis_delta = (row[cfg.flow_axis]-last[cfg.flow_axis]) * cfg.flow_sign
                    prediction_error = np.hypot(row['x']-predicted[0], row['y']-predicted[1])
                    direction_ok = axis_delta >= -cfg.tracking_reverse_tolerance_px * gap
                    prediction_ok = (not has_prediction or
                                     prediction_error <= cfg.tracking_prediction_tolerance_px * gap)
                    if (distance <= cfg.max_link_distance_px * gap and
                            ratio <= cfg.max_area_ratio and direction_ok and prediction_ok):
                        cost[a, b] = prediction_error if has_prediction else distance
            eligible = cost < 1e12
            margin = cfg.tracking_ambiguity_margin_px
            ambiguous_rows = np.zeros(len(rows), dtype=bool)
            ambiguous_tracks = np.zeros(len(active), dtype=bool)
            for b in range(len(rows)):
                values = np.sort(cost[eligible[:, b], b])
                ambiguous_rows[b] = len(values) > 1 and values[1]-values[0] <= margin
            for a in range(len(active)):
                values = np.sort(cost[a, eligible[a]])
                ambiguous_tracks[a] = len(values) > 1 and values[1]-values[0] <= margin
            for a, tid in enumerate(active):
                if ambiguous_tracks[a] or np.any(eligible[a] & ambiguous_rows):
                    self.tracks[tid]['ambiguous'] = True
            for b, row in enumerate(rows):
                row['ambiguous_link'] = bool(ambiguous_rows[b] or np.any(eligible[:, b] & ambiguous_tracks))
            ai, bi = linear_sum_assignment(cost)
            for a, b in zip(ai, bi):
                if not eligible[a, b]:
                    continue
                tid = active[a]
                track = self.tracks[tid]
                if frame-track['rows'][-1]['frame'] > 1:
                    track['gap'] = True
                track['ambiguous'] |= flag(rows[b].get('ambiguous_link', False))
                if flag(rows[b].get('cluster_candidate', False)):
                    track['cluster'] = True
                    track['cluster_reasons'].update(
                        part for part in str(rows[b].get('cluster_flags', '')).split(';')
                        if part and part != 'nan'
                    )
                assignments.append((tid, rows[b])); used.add(b)
        for b, row in enumerate(rows):
            if b not in used:
                tid = self.next_id; self.next_id += 1
                self.active.add(tid)
                self.tracks[tid] = {'rows': [],
                                    'ambiguous': flag(row.get('ambiguous_link', False)), 'gap': False,
                                    'cluster': flag(row.get('cluster_candidate', False)),
                                    'cluster_reasons': set(
                                        part for part in str(row.get('cluster_flags', '')).split(';')
                                        if part and part != 'nan'
                                    ),
                                    'best_score': None, 'best': None}
                assignments.append((tid, row))
        for tid, row in assignments:
            row['track_id'] = tid
            self.tracks[tid]['rows'].append(row)
        return assignments


def square_crop(raw, masks, row, cfg):
    side = max(row['y1']-row['y0'], row['x1']-row['x0']) + 2*cfg.crop_margin_px
    y = int(np.floor((row['y0']+row['y1']-side)/2))
    x = int(np.floor((row['x0']+row['x1']-side)/2))
    h, w = raw.shape
    crop = np.full((side, side), np.median(np.concatenate((raw[0], raw[-1], raw[:, 0], raw[:, -1]))), dtype=raw.dtype)
    mask = np.zeros((side, side), dtype=np.uint8)
    ya, yb, xa, xb = max(0, y), min(h, y+side), max(0, x), min(w, x+side)
    dest = np.s_[ya-y:yb-y, xa-x:xb-x]
    crop[dest] = raw[ya:yb, xa:xb]
    mask[dest] = (masks[ya:yb, xa:xb] == row['label']).astype(np.uint8)
    return crop, mask, x, y


def complete_passage(track, shape, start, stop, cfg):
    rows = track['rows']; first, last = rows[0], rows[-1]
    if first['frame'] <= start or last['frame'] >= stop-1:
        return False
    axis = cfg.flow_axis; length = shape[1 if axis == 'x' else 0]
    near_low = lambda row: row[axis+'0'] <= cfg.passage_edge_px
    near_high = lambda row: row[axis+'1'] >= length-cfg.passage_edge_px
    return bool((near_low(first) and near_high(last)) if cfg.flow_sign > 0 else
                (near_high(first) and near_low(last)))


def region_passage(track, rec, cfg):
    rows = track['rows']
    if cfg.detection_roi is None:
        return rows, complete_passage(track, rec.shape, rec.start, rec.stop, cfg)
    x0,x1,y0,y1 = cfg.detection_roi
    h,w = rec.shape
    x0,x1,y0,y1 = x0*w,x1*w,y0*h,y1*h
    inside = [r for r in rows if r['x1'] > x0 and r['x0'] < x1
              and r['y1'] > y0 and r['y0'] < y1]
    if not inside:
        return [], False
    # Require observations entirely before entry and entirely after exit.
    axis = cfg.flow_axis
    low, high = (x0,x1) if axis == 'x' else (y0,y1)
    before = [r for r in rows if r['frame'] < inside[0]['frame']]
    after = [r for r in rows if r['frame'] > inside[-1]['frame']]
    if cfg.flow_sign < 0:
        complete = any(r[axis+'0'] >= high for r in before) and any(r[axis+'1'] <= low for r in after)
    else:
        complete = any(r[axis+'1'] <= low for r in before) and any(r[axis+'0'] >= high for r in after)
    return inside, bool(complete)


def passage_table(tracks, rec, cfg):
    """Return one temporal anchor per track inside the configured detection ROI."""
    records = []
    times = np.asarray(rec.times, dtype=float)
    for track_id, track in tracks.items():
        if cfg.reject_cluster_tracks and track.get('cluster', False):
            continue
        rows, complete = region_passage(track, rec, cfg)
        if not rows:
            continue
        last = rows[-1]
        next_index = int(last['frame']) + 1
        start = float(rows[0]['time_s'])
        end = float(times[next_index] if next_index < len(times)
                    else last['time_s'] + 1 / rec.fps)
        records.append({
            'track_id': int(track_id),
            'passage_start_s': start,
            'passage_end_s': end,
            'passage_center_s': (start + end) / 2,
            'complete_passage': bool(complete),
        })
    return pd.DataFrame(records)


def align_peak_times(tracks, peaks, rec, cfg):
    """Correct constant offset and linear clock drift between APD and CINE time.

    The fit uses mutual nearest neighbours between APD peak maxima and cell
    passage centres. A robust affine model is then applied to the complete peak
    interval. Raw event times are retained in ``*_uncorrected`` columns.
    """
    aligned = peaks.copy()
    event_columns = ('event_onset_s', 'event_end_s', 'event_time_s')
    for column in event_columns:
        raw_column = f'{column}_uncorrected'
        if raw_column not in aligned.columns and column in aligned.columns:
            aligned[raw_column] = aligned[column]

    diagnostics = {
        'alignment_status': 'disabled',
        'alignment_pairs': 0,
        'alignment_reference_s': np.nan,
        'alignment_offset_ms': 0.0,
        'alignment_drift_ppm': 0.0,
        'alignment_rmse_ms': np.nan,
    }
    if not cfg.correct_time_drift or aligned.empty:
        for key, value in diagnostics.items():
            aligned[key] = value
        return aligned, diagnostics

    passages = passage_table(tracks, rec, cfg)
    if passages.empty:
        diagnostics['alignment_status'] = 'no_passages'
        for key, value in diagnostics.items():
            aligned[key] = value
        return aligned, diagnostics

    peak_times = aligned['event_time_s_uncorrected'].to_numpy(dtype=float)
    cell_times = passages['passage_center_s'].to_numpy(dtype=float)
    distances = np.abs(peak_times[:, None] - cell_times[None, :])
    nearest_cell = distances.argmin(axis=1)
    nearest_peak = distances.argmin(axis=0)
    max_gap_s = cfg.drift_search_ms / 1000
    pairs = [(peak_index, cell_index)
             for peak_index, cell_index in enumerate(nearest_cell)
             if nearest_peak[cell_index] == peak_index
             and distances[peak_index, cell_index] <= max_gap_s]

    if len(pairs) < cfg.drift_min_pairs:
        diagnostics.update(alignment_status='insufficient_pairs', alignment_pairs=len(pairs))
        for key, value in diagnostics.items():
            aligned[key] = value
        return aligned, diagnostics

    peak_index, cell_index = np.asarray(pairs, dtype=int).T
    x = peak_times[peak_index]
    y = cell_times[cell_index]
    reference = float(np.median(x))
    keep = np.ones(len(x), dtype=bool)
    beta = np.zeros(2)
    for _ in range(8):
        design = np.column_stack((np.ones(keep.sum()), x[keep] - reference))
        beta = np.linalg.lstsq(design, y[keep] - x[keep], rcond=None)[0]
        residual = (y - x) - (beta[0] + beta[1] * (x - reference))
        center = float(np.median(residual[keep]))
        mad = float(1.4826 * np.median(np.abs(residual[keep] - center)))
        cutoff = max(cfg.drift_residual_ms / 1000, 3 * mad)
        updated = np.abs(residual - center) <= cutoff
        if updated.sum() < cfg.drift_min_pairs or np.array_equal(updated, keep):
            break
        keep = updated

    design = np.column_stack((np.ones(keep.sum()), x[keep] - reference))
    beta = np.linalg.lstsq(design, y[keep] - x[keep], rcond=None)[0]
    residual = (y[keep] - x[keep]) - (beta[0] + beta[1] * (x[keep] - reference))
    drift_ppm = float(beta[1] * 1e6)
    if abs(drift_ppm) > cfg.drift_max_ppm:
        diagnostics.update(alignment_status='drift_limit_exceeded',
                           alignment_pairs=int(keep.sum()),
                           alignment_reference_s=reference,
                           alignment_drift_ppm=drift_ppm)
        for key, value in diagnostics.items():
            aligned[key] = value
        return aligned, diagnostics

    diagnostics.update(
        alignment_status='applied',
        alignment_pairs=int(keep.sum()),
        alignment_reference_s=reference,
        alignment_offset_ms=float(beta[0] * 1000),
        alignment_drift_ppm=drift_ppm,
        alignment_rmse_ms=float(np.sqrt(np.mean(residual ** 2)) * 1000),
    )
    for column in event_columns:
        raw_column = f'{column}_uncorrected'
        if raw_column not in aligned.columns:
            continue
        raw = aligned[raw_column].to_numpy(dtype=float)
        aligned[column] = raw + beta[0] + beta[1] * (raw - reference)
    for key, value in diagnostics.items():
        aligned[key] = value
    return aligned, diagnostics


def classify(tracks, peaks, rec, cfg):
    peaks, alignment = align_peak_times(tracks, peaks, rec, cfg)
    if 'peak_id' not in peaks.columns:
        peaks['peak_id'] = np.arange(len(peaks), dtype=int)
    pad = cfg.match_padding_ms / 1000
    peak_start = peaks.get('event_onset_s', peaks.event_time_s).to_numpy(dtype=float)
    peak_end = peaks.get('event_end_s', peaks.event_time_s).to_numpy(dtype=float)
    peak_time = peaks.get('event_time_s', pd.Series(dtype=float)).to_numpy(dtype=float)
    times = np.asarray(getattr(rec, 'times', []), dtype=float)
    candidates, candidate_distance = {}, {}
    intervals, passage_intervals, passages = {}, {}, {}
    for tid, tr in tracks.items():
        rows, complete = region_passage(tr, rec, cfg)
        passages[tid] = (rows, complete)
        intervals[tid] = (float('nan'), float('nan'))
        passage_intervals[tid] = (float('nan'), float('nan'))
        candidates[tid] = []
        if cfg.reject_cluster_tracks and tr.get('cluster', False):
            continue
        if not rows:
            continue
        last = rows[-1]
        next_index = last['frame'] + 1
        passage_start = rows[0]['time_s']
        passage_end = times[next_index] if next_index < len(times) else last['time_s'] + 1/rec.fps
        passage_intervals[tid] = (passage_start, passage_end)
        lo, hi = passage_start - pad, passage_end + pad
        intervals[tid] = (lo, hi)
        passage_center = (passage_start + passage_end) / 2
        # Match against observed frame intervals. Peak maxima alone are too narrow.
        for p in range(len(peaks)):
            best_distance = np.inf
            for row in rows:
                i = row['frame'] + 1
                end = times[i] if i < len(times) else row['time_s'] + 1/rec.fps
                distance = max(float(row['time_s']) - peak_end[p], peak_start[p] - end, 0.0)
                best_distance = min(best_distance, distance)
            if best_distance <= pad:
                candidates[tid].append(p)
                # Interval distance is primary; centre distance resolves overlaps.
                candidate_distance[tid, p] = (
                    best_distance + 1e-3 * abs(peak_time[p] - passage_center)
                )
    owners = {p: [tid for tid, ids in candidates.items() if p in ids] for p in range(len(peaks))}

    # Global minimum-cost matching enforces one peak per cell and one cell per peak.
    track_ids = list(tracks)
    assigned = {}
    assignment_distance_ms = {}
    if track_ids and len(peaks):
        cost = np.full((len(track_ids), len(peaks)), 1e9, dtype=float)
        for row_index, tid in enumerate(track_ids):
            for peak_index in candidates[tid]:
                cost[row_index, peak_index] = candidate_distance[tid, peak_index]
        row_index, peak_index = linear_sum_assignment(cost)
        for r, p in zip(row_index, peak_index):
            if cost[r, p] >= 1e9:
                continue
            tid = track_ids[r]
            assigned[tid] = int(p)
            assignment_distance_ms[tid] = float(candidate_distance[tid, p] * 1000)

    result = []
    for tid, tr in tracks.items():
        candidate_indices = candidates[tid]
        assigned_peak = assigned.get(tid)
        lo, hi = intervals[tid]
        reasons = []
        quality_notes = []
        blocking_reasons = []
        if len(tr['rows']) < cfg.min_track_detections:
            reasons.append('short_track')
            quality_notes.append('short_track')
        if tr['ambiguous']:
            reasons.append('ambiguous_tracking')
            quality_notes.append('ambiguous_tracking')
        if tr['gap']:
            reasons.append('tracking_gap')
            quality_notes.append('tracking_gap')
        if cfg.reject_cluster_tracks and tr.get('cluster', False):
            detail = ','.join(sorted(tr.get('cluster_reasons', set())))
            reason = 'cluster_or_touching_cells' + (f'[{detail}]' if detail else '')
            reasons.append(reason)
            blocking_reasons.append(reason)
        region_rows, complete = passages[tid]
        if not region_rows:
            reasons.append('no_detection_region_crossing')
            quality_notes.append('no_detection_region_crossing')
        if not complete:
            reasons.append('incomplete_passage')
            quality_notes.append('incomplete_passage')
        if region_rows and any(b['frame'] != a['frame']+1 for a,b in zip(region_rows,region_rows[1:])):
            reasons.append('interrupted_region_passage')
            quality_notes.append('interrupted_region_passage')
        incomplete_apd = (np.isfinite(lo) and
                          (lo + cfg.apd_delay_s < rec.t[0] or hi + cfg.apd_delay_s > rec.t[-1]))
        if incomplete_apd:
            reasons.append('incomplete_apd_coverage')
            quality_notes.append('incomplete_apd_coverage')
        if tr['best'] is None or tr['best']['edge'] <= cfg.edge_margin_px:
            reasons.append('no_complete_cell_crop')
            blocking_reasons.append('no_complete_cell_crop')
        if len(candidate_indices) > 1:
            reasons.append('multiple_peak_candidates')
            quality_notes.append('multiple_peak_candidates')
        if assigned_peak is not None and len(owners[assigned_peak]) > 1:
            reasons.append('shared_peak_candidate')
            quality_notes.append('shared_peak_candidate')
        if assigned_peak is None and candidate_indices:
            reasons.append('candidate_assigned_elsewhere')
            quality_notes.append('candidate_assigned_elsewhere')
        category = ('uneindeutig' if blocking_reasons else
                    ('zelltyp_1_peak' if assigned_peak is not None else 'zelltyp_2_kein_peak'))
        best = tr['best'] or tr['rows'][0]
        passage_start, passage_end = passage_intervals[tid]
        def peak_value(column, fallback=None):
            if assigned_peak is None:
                return float('nan')
            selected = column if column in peaks.columns else fallback
            return float(peaks.iloc[assigned_peak][selected]) if selected in peaks.columns else float('nan')
        peak_event_time = peak_value('event_time_s')
        peak_apd_time = peak_value('apd_time_s')
        peak_event_onset = peak_value('event_onset_s', 'event_time_s')
        peak_event_end = peak_value('event_end_s', 'event_time_s')
        peak_frame = float('nan')
        peak_frame_time = float('nan')
        peak_minus_frame_ms = float('nan')
        if np.isfinite(peak_event_time) and len(times):
            right = min(len(times)-1, int(np.searchsorted(times, peak_event_time)))
            left = max(0, right-1)
            peak_frame = left if abs(peak_event_time-times[left]) <= abs(peak_event_time-times[right]) else right
            peak_frame_time = times[int(peak_frame)]
            peak_minus_frame_ms = 1000 * (peak_event_time-peak_frame_time)
        assigned_peak_id = (int(peaks.iloc[assigned_peak]['peak_id'])
                            if assigned_peak is not None else None)
        candidate_peak_ids = [int(peaks.iloc[p]['peak_id']) for p in candidate_indices]
        result.append(dict(track_id=tid, category=category,
                           reason=';'.join(reasons) or 'automatic_assignment',
                           blocking_reason=';'.join(blocking_reasons),
                           quality_notes=';'.join(quality_notes),
                           start_frame=tr['rows'][0]['frame'], end_frame=tr['rows'][-1]['frame'],
                           n_detections=len(tr['rows']),
                           passage_start_frame=region_rows[0]['frame'] if region_rows else float('nan'),
                           passage_end_frame=region_rows[-1]['frame'] if region_rows else float('nan'),
                           passage_start_s=passage_start, passage_end_s=passage_end,
                           passage_center_s=(passage_start+passage_end)/2,
                           match_start_s=lo, match_end_s=hi,
                           peak_ids='' if assigned_peak_id is None else str(assigned_peak_id),
                           candidate_peak_ids=';'.join(map(str, candidate_peak_ids)),
                           assignment_distance_ms=assignment_distance_ms.get(tid, float('nan')),
                           representative_frame=best['frame'],
                           representative_time_s=best['time_s'],
                           peak_event_time_s=peak_event_time, peak_apd_time_s=peak_apd_time,
                           peak_event_onset_s=peak_event_onset, peak_event_end_s=peak_event_end,
                           peak_frame=peak_frame, peak_frame_time_s=peak_frame_time,
                           peak_minus_frame_ms=peak_minus_frame_ms,
                           alignment_status=alignment['alignment_status'],
                           alignment_offset_ms=alignment['alignment_offset_ms'],
                           alignment_drift_ppm=alignment['alignment_drift_ppm']))
    columns = ['track_id', 'category', 'reason', 'blocking_reason', 'quality_notes',
               'start_frame', 'end_frame', 'n_detections',
               'passage_start_frame', 'passage_end_frame', 'passage_start_s', 'passage_end_s',
               'passage_center_s', 'match_start_s', 'match_end_s', 'peak_ids',
               'candidate_peak_ids', 'assignment_distance_ms',
               'representative_frame', 'representative_time_s', 'peak_event_time_s',
               'peak_apd_time_s', 'peak_event_onset_s', 'peak_event_end_s', 'peak_frame',
               'peak_frame_time_s', 'peak_minus_frame_ms', 'alignment_status',
               'alignment_offset_ms', 'alignment_drift_ppm']
    table = pd.DataFrame(result, columns=columns)
    audit = peaks.copy()
    audit['candidate_track_ids'] = [';'.join(map(str, owners[p])) for p in range(len(peaks))]
    audit['candidate_count'] = [len(owners[p]) for p in range(len(peaks))]
    assigned_track_by_peak = {peak_index: track_id for track_id, peak_index in assigned.items()}
    audit['assigned_track_id'] = [assigned_track_by_peak.get(p, np.nan) for p in range(len(peaks))]
    audit['assignment_status'] = np.where(audit['assigned_track_id'].notna(), 'assigned', 'unassigned')
    return table, audit


def reclassify_exported(table, detections, peaks, rec, cfg):
    """Re-run APD association for an exported run without segmenting images again.

    Existing crops and their paths are retained. Track-quality flags are rebuilt
    from the exported detections and the original automatic reasons.
    """
    required = {'track_id', 'frame', 'time_s', 'x', 'y', 'x0', 'x1', 'y0', 'y1', 'edge'}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f'detections.csv is missing columns: {sorted(missing)}')
    original = table.copy()
    tracks = {}
    for row in original.itertuples(index=False):
        tid = int(row.track_id)
        group = detections[detections.track_id == tid].sort_values('frame')
        if group.empty:
            continue
        rows = group.to_dict('records')
        representative = int(getattr(row, 'representative_frame', rows[0]['frame']))
        best_rows = [r for r in rows if int(r['frame']) == representative]
        reason = str(getattr(row, 'reason', '') or '')
        cluster_rows = group.loc[group.get('cluster_candidate', False).fillna(False).astype(bool)] \
            if 'cluster_candidate' in group else group.iloc[:0]
        cluster_reasons = {
            part
            for value in cluster_rows.get('cluster_flags', pd.Series(dtype=str)).fillna('')
            for part in str(value).split(';') if part
        }
        tracks[tid] = {
            'rows': rows,
            'ambiguous': 'ambiguous_tracking' in reason,
            'gap': any(int(b['frame']) != int(a['frame']) + 1 for a, b in zip(rows, rows[1:])),
            'cluster': bool(len(cluster_rows)) or 'cluster_or_touching_cells' in reason,
            'cluster_reasons': cluster_reasons,
            'best': best_rows[0] if best_rows else rows[0],
            'best_score': None,
        }
    revised, audit = classify(tracks, peaks, rec, cfg)
    carry = [c for c in original.columns if c not in revised.columns and c != 'track_id']
    if carry:
        revised = revised.merge(original[['track_id'] + carry], on='track_id', how='left', validate='one_to_one')
    return revised, audit


def reclassify_run(run_dir, rec, cfg=None, redetect_peaks=True):
    """Reclassify an existing run and write auditable replacement CSV files.

    When ``redetect_peaks`` is true, the current peak settings are applied to
    the raw APD signal. Set it to false only to reuse the exported peaks.csv.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if cfg is None:
        saved = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        cfg = Config(**saved)
    cells = pd.read_csv(run_dir / 'cells.csv')
    detections = pd.read_csv(run_dir / 'detections.csv')
    peaks = detect_peaks(rec, cfg)[0] if redetect_peaks else pd.read_csv(run_dir / 'peaks.csv')
    revised, audit = reclassify_exported(cells, detections, peaks, rec, cfg)
    revised['previous_category'] = cells.set_index('track_id').reindex(revised.track_id)['category'].to_numpy()
    revised['previous_reason'] = cells.set_index('track_id').reindex(revised.track_id)['reason'].to_numpy()
    revised.to_csv(run_dir / 'cells_reclassified.csv', index=False)
    audit.to_csv(run_dir / 'peaks_reclassified.csv', index=False)
    review = find_suspicious_track_cases(revised, detections, cfg)
    review.to_csv(run_dir / 'track_proximity_review.csv', index=False)
    (run_dir / 'reclassification_config.json').write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return revised, audit


def _parse_peak_ids(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [int(float(part.strip())) for part in str(value).split(';') if part.strip()]


def assignment_summary(cells, peaks):
    """Return category counts and an audit of detected versus assigned peaks."""
    order = ['zelltyp_1_peak', 'zelltyp_2_kein_peak', 'uneindeutig']
    labels = {
        'zelltyp_1_peak': 'With APD peak',
        'zelltyp_2_kein_peak': 'Without APD peak',
        'uneindeutig': 'Ambiguous',
    }
    counts = cells['category'].value_counts().reindex(order, fill_value=0)
    total = len(cells)
    category_summary = pd.DataFrame({
        'category': order,
        'label': [labels[value] for value in order],
        'count': counts.to_numpy(dtype=int),
        'percent': (100 * counts / total if total else counts.astype(float)).round(2).to_numpy(),
    })

    detected = set(peaks['peak_id'].dropna().astype(int))
    assigned_values = cells.loc[cells['category'].eq('zelltyp_1_peak'), 'peak_ids']
    assigned_list = [peak_id for value in assigned_values for peak_id in _parse_peak_ids(value)]
    assigned = set(assigned_list)
    peak_summary = pd.DataFrame({
        'metric': [
            'Detected APD peaks',
            'Tracks classified with a peak',
            'Unique assigned peak IDs',
            'Detected peaks without a cell',
            'Peak IDs assigned more than once',
            'Assigned IDs missing from the peak table',
        ],
        'count': [
            len(detected),
            int(cells['category'].eq('zelltyp_1_peak').sum()),
            len(assigned & detected),
            len(detected - assigned),
            len(assigned_list) - len(set(assigned_list)),
            len(assigned - detected),
        ],
    })
    return category_summary, peak_summary


def apply_manual_assignments(cells, peaks, assignments, rec=None):
    """Apply reviewed labels without discarding the automatic audit columns.

    ``assignments`` maps a track ID to ``(category, reason)`` or to
    ``(category, reason, peak_id)``. An explicit peak ID is required when more
    than one candidate is available.
    """
    result = cells.copy()
    if 'automatic_category' not in result:
        result['automatic_category'] = result['category']
    if 'automatic_reason' not in result:
        result['automatic_reason'] = result['reason']
    if 'manual_category' not in result:
        result['manual_category'] = pd.NA
    if 'manual_reason' not in result:
        result['manual_reason'] = ''

    allowed = {'zelltyp_1_peak', 'zelltyp_2_kein_peak', 'uneindeutig'}
    peak_fields = {
        'peak_event_time_s': 'event_time_s',
        'peak_apd_time_s': 'apd_time_s',
        'peak_event_onset_s': 'event_onset_s',
        'peak_event_end_s': 'event_end_s',
    }
    known_tracks = set(result['track_id'].astype(int))
    for track_id, decision in assignments.items():
        if int(track_id) not in known_tracks:
            raise ValueError(f'Track {track_id} does not exist.')
        if len(decision) not in (2, 3):
            raise ValueError(f'Track {track_id}: use (category, reason) or (category, reason, peak_id).')
        category, reason = decision[:2]
        if category not in allowed:
            raise ValueError(f'Track {track_id}: invalid category {category!r}.')
        if not str(reason).strip():
            raise ValueError(f'Track {track_id}: a review reason is required.')
        index = result.index[result['track_id'].astype(int).eq(int(track_id))][0]
        peak_id = int(decision[2]) if len(decision) == 3 else None

        if category == 'zelltyp_1_peak':
            if peak_id is None:
                ids = _parse_peak_ids(result.at[index, 'peak_ids'])
                if not ids and 'candidate_peak_ids' in result:
                    ids = _parse_peak_ids(result.at[index, 'candidate_peak_ids'])
                if len(ids) != 1:
                    raise ValueError(f'Track {track_id}: specify one peak_id explicitly; candidates={ids}.')
                peak_id = ids[0]
            selected = peaks.loc[peaks['peak_id'].astype(int).eq(peak_id)]
            if len(selected) != 1:
                raise ValueError(f'Peak {peak_id} was not found exactly once.')
            peak = selected.iloc[0]
            result.at[index, 'peak_ids'] = str(peak_id)
            for target, source in peak_fields.items():
                if source in peak.index:
                    result.at[index, target] = peak[source]
            if rec is not None:
                frame = int(np.argmin(np.abs(np.asarray(rec.times) - peak['event_time_s'])))
                result.at[index, 'peak_frame'] = frame
                result.at[index, 'peak_frame_time_s'] = rec.times[frame]
                result.at[index, 'peak_minus_frame_ms'] = 1000 * (peak['event_time_s'] - rec.times[frame])
        else:
            result.at[index, 'peak_ids'] = ''
            for target in peak_fields:
                result.at[index, target] = np.nan
            for target in ('peak_frame', 'peak_frame_time_s', 'peak_minus_frame_ms'):
                if target in result:
                    result.at[index, target] = np.nan

        result.at[index, 'category'] = category
        result.at[index, 'reason'] = 'manual_override'
        result.at[index, 'manual_category'] = category
        result.at[index, 'manual_reason'] = str(reason).strip()

    assigned = [peak_id for value in result.loc[result['category'].eq('zelltyp_1_peak'), 'peak_ids']
                for peak_id in _parse_peak_ids(value)]
    duplicates = sorted({peak_id for peak_id in assigned if assigned.count(peak_id) > 1})
    if duplicates:
        raise ValueError(f'Peak IDs assigned to more than one cell: {duplicates}')
    return result


def plot_assignments(table, peaks, limits=None):
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    colors = {'zelltyp_1_peak': 'tab:red', 'zelltyp_2_kein_peak': 'tab:blue', 'uneindeutig': 'gray'}
    if limits is None and len(table):
        bounds = table[['match_start_s', 'match_end_s']].to_numpy(dtype=float).ravel()
        bounds = bounds[np.isfinite(bounds)]
        if len(bounds):
            left, right = float(bounds.min()), float(bounds.max())
            margin = max(0.05 * (right-left), 5e-5)
            limits = (left-margin, right+margin)
    for row in table.itertuples():
        color = colors[row.category]
        center = getattr(row, 'passage_center_s', (row.match_start_s+row.match_end_s)/2)
        ax.plot([row.match_start_s, row.match_end_s], [row.track_id]*2,
                color=color, lw=3, zorder=2)
        ax.scatter(center, row.track_id, color=color, s=34, zorder=3)
    for peak in peaks.itertuples():
        onset = getattr(peak, 'event_onset_s', peak.event_time_s)
        end = getattr(peak, 'event_end_s', peak.event_time_s)
        if limits is None or (end >= limits[0] and onset <= limits[1]):
            ax.axvspan(onset, end, color='orange', alpha=0.2, zorder=0)
            ax.axvline(peak.event_time_s, color='orange', alpha=0.5, lw=0.7, zorder=1)
    if limits is not None:
        ax.set_xlim(*limits)
    ax.set(xlabel='NI time after APD delay and drift correction [s]', ylabel='Track ID',
           title='Dot: passage centre | Red: peak | Blue: no peak | Grey: ambiguous | Orange: APD event')
    return fig


def find_suspicious_track_cases(cells, detections, cfg):
    """Find nearby simultaneous tracks and plausible track fragments.

    Simultaneous close detections may be adjacent cells or Cellpose
    over-segmentation. Sequential close detections may be fragments of one
    biological trajectory. The returned table is a review list, not an
    automatic merge decision.
    """
    cell_columns = {'track_id', 'passage_center_s', 'category'}
    detection_columns = {'track_id', 'frame', 'time_s', 'x', 'y', 'area'}
    if missing := cell_columns - set(cells.columns):
        raise ValueError(f'Cell table is missing columns: {sorted(missing)}')
    if missing := detection_columns - set(detections.columns):
        raise ValueError(f'Detection table is missing columns: {sorted(missing)}')

    columns = [
        'case_id', 'case_track_ids', 'case_type', 'track_a', 'track_b',
        'track_a_category', 'track_b_category', 'passage_delta_ms',
        'common_frames', 'frame_gap', 'spatial_distance_px', 'area_ratio',
        'review_frame', 'review_start_frame', 'review_end_frame',
    ]
    valid_cells = cells.dropna(subset=['passage_center_s']).copy()
    if len(valid_cells) < 2:
        return pd.DataFrame(columns=columns)

    cell_rows = {int(row.track_id): row for row in valid_cells.itertuples(index=False)}
    groups = {
        int(track_id): group.sort_values('frame').copy()
        for track_id, group in detections.groupby('track_id')
        if int(track_id) in cell_rows
    }
    track_ids = sorted(groups)
    records = []
    for position, track_a in enumerate(track_ids):
        row_a = cell_rows[track_a]
        det_a = groups[track_a]
        for track_b in track_ids[position + 1:]:
            row_b = cell_rows[track_b]
            det_b = groups[track_b]
            review_track_a, review_track_b = track_a, track_b
            review_row_a, review_row_b = row_a, row_b
            passage_delta_ms = abs(row_b.passage_center_s - row_a.passage_center_s) * 1000
            common = det_a[['frame', 'x', 'y', 'area']].merge(
                det_b[['frame', 'x', 'y', 'area']], on='frame', suffixes=('_a', '_b')
            )

            case_type = None
            common_frames = len(common)
            frame_gap = 0
            spatial_distance = np.nan
            area_ratio = np.nan
            review_frame = None
            if common_frames and passage_delta_ms <= cfg.review_passage_delta_ms:
                distances = np.hypot(common.x_a - common.x_b, common.y_a - common.y_b)
                closest = int(np.argmin(distances.to_numpy()))
                spatial_distance = float(distances.iloc[closest])
                area_a = float(common.iloc[closest].area_a)
                area_b = float(common.iloc[closest].area_b)
                area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
                review_frame = int(common.iloc[closest].frame)
                if spatial_distance <= cfg.review_close_distance_px:
                    case_type = 'simultaneous_close'
            elif not common_frames:
                first_id, second_id = ((track_a, track_b)
                                       if det_a.frame.min() <= det_b.frame.min()
                                       else (track_b, track_a))
                first, second = groups[first_id], groups[second_id]
                frame_gap = int(second.frame.min() - first.frame.max())
                if 0 < frame_gap <= cfg.review_split_gap_frames:
                    end = first.iloc[-1]
                    start = second.iloc[0]
                    spatial_distance = float(np.hypot(start.x - end.x, start.y - end.y))
                    area_ratio = max(float(start.area), float(end.area)) / max(
                        min(float(start.area), float(end.area)), 1.0
                    )
                    allowed_distance = cfg.max_link_distance_px * frame_gap
                    if spatial_distance <= allowed_distance:
                        case_type = 'possible_track_split'
                        review_track_a, review_track_b = first_id, second_id
                        review_row_a = cell_rows[review_track_a]
                        review_row_b = cell_rows[review_track_b]
                        review_frame = int(first.frame.max())

            if case_type is None:
                continue
            start_frame = max(0, review_frame - cfg.review_context_frames)
            stop_limit = int(getattr(cfg, 'stop_frame', 0) or (detections.frame.max() + 1)) - 1
            end_frame = min(stop_limit, review_frame + cfg.review_context_frames)
            records.append({
                'case_type': case_type,
                'track_a': int(review_track_a),
                'track_b': int(review_track_b),
                'track_a_category': review_row_a.category,
                'track_b_category': review_row_b.category,
                'passage_delta_ms': float(passage_delta_ms),
                'common_frames': int(common_frames),
                'frame_gap': int(frame_gap),
                'spatial_distance_px': spatial_distance,
                'area_ratio': area_ratio,
                'review_frame': int(review_frame),
                'review_start_frame': int(start_frame),
                'review_end_frame': int(end_frame),
            })

    if not records:
        return pd.DataFrame(columns=columns)

    review = pd.DataFrame(records)
    parent = {track_id: track_id for track_id in set(review.track_a) | set(review.track_b)}

    def find(track_id):
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for row in review.itertuples(index=False):
        union(int(row.track_a), int(row.track_b))
    roots = sorted({find(track_id) for track_id in parent})
    case_number = {root: index for index, root in enumerate(roots)}
    review['case_id'] = [case_number[find(int(track_id))] for track_id in review.track_a]
    members = {}
    for case_id, group in review.groupby('case_id'):
        ids = sorted(set(group.track_a.astype(int)) | set(group.track_b.astype(int)))
        members[case_id] = ';'.join(map(str, ids))
    review['case_track_ids'] = review.case_id.map(members)
    return review[columns].sort_values(
        ['review_frame', 'case_id', 'track_a', 'track_b']
    ).reset_index(drop=True)


def _tracks_for_case(cases, case_id):
    selected = cases.loc[cases['case_id'].eq(case_id)]
    if selected.empty:
        raise ValueError(f'Case {case_id} does not exist.')
    return sorted(set(selected.track_a.astype(int)) | set(selected.track_b.astype(int)))


def plot_track_case(rec, detections, cases, case_id, context_frames=None, max_panels=12,
                    show_all_tracks=True):
    """Plot one suspicious case, with unrelated detections shown in grey."""
    track_ids = _tracks_for_case(cases, case_id)
    selected = detections.loc[detections.track_id.astype(int).isin(track_ids)].copy()
    if selected.empty:
        raise ValueError(f'No detections found for case {case_id}.')
    context = 2 if context_frames is None else int(context_frames)
    start = max(int(rec.start), int(selected.frame.min()) - context)
    end = min(int(rec.stop) - 1, int(selected.frame.max()) + context)
    frame_ids = np.arange(start, end + 1, dtype=int)
    if len(frame_ids) > max_panels:
        frame_ids = np.unique(np.linspace(start, end, max_panels).round().astype(int))
    columns = min(4, len(frame_ids))
    rows = int(np.ceil(len(frame_ids) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.2 * rows), squeeze=False,
                             constrained_layout=True)
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, max(len(track_ids), 2)))
    color_by_track = dict(zip(track_ids, colors))
    from matplotlib.patches import Rectangle
    for ax, frame in zip(axes.ravel(), frame_ids):
        ax.imshow(rec.frame(frame), cmap='gray')
        frame_rows = (detections.loc[detections.frame.eq(frame)] if show_all_tracks
                      else selected.loc[selected.frame.eq(frame)])
        for detection in frame_rows.itertuples(index=False):
            track_id = int(detection.track_id)
            highlighted = track_id in color_by_track
            color = color_by_track[track_id] if highlighted else '0.65'
            cluster = bool(getattr(detection, 'cluster_candidate', False))
            ax.add_patch(Rectangle(
                (detection.x0, detection.y0), detection.x1 - detection.x0,
                detection.y1 - detection.y0, fill=False, edgecolor=color,
                linewidth=2 if highlighted else 1,
                linestyle='--' if cluster else '-',
            ))
            label = f'T{track_id}' + (' CLUSTER' if cluster else '')
            ax.text(detection.x0, max(0, detection.y0 - 2), label,
                    color=color, fontsize=9 if highlighted else 7,
                    weight='bold' if highlighted else 'normal')
        ax.set_title(f'Frame {frame} | {rec.times[frame]:.6f} s')
        ax.axis('off')
    for ax in axes.ravel()[len(frame_ids):]:
        ax.axis('off')
    suffix = ' | coloured: selected case | grey: other detections' if show_all_tracks else ''
    fig.suptitle(f'Suspicious track case {case_id}: tracks {track_ids}{suffix}')
    return fig


def export_track_case_gif(rec, detections, cases, case_id, output_path,
                          context_frames=None, playback_fps=8, scale=3, max_frames=120,
                          show_all_tracks=True):
    """Write a slowed annotated GIF for one suspicious-track case."""
    from PIL import Image, ImageDraw
    track_ids = _tracks_for_case(cases, case_id)
    selected = detections.loc[detections.track_id.astype(int).isin(track_ids)].copy()
    context = 2 if context_frames is None else int(context_frames)
    start = max(int(rec.start), int(selected.frame.min()) - context)
    end = min(int(rec.stop) - 1, int(selected.frame.max()) + context)
    frame_ids = np.arange(start, end + 1, dtype=int)
    if len(frame_ids) > max_frames:
        frame_ids = np.unique(np.linspace(start, end, max_frames).round().astype(int))
    palette = plt.get_cmap('tab10')(np.linspace(0, 1, max(len(track_ids), 2)))[:, :3]
    colors = {track_id: tuple((255 * color).astype(np.uint8))
              for track_id, color in zip(track_ids, palette)}
    images = []
    for frame in frame_ids:
        gray = (255 * normalize(rec.frame(frame))).astype(np.uint8)
        image = Image.fromarray(gray, mode='L').convert('RGB')
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        frame_rows = (detections.loc[detections.frame.eq(frame)] if show_all_tracks
                      else selected.loc[selected.frame.eq(frame)])
        for detection in frame_rows.itertuples(index=False):
            track_id = int(detection.track_id)
            box = tuple(int(value * scale) for value in
                        (detection.x0, detection.y0, detection.x1, detection.y1))
            highlighted = track_id in colors
            color = colors[track_id] if highlighted else (150, 150, 150)
            draw.rectangle(box, outline=color, width=max(2, scale) if highlighted else 1)
            cluster = bool(getattr(detection, 'cluster_candidate', False))
            label = f'T{track_id}' + (' CLUSTER' if cluster else '')
            draw.text((box[0], max(0, box[1] - 10)), label, fill=color)
        draw.text((5, 5), f'Frame {frame}', fill=(255, 255, 0))
        images.append(image)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:], loop=0,
                   duration=max(1, round(1000 / playback_fps)))
    return output_path


def export_dataset(rec, background, model, cfg):
    # Changing timing/input settings after prepare must never silently reuse an old clock.
    for key in ('cine_path', 'apd_path', 'sync_mode', 'active_low', 'first_frame_time_s',
                'sync_correction_s', 'start_frame', 'stop_frame', 'background_samples'):
        if getattr(cfg, key) != getattr(rec.cfg, key):
            raise ValueError(f'{key} changed after prepare(); run prepare() again.')
    validate(cfg)
    root = Path(cfg.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = root / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir()
    manifest = {'status': 'running', 'config': asdict(cfg), 'cine_fps': rec.fps,
                'cine_frames': rec.count, 'ni_sample_rate_hz': 1/rec.dt,
                'sync_method': cfg.sync_mode, 'clock_formula': 'NI trigger + CINE time relative to trigger + correction' if cfg.sync_mode == 'trigger' else cfg.sync_mode,
                'label_meaning': 'APD association, not independently validated biological ground truth',
                'assumption': 'Configured ROI is a provisional detection region; peak intervals estimated at a fraction of local prominence',
                'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'cellpose', 'pims')}}
    manifest_path = run / 'manifest.json'
    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    write_manifest()
    (run / 'config.json').write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding='utf-8'
    )
    try:
        peaks, _, _ = detect_peaks(rec, cfg)
        fig, _ = plot_peaks(rec, cfg); fig.savefig(run/'apd_peak_control.png', dpi=160); plt.close(fig)
        tifffile.imwrite(run/'background_normalized.tif', background)
        tracker = Tracker(cfg, rec.shape)
        screened = []
        with tempfile.TemporaryDirectory(prefix='representatives_', dir=run) as cache_dir:
            cache = Path(cache_dir)
            for frame in range(rec.start, rec.stop):
                raw = rec.frame(frame)
                masks, _, candidate = segment(raw, background, model, cfg)
                rows = objects(masks, frame, rec.times[frame], cfg)
                screened.append({'frame': frame, 'time_s': rec.times[frame],
                                 'candidate_pixels': int(candidate.sum()), 'n_cells': len(rows)})
                for tid, row in tracker.update(rows, frame):
                    tr = tracker.tracks[tid]
                    # Prefer a complete, isolated cell, then maximum distance from image borders.
                    score = (not row.get('cluster_candidate', False),
                             row['edge'] > cfg.edge_margin_px, len(rows) == 1,
                             row['edge'], row['area'])
                    if tr['best_score'] is None or score > tr['best_score']:
                        crop, mask, x, y = square_crop(raw, masks, row, cfg)
                        np.savez(cache/f'{tid}.npz', crop=crop, mask=mask)
                        tr['best_score'], tr['best'] = score, dict(row, crop_x0=x, crop_y0=y)
                if (frame-rec.start) % 500 == 0:
                    print(f'Frame {frame}/{rec.stop-1}; Tracks {len(tracker.tracks)}', flush=True)
            table, audit = classify(tracker.tracks, peaks, rec, cfg)
            for category in ('zelltyp_1_peak', 'zelltyp_2_kein_peak', 'uneindeutig'):
                for kind in ('crops', 'masks', 'masked_crops'):
                    (run/category/kind).mkdir(parents=True)
            for idx, row in table.iterrows():
                tid = int(row.track_id)
                stem = f'cell_{tid:06d}_frame_{int(row.representative_frame):08d}'
                with np.load(cache/f'{tid}.npz', allow_pickle=False) as data:
                    crop, mask = data['crop'], data['mask']
                for kind, arr in [('crops', crop), ('masks', mask), ('masked_crops', np.where(mask, crop, 0))]:
                    relative = Path(row.category)/kind/(stem+'.tif')
                    tifffile.imwrite(run/relative, arr, photometric='minisblack')
                    table.loc[idx, kind+'_path'] = relative.as_posix()
                best = tracker.tracks[tid]['best']
                table.loc[idx, 'crop_x0'] = best['crop_x0']
                table.loc[idx, 'crop_y0'] = best['crop_y0']
        points = [r for tr in tracker.tracks.values() for r in tr['rows']]
        detection_table = pd.DataFrame(
            points,
            columns=['track_id','frame','time_s','label','area','x','y',
                     'x0','y0','x1','y1','edge','mask_elongation','nearby_mask',
                     'cluster_candidate','cluster_flags','ambiguous_link'],
        )
        detection_table.to_csv(run/'detections.csv', index=False)
        review = find_suspicious_track_cases(table, detection_table, cfg)
        review.to_csv(run/'track_proximity_review.csv', index=False)
        pd.DataFrame(screened).to_csv(run/'frame_screening.csv', index=False)
        table.to_csv(run/'cells.csv', index=False)
        time_columns = ['track_id', 'category', 'reason', 'blocking_reason', 'quality_notes',
                        'passage_start_frame', 'passage_end_frame',
                        'passage_start_s', 'passage_end_s', 'passage_center_s', 'match_start_s',
                        'match_end_s', 'representative_frame', 'representative_time_s', 'peak_ids',
                        'candidate_peak_ids', 'assignment_distance_ms',
                        'peak_event_time_s', 'peak_apd_time_s', 'peak_event_onset_s',
                        'peak_event_end_s', 'peak_frame', 'peak_frame_time_s', 'peak_minus_frame_ms',
                        'alignment_status', 'alignment_offset_ms', 'alignment_drift_ppm']
        table[time_columns].to_csv(run/'cell_timepoints.csv', index=False)
        audit.to_csv(run/'peaks.csv', index=False)
        fig = plot_assignments(table, audit); fig.savefig(run/'assignment_control.png', dpi=160); plt.close(fig)
        alignment = ({key: (audit.iloc[0][key].item()
                            if isinstance(audit.iloc[0][key], np.generic)
                            else audit.iloc[0][key]) for key in (
            'alignment_status', 'alignment_pairs', 'alignment_reference_s',
            'alignment_offset_ms', 'alignment_drift_ppm', 'alignment_rmse_ms')}
            if len(audit) else {})
        manifest.update(status='complete', counts=table.category.value_counts().to_dict(),
                        time_alignment=alignment,
                        suspicious_track_cases=int(review.case_id.nunique()) if len(review) else 0)
        write_manifest()
        print(f'Saved: {run}\n{table.category.value_counts().to_string()}')
        return run, table
    except Exception as exc:
        manifest.update(status='failed', error=str(exc)); write_manifest()
        raise


def select_file(title, filetypes):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    try:
        return filedialog.askopenfilename(parent=root, title=title, filetypes=filetypes)
    finally:
        root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON file containing Config fields')
    parser.add_argument('--cine'); parser.add_argument('--apd'); parser.add_argument('--model')
    parser.add_argument('--output'); parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    values = json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else {}
    for arg, key in [('cine', 'cine_path'), ('apd', 'apd_path'), ('model', 'pretrained_model'), ('output', 'output_root')]:
        if getattr(args, arg): values[key] = getattr(args, arg)
    cfg = Config(**values)
    if not cfg.cine_path: cfg.cine_path = select_file('Select CINE file', [('CINE', '*.cine')])
    if not cfg.apd_path: cfg.apd_path = select_file('Select APD file', [('NumPy', '*.npz')])
    rec, background = prepare(cfg)
    try:
        plot_peaks(rec, cfg)
        if args.export:
            model = load_model(cfg)
            export_dataset(rec, background, model, cfg)
        plt.show()
    finally:
        rec.close()


if __name__ == '__main__':
    main()
