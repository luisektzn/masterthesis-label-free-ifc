"""Classification figures of the thesis, drawn from the result files of notebooks 02 to 07.

Nothing is trained here. Every function reads the saved metrics or predictions and returns a matplotlib figure in
the style of ``thesis_style``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import thesis_style as S

CLS = ['HeLa', 'NIH3T3']
MODELS = ['CNN', 'ChipNet', 'Control']
MODEL_SHORT = {'CNN': 'CNN (64 × 64)', 'ChipNet': 'Chip network (8 × 8)', 'Control': 'Feature control'}


def small() -> float:
    return mpl.rcParams['xtick.labelsize']


def ms(size: float) -> float:
    """Marker size, slightly larger than the matplotlib default."""
    return size * 1.1


def read(results_dir: Path, name: str, kind: str = 'metrics') -> pd.DataFrame:
    """<name>_metrics.csv or <name>_predictions.csv.gz of one notebook (empty table if the file is missing)."""
    path = Path(results_dir) / (f'{name}_{kind}.csv.gz' if kind == 'predictions' else f'{name}_{kind}.csv')
    if not path.is_file():
        print('missing:', path)
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def select(df: pd.DataFrame, scenario: str | None = None, model: str | None = None) -> pd.DataFrame:
    if len(df) and scenario is not None:
        df = df[df.scenario.eq(scenario)]
    if len(df) and model is not None:
        df = df[df.model.eq(model)]
    return df


# ---------------------------------------------------------------- confusion matrix
def confusion_counts(pred: pd.DataFrame) -> np.ndarray:
    """Confusion matrix of all folds, averaged over the seeds (counts per seed). The percentages then equal the
    recalls calculated from the predictions of all seeds."""
    y, p = pred.y.to_numpy(), pred.pred.to_numpy()
    cm = np.array([[np.sum((y == i) & (p == j)) for j in (0, 1)] for i in (0, 1)], float)
    return cm / max(int(pred.seed.nunique()), 1)


def confusion_figure(pred: pd.DataFrame, title: str):
    """Confusion matrix normalized per true class (recall), annotated with percentage and mean count."""
    fig, ax = plt.subplots(figsize=S.figsize('half', 0.85), layout='constrained')
    cm = confusion_counts(pred)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap=S.CMAP_SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{norm[i, j]:.0%}\n(n = {int(round(cm[i, j]))})', ha='center', va='center',
                    color='white' if norm[i, j] < 0.55 else S.INK, fontsize=small())
    ax.set_xticks(range(2), CLS)
    ax.set_yticks(range(2), CLS)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=small())
    return fig


# ---------------------------------------------------------------- summary of several evaluations
def dots(ax, df, x_order, colors, markers, labels, width=0.62, seed=0):
    """Single runs (folds and seeds) as small points, mean ± standard deviation as a large marker."""
    rng = np.random.default_rng(seed)
    models = [m for m in MODELS if (df.model == m).any()]
    n = max(len(models), 1)
    handles = {}
    for i, key in enumerate(x_order):
        for k, mdl in enumerate(models):
            v = df.loc[(df.key == key) & (df.model == mdl), 'macro_f1'].dropna().to_numpy(float)
            if not len(v):
                continue
            pos = i + (k - (n - 1) / 2) * width / n
            c, mk = colors[mdl], markers[mdl]
            ax.scatter(pos + rng.uniform(-0.035, 0.035, len(v)), v, s=ms(3) ** 2, color=c, alpha=0.4,
                       linewidths=0, zorder=2)
            ax.errorbar(pos, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else None, fmt=mk, ms=ms(6.5), color=c,
                        mfc=c, mec=S.INK_2, mew=0.8, capsize=ms(2.5), elinewidth=1.0, zorder=3)
            handles[mdl] = Line2D([], [], ls='none', color=c, marker=mk, ms=ms(6), mfc=c, mec=S.INK_2,
                                  label=labels[mdl])
    ax.set_xticks(range(len(x_order)))
    ax.set_xlim(-0.6, len(x_order) - 0.4)
    S.ygrid(ax)
    return [handles[m] for m in models if m in handles]


def comparison_figure(parts: dict, labels: dict, separators=()):
    """Macro-F1 of the CNN, the chip network and the feature control for several evaluations.

    ``parts``: {key: metrics table}, ``labels``: {key: x label}, ``separators``: positions after which a vertical
    line is drawn.
    """
    m = pd.concat([v.assign(key=k) for k, v in parts.items() if len(v)], ignore_index=True)
    order = [k for k in parts if (m.key == k).any()]
    fig, ax = plt.subplots(figsize=S.figsize('full', 0.5 if len(order) <= 3 else 0.44), layout='constrained')
    handles = dots(ax, m, order, S.MODEL_COLORS, S.MODEL_MARKERS, MODEL_SHORT)
    ax.set_xticklabels([labels[k] for k in order], fontsize=small() - 1)
    for g in separators:
        if 0 < g < len(order):
            ax.axvline(g - 0.5, color=S.GRID, lw=0.8)
    vals = m.macro_f1.dropna()
    ax.set_ylim(min(0.4, vals.min() - 0.05, 0.2), 1.02)
    ax.set_ylabel('Macro-F1')
    ax.axhline(0.5, color=S.NEUTRAL, lw=0.8, ls=':', zorder=1)
    ax.annotate(' chance', (1, 0.5), xycoords=('axes fraction', 'data'), va='center', ha='left', fontsize=small(),
                color=S.INK_2, annotation_clip=False)
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncols=3, frameon=False,
              handlelength=1.8, columnspacing=1.2, borderaxespad=0.2)
    return fig


# ---------------------------------------------------------------- input size
def input_size_figure(metrics: pd.DataFrame, px_um: float):
    """Macro-F1 of the CNN and the fully connected network against the input size (sans-serif, as in the thesis).
    The tick labels give the input size and the effective width of one input pixel, limited to the camera pixel
    size: larger inputs only interpolate the image (grey labels)."""
    side_um = float(metrics.side_um.dropna().iloc[0])
    rc = {'font.family': 'sans-serif', 'font.size': 18, 'axes.labelsize': 20, 'xtick.labelsize': 16,
          'ytick.labelsize': 17, 'legend.fontsize': 17}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(11, 6.0), layout='constrained')
        sizes = sorted(metrics.input_size[metrics.input_size > 0].unique())
        pos = {n: i for i, n in enumerate(sizes)}                     # equal spacing on the x axis
        handles = []
        for net, kind, mk, lab in [('CNN', 'CNN', 'D', 'CNN'), ('Fully connected', 'ChipNet', 'o', 'Chip network')]:
            g = metrics[metrics.network.eq(net)].groupby('input_size').macro_f1.agg(['mean', 'std'])
            col = S.MODEL_COLORS[kind]
            ax.errorbar([pos[n] for n in g.index], g['mean'], yerr=g['std'], color=col, marker=mk, ms=11, lw=2.5,
                        capsize=4)
            handles.append(Line2D([], [], color=col, marker=mk, ms=11, ls='none', label=lab))
        ax.axhline(0.5, color='0.45', ls=':', lw=1.5)
        ax.text(len(sizes) - 0.45, 0.5, 'chance', va='center', ha='left', color='0.45', fontsize=15)
        n_crop = side_um / px_um                                       # camera pixels per crop side
        labels = [f'{int(n)}×{int(n)}\n' + (f'{side_um / n:.1f} µm' if n <= n_crop else f'{px_um:.2f} µm')
                  for n in sizes]
        ax.set_xticks(range(len(sizes)), labels)
        for lab, n in zip(ax.get_xticklabels(), sizes):
            if n > n_crop:
                lab.set_color('0.6')                                  # interpolated, camera resolution
        ax.set_xlim(-0.5, len(sizes) - 0.5)
        ax.set_ylim(0.45, 0.9)
        ax.set_xlabel('Input size and resolution (µm per input pixel)')
        ax.set_ylabel('Macro-F1')
        S.ygrid(ax)
        ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncols=2, frameon=False,
                  handletextpad=0.3, columnspacing=1.5)
    return fig
