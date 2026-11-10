"""Figure style of the thesis (fonts, sizes, colours), shared by all figures of this folder.

Colour rules
------------
* All colours are sampled from perceptually uniform colour maps, so the categories remain distinguishable in
  greyscale print and for readers with red-green colour vision deficiency.
* Cell types are sampled from ``viridis`` (HeLa dark blue, NIH3T3 light green). Ordered variants (flow rates) are
  sampled from ``magma``. Magnitudes (velocity maps) use ``viridis``.
* Text, axes and annotations are drawn in dark grey, never in a data colour.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager

# ---------------------------------------------------------------- neutral colours
INK = '#0b0b0b'          # main text
INK_2 = '#52514e'        # axes, secondary text
GRID = '#e6e5e1'         # grid lines
NEUTRAL = '#a09f9a'      # reference lines
SURFACE = '#ffffff'


def _sample(cmap: str, value: float) -> str:
    """Hex colour of a matplotlib colour map at position ``value`` (0 to 1)."""
    return mpl.colors.to_hex(mpl.colormaps[cmap](value))


# ---------------------------------------------------------------- categorical colours
CELL_COLORS = {'HeLa': _sample('viridis', 0.25), 'NIH3T3': _sample('viridis', 0.75)}
ANALYTICAL_COLOR = _sample('viridis', 0.50)        # analytical model of fully developed flow
DEVELOPED_COLOR = _sample('plasma', 0.70)          # dashed reference line in the COMSOL axis figures


def levels(n: int, cmap: str = 'magma', lo: float = 0.2, hi: float = 0.8) -> list[str]:
    """n ordered colours from a perceptually uniform map (flow rates)."""
    return [_sample(cmap, v) for v in np.linspace(lo, hi, n)] if n > 1 else [_sample(cmap, (lo + hi) / 2)]


CMAP_SEQ = mpl.colormaps['viridis']               # magnitudes

# ---------------------------------------------------------------- figure sizes
CM = 1 / 2.54
TEXT_WIDTH_IN = 15.5 * CM            # text width of the thesis
HALF_WIDTH_IN = 7.5 * CM


def figsize(width: str | float = 'full', aspect: float = 0.62) -> tuple[float, float]:
    """Figure size in inches. ``width`` is 'full', 'half' or a width in cm."""
    if width == 'full':
        w = TEXT_WIDTH_IN
    elif width == 'half':
        w = HALF_WIDTH_IN
    else:
        w = float(width) * CM
    return (w, w * aspect)


# ---------------------------------------------------------------- font
def _register_latin_modern() -> str | None:
    """Find Latin Modern Roman (the font of the thesis, package lmodern) in common TeX installations and register
    it with matplotlib."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    if 'Latin Modern Roman' in names:
        return 'Latin Modern Roman'
    patterns = [
        '/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf',
        '/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf',
        '/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf',
        '/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf',
        os.path.expanduser('~/Library/Fonts/lmroman10-*.otf'),
        'C:/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf',
        os.path.expandvars('%LOCALAPPDATA%/Programs/MiKTeX/fonts/opentype/public/lm/lmroman10-*.otf'),
        str(Path(__file__).with_name('fonts') / 'lmroman10-*.otf'),
    ]
    found = False
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                font_manager.fontManager.addfont(path)
                found = True
            except Exception:
                pass
    return 'Latin Modern Roman' if found else None


def apply(font_size: float = 9) -> None:
    """Set the thesis style for all following figures."""
    import logging
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    lm = _register_latin_modern()
    serif = ([lm] if lm else []) + ['CMU Serif', 'DejaVu Serif', 'Times New Roman']
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': serif, 'mathtext.fontset': 'cm',
        'font.size': font_size, 'axes.labelsize': font_size, 'axes.titlesize': font_size,
        'legend.fontsize': font_size - 1, 'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1, 'figure.titlesize': font_size + 1,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
        'axes.edgecolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.axisbelow': True, 'axes.prop_cycle': mpl.cycler(color=list(CELL_COLORS.values())),
        'lines.linewidth': 1.4, 'lines.markersize': 4.5, 'lines.markeredgewidth': 0.6,
        'errorbar.capsize': 2.5,
        'patch.linewidth': 0.6, 'legend.frameon': False, 'legend.handlelength': 1.4,
        'image.cmap': 'gray', 'image.interpolation': 'nearest',
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'savefig.facecolor': SURFACE,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True,
        'figure.constrained_layout.use': False,
    })


def ygrid(ax) -> None:
    """Light horizontal grid lines behind the data."""
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
