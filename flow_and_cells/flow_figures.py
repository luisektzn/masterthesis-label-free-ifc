"""Figures of the flow and cell-dynamics part of the thesis, from saved tables only.

Nothing is segmented or tracked here. The figures read
    flow_theory.py and comsol_analysis.py           (analytical model, COMSOL exports in comsol/)
    results/measured/measured_summary.csv           (measured_summary.py)
    cells_primary.csv, detection_morphology.csv     (tables of the cell-tracking run, CFG.v4_tables)
    lateral_position_tracks.csv                     (lateral position per track, CFG.lateral_csv)
    static deformation reference                    (CFG.static_csv)

Every figure is a function fig(talk) -> Figure, registered with its thesis file name, caption and inputs.
Usage (see 04_figures.ipynb):
    import flow_figures as F
    F.CFG.v4_tables = Path('.../tables')
    F.make('res_flow_cross_section')     # one figure: thesis PDF/PNG and talk PNG
    F.make_all()                         # all figures
    F.overview()                         # which figures exist, which inputs are missing

Output (CFG.out, default results/figures):
    thesis/<name>.pdf, .png (300 dpi), <name>_caption.txt    Latin Modern, half (7.5 cm) or full (15.5 cm) text width
    talk/<name>.png                                          sans serif, large fonts
    tables/*.csv                                             numbers behind some figures
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import comsol_analysis as ca
import flow_theory as ft
import thesis_style as ts

HERE = Path(__file__).resolve().parent
CLS = ['HeLa', 'NIH3T3']
MODES = ['BF', 'Ph2']
MODE_LABELS = {'BF': 'bright field', 'Ph2': 'Ph2'}
FLOWS = [30, 150, 300]


def _env_path(name):
    v = os.environ.get(name)
    return Path(v) if v else None


# =====================================================================================
# 1  Configuration
# =====================================================================================
@dataclass
class Config:
    # ---- inputs
    v4_tables: Path | None = _env_path('CELL_TABLES')         # tables of the tracking run (cells_primary.csv, ...)
    lateral_csv: Path | None = _env_path('LATERAL_CSV')       # lateral_position_tracks.csv
    static_csv: Path | None = _env_path('STATIC_CSV')         # static deformation reference
    static_label: str = 'static Phantom'                      # name of the static reference in the caption
    # ---- output
    out: Path = Path(os.environ.get('FLOW_RESULTS', HERE / 'results')) / 'figures'
    # ---- thesis style
    thesis_font: float = 9
    # ---- talk style
    talk: bool = True
    talk_font: float = 16
    talk_fonts: tuple = ('Calibri', 'Carlito', 'DejaVu Sans')
    talk_size: tuple = (8.5, 5.4)
    talk_size_wide: tuple = (11.5, 5.4)
    talk_dpi: int = 200
    marker_scale: float = 1.0
    # ---- display in the notebook
    show: bool = True
    show_talk: bool = True
    # ---- content
    excluded_flow_rates: tuple = (600.0,)           # chip failure
    min_tracks: int = 10                            # groups with fewer tracks are drawn as single points
    flow_direction_deg: float = 180.0               # flow from right to left in the images
    rotation_min_det: int = 5
    rotation_d_min: float = 0.10
    rotation_example_flow: float = 30.0
    rotation_n_examples: int = 20
    size_bin_um: float = 1.0
    size_min_per_bin: int = 50
    # Motion blur of the Andor Zyla: flow rates and exposure time as in the thesis figure res_motion_blur_zyla.
    # zyla_measured adds measured cells as points, (t_exp in µs, blur in µm), e.g. ((700.8, 4.75),).
    zyla_px_um: float = 0.325
    zyla_flows: tuple = (0.3, 1, 10, 30)
    zyla_t_exp_us: tuple = (700.8,)
    zyla_measured: tuple = ()


CFG = Config()
TALK = False                                        # set by style()

# Channel, focusing and camera
WALL_HALF_UM = 39.6 / 2          # half of the measured channel width (Results)
STREAM_HALF_UM = 4.7             # half of the expected sample stream width (Results)
PX_UM = 0.925                    # Phantom TMX pixel size
ONE_PIXEL_D = (0.025, 0.033)     # D_ellipse for one pixel between the axes (Results)
A_CH = ft.W_CH * ft.H_CH
MEASURED_COLOR = ts.INK
COMSOL_STYLE = dict(color=ts.INK_2, marker='D', mfc='white', ls='none')


# =====================================================================================
# 2  Style, sizes, legends, saving
# =====================================================================================
def style(talk: bool = False) -> None:
    """Thesis: Latin Modern at CFG.thesis_font. Talk: the same colours, sans serif and larger."""
    global TALK
    TALK = bool(talk)
    if TALK:
        ts.apply(font_size=CFG.talk_font)
        mpl.rcParams.update({
            'font.family': 'sans-serif', 'font.sans-serif': list(CFG.talk_fonts), 'mathtext.fontset': 'dejavusans',
            'lines.linewidth': 2.4, 'axes.linewidth': 1.0,
            'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'xtick.major.size': 5, 'ytick.major.size': 5,
            'legend.fontsize': CFG.talk_font - 1, 'xtick.labelsize': CFG.talk_font - 1,
            'ytick.labelsize': CFG.talk_font - 1, 'grid.linewidth': 0.9,
        })
    else:
        ts.apply(font_size=CFG.thesis_font)


def single(width='half', aspect=0.8, wide=False):
    """One diagram. Thesis: 'half' (7.5 cm) or 'full' (15.5 cm) text width. Talk: CFG.talk_size(_wide)."""
    size = (CFG.talk_size_wide if wide else CFG.talk_size) if TALK else ts.figsize(width, aspect)
    return plt.subplots(figsize=size, layout='constrained')


def small() -> float:
    return mpl.rcParams['xtick.labelsize']


def lw(thesis: float = 1.4) -> float:
    return thesis * 1.7 if TALK else thesis


def ms(thesis: float = 5.0) -> float:
    """Marker size: a bit larger than the defaults, and doubled on the slides."""
    return thesis * CFG.marker_scale * (2.0 if TALK else 1.1)


def legend(ax, handles=None, loc='top', ncols=None, **kw):
    """Legend of one diagram. loc='top': above the axes. Otherwise a matplotlib location."""
    if handles is None:
        handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return None
    if loc == 'top':
        return ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.01),
                         ncols=ncols or min(len(handles), 2), frameon=False,
                         handlelength=1.8, columnspacing=1.2, borderaxespad=0.2, **kw)
    return ax.legend(handles=handles, loc=loc, ncols=ncols or 1, frameon=False, handlelength=1.8, **kw)


def end_label(ax, x, y, text, color=None, dx=4, **kw) -> None:
    """Direct label right of the last point of a line."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords='offset points', va='center', ha='left',
                color=color or ts.INK_2, fontsize=small(), annotation_clip=False, **kw)


def handle(kind='line', color=None, label='', **kw):
    color = color or ts.INK_2
    if kind == 'patch':
        return Patch(facecolor=color, edgecolor='none', label=label, **kw)
    if kind == 'marker':
        kw.setdefault('marker', 'o')
        kw.setdefault('ms', ms(6))
        return Line2D([], [], ls='none', color=color, label=label, **kw)
    kw.setdefault('lw', lw())
    return Line2D([], [], color=color, label=label, **kw)


def out_dir(kind: str) -> Path:
    d = Path(CFG.out) / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name: str, caption: str, talk: bool) -> Path:
    if talk:
        path = out_dir('talk') / f'{name}.png'
        fig.savefig(path, dpi=CFG.talk_dpi, bbox_inches='tight', pad_inches=0.05)
    else:
        d = out_dir('thesis')
        fig.savefig(d / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
        path = d / f'{name}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02)
        if caption:
            (d / f'{name}_caption.txt').write_text(caption, encoding='utf-8')
    plt.close(fig)
    return path


def _display(path: Path, width=None) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
        if get_ipython() is not None:
            display(Image(filename=str(path), width=width))
    except ImportError:
        pass


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = out_dir('tables') / f'{name}.csv'
    df.to_csv(path, index=False, float_format='%.4g')
    return path


# =====================================================================================
# 3  Registry: every figure is a function fig(talk) -> Figure, with inputs and caption
# =====================================================================================
FIGURES: dict[str, dict] = {}
SECTIONS = {'chip': '1 Chip and flow', 'position': '2 Where the cells go',
            'deformation': '3 What the flow does to the cells', 'appendix': '4 Appendix'}


def figure(name, section, caption, needs=lambda: [], width='half'):
    """Register a figure. caption: thesis caption (string or function). needs(): input files.
    width: thesis width, 'half' or 'full'."""
    def deco(fn):
        FIGURES[name] = dict(fn=fn, section=section, caption=caption, needs=needs, width=width)
        return fn
    return deco


def missing_inputs(name) -> list[str]:
    out = []
    for p in FIGURES[name]['needs']():
        if p is None or not Path(p).exists():
            out.append(str(p) if p is not None else 'path not set (None)')
    return out


def make(name: str, thesis: bool = True, talk: bool | None = None) -> dict:
    """Write the thesis and the talk version of one figure. Skips the figure if an input is missing."""
    spec = FIGURES[name]
    miss = missing_inputs(name)
    if miss:
        print(f'{name}: skipped, missing input:\n    ' + '\n    '.join(miss))
        return {}
    files = {}
    do_talk = CFG.talk if talk is None else talk
    for t in [False] * thesis + [True] * do_talk:
        style(t)
        try:
            fig = spec['fn'](t)
        except FileNotFoundError as e:
            print(f'{name}: skipped, {e}')
            plt.close('all')
            return files
        if fig is None:
            print(f'{name}: not drawn (no data)')
            break
        cap = spec['caption']() if callable(spec['caption']) else spec['caption']
        path = _save(fig, name, cap, t)
        files['talk' if t else 'thesis'] = path
        if CFG.show and (not t or CFG.show_talk):
            print(('Talk: ' if t else 'Thesis: ') + str(path))
            _display(path, width=None if t else (380 if spec['width'] == 'half' else 760))
    style(False)
    return files


def make_section(section: str) -> None:
    for name, spec in FIGURES.items():
        if spec['section'] == section:
            make(name)


def make_all() -> None:
    for s in SECTIONS:
        make_section(s)


def overview() -> pd.DataFrame:
    rows = []
    for name, spec in FIGURES.items():
        miss = missing_inputs(name)
        rows.append(dict(section=SECTIONS[spec['section']], figure=name, thesis_width=spec['width'],
                         ready=not miss, missing='; '.join(Path(m).name for m in miss),
                         thesis_file=(Path(CFG.out) / 'thesis' / f'{name}.pdf').exists(),
                         talk_file=(Path(CFG.out) / 'talk' / f'{name}.png').exists()))
    return pd.DataFrame(rows)


# =====================================================================================
# 4  Loading (cached, nothing is recomputed from raw data)
# =====================================================================================
@lru_cache(maxsize=32)
def _read(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def read(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'missing input {p}')
    return _read(str(p), p.stat().st_mtime).copy()


def v4(name: str) -> Path | None:
    """Path of <name>.csv or <name>.csv.gz in the table folder of the tracking run."""
    if CFG.v4_tables is None:
        return None
    for suffix in ('.csv', '.csv.gz'):
        p = Path(CFG.v4_tables) / (name + suffix)
        if p.exists():
            return p
    return Path(CFG.v4_tables) / (name + '.csv')


def _no_failed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~df.flow_rate_ul_min.isin(CFG.excluded_flow_rates)].copy()


def cells() -> pd.DataFrame:
    """Retained tracks (track medians)."""
    return _no_failed(read(v4('cells_primary')))


def detections() -> pd.DataFrame:
    """Single detections of the retained tracks."""
    return _no_failed(read(v4('detection_morphology')))


def lateral() -> pd.DataFrame:
    return _no_failed(read(CFG.lateral_csv))


def static_reference() -> pd.DataFrame:
    """Median and IQR of D_ellipse of the static cells per cell line (permissive filter, all days)."""
    s = read(CFG.static_csv)
    if 'set' in s:
        a = s.loc[s['set'].str.contains('permissive') & s['date'].astype(str).eq('all')]
        return a.set_index('cell_type')[['D_median', 'D_q25', 'D_q75']]
    return s.set_index(s.columns[0])[['D_median', 'D_q25', 'D_q75']]


def flows_of(df) -> list:
    return sorted(df.flow_rate_ul_min.unique())


def flow_colors() -> dict:
    return dict(zip(FLOWS, ts.levels(len(FLOWS), 'magma', 0.2, 0.7)))


@lru_cache(maxsize=1)
def comsol_axis() -> pd.DataFrame:
    """COMSOL cut line on the channel axis at mid-height, 300 µL/min (comsol_analysis.read_axis)."""
    return ca.read_axis(ca.AXIS_FILE)


def comsol_numbers() -> dict:
    """Median axis velocity between s = 1.0 and 2.3 mm, pressure drop from the entrance to the outlet, length."""
    s = ca.axis_summary(comsol_axis())
    return dict(u_axis_m_s=s['U_dev_median_m_s'], dp_entrance_outlet_bar=s['dp_entrance_outlet_bar'],
                length_mm=s['length_entrance_outlet_mm'])


@lru_cache(maxsize=1)
def comsol() -> dict:
    """Measured diameters and velocities, COMSOL particle tracks and the edge stress along the simulated paths."""
    flow = ca.load_measured()
    data = ca.load_all_paths()
    used = ca.used_diameters(data)
    paths = ca.table_paths(data, used)
    return dict(flow=flow, data=data, used=used, paths=paths, summary=ca.path_stress_summary(paths))


AXIS_FILE = lambda: [ca.AXIS_FILE]
COMSOL_NEEDS = lambda: [ca.PATH_FILES[15], ca.MEASURED_SUMMARY]
CELLS_NEEDS = lambda: [v4('cells_primary')]


# =====================================================================================
# 5  Chip and flow (Results, Sections sec:res-stream and sec:res-flow, sec:res-loads)
# =====================================================================================
@figure('res_flow_cross_section', 'chip',
        'Velocity of fully developed flow over the cross-section of the 40 × 40 µm focusing channel, normalized by '
        'the mean velocity U, Eq. (poiseuille_rectangular_velocity). Dashed lines: width of the sample stream after '
        'Lee et al. (9.4 µm at a sheath-to-sample ratio of 2), which extends over the full depth in 2D focusing.')
def fig_flow_cross_section(talk):
    fig, ax = single('half', 0.9)
    n = 161
    y = np.linspace(-ft.W_CH / 2, ft.W_CH / 2, n)
    z = np.linspace(0, ft.H_CH, n)
    Y, Z = np.meshgrid(y, z)
    U = ft.flow_per_pressure_gradient(ft.W_CH, ft.H_CH) / (ft.W_CH * ft.H_CH)
    u = ft.velocity_rect(Y, Z, ft.W_CH, ft.H_CH, 1.0) / U
    im = ax.imshow(u, extent=[y[0] / ft.UM, y[-1] / ft.UM, -20, 20], origin='lower', cmap=ts.CMAP_SEQ,
                   vmin=0, vmax=2.1, interpolation='bilinear')
    # contour labels: white on the dark outer levels, dark on the bright inner levels
    for levels, col in [([0.5, 1.0], 'white'), ([1.5, 2.0], ts.INK)]:
        cs = ax.contour(Y / ft.UM, Z / ft.UM - 20, u, levels=levels, colors=col, linewidths=lw(0.6))
        ax.clabel(cs, fmt='%.1f', fontsize=small() - 1)
    wf = ft.lee_width(1.0, 1.0)[0] / ft.UM
    for s in (-1, 1):
        ax.plot([s * wf / 2] * 2, [-20, 20], color=ts.INK, lw=lw(1.2), ls='--')
    ax.set_xlabel('Lateral position $y$ (µm)')
    ax.set_ylabel('Depth $z$ from mid-plane (µm)')
    ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('$u_x/U$')
    cb.outline.set_visible(False)
    legend(ax, [handle(color=ts.INK, ls='--', lw=lw(1.2), label=f'sample stream ({wf:.1f} µm wide, full depth)')],
           ncols=1)
    return fig


def _velocity_lines(ax):
    """U and U_max against the total flow rate, with the COMSOL axis velocity at 300 µL/min."""
    q = np.linspace(0, ft.Q_FAILED, 200)
    U = q * ft.UL_MIN / A_CH
    k = ft.u_max_over_u()
    col = ts.ANALYTICAL_COLOR
    ax.plot(q, k * U, color=col, lw=lw(1.6), ls='--')
    ax.plot(q, U, color=col, lw=lw(1.6))
    end_label(ax, q[-1], k * U[-1], '$U_\\mathrm{max}$')
    end_label(ax, q[-1], U[-1], '$U$')
    ax.plot(300, comsol_numbers()['u_axis_m_s'], ms=ms(6.5), mew=lw(1.2), zorder=6, **COMSOL_STYLE)
    ax.axvline(ft.Q_FAILED, color=ts.NEUTRAL, lw=lw(0.8), ls=':')
    ax.text(ft.Q_FAILED, 0.02, 'bonding failed ', transform=ax.get_xaxis_transform(), ha='right', va='bottom',
            color=ts.INK_2, fontsize=small())
    ax.set_xlim(0, ft.Q_FAILED * 1.05)
    ax.set_ylim(0)
    ax.set_xticks(FLOWS + [ft.Q_FAILED])
    ax.set_xlabel('Total flow rate $Q$ (µL/min)')
    ax.set_ylabel('Velocity (m/s)')
    ts.ygrid(ax)


@figure('res_flow_velocity_model', 'chip',
        lambda: 'Mean velocity U = Q/A and centreline velocity U_max = 2.096 U against the total flow rate. Diamond: '
                f'COMSOL axis velocity between s = 1.0 and 2.3 mm at 300 µL/min (median, '
                f'{comsol_numbers()["u_axis_m_s"]:.2f} m/s).',
        needs=AXIS_FILE)
def fig_flow_velocity_model(talk):
    """As res_flow_velocity, but without the measured cell velocities."""
    fig, ax = single('half', 0.8)
    _velocity_lines(ax)
    legend(ax, [handle(color=ts.ANALYTICAL_COLOR, label='$U$ (solid), $U_\\mathrm{max}$ (dashed)'),
                handle('marker', ts.INK_2, 'COMSOL (axis)', marker='D', mfc='white', ms=ms(6.5))], loc='upper left')
    return fig


@figure('res_flow_velocity', 'chip',
        lambda: 'Mean velocity U = Q/A and centreline velocity U_max = 2.096 U against the total flow rate. Points: '
                'lowest and highest measured median cell velocity over cell lines and imaging modes. Diamond: '
                f'COMSOL axis velocity between s = 1.0 and 2.3 mm at 300 µL/min (median, '
                f'{comsol_numbers()["u_axis_m_s"]:.2f} m/s).',
        needs=AXIS_FILE)
def fig_flow_velocity(talk):
    fig, ax = single('half', 0.8)
    _velocity_lines(ax)
    for qq, (lo, hi) in ft.MEASURED_CELL_VELOCITY.items():
        ax.plot([qq, qq], [lo, hi], color=MEASURED_COLOR, lw=lw(1.2), zorder=4)
        ax.plot([qq, qq], [lo, hi], 'o', color=MEASURED_COLOR, ms=ms(6), zorder=5)
    legend(ax, [handle(color=ts.ANALYTICAL_COLOR, label='$U$ (solid), $U_\\mathrm{max}$ (dashed)'),
                handle('marker', MEASURED_COLOR, 'measured cells'),
                handle('marker', ts.INK_2, 'COMSOL (axis)', marker='D', mfc='white', ms=ms(6.5))], loc='upper left')
    return fig


@figure('res_flow_reynolds', 'chip',
        'Reynolds number Re = ρUD_h/η with the mean velocity against the total flow rate. Filled points: operating '
        'points of the cell experiments. Open point: 600 µL/min, where the bonding failed.')
def fig_flow_reynolds(talk):
    ops = ft.table_operating_points()
    fig, ax = single('half', 0.8)
    q = np.linspace(1, ft.Q_FAILED, 300)
    Re = ft.RHO * q * ft.UL_MIN / A_CH * ft.W_CH / ft.ETA
    ax.plot(q, Re, color=ts.ANALYTICAL_COLOR, lw=lw(1.6), zorder=2)
    for _, r in ops.iterrows():
        ax.plot(r.Q_total_ul_min, r.Re_mean, 'o', color=MEASURED_COLOR, ms=ms(6),
                mfc='white' if r.failed else MEASURED_COLOR, mew=lw(1.0), zorder=4)
        ax.annotate(f'{r.Re_mean:.1f}' if r.Re_mean < 100 else f'{r.Re_mean:.0f}', (r.Q_total_ul_min, r.Re_mean),
                    xytext=(-6, 6), textcoords='offset points', ha='right', color=ts.INK_2, fontsize=small())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Total flow rate $Q$ (µL/min)')
    ax.set_ylabel('Re $= \\rho U D_\\mathrm{h}/\\eta$')
    ax.set_xticks([1, 3, 10, 30, 100, 300, 600], ['1', '3', '10', '30', '100', '300', '600'])
    legend(ax, [handle(color=ts.ANALYTICAL_COLOR, label='Re(Q)'),
                handle('marker', MEASURED_COLOR, 'cell experiments'),
                handle('marker', MEASURED_COLOR, 'bonding failed', mfc='white')], loc='upper left')
    return fig


@figure('res_pressure_drop', 'chip',
        lambda: 'Estimated pressure drop over the focusing channel, from the junction to the outlet hole at its end '
                '(lower bound, without filters, inlets, outlet hole and tubing). Diamond: COMSOL, entrance to outlet '
                f'({comsol_numbers()["length_mm"]:.2f} mm), 300 µL/min.',
        needs=AXIS_FILE)
def fig_pressure_drop(talk):
    ops = ft.table_operating_points()
    fig, ax = single('half', 0.8)
    x = np.arange(len(ops))
    ax.bar(x, ops.dp_total_bar, 0.6, color=ts.NEUTRAL, edgecolor='white', lw=1.0)
    cv = comsol_numbers()['dp_entrance_outlet_bar']
    i300 = int(np.flatnonzero(ops.Q_total_ul_min.eq(300))[0])
    ax.plot(i300, cv, ms=ms(6.5), mew=lw(1.2), zorder=5, **COMSOL_STYLE)
    for i, r in ops.iterrows():
        top = max(r.dp_total_bar, cv if i == i300 else 0) + 0.04 * ops.dp_total_bar.max()
        ax.text(i, top, f'{r.dp_total_bar:.2f} bar' if r.dp_total_bar < 1 else f'{r.dp_total_bar:.1f} bar',
                ha='center', va='bottom', color=ts.INK, fontsize=small())
    ax.set_xticks(x, [f'{int(q)}' + ('\n(failed)' if f else '') for q, f in zip(ops.Q_total_ul_min, ops.failed)])
    ax.set_ylim(0, ops.dp_total_bar.max() * 1.25)
    ax.set_xlabel('Total flow rate $Q$ (µL/min)')
    ax.set_ylabel('Pressure drop (bar)')
    ts.ygrid(ax)
    legend(ax, [handle('patch', ts.NEUTRAL, 'model, focusing channel'),
                handle('marker', ts.INK_2, 'COMSOL', marker='D', mfc='white', ms=ms(6.5))], loc='upper left')
    return fig


@figure('res_stream_width_ratio', 'chip',
        'Width of the focused stream against the sheath-to-sample ratio, with the model of Lee et al. At ratio 2, all '
        'images at total flow rates of 3, 30 and 150 µL/min are combined (n = 9). Ratios 3 and 4 were measured at a '
        'sample flow rate of 10 µL/min. Mean and standard deviation over the images. Band: measured cell diameters.')
def fig_stream_width_ratio(talk):
    sw = ft.table_stream_width()
    band = mpl.colors.to_rgba(ts.CELL_COLORS['NIH3T3'], 0.15)
    fig, ax = single('half', 0.8)
    ratio = np.linspace(0.5, 6, 120)
    ax.axhspan(14.2, 18.6, color=band, zorder=0, lw=0)                 # measured median cell diameters (Results)
    ax.plot(ratio, [ft.lee_width(1.0, r / 2)[0] / ft.UM for r in ratio], color=ts.ANALYTICAL_COLOR, lw=lw(1.6))
    ax.plot(ratio, ft.W_CH / ft.UM / (1 + ratio), color=ts.NEUTRAL, lw=lw(1.0), ls='--')
    m = ft.table_stream_by_ratio(sw)
    ax.errorbar(m.ratio, m.stream_width_um, m.stream_width_sd_um, fmt='o', color=MEASURED_COLOR, ms=ms(3.5),
                capsize=ms(2.5), elinewidth=lw(0.9), zorder=4)
    ax.set_xlim(0.5, 6)
    ax.set_ylim(0, 28)
    ax.set_xlabel('Sheath-to-sample ratio $Q_\\mathrm{sheath}/Q_\\mathrm{i}$')
    ax.set_ylabel('Stream width $w_\\mathrm{f}$ (µm)')
    ts.ygrid(ax)
    legend(ax, [handle(color=ts.ANALYTICAL_COLOR, label='Lee et al. model'),
                handle(color=ts.NEUTRAL, ls='--', lw=lw(1.0), label='without $\\gamma$'),
                handle('marker', MEASURED_COLOR, 'measured'),
                handle('patch', mpl.colors.to_rgba(ts.CELL_COLORS['NIH3T3'], 0.35), 'cell diameters')], ncols=2)
    return fig


@figure('res_shear_midplane', 'chip',
        'Shear stress of fully developed flow in the mid-plane, Eq. (shear_stress_rectangular), at 30, 150 and '
        '300 µL/min. Grey band: half width of the sample stream. Dotted line: edge of a centred 15 µm cell.')
def fig_shear_midplane(talk):
    ops = ft.table_operating_points()
    fc = flow_colors()
    fig, ax = single('half', 0.8)
    y = np.linspace(0, ft.W_CH / 2, 300)
    wf = ops.loc[ops.Q_total_ul_min == 300, 'stream_width_lee_um'].iloc[0]
    ax.axvspan(0, wf / 2, color=ts.GRID, zorder=0, lw=0)
    for _, r in ops[~ops.failed].iterrows():
        q = int(r.Q_total_ul_min)
        t = ft.shear_midplane(y, ft.W_CH, ft.H_CH, r.dp_per_L_kPa_mm * 1e6)
        ax.plot(y / ft.UM, t, color=fc[q], lw=lw())
        end_label(ax, ft.W_CH / 2 / ft.UM, t[-1], f'{q} µL/min')
    ax.axvline(7.5, color=ts.INK_2, lw=lw(0.9), ls=':')
    ax.set_xlim(0, ft.W_CH / 2 / ft.UM)
    ax.set_ylim(0)
    ax.set_xlabel('Distance from the centreline $y$ (µm)')
    ax.set_ylabel('Shear stress (Pa)')
    ts.ygrid(ax)
    legend(ax, [handle('patch', ts.GRID, 'sample stream'),
                handle(color=ts.INK_2, ls=':', lw=lw(0.9), label='edge of a centred 15 µm cell')], ncols=1)
    return fig


@figure('res_capillary_number', 'chip',
        'Capillary number at 300 µL/min for apparent moduli of 0.3 to 1.5 kPa, at the edge of a centred cell. '
        'Dotted line: Ca ≈ 0.3 (Müller et al.).')
def fig_capillary_number(talk):
    loads = ft.table_loads(300)
    col = ts.ANALYTICAL_COLOR
    fig, ax = single('half', 0.8)
    d = loads.d_um
    ax.fill_between(d, loads.Ca_min, loads.Ca_max, color=mpl.colors.to_rgba(col, 0.25), lw=0)
    ax.plot(d, loads.Ca_max, color=col, lw=lw(1.2))
    ax.plot(d, loads.Ca_min, color=col, lw=lw(1.2), ls='--')
    ax.axhline(0.3, color=ts.NEUTRAL, lw=lw(0.8), ls=':')
    ax.set_xlabel('Cell diameter $d$ (µm)')
    ax.set_ylabel('Ca $= \\eta\\dot\\gamma/G$')
    ax.set_xlim(d.iloc[0] - 1, d.iloc[-1] + 1)
    ax.set_ylim(0)
    ts.ygrid(ax)
    legend(ax, [handle(color=col, lw=lw(1.2), label='$E_\\mathrm{app}$ = 0.3 kPa'),
                handle(color=col, ls='--', lw=lw(1.2), label='$E_\\mathrm{app}$ = 1.5 kPa'),
                handle(color=ts.NEUTRAL, ls=':', lw=lw(0.8), label='Ca = 0.3 (Müller et al.)')], loc='upper left')
    return fig


@figure('res_motion_blur_phantom', 'chip',
        'Motion blur against exposure time for the highest measured cell velocity at each flow rate, Eq. '
        '(motion_blur). Dotted line: one pixel (0.925 µm). Vertical lines: exposure times of 0.2 and 0.7 µs.')
def fig_motion_blur_phantom(talk):
    blur = ft.table_motion_blur().set_index('Q_total_ul_min')
    fc = flow_colors()
    fig, ax = single('half', 0.8)
    t = np.linspace(0, 1.0, 100)                                  # µs
    for q in FLOWS:
        v = blur.loc[q, 'v_m_s']
        ax.plot(t, v * t, color=fc[q], lw=lw())                    # m/s * µs = µm
        end_label(ax, 1.0, v, f'{q} µL/min')
    ax.axhline(PX_UM, color=ts.INK_2, lw=lw(0.8), ls=':')
    for te in ft.T_EXP:
        ax.axvline(te * 1e6, color=ts.NEUTRAL, lw=lw(0.8), zorder=0)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0)
    ax.set_xlabel('Exposure time $t_\\mathrm{exp}$ (µs)')
    ax.set_ylabel('Motion blur (µm)')
    ts.ygrid(ax)
    legend(ax, [handle(color=ts.INK_2, ls=':', lw=lw(0.8), label='one pixel (0.925 µm)'),
                handle(color=ts.NEUTRAL, lw=lw(0.8), label='exposure times used')], ncols=1)
    return fig


@figure('res_motion_blur_zyla', 'chip',
        lambda: 'Motion blur $\\Delta x = U_\\mathrm{max}\\,t_\\mathrm{exp}$ (Eq. motion_blur) against exposure time for '
        f'the Andor Zyla 4.2P, with the centreline velocity $U_\\mathrm{{max}} = 2.096\\,Q/A$ at total flow rates of '
        f'{CFG.zyla_flows[0]:g} to {CFG.zyla_flows[-1]:g} µL/min. Dotted line: one pixel ({CFG.zyla_px_um:g} µm). '
        + (f'Vertical line: exposure time {CFG.zyla_t_exp_us[0]:g} µs. ' if CFG.zyla_t_exp_us else '')
        + ('Point: measured cell (6.77 mm/s).' if CFG.zyla_measured else ''))
def fig_motion_blur_zyla(talk):
    k = ft.u_max_over_u()
    flows = list(CFG.zyla_flows)
    cols = dict(zip(flows, ts.levels(len(flows), 'magma', 0.2, 0.7)))
    fig, ax = single('half', 0.8)
    t = np.geomspace(1, 1000, 200)                                # µs
    for q in flows:
        u_max = k * q * ft.UL_MIN / A_CH                           # m/s
        ax.plot(t, u_max * t, color=cols[q], lw=lw())              # m/s * µs = µm
        end_label(ax, t[-1], u_max * t[-1], f'{q:g} µL/min')
    ax.axhline(CFG.zyla_px_um, color=ts.INK_2, lw=lw(0.8), ls=':')
    for te in CFG.zyla_t_exp_us:
        ax.axvline(te, color=ts.NEUTRAL, lw=lw(0.8), zorder=0)
    for te, b in CFG.zyla_measured:
        ax.plot(te, b, 'o', color=MEASURED_COLOR, ms=ms(6), zorder=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 1000)
    plain = mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}')
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.set_xlabel('Exposure time $t_\\mathrm{exp}$ (µs)')
    ax.set_ylabel('Motion blur (µm)')
    ts.ygrid(ax)
    h = [handle(color=ts.INK_2, ls=':', lw=lw(0.8), label=f'one pixel ({CFG.zyla_px_um:g} µm)')]
    if CFG.zyla_t_exp_us:
        h.append(handle(color=ts.NEUTRAL, lw=lw(0.8), label='exposure time used'))
    if CFG.zyla_measured:
        h.append(handle('marker', MEASURED_COLOR, 'measured cell'))
    legend(ax, h, ncols=1)
    return fig


# =====================================================================================
# 6  Where the cells go (Results, Sections sec:res-comsol, sec:res-particles, sec:res-position)
# =====================================================================================
@figure('res_comsol_axis_velocity', 'position',
        'COMSOL half model at 300 µL/min: velocity magnitude on the channel axis at mid-height against the distance '
        'from the entrance of the focusing channel. Dashed: centreline velocity of fully developed flow in a '
        '40 × 40 µm channel.', needs=AXIS_FILE)
def fig_comsol_axis_velocity(talk):
    a = comsol_axis()
    u_max = ft.table_operating_points().set_index('Q_total_ul_min').loc[300, 'U_max_m_s']
    fig, ax = single('half', 0.8)
    ax.plot(a['s'], a['U'], color=ts.INK, lw=lw())
    ax.plot([0, a.s.max()], [u_max, u_max], color=ts.DEVELOPED_COLOR, ls='--', lw=lw())
    ax.axvline(0, color=ts.NEUTRAL, lw=lw(0.8), ls=':')
    ax.set_xlim(a['s'].min(), a.s.max() + 0.05)
    ax.set_ylim(0, u_max * 1.15)
    ax.set_xlabel('Distance from the channel entrance $s$ (mm)')
    ax.set_ylabel('Velocity on the axis (m/s)')
    ts.ygrid(ax)
    legend(ax, [handle(color=ts.INK, label='COMSOL'),
                handle(color=ts.DEVELOPED_COLOR, ls='--', label='fully developed flow')], loc='center left')
    return fig


@figure('res_comsol_axis_pressure', 'position',
        'COMSOL half model at 300 µL/min: pressure relative to the outlet on the channel axis at mid-height against '
        'the distance from the entrance of the focusing channel. Dashed: pressure gradient of fully developed flow in '
        'a 40 × 40 µm channel.', needs=AXIS_FILE)
def fig_comsol_axis_pressure(talk):
    a = comsol_axis()
    s_out = float(a.s.max())
    dpdl = ft.table_operating_points().set_index('Q_total_ul_min').loc[300, 'dp_per_L_kPa_mm'] * 1e3   # Pa/mm
    fig, ax = single('half', 0.8)
    ax.plot(a['s'], a['p_rel'] / 1e5, color=ts.INK, lw=lw())
    ax.plot([0, s_out], [dpdl * s_out / 1e5, 0], color=ts.DEVELOPED_COLOR, ls='--', lw=lw())
    ax.axvline(0, color=ts.NEUTRAL, lw=lw(0.8), ls=':')
    ax.set_xlim(a['s'].min(), s_out + 0.05)
    ax.set_ylim(0)
    ax.set_xlabel('Distance from the channel entrance $s$ (mm)')
    ax.set_ylabel('Pressure (bar)')
    ts.ygrid(ax)
    legend(ax, [handle(color=ts.INK, label='COMSOL'),
                handle(color=ts.DEVELOPED_COLOR, ls='--', label='fully developed flow')], loc='lower left')
    return fig


def _mark_positions(ax):
    for s, txt, ha in [(0.0, 'entrance ', 'right'), (ca.S_IMG_MM, ' imaging', 'left')]:
        ax.axvline(s, color=ts.NEUTRAL, lw=lw(0.8), ls=':', zorder=1)
        ax.text(s, 1.0, txt, transform=ax.get_xaxis_transform(), ha=ha, va='bottom', fontsize=small(),
                color=ts.INK_2)


@figure('res_comsol_z_d15', 'position',
        'Simulated trajectories of 15 µm particles at 300 µL/min: height of the particle centre against the distance '
        'from the entrance of the focusing channel. Shown are the particles that reach the outlet and start at least '
        'one radius away from floor and ceiling. Colour: start height. Dotted lines: channel entrance and assumed '
        'imaging position. Dashed line: mid-height.', needs=COMSOL_NEEDS)
def fig_comsol_z_d15(talk):
    d, tab = comsol()['data'][15]
    fig, ax = single('half', 0.8)
    norm = plt.Normalize(0, ca.H_UM)
    for _, r in tab[tab['valid']].iterrows():
        g = d[d['particle'] == r['particle']].sort_values('t')
        ax.plot(g['s'], g['z_um'], color=ts.CMAP_SEQ(norm(r['z0_um'])), lw=lw(0.9))
    ax.axhline(ca.H_UM / 2, color=ts.INK_2, lw=lw(0.8), ls='--')
    ax.set_ylim(0, ca.H_UM)
    ax.set_ylabel('Height $z$ (µm)')
    ax.set_xlim(d['s'].min(), d['s'].max() + 0.05)
    ax.set_xlabel('Distance from the channel entrance $s$ (mm)')
    _mark_positions(ax)
    ts.ygrid(ax)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ts.CMAP_SEQ), ax=ax, shrink=0.9)
    cb.set_label('Start height $z_0$ (µm)')
    cb.outline.set_visible(False)
    legend(ax, [handle(color=ts.INK_2, ls='--', lw=lw(0.8), label='mid-height')], loc='lower left')
    return fig


@figure('lateral_position_pooled', 'position',
        'Lateral position of the cell centres relative to the channel axis (track medians), all flow rates and both '
        'imaging modes pooled. Grey band: expected width of the sample stream (±4.7 µm). Solid lines: channel walls.',
        needs=lambda: [CFG.lateral_csv], width='full')
def fig_lateral_position(talk):
    lat = lateral()
    fig, ax = single('full', 0.42)
    bins = np.arange(-WALL_HALF_UM, WALL_HALF_UM + 1e-9, 1.0)
    ax.axvspan(-STREAM_HALF_UM, STREAM_HALF_UM, color=ts.GRID, lw=0, zorder=0)
    for s in (-1, 1):
        ax.axvline(s * WALL_HALF_UM, color=ts.INK_2, lw=lw(1.2))
    h = []
    for ct in CLS:
        v = lat.loc[lat.cell_type.eq(ct), 'y_rel_um'].dropna()
        if len(v) >= CFG.min_tracks:
            ax.hist(v, bins=bins, density=True, histtype='step', lw=lw(1.5), color=ts.CELL_COLORS[ct])
            h.append(handle(color=ts.CELL_COLORS[ct], lw=lw(1.5), label=f'{ct} ({len(v)} tracks)'))
    h += [handle('patch', ts.GRID, 'sample stream'), handle(color=ts.INK_2, lw=lw(1.2), label='channel wall')]
    ax.set_xlim(-WALL_HALF_UM - 1, WALL_HALF_UM + 1)
    ax.set_xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    ax.set_xlabel('Lateral position $y$ (µm)')
    ax.set_ylabel('Density')
    ts.ygrid(ax)
    legend(ax, h, ncols=2 if TALK else 4)
    save_table(lat.groupby('cell_type').y_rel_um.describe().reset_index(), 'lateral_position_pooled')
    return fig


# =====================================================================================
# 7  What the flow does to the cells (Results, Sections sec:res-particles and sec:res-size-deformation)
# =====================================================================================
def stress_table() -> pd.DataFrame:
    """Shear stress at the edge of a centred cell for every measured condition, and the COMSOL-path values at 300."""
    c = comsol()
    f = c['flow'].copy()
    f['sigma_c_centred_Pa'] = [float(ca.edge_stress_analytic(q, d)) for q, d in zip(f['Q'], f['d_um'])]
    cs = c['summary']
    for col in ['edge_med', 'edge_q25', 'edge_q75']:
        f[f'sigma_c_comsol_pos_{col[5:]}_Pa'] = [cs.loc[x, col] if q == ca.Q_COMSOL and x in cs.index else np.nan
                                                 for x, q in zip(f['cell'], f['Q'])]
    return f.sort_values(['cell', 'Q']).reset_index(drop=True)


@figure('res_cellmodel_stress', 'deformation',
        'Shear stress at the edge of a cell against the total flow rate (Eq. shear_stress_rectangular), for the '
        'measured median diameter at 300 µL/min. Lines: centred cell, dashed above the measured range. Open '
        'diamonds: the same equation evaluated at the lateral positions of the simulated 15 µm particles between '
        'the entrance and the imaging position (median and interquartile range over the particles).',
        needs=COMSOL_NEEDS, width='full')
def fig_cellmodel_stress(talk):
    c = comsol()
    save_table(stress_table(), 'stress_conditions')
    q_max = 600
    fig, ax = single('full', 0.5)
    q_in, q_out = np.linspace(0, ca.Q_COMSOL, 100), np.linspace(ca.Q_COMSOL, q_max, 100)
    for k, ct in enumerate(ca.CELLS):
        col, dc = ts.CELL_COLORS[ct], ca.CELLS[ct]['d_um']
        ax.plot(q_in, ca.edge_stress_analytic(q_in, dc), color=col, lw=lw())
        ax.plot(q_out, ca.edge_stress_analytic(q_out, dc), color=col, ls='--', lw=lw())
        end_label(ax, q_max, float(ca.edge_stress_analytic(q_max, dc)), f'{ct} ($d$ = {dc:.1f} µm)', col)
        if ct in c['summary'].index:
            r = c['summary'].loc[ct]
            ax.errorbar(ca.Q_COMSOL + (k - 0.5) * 12, r['edge_med'],
                        yerr=[[r['edge_med'] - r['edge_q25']], [r['edge_q75'] - r['edge_med']]],
                        fmt='D', color=col, mfc='white', ms=ms(5), mew=lw(1.0), capsize=ms(3), zorder=5)
    ax.set_xlim(0, q_max)
    ax.set_ylim(0, None)
    ax.set_xticks([0, 30, 150, 300, 450, 600])
    ax.set_xlabel('Total flow rate $Q$ (µL/min)')
    ax.set_ylabel('Shear stress at the cell edge (Pa)')
    ts.ygrid(ax)
    legend(ax, [handle(label='centred cell'), handle(ls='--', label='above the measured range'),
                handle('marker', ts.INK_2, 'at the simulated particle positions', marker='D', mfc='white', ms=ms(5))],
           ncols=3)
    return fig


def _violin(ax, v, pos, color, width):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return
    if len(v) >= CFG.min_tracks and np.ptp(v) > 0:
        body = ax.violinplot([v], positions=[pos], widths=0.88 * width, showextrema=False, points=150)['bodies'][0]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.3)
        ax.boxplot([v], positions=[pos], widths=0.26 * width, manage_ticks=False, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor=color, edgecolor=ts.INK_2, linewidth=lw(0.8)),
                   medianprops=dict(color=ts.INK, linewidth=lw(1.2)), whiskerprops=dict(linewidth=lw(0.8)),
                   capprops=dict(linewidth=lw(0.8)))
    else:                                             # few tracks: single points
        ax.scatter(np.full(len(v), pos), v, s=ms(3.5) ** 2, color=color, zorder=3)


def _violins(ax, df, metric, flows):
    width = 0.76 / len(CLS)
    offs = [(k - (len(CLS) - 1) / 2) * width for k in range(len(CLS))]
    for j, q in enumerate(flows):
        for k, ct in enumerate(CLS):
            g = df.loc[df.cell_type.eq(ct) & df.flow_rate_ul_min.eq(q), metric]
            _violin(ax, g, j + offs[k], ts.CELL_COLORS[ct], width)
    return offs


def _top(df, metric, q=0.99, pad=1.08):
    """Upper y limit at the 99th percentile of the groups. All values are still part of the statistics."""
    return float(df.groupby(['cell_type', 'flow_rate_ul_min'])[metric].quantile(q).max()) * pad


def _cell_handles():
    return [handle('patch', ts.CELL_COLORS[ct], ct, alpha=0.6) for ct in CLS]


@figure('res_cif_deformation', 'deformation',
        lambda: ('Deformation $D_\\mathrm{ellipse}$ of cells in flow (track medians, bright field and Ph2 pooled) and of '
                 f'static cells (squares: median and interquartile range of the {CFG.static_label} reference, '
                 'permissive filter). Violins: density, normalized per group. Boxes: interquartile range and median. '
                 'Grey band: one pixel difference between the major and the minor axis at 0.925 µm/px. The y axis '
                 'ends at the 99th percentile of the groups.'),
        needs=lambda: [v4('cells_primary'), CFG.static_csv], width='full')
def fig_cif_deformation(talk):
    c = cells()
    st = static_reference()
    flows = flows_of(c)
    fig, ax = single('full', 0.46)
    ax.axhspan(*ONE_PIXEL_D, color=ts.GRID, zorder=0, lw=0)
    offs = _violins(ax, c, 'deformation', flows)
    xs = -1.0
    for k, ct in enumerate(CLS):
        if ct in st.index:
            r = st.loc[ct]
            ax.errorbar(xs + offs[k], r.D_median, yerr=[[r.D_median - r.D_q25], [r.D_q75 - r.D_median]], fmt='s',
                        ms=ms(6), color=ts.CELL_COLORS[ct], capsize=ms(3), elinewidth=lw(1.0), zorder=3)
    ax.set_xticks([xs] + list(range(len(flows))), ['static'] + [f'{q:g}' for q in flows])
    ax.set_xlim(xs - 0.6, len(flows) - 0.4)
    ax.axvline(-0.5, color=ts.GRID, lw=lw(0.8))
    ax.set_ylim(0, _top(c, 'deformation'))
    ax.set_xlabel('Total flow rate $Q$ (µL/min)')
    ax.set_ylabel(r'$D_\mathrm{ellipse}$')
    ts.ygrid(ax)
    legend(ax, _cell_handles() + [handle('marker', ts.INK_2, 'static reference', marker='s'),
                                  handle('patch', ts.GRID, 'one pixel')], ncols=2 if TALK else 4)
    return fig


def _size_binned(g):
    """Median and interquartile range of D_ellipse in classes of the equivalent diameter (CFG.size_bin_um wide,
    at least CFG.size_min_per_bin tracks)."""
    edges = np.arange(np.floor(g.equivalent_diameter_um.min()),
                      g.equivalent_diameter_um.max() + CFG.size_bin_um, CFG.size_bin_um)
    s = g.groupby(pd.cut(g.equivalent_diameter_um, edges), observed=True).deformation.agg(
        ['size', 'median', lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
    s.columns = ['size', 'median', 'q25', 'q75']
    s['d'] = [iv.mid for iv in s.index]
    return s[s['size'] >= CFG.size_min_per_bin]


@figure('res_deformation_diameter', 'deformation',
        lambda: 'Deformation $D_\\mathrm{ellipse}$ against the equivalent diameter $d_\\mathrm{eq}$ (track medians, all '
                'flow rates, bright field and Ph2 pooled). Lines: median in classes of '
                f'{CFG.size_bin_um:g} µm with at least {CFG.size_min_per_bin} tracks. Bands: interquartile range '
                'within each class.',
        needs=CELLS_NEEDS)
def fig_deformation_diameter(talk):
    c = cells().dropna(subset=['equivalent_diameter_um', 'deformation'])
    fig, ax = single('half', 0.8)
    h = []
    for ct in CLS:
        g = c.loc[c.cell_type.eq(ct)]
        b = _size_binned(g)
        if len(b):
            col = ts.CELL_COLORS[ct]
            ax.fill_between(b.d, b.q25, b.q75, color=col, alpha=0.2, lw=0)
            ax.plot(b.d, b['median'], color=col, lw=lw(1.6))
            h.append(handle(color=col, lw=lw(1.6), label=f'{ct} (n = {len(g)})'))
    ax.set_ylim(0, None)
    ax.set_xlabel(r'Equivalent diameter $d_\mathrm{eq}$ (µm)')
    ax.set_ylabel(r'$D_\mathrm{ellipse}$ (track median)')
    ts.ygrid(ax)
    legend(ax, h, loc='lower left')
    return fig


# ---------------------------------------------------------------- rotation
def _theta(det):
    """Signed angle of the major axis to the flow direction, -90° to 90°."""
    major_vs_x = 90.0 - np.rad2deg(det.orientation_rad.astype(float))
    return (major_vs_x - CFG.flow_direction_deg % 180.0 + 90.0) % 180.0 - 90.0


def _unwrap(theta_deg):
    return np.rad2deg(np.unwrap(np.deg2rad(2 * np.asarray(theta_deg))) / 2)


@lru_cache(maxsize=1)
def _rotation_cached(key):
    det = detections()
    c = cells()
    det = det.loc[det.identity_kind.eq('tracked') & det.cell_uid.isin(c.cell_uid)] if 'identity_kind' in det else det
    det = det.dropna(subset=['orientation_rad', 'time_s']).copy()
    det['theta_deg'] = _theta(det)
    rows = []
    for uid, g in det.groupby('cell_uid', sort=False):
        g = g.sort_values('time_s')
        t = (g.time_s.to_numpy() - g.time_s.iloc[0]) * 1e3
        th = _unwrap(g.theta_deg)
        r = dict(cell_uid=uid, n_det=len(g), D_median=g.deformation.median(), duration_ms=t[-1],
                 total_rotation_deg=th[-1] - th[0], rate_deg_ms=np.nan)
        if len(g) >= 3 and t[-1] > 0:
            r['rate_deg_ms'] = np.polyfit(t, th, 1)[0]
        rows.append(r)
    tracks = pd.DataFrame(rows).merge(c[['cell_uid', 'cell_type', 'imaging_mode', 'flow_rate_ul_min']]
                                      .drop_duplicates('cell_uid'), on='cell_uid', how='left')
    tracks['eligible'] = ((tracks.n_det >= CFG.rotation_min_det) & (tracks.D_median >= CFG.rotation_d_min)
                          & tracks.rate_deg_ms.notna())
    return det, tracks


def rotation():
    """Detections with the signed angle, and one row per track with its rotation."""
    return _rotation_cached((str(CFG.v4_tables), CFG.rotation_min_det, CFG.rotation_d_min, CFG.flow_direction_deg))


@figure('res_rotation_angle', 'deformation',
        lambda: f'Change of the angle of the major axis to the flow direction along single tracks at '
                f'{CFG.rotation_example_flow:g} µL/min for HeLa (left) and NIH3T3 (right), bright field and Ph2 pooled. '
                f'Random sample of {CFG.rotation_n_examples} tracks per cell line with at least '
                f'{CFG.rotation_min_det} detections and a track median $D_\\mathrm{{ellipse}}$ of at least '
                f'{CFG.rotation_d_min:g}. The angle was unwrapped with a period of 180°.',
        needs=lambda: [v4('cells_primary'), v4('detection_morphology')], width='full')
def fig_rotation_angle(talk):
    det, tr = rotation()
    el = tr.loc[tr.eligible]
    save_table(el.groupby(['cell_type', 'imaging_mode', 'flow_rate_ul_min']).agg(
        n_tracks=('cell_uid', 'size'), abs_rate_median_deg_ms=('rate_deg_ms', lambda s: s.abs().median()),
        frac_total_over_45deg=('total_rotation_deg', lambda s: (s.abs() > 45).mean())).reset_index(),
        'rotation_summary')
    size = CFG.talk_size_wide if TALK else ts.figsize('full', 0.4)
    fig, axes = plt.subplots(1, len(CLS), figsize=size, sharey=True, layout='constrained')
    for ax, ct in zip(axes, CLS):
        pool = el.loc[el.cell_type.eq(ct) & el.flow_rate_ul_min.eq(CFG.rotation_example_flow)]
        pick = pool.sample(min(CFG.rotation_n_examples, len(pool)), random_state=0) if len(pool) else pool
        for uid in pick.cell_uid:
            g = det.loc[det.cell_uid.eq(uid)].sort_values('time_s')
            t = (g.time_s - g.time_s.iloc[0]) * 1e3
            th = _unwrap(g.theta_deg)
            ax.plot(t, th - th[0], color=ts.CELL_COLORS[ct], lw=lw(0.9), alpha=0.8)
        ax.axhline(0, color=ts.NEUTRAL, lw=lw(0.8))
        ax.set_title(f'{ct}, {CFG.rotation_example_flow:g} µL/min (n = {len(pick)})', fontsize=small())
        ax.set_xlabel('Time in the field of view (ms)')
        ts.ygrid(ax)
    axes[0].set_ylabel('Change of the major-axis angle (°)')
    return fig


@figure('app_deformation_modes', 'appendix',
        'Deformation $D_\\mathrm{ellipse}$ (track medians) per cell line and total flow rate in (a) bright field and '
        '(b) Ph2. Violins: density, normalized per group. Boxes: interquartile range and median. Groups with very '
        'few tracks are shown as single points. Numbers below the axis: tracks per group (left HeLa, right NIH3T3). '
        'The y axis ends at the 99th percentile of the groups, the scale is the same in both panels.',
        needs=CELLS_NEEDS, width='full')
def fig_deformation_modes(talk):
    c = cells()
    flows = flows_of(c)
    size = CFG.talk_size_wide if TALK else ts.figsize('full', 0.46)
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True, layout='constrained')
    top = _top(c, 'deformation')
    for ax, mode in zip(axes, MODES):
        sub = c.loc[c.imaging_mode.eq(mode)]
        offs = _violins(ax, sub, 'deformation', flows)
        n = sub.groupby(['cell_type', 'flow_rate_ul_min']).size()
        for j, q in enumerate(flows):
            for k, ct in enumerate(CLS):
                ax.text(j + offs[k], -0.115, str(int(n.get((ct, q), 0))), transform=ax.get_xaxis_transform(),
                        ha='center', va='top', color=ts.INK_2, fontsize=small() * 0.8)
        ax.set_xticks(range(len(flows)), [f'{q:g}' for q in flows])
        ax.set_xlim(-0.6, len(flows) - 0.4)
        ax.set_ylim(0, top)
        ax.set_xlabel('Total flow rate $Q$ (µL/min)', labelpad=14)
        ax.set_title(MODE_LABELS[mode].capitalize() if mode == 'BF' else MODE_LABELS[mode], fontsize=small())
        ts.ygrid(ax)
    axes[0].set_ylabel(r'$D_\mathrm{ellipse}$ (track median)')
    fig.legend(handles=_cell_handles(), loc='outside upper center', ncols=2, frameon=False, handlelength=1.8)
    return fig


if __name__ == '__main__':
    mpl.use('Agg')
    CFG.show = False
    make_all()
    print(overview().to_string(index=False))
