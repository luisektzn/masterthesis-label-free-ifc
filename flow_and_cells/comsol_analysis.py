"""Evaluation of the COMSOL simulation at 300 µL/min (half model): flow field on cut lines, particle trajectories,
forces, particle acceleration and the viscous stress on cells along the simulated paths.

Inputs (folder COMSOL_DIR, default comsol/ next to this file, or the environment variable COMSOL_DIR)
    axis_300.txt               cut line on the channel axis at mid-height, columns x y z p spf.U (mm, Pa, m/s)
    velocity_cross_300.txt     cut line across the channel at the imaging position, mid-height,
                               columns x y z spf.U spf.sr (mm, m/s, 1/s)
    path_300_d15_v3.csv.gz     particle tracks of the 15 µm and 20 µm particles, long format written by
    path_300_d20_v3.csv.gz     comsol_convert.py (positions in mm, velocities in m/s, forces in N)
and results/measured/measured_summary.csv (measured_summary.py) for the measured cell diameters and velocities.

Sections of the thesis that use the numbers
    sec:res-flow, sec:res-comsol     axis_summary, cross_summary
    sec:res-particles                particle_summary, force_summary, path_stress_summary
    sec:res-loads                    acceleration_summary, table_channel
    sec:res-position                 particle_summary (heights, distance from the axis)

Models (equation labels of the thesis)
    shear stress at the cell edge    sigma_c = tau(|x| + d/2), Eq. shear_stress_rectangular (Gerum et al. 2022, Eq. 4),
                                     x = lateral offset of the particle from the axis
    extensional stress               sigma_e = 2 eta |dv/dy|, component of Eq. viscous_stress_tensor along the axis

Run: python comsol_analysis.py, or open 03_comsol_analysis.ipynb.
Output: results/comsol/*.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import flow_theory as ft
import measured_summary as ms

HERE = Path(__file__).resolve().parent
COMSOL_DIR = Path(os.environ.get('COMSOL_DIR', HERE / 'comsol'))
RESULTS_DIR = Path(os.environ.get('FLOW_RESULTS', HERE / 'results')) / 'comsol'

# =====================================================================================
# 1  Settings
# =====================================================================================
Q_COMSOL = 300                               # total flow rate of the COMSOL model (µL/min)
AXIS_FILE = COMSOL_DIR / 'axis_300.txt'
CROSS_FILE = COMSOL_DIR / 'velocity_cross_300.txt'
PATH_FILES = {15: COMSOL_DIR / 'path_300_d15_v3.csv.gz',   # key: simulated particle diameter (µm)
              20: COMSOL_DIR / 'path_300_d20_v3.csv.gz'}
MEASURED_SUMMARY = ms.SUMMARY_CSV

X_AXIS_MM = -14.5            # symmetry plane = channel axis (COMSOL coordinates)
Y_ENTRANCE_MM = -0.7455      # entrance of the focusing channel: steepest increase of the axis velocity
H_UM = 40.0                  # channel height
S_IMG_MM = 1.5               # assumed imaging position, distance from the entrance (Methods)
ETA = ft.ETA                 # Pa s
RHO_P = 1050.0               # particle density (Methods, sec:methods_particles)

DEVELOPED_MM = (1.0, 2.3)    # developed part of the focusing channel on the axis
NEAR_ENTRANCE_MM = 0.3       # extensional stress: peak within |s| < 0.3 mm
CHANNEL_MM = (0.3, 2.9)      # forces: samples in the focusing channel
AGREE_PERCENT = 0.5          # cross line: agreement with Eq. poiseuille_rectangular_velocity

# Cell lines: d_um is replaced by the measured median diameter at Q_COMSOL (load_measured). The simulated
# diameter closest to it is used for the stress along the paths (d_sim = None: automatic).
CELLS = {'HeLa': dict(d_um=14.6, d_sim=None), 'NIH3T3': dict(d_um=17.7, d_sim=None)}


# =====================================================================================
# 2  Reading
# =====================================================================================
def read_comsol_table(path) -> np.ndarray:
    """Numbers of a COMSOL text export. Header lines start with %, columns are separated by spaces, tabs or commas."""
    rows = []
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        s = line.strip()
        if not s or s.startswith('%'):
            continue
        try:
            rows.append([float(v) for v in s.replace(',', ' ').replace(';', ' ').split()])
        except ValueError:
            continue
    return np.array(rows)


def read_axis(path=AXIS_FILE) -> pd.DataFrame:
    """Axis cut line: s (mm downstream of the entrance), velocity U (m/s), pressure p and p_rel relative to the
    outlet (Pa). The entrance is the position of the steepest increase of the axis velocity (Methods)."""
    a = pd.DataFrame(read_comsol_table(path)[:, :5], columns=['x', 'y', 'z', 'p', 'U'])
    a = a.sort_values('y', ascending=False).reset_index(drop=True)
    dUdy = np.gradient(a['U'].to_numpy(), a['y'].to_numpy())
    a['s'] = float(a['y'][np.argmax(np.abs(dUdy))]) - a['y']
    a['p_rel'] = a['p'] - a['p'].iloc[-1]
    return a


def read_cross(path=CROSS_FILE) -> pd.DataFrame:
    """Cross cut line at the imaging position: distance from the axis (µm), U (m/s), shear rate (1/s)."""
    a = read_comsol_table(path)
    out = pd.DataFrame({'y_um': np.abs(a[:, 0] - X_AXIS_MM) * 1e3, 'U': a[:, 3], 'sr': a[:, 4]})
    return out.sort_values('y_um').reset_index(drop=True)


def load_paths(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Particle tracks (long format of comsol_convert.py) and one row per particle.
    passed: the particle reaches the outlet. valid: passed and the centre starts at least one radius away
    from floor and ceiling (Methods)."""
    d = pd.read_csv(path)
    d['s'] = Y_ENTRANCE_MM - d['qy']                       # mm downstream of the entrance
    d['x_um'] = (d['qx'] - X_AXIS_MM) * 1e3
    d['z_um'] = d['qz'] * 1e3
    d_sim = float(Path(path).name.split('_d')[1].split('_')[0].split('.')[0])
    s_out = d['s'].max()
    rows = []
    for pid, g in d.groupby('particle'):
        g = g.sort_values('t')
        z0 = g['z_um'].iloc[0]
        rows.append(dict(particle=pid, x0_um=g['x_um'].iloc[0], z0_um=z0,
                         passed=g['s'].max() > s_out - 0.05,
                         wall_start=(z0 < d_sim / 2) or (z0 > H_UM - d_sim / 2)))
    tab = pd.DataFrame(rows)
    tab['valid'] = tab['passed'] & ~tab['wall_start']
    return d, tab


def load_all_paths() -> dict:
    """{d_sim: (tracks, particle table)} for every path file that exists."""
    data = {k: load_paths(f) for k, f in PATH_FILES.items() if Path(f).exists()}
    if not data:
        raise FileNotFoundError(f'no COMSOL path file found in {COMSOL_DIR}')
    return data


def load_measured(path=MEASURED_SUMMARY) -> pd.DataFrame:
    """Measured median diameter and velocity per cell line and flow rate (measured_summary.py). Sets
    CELLS[c]['d_um'] to the measured median diameter at Q_COMSOL."""
    m = pd.read_csv(path)
    flow = pd.DataFrame({'cell': m['cell_type'], 'Q': m['flow_rate_ul_min'].astype(int),
                         'd_um': m['equivalent_diameter_um_median'], 'v_m_s': m['speed_m_s_median'],
                         'n_tracks': m['n_tracks']})
    for c in CELLS:
        row = flow[(flow['cell'] == c) & (flow['Q'] == Q_COMSOL)]
        if len(row) and np.isfinite(row['d_um'].iloc[0]):
            CELLS[c]['d_um'] = round(float(row['d_um'].iloc[0]), 1)
    return flow


def used_diameters(data) -> dict:
    """Simulated diameter used for each cell line: the one closest to the measured diameter at Q_COMSOL."""
    return {c: (p['d_sim'] if p.get('d_sim') in data else min(data, key=lambda k: abs(k - p['d_um'])))
            for c, p in CELLS.items()}


# =====================================================================================
# 3  Flow field on the cut lines (sec:res-flow, sec:res-comsol)
# =====================================================================================
def axis_summary(a: pd.DataFrame) -> dict:
    """Axis velocity and pressure compared with fully developed flow at Q_COMSOL.
    U_dev is the median axis velocity in DEVELOPED_MM. s90 and s99 are the first positions downstream of the
    entrance where U reaches 90 % and 99 % of U_dev (and, for comparison, of the analytical U_max).
    The COMSOL pressure gradient is the slope of a straight line fitted to p(s) in DEVELOPED_MM."""
    ops = ft.table_operating_points().set_index('Q_total_ul_min')
    u_max = float(ops.loc[Q_COMSOL, 'U_max_m_s'])
    dev = a[(a.s >= DEVELOPED_MM[0]) & (a.s <= DEVELOPED_MM[1])]
    u_dev = float(dev.U.median())
    down = a[a.s >= 0].sort_values('s')

    def first_s(level):
        hit = down.loc[down.U >= level, 's']
        return float(hit.iloc[0]) if len(hit) else np.nan

    slope = -np.polyfit(dev.s, dev.p, 1)[0]                    # Pa/mm, positive for falling pressure
    return {
        'U_dev_median_m_s': u_dev, 'U_dev_min_m_s': float(dev.U.min()), 'U_dev_max_m_s': float(dev.U.max()),
        'U_max_analytical_m_s': u_max,
        's90_mm': first_s(0.9 * u_dev), 's99_mm': first_s(0.99 * u_dev),
        's90_analytical_mm': first_s(0.9 * u_max), 's99_analytical_mm': first_s(0.99 * u_max),
        'dpdx_comsol_kPa_mm': slope / 1e3, 'dpdx_analytical_kPa_mm': float(ops.loc[Q_COMSOL, 'dp_per_L_kPa_mm']),
        'dp_entrance_outlet_bar': float(np.interp(0.0, a['s'], a['p_rel'])) / 1e5,
        'length_entrance_outlet_mm': float(a.s.max()),
        'dp_analytical_bar': float(ops.loc[Q_COMSOL, 'dp_total_bar']),
    }


def cross_summary(c: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Cross line at the imaging position against Eq. poiseuille_rectangular_velocity in the mid-plane.
    y_agree_um is the largest distance from the axis up to which all points agree within AGREE_PERCENT."""
    u_a = ft.velocity_rect(c.y_um * ft.UM, ft.H_CH / 2, ft.W_CH, ft.H_CH, ft.dpdx_channel(Q_COMSOL))
    c = c.assign(U_analytical=u_a, diff_m_s=c.U - u_a, diff_percent=(c.U - u_a) / u_a * 100)
    inside = c[c.y_um < ft.W_CH / 2 / ft.UM]
    bad = inside.loc[inside.diff_percent.abs() > AGREE_PERCENT, 'y_um']
    y_agree = float(inside.loc[inside.y_um < bad.min(), 'y_um'].max()) if len(bad) else float(inside.y_um.max())
    outer = inside[inside.y_um > y_agree]
    return ({'y_agree_um': y_agree,
             'diff_beyond_min_m_s': float(outer.diff_m_s.min()) if len(outer) else np.nan,
             'diff_beyond_max_m_s': float(outer.diff_m_s.max()) if len(outer) else np.nan}, c)


# =====================================================================================
# 4  Particle trajectories and forces (sec:res-particles, sec:res-position)
# =====================================================================================
def particle_table(d: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    """One row per particle: positions and velocities at the entrance and at the imaging position, peak
    extensional stress near the entrance and the time at which the particle leaves the model."""
    rows = []
    for _, r in tab.iterrows():
        g = d[d['particle'] == r['particle']].sort_values('t')
        out = r.to_dict()
        if r['passed']:
            s = g['s'].to_numpy()
            sig_e = 2 * ETA * g['vy'].abs()
            out.update(t_out_ms=g['t'].iloc[-1] * 1e3,
                       z_entrance_um=np.interp(0.0, s, g['z_um']),
                       z_img_um=np.interp(S_IMG_MM, s, g['z_um']),
                       abs_x_img_um=abs(np.interp(S_IMG_MM, s, g['x_um'])),
                       v_p_img_m_s=np.interp(S_IMG_MM, s, g['v_p']),
                       u_f_img_m_s=np.interp(S_IMG_MM, s, g['u_f']),
                       sigma_e_peak_Pa=float(sig_e[np.abs(g['s']) < NEAR_ENTRANCE_MM].max()))
            out['slip_percent'] = (out['v_p_img_m_s'] - out['u_f_img_m_s']) / out['u_f_img_m_s'] * 100
        rows.append(out)
    return pd.DataFrame(rows)


def _rng(v):
    v = pd.Series(v).dropna()
    return (v.min(), v.median(), v.max()) if len(v) else (np.nan,) * 3


def particle_summary(data) -> pd.DataFrame:
    """Per simulated diameter: numbers of particles, and minimum, median and maximum of the quantities of
    particle_table over the valid particles (start heights also over all particles, slip also over all
    passed particles)."""
    rows = []
    for k, (d, tab) in data.items():
        p = particle_table(d, tab)
        v = p[p['valid']]
        row = {'d_sim_um': k, 'n_particles': len(p), 'n_passed': int(p['passed'].sum()), 'n_valid': len(v)}
        for name, values in [('z0_all_um', p['z0_um']), ('z0_valid_um', v['z0_um']),
                             ('z_entrance_um', v['z_entrance_um']), ('z_img_um', v['z_img_um']),
                             ('abs_x_img_um', v['abs_x_img_um']), ('v_p_img_m_s', v['v_p_img_m_s']),
                             ('slip_valid_percent', v['slip_percent']),
                             ('slip_passed_percent', p.loc[p['passed'], 'slip_percent']),
                             ('sigma_e_peak_Pa', v['sigma_e_peak_Pa']),
                             ('t_out_passed_ms', p.loc[p['passed'], 't_out_ms'])]:
            row[f'{name}_min'], row[f'{name}_median'], row[f'{name}_max'] = _rng(values)
        rows.append(row)
    return pd.DataFrame(rows).set_index('d_sim_um')


def force_summary(data) -> pd.DataFrame:
    """Per simulated diameter: median and maximum magnitude of the total force, the drag force and the lift
    force (nN) over all samples of the valid particles in the focusing channel (CHANNEL_MM)."""
    rows = []
    for k, (d, tab) in data.items():
        g = d[d['particle'].isin(tab.loc[tab['valid'], 'particle'])
              & d['s'].between(*CHANNEL_MM)]
        row = {'d_sim_um': k, 'n_samples': len(g)}
        for name, cols in [('total', ['Ftx', 'Fty', 'Ftz']), ('drag', ['FDx', 'FDy', 'FDz']),
                           ('lift', ['FLx', 'FLy', 'FLz'])]:
            f = np.sqrt((g[cols] ** 2).sum(axis=1)) * 1e9
            row[f'F_{name}_median_nN'], row[f'F_{name}_max_nN'] = f.median(), f.max()
        rows.append(row)
    return pd.DataFrame(rows).set_index('d_sim_um')


def particle_acceleration(d: pd.DataFrame, tab: pd.DataFrame, d_sim_um: float = 15.0,
                          selection: str = 'valid') -> pd.DataFrame:
    """Acceleration of the particles at the channel entrance (Results sec:res-loads).
    v_up = median particle velocity for -1.0 < s < -0.5 mm, v_dn = median for 1.0 < s < 1.5 mm. The velocity is
    smoothed with a running median of 5 samples and made monotonic (running maximum). t10 and t90 are the times at
    which it reaches v_up + 0.1 (v_dn - v_up) and v_up + 0.9 (v_dn - v_up).
    a_mean = 0.8 (v_dn - v_up) / (t90 - t10), a_max = largest increase over 5 samples (50 µs) divided by the time.
    F = m a with m = RHO_P pi d^3 / 6.
    selection: 'valid' (particles retained for the evaluation, as stated in the Results) or 'reached_imaging'
    (all particles that reach s = 1.5 mm, as in the first version of this evaluation)."""
    m_p = RHO_P * np.pi / 6 * (d_sim_um * 1e-6) ** 3
    ids = tab.loc[tab['valid'], 'particle'] if selection == 'valid' else tab['particle']
    rows = []
    for pid in ids:
        g = d[d['particle'] == pid].sort_values('t')
        if g['s'].max() < S_IMG_MM:
            continue
        t, s, v = g['t'].to_numpy(), g['s'].to_numpy(), g['v_p'].to_numpy()
        v_up = np.median(v[(s > -1.0) & (s < -0.5)])
        v_dn = np.median(v[(s > 1.0) & (s < 1.5)])
        vm = np.maximum.accumulate(pd.Series(v).rolling(5, center=True, min_periods=1).median().to_numpy())
        i10 = np.argmax(vm >= v_up + 0.1 * (v_dn - v_up))
        i90 = np.argmax(vm >= v_up + 0.9 * (v_dn - v_up))
        dt = t[i90] - t[i10]
        k = 5
        a_max = ((vm[k:] - vm[:-k]) / (t[k:] - t[:-k])).max()
        rows.append(dict(particle=pid, v_up_m_s=v_up, v_dn_m_s=v_dn, dt_10_90_us=dt * 1e6, s10_mm=s[i10],
                         s90_mm=s[i90], a_mean_m_s2=0.8 * (v_dn - v_up) / dt, a_max_m_s2=a_max))
    r = pd.DataFrame(rows)
    r['F_mean_nN'] = m_p * r['a_mean_m_s2'] * 1e9
    r['F_max_nN'] = m_p * r['a_max_m_s2'] * 1e9
    r.attrs['mass_kg'] = m_p
    return r


# =====================================================================================
# 5  Viscous stress on the cells (sec:res-particles, Discussion)
# =====================================================================================
def edge_stress_analytic(q_ul, d_um, offset_um=0.0):
    """Shear stress at the outer edge of a cell of diameter d_um with the lateral offset offset_um from the axis,
    Eq. shear_stress_rectangular (mid-plane, cell-free fully developed flow)."""
    y = np.minimum((np.abs(np.asarray(offset_um, float)) + d_um / 2) * ft.UM, ft.W_CH / 2)
    return ft.shear_midplane(y, ft.W_CH, ft.H_CH, ft.dpdx_channel(q_ul))


def stress_history(g: pd.DataFrame, d_um: float, q_ul: float = Q_COMSOL) -> pd.DataFrame:
    """Stress along one particle path: shear stress at the outer cell edge in the focusing channel (s >= 0, lateral
    offset of the particle included) and extensional stress 2 eta |dv/dy| at the particle centre."""
    g = g.sort_values('t')
    s = g['s'].to_numpy()
    sig_edge = edge_stress_analytic(q_ul, d_um, g['x_um'].to_numpy())
    return pd.DataFrame({'t': g['t'].to_numpy(), 's': s, 'sigma_edge': np.where(s >= 0, sig_edge, np.nan),
                         'sigma_e': 2 * ETA * g['vy'].abs().to_numpy()})


def table_paths(data, used) -> pd.DataFrame:
    """Per cell line and valid particle: median shear stress at the cell edge between 0.3 mm and the imaging
    position, and the peak extensional stress near the entrance."""
    rows = []
    for c, k in used.items():
        d, tab = data[k]
        for pid in tab.loc[tab['valid'], 'particle']:
            g = d[d['particle'] == pid]
            h = stress_history(g, CELLS[c]['d_um'])
            near = (h['s'] > -NEAR_ENTRANCE_MM) & (h['s'] < NEAR_ENTRANCE_MM)
            ch = (h['s'] > NEAR_ENTRANCE_MM) & (h['s'] <= S_IMG_MM)
            rows.append(dict(cell=c, d_sim_um=k, particle=pid,
                             sigma_e_peak_Pa=float(h.loc[near, 'sigma_e'].max()),
                             sigma_edge_channel_median_Pa=float(h.loc[ch, 'sigma_edge'].median())))
    return pd.DataFrame(rows)


def path_stress_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Per cell line: median, interquartile range (q25, q75) and range (min, max) over the particles of the edge
    stress and of the peak extensional stress."""
    g = paths.groupby('cell')
    out = {}
    for key, col in [('edge', 'sigma_edge_channel_median_Pa'), ('ext', 'sigma_e_peak_Pa')]:
        out[f'{key}_med'] = g[col].median()
        out[f'{key}_q25'] = g[col].quantile(.25)
        out[f'{key}_q75'] = g[col].quantile(.75)
        out[f'{key}_min'] = g[col].min()
        out[f'{key}_max'] = g[col].max()
    return pd.DataFrame(out)


def table_channel(flow: pd.DataFrame) -> pd.DataFrame:
    """Per cell line and flow rate: shear stress at the edge of a centred cell with the measured median diameter
    of the condition, and the passage time from the entrance to the imaging position with the measured median
    velocity."""
    rows = []
    for _, r in flow.sort_values(['cell', 'Q']).iterrows():
        rows.append(dict(cell=r['cell'], Q_ul_min=r['Q'], d_um=r['d_um'], n_tracks=r['n_tracks'],
                         sigma_c_centred_Pa=float(edge_stress_analytic(r['Q'], r['d_um'])),
                         v_measured_m_s=r['v_m_s'], t_entrance_to_imaging_ms=S_IMG_MM / r['v_m_s']))
    return pd.DataFrame(rows)


# =====================================================================================
# 6  Main
# =====================================================================================
def run(save=True) -> dict:
    """All evaluations. Parts whose input file is missing are skipped with a message."""
    res = {}
    if AXIS_FILE.exists():
        res['axis'] = read_axis()
        res['axis_summary'] = pd.Series(axis_summary(res['axis']))
    else:
        print('axis cut line not found:', AXIS_FILE)
    if CROSS_FILE.exists():
        summary, table = cross_summary(read_cross())
        res['cross_summary'], res['cross'] = pd.Series(summary), table
    else:
        print('cross cut line not found:', CROSS_FILE)
    data = load_all_paths()
    res['particles'] = particle_summary(data)
    res['forces'] = force_summary(data)
    if 15 in data:
        res['acceleration'] = particle_acceleration(*data[15], d_sim_um=15.0)
    flow = load_measured()
    used = used_diameters(data)
    res['used'] = used
    res['paths'] = table_paths(data, used)
    res['path_stress'] = path_stress_summary(res['paths'])
    res['channel'] = table_channel(flow)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for name in ['axis_summary', 'cross_summary', 'particles', 'forces', 'acceleration', 'paths',
                     'path_stress', 'channel']:
            if name in res:
                obj = res[name]
                keep_index = not isinstance(obj.index, pd.RangeIndex)
                obj.to_csv(RESULTS_DIR / f'{name}.csv', index=keep_index, float_format='%.6g')
    return res


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    r = run()
    for key in ['axis_summary', 'cross_summary', 'particles', 'forces', 'path_stress', 'channel']:
        if key in r:
            print(f'\n{key}\n', r[key].T.round(3).to_string() if key == 'particles' else r[key].round(3).to_string())
    if 'acceleration' in r:
        print('\nacceleration (median)\n', r['acceleration'].median(numeric_only=True).round(3).to_string())
    print('\nwritten to', RESULTS_DIR)
