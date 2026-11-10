"""Analytical flow model of the focusing channel.

Every function names the equation label of the Background chapter (02_background) or the estimate
described in Methods (sec:methods_operating, sec:methods_flow_deformation), so that each number in the
Results can be traced back to the text. The module only computes numbers. It draws no figures.

Run
    python flow_theory.py                writes results/flow_theory/*.csv and checks.txt
or open 01_flow_theory.ipynb, which explains every equation and calls the functions below.

Outputs (RESULTS_DIR = results/flow_theory, or the folder in the environment variable FLOW_RESULTS)
    operating_points.csv       Table tab:res-flow (velocities, Re, pressure gradient, pressure drop)
    hydraulic_resistance.csv   exact series against Eq. hydraulic_resistance_rectangular
    stream_width.csv           Lee et al. model against the measured stream width, per condition
    stream_width_by_ratio.csv  Table tab:res-stream (conditions with the same ratio combined)
    cell_loads.csv             Section sec:res-loads (Re_p, shear stress, Ca, focusing length, forces)
    motion_blur.csv            motion blur of the Phantom recordings (Eqs. motion_blur, max_exposure_time)
    checks.txt                 every value of the thesis text that this module reproduces
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get('FLOW_RESULTS', HERE / 'results')) / 'flow_theory'

UL_MIN = 1e-9 / 60.0          # 1 µL/min in m^3/s
UM = 1e-6                     # 1 µm in m

# =====================================================================================
# 1  Input parameters (Methods, sec:flow_focusing and sec:methods_operating)
# =====================================================================================
RHO = 1000.0                  # kg/m^3, PBS treated as water (Bruus 2008)
ETA = 1.0e-3                  # Pa s (Bruus 2008)

W_CH, H_CH, L_CH = 40 * UM, 40 * UM, 3.0e-3      # focusing channel, design values
# The outlet hole was punched at the end of the 3.0 mm focusing channel in all experiments, so the
# fluid leaves the chip there. The pressure drop on the chip is that of the focusing channel alone.

Q_TOTAL = [30, 150, 300]                         # total flow rates of the cell experiments (µL/min)
Q_FAILED = 600                                   # the bonding failed at this flow rate (Results)
SAMPLE_FRACTION = 1 / 3                          # sample-to-sheath ratio 1:2 (Methods)

D_CELLS = np.array([10, 15, 20, 25]) * UM        # cell diameters for the load estimates (Results)
E_APP = (0.3e3, 1.5e3)                           # Pa, apparent Young's modulus (Methods)
F_LIFT = (0.02, 0.05)                            # lift coefficients (Amini et al. 2014, p. 2743)

PIXEL = 0.925 * UM                               # Phantom TMX, pixel size in the object plane
T_EXP = (0.2e-6, 0.7e-6)                         # exposure times drawn in the motion-blur figure

N_TERMS = 200                                    # series terms (Methods)

# =====================================================================================
# 2  Measured values used by the model
# =====================================================================================
# Stream width per condition (Results, tab:res-stream). q_sheath is the TOTAL sheath flow rate, q_sample the
# sample flow rate (µL/min), channel_width_um the measured channel width of the condition. The values are the
# output of the stream-width image analysis (channel_stream_by_condition.csv).
MEASURED_STREAM = pd.DataFrame({
    'q_sheath': [2, 20, 100, 30, 40],
    'q_sample': [1, 10, 50, 10, 10],
    'n_images': [3, 3, 3, 3, 4],
    'channel_width_um': [39.35, 39.30, 40.12, 39.65, 39.48],
    'stream_width_um': [6.91, 8.97, 10.43, 6.14, 5.68],
    'stream_width_sd_um': [1.82, 0.45, 0.64, 0.44, 1.15],
})
# Lowest and highest median cell velocity over cell lines and imaging modes (m/s), from the tracked cells
# (Results, paragraph "Cell velocity, acceleration and pressure"). Used for the Phantom motion blur.
MEASURED_CELL_VELOCITY = {30: (0.35, 0.44), 150: (1.48, 1.89), 300: (2.88, 3.72)}


# =====================================================================================
# 3  Physics (equation labels of the Background chapter)
# =====================================================================================
def _cosh_ratio(a, b):
    """cosh(a)/cosh(b) without overflow."""
    return np.exp(a - b) * (1 + np.exp(-2 * a)) / (1 + np.exp(-2 * b))


def _sinh_cosh_ratio(a, b):
    """sinh(a)/cosh(b) without overflow (a, b >= 0)."""
    return np.exp(a - b) * (1 - np.exp(-2 * a)) / (1 + np.exp(-2 * b))


def velocity_rect(y, z, w, h, dpdx, eta=ETA, n_terms=N_TERMS):
    """Eq. poiseuille_rectangular_velocity: u_x(y, z) for -w/2 < y < w/2 and 0 < z < h.
    dpdx is the pressure drop per length Delta p / L (positive)."""
    y, z = np.broadcast_arrays(np.asarray(y, float), np.asarray(z, float))
    u = np.zeros_like(y)
    for n in range(1, 2 * n_terms, 2):
        u += (1 / n ** 3) * (1 - _cosh_ratio(n * np.pi * np.abs(y) / h, n * np.pi * w / (2 * h))) \
             * np.sin(n * np.pi * z / h)
    return 4 * h ** 2 * dpdx / (np.pi ** 3 * eta) * u


def flow_per_pressure_gradient(w, h, eta=ETA, n_terms=N_TERMS):
    """Q / (Delta p / L), from Eq. volume_flow_rate applied to Eq. poiseuille_rectangular_velocity.
    The integral of the series is the denominator of Eq. velocity_ratio_gamma."""
    n = np.arange(1, 2 * n_terms, 2)
    s = np.sum(np.tanh(n * np.pi * w / (2 * h)) / n ** 5)
    return h ** 3 * w / (12 * eta) * (1 - 192 * h / (np.pi ** 5 * w) * s)


def r_hyd_exact(w, h, length, eta=ETA):
    """Hydraulic resistance from the full series (Eq. hagen_poiseuille_general), with the smaller side as h.
    All pressure values of the thesis use this function."""
    w, h = max(w, h), min(w, h)
    return length / flow_per_pressure_gradient(w, h, eta)


def r_hyd_approx(w, h, length, eta=ETA):
    """Eq. hydraulic_resistance_rectangular (valid for h < w). Only for comparison with the exact series."""
    w, h = max(w, h), min(w, h)
    return 12 * eta * length / (h ** 3 * w) / (1 - 0.630 * h / w)


def u_max_over_u(w=W_CH, h=H_CH):
    """Ratio of the centreline velocity to the mean velocity, Eq. poiseuille_rectangular_velocity."""
    q = flow_per_pressure_gradient(w, h)                 # for dp/dx = 1
    u_mean = q / (w * h)
    u_max = velocity_rect(0.0, h / 2, w, h, 1.0)
    return float(u_max / u_mean)


def dpdx_channel(q_ul):
    """Pressure gradient of fully developed flow in the focusing channel (Pa/m) at the total flow rate q_ul."""
    return r_hyd_exact(W_CH, H_CH, L_CH) * np.asarray(q_ul, float) * UL_MIN / L_CH


def shear_midplane(y, w, h, dpdx, n_terms=N_TERMS):
    """Eq. shear_stress_rectangular (Gerum et al. 2022, Eq. 4): |tau(y)| in the mid-plane."""
    y = np.asarray(y, float)
    t = np.zeros_like(y)
    for n in range(1, 2 * n_terms, 2):
        t += (-1) ** ((n - 1) // 2) / n ** 2 * _sinh_cosh_ratio(n * np.pi * np.abs(y) / h, n * np.pi * w / (2 * h))
    return np.abs(4 * h / np.pi ** 2 * dpdx * t)


def lee_gamma(wf, wo=W_CH, h=H_CH, n_terms=N_TERMS):
    """Eq. velocity_ratio_gamma (Lee et al. 2006): mean velocity in the focused stream over the mean velocity
    in the outlet channel, for a centred stream of width wf."""
    k = 2 * np.arange(n_terms) + 1
    num = 1 - 192 * h / (np.pi ** 5 * wf) * np.sum(
        _sinh_cosh_ratio(k * np.pi * wf / (2 * h), k * np.pi * wo / (2 * h)) / k ** 5)
    den = 1 - 192 * h / (np.pi ** 5 * wo) * np.sum(np.tanh(k * np.pi * wo / (2 * h)) / k ** 5)
    return num / den


def lee_width(q_sample, q_sheath_each, wo=W_CH, h=H_CH):
    """Eqs. hydrodynamic_focusing_width and velocity_ratio_gamma, solved together by bisection (Methods).
    Returns (wf, gamma). q_sheath_each is the flow rate of ONE sheath stream."""
    target = q_sample / (q_sample + 2 * q_sheath_each)
    lo, hi = 1e-4 * wo, wo
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f = mid / wo - target / lee_gamma(mid, wo, h)
        lo, hi = (mid, hi) if f < 0 else (lo, mid)
    wf = 0.5 * (lo + hi)
    return wf, lee_gamma(wf, wo, h)


def lee_gamma_numeric(wf, wo=W_CH, h=H_CH, n=801):
    """Check of lee_gamma: gamma from direct integration of the depth-averaged velocity profile."""
    y = np.linspace(-wo / 2, wo / 2, n)
    z = np.linspace(0, h, n)
    Y, Z = np.meshgrid(y, z)
    u_bar = np.trapezoid(velocity_rect(Y, Z, wo, h, 1.0), z, axis=0)
    inside = np.abs(y) <= wf / 2
    return np.trapezoid(u_bar[inside], y[inside]) / (y[inside][-1] - y[inside][0]) / (np.trapezoid(u_bar, y) / wo)


def pool_groups(means, sds, ns):
    """Mean and standard deviation over all images of several conditions, from the mean, standard deviation
    and number of images of each condition (exact for the sample standard deviation)."""
    means, sds, ns = map(np.asarray, (means, sds, ns))
    n = ns.sum()
    mean = np.sum(ns * means) / n
    ss = np.sum((ns - 1) * sds ** 2 + ns * (means - mean) ** 2)
    return mean, np.sqrt(ss / (n - 1)), int(n)


# =====================================================================================
# 4  Tables
# =====================================================================================
def table_operating_points():
    """Table tab:res-flow: velocities, Re, pressure gradient, pressure drop and wall shear stress per
    operating point. The pressure drop is a lower bound (filters, inlets, outlet hole and tubing excluded)."""
    A = W_CH * H_CH
    D_h = 2 * W_CH * H_CH / (W_CH + H_CH)
    k_max = u_max_over_u()
    rows = []
    for q_ul in Q_TOTAL + [Q_FAILED]:
        Q = q_ul * UL_MIN
        U = Q / A
        dpdx = dpdx_channel(q_ul)
        wf, gamma = lee_width(SAMPLE_FRACTION * Q, (1 - SAMPLE_FRACTION) * Q / 2)
        rows.append({
            'Q_total_ul_min': q_ul, 'Q_sample_ul_min': q_ul * SAMPLE_FRACTION,
            'Q_sheath_ul_min': q_ul * (1 - SAMPLE_FRACTION),
            'U_mean_m_s': U, 'U_max_m_s': k_max * U, 'U_stream_m_s': gamma * U,
            'Re_mean': RHO * U * D_h / ETA, 'Re_max': RHO * k_max * U * D_h / ETA,   # Eq. reynolds_number
            'dp_per_L_kPa_mm': dpdx / 1e6,
            'dp_total_bar': dpdx * L_CH / 1e5,                                     # Eq. hagen_poiseuille_general
            'tau_wall_mean_Pa': dpdx * W_CH * H_CH / (2 * (W_CH + H_CH)),         # force balance (Methods)
            'tau_wall_centre_Pa': float(shear_midplane(W_CH / 2, W_CH, H_CH, dpdx)),
            'residence_time_ms': L_CH / U * 1e3,
            'stream_width_lee_um': wf / UM, 'gamma': gamma,
            'failed': q_ul == Q_FAILED,
        })
    return pd.DataFrame(rows)


def table_resistances():
    """Hydraulic resistance of the focusing channel: exact series against Eq. hydraulic_resistance_rectangular.
    attrs['square_coefficient'] is k in R = k eta L / h^4 (Bruus 2008: 28.4)."""
    r_ex, r_ap = r_hyd_exact(W_CH, H_CH, L_CH), r_hyd_approx(W_CH, H_CH, L_CH)
    df = pd.DataFrame([{'section': 'focusing channel', 'width_um': W_CH / UM, 'height_um': H_CH / UM,
                        'length_mm': L_CH * 1e3, 'R_exact_Pa_s_m3': r_ex, 'R_approx_Pa_s_m3': r_ap,
                        'approx_over_exact': r_ap / r_ex}])
    df.attrs['square_coefficient'] = r_hyd_exact(W_CH, H_CH, 1.0) * H_CH ** 4 / ETA
    return df


def table_stream_width():
    """Lee et al. model against the measured stream width, per condition. lee_um uses the measured channel
    width of the condition, lee_nominal_40um the design width. without_gamma_um is w_o Q_i / Q."""
    m = MEASURED_STREAM.copy()
    m['Q_total_ul_min'] = m.q_sheath + m.q_sample
    m['ratio'] = m.q_sheath / m.q_sample
    m['Re_mean'] = RHO * (m.Q_total_ul_min * UL_MIN / (W_CH * H_CH)) * W_CH / ETA
    lee, lee_nom, gam = [], [], []
    for _, r in m.iterrows():
        wf, g = lee_width(r.q_sample, r.q_sheath / 2, wo=r.channel_width_um * UM)
        lee.append(wf / UM)
        gam.append(g)
        lee_nom.append(lee_width(r.q_sample, r.q_sheath / 2)[0] / UM)
    m['lee_um'] = lee
    m['lee_nominal_40um'] = lee_nom
    m['gamma'] = gam
    m['without_gamma_um'] = m.channel_width_um / (1 + m.ratio)
    m['measured_minus_lee_um'] = m.stream_width_um - m.lee_um
    return m


def table_stream_by_ratio(sw):
    """Table tab:res-stream: all images with the same sheath-to-sample ratio combined (ratio 2: total flow rates
    of 3, 30 and 150 µL/min). lee_measured_um is the image-weighted mean of the per-condition predictions with the
    measured channel widths, lee_nominal_40um the prediction for the design width (figure res_stream_width_ratio)."""
    rows = []
    for ratio, g in sw.groupby('ratio'):
        mean, sd, n = pool_groups(g.stream_width_um, g.stream_width_sd_um, g.n_images)
        rows.append({'ratio': ratio,
                     'Q_total_ul_min': ', '.join(f'{int(q)}' for q in g.Q_total_ul_min),
                     'n_images': n, 'stream_width_um': mean, 'stream_width_sd_um': sd,
                     'lee_measured_um': np.average(g.lee_um, weights=g.n_images),
                     'lee_nominal_40um': lee_width(1.0, ratio / 2)[0] / UM})
    return pd.DataFrame(rows)


def table_loads(q_ul=300):
    """Section sec:res-loads: size-dependent quantities for the cell diameters D_CELLS at the total flow rate q_ul
    (cell-free, fully developed flow)."""
    Q = q_ul * UL_MIN
    U = Q / (W_CH * H_CH)
    k = u_max_over_u()
    U_max = k * U
    dpdx = dpdx_channel(q_ul)
    wf, _ = lee_width(SAMPLE_FRACTION * Q, (1 - SAMPLE_FRACTION) * Q / 2)
    Re = RHO * U * W_CH / ETA
    G = np.array(E_APP) / 3                                                   # G = E_app / 3 (Methods)
    rows = []
    for d in D_CELLS:
        sig_c = float(shear_midplane(d / 2, W_CH, H_CH, dpdx))                 # edge of a centred cell
        y_off = min(wf / 2 + d / 2, W_CH / 2)                                 # centre at the stream boundary
        sig_off = float(shear_midplane(y_off, W_CH, H_CH, dpdx))
        gdot = sig_c / ETA
        rows.append({
            'Q_total_ul_min': q_ul, 'd_um': d / UM, 'chi': d / W_CH,              # Eq. confinement-ratio
            'Re_p_mean': Re * (d / W_CH) ** 2,                                   # Eq. particle_reynolds_number
            'Re_p_max': Re * k * (d / W_CH) ** 2,
            'sigma_centred_Pa': sig_c, 'sigma_stream_edge_Pa': sig_off,          # Eq. shear_stress_rectangular
            'shear_rate_centred_1_s': gdot,
            'Ca_min': ETA * gdot / G.max(), 'Ca_max': ETA * gdot / G.min(),      # Eq. ca_number
            'L_f_min_um': np.pi * ETA * W_CH ** 2 / (RHO * U_max * d ** 2 * max(F_LIFT)) / UM,  # Eq. focusing_length
            'L_f_max_um': np.pi * ETA * W_CH ** 2 / (RHO * U_max * d ** 2 * min(F_LIFT)) / UM,
            'F_L_min_nN': min(F_LIFT) * RHO * U_max ** 2 * d ** 4 / W_CH ** 2 * 1e9,          # Eq. inertial_lift_force
            'F_L_max_nN': max(F_LIFT) * RHO * U_max ** 2 * d ** 4 / W_CH ** 2 * 1e9,
            'F_D_per_U_rel_pN_s_per_mm': 3 * np.pi * ETA * d * 1e12 / 1e3,      # Eq. stokes_drag
        })
    return pd.DataFrame(rows)


def particle_reynolds(d_um, q_ul=300, use_max=False):
    """Eq. particle_reynolds_number for any diameter (µm), with the mean or the centreline velocity."""
    U = q_ul * UL_MIN / (W_CH * H_CH) * (u_max_over_u() if use_max else 1.0)
    return RHO * U * W_CH / ETA * (np.asarray(d_um, float) * UM / W_CH) ** 2


def table_motion_blur():
    """Motion blur of the Phantom recordings (Eqs. motion_blur and max_exposure_time) for the highest measured
    median cell velocity at each flow rate."""
    rows = []
    for q_ul in Q_TOTAL:
        v = max(MEASURED_CELL_VELOCITY[q_ul])
        rows.append({'Q_total_ul_min': q_ul, 'v_m_s': v,
                     **{f'blur_um_{t * 1e6:g}us': v * t / UM for t in T_EXP},
                     't_exp_max_1px_us': PIXEL / v * 1e6})
    return pd.DataFrame(rows)


# =====================================================================================
# 5  Comparison with the thesis text
# =====================================================================================
def checks(ops, res, sw, sw_ratio, loads):
    """Every value of the thesis text that follows from this module. The value in the text is given in
    brackets. loads is table_loads(300)."""
    o = ops.set_index('Q_total_ul_min')
    r = sw_ratio.set_index('ratio')
    lines = [
        ('Background: Bruus coefficient for a square channel (28.4)', res.attrs['square_coefficient']),
        ('Results: R_approx / R_exact for the square channel', res.approx_over_exact.iloc[0]),
        ('Results: U_max/U (2.096)', u_max_over_u()),
        ('Results: gamma at ratio 2 (1.42)', lee_width(1, 1)[1]),
        ('Check: gamma from direct integration', lee_gamma_numeric(lee_width(1, 1)[0])),
        ('Results: w_f/w_o at ratio 2 (0.235)', lee_width(1, 1)[0] / W_CH),
        ('Results: w_f at ratio 2/3/4, 40 um (9.4/7.0/5.6 um)',
         tuple(lee_width(1, x / 2)[0] / UM for x in (2, 3, 4))),
        ('tab:res-stream: Lee model, measured widths (9.30/6.94/5.51)', tuple(r.lee_measured_um.round(2))),
        ('tab:res-stream: stream width ratio 2 (8.77 +- 1.82, n = 9)',
         (r.loc[2, 'stream_width_um'], r.loc[2, 'stream_width_sd_um'], r.loc[2, 'n_images'])),
        ('tab:res-flow: U 30/150/300/600 (0.313/1.56/3.13/6.25 m/s)', tuple(o.U_mean_m_s.round(3))),
        ('tab:res-flow: U_max (0.655/3.28/6.55/13.1 m/s)', tuple(o.U_max_m_s.round(3))),
        ('tab:res-flow: Re (12.5/62.5/125/250)', tuple(o.Re_mean.round(1))),
        ('Results: Re with U_max (26/131/262)', tuple(o.loc[[30, 150, 300], 'Re_max'].round(0))),
        ('tab:res-flow: dp/L (5.56/27.8/55.6/111 kPa/mm)', tuple(o.dp_per_L_kPa_mm.round(2))),
        ('tab:res-flow: dp (0.17/0.83/1.7/3.3 bar)', tuple(o.dp_total_bar.round(2))),
        ('Results: residence time 30/150/300 (9.6/1.92/0.96 ms)', tuple(o.loc[[30, 150, 300], 'residence_time_ms'].round(2))),
        ('Results: wall shear stress, centre of side wall, 300 (751 Pa)', o.loc[300, 'tau_wall_centre_Pa']),
        ('Results: centre of side wall / mean wall shear (about 1.35)',
         o.loc[300, 'tau_wall_centre_Pa'] / o.loc[300, 'tau_wall_mean_Pa']),
        ('Results: chi for 10 to 25 um (0.25 to 0.63)', (loads.chi.min(), loads.chi.max())),
        ('Results: Re_p with U (7.8 to 48.8)', (loads.Re_p_mean.min(), loads.Re_p_mean.max())),
        ('Results: Re_p with U_max (16.4 to 102)', (loads.Re_p_max.min(), loads.Re_p_max.max())),
        ('Results: Re_p for 14.2 and 18.6 um (15.8 and 27.0)', tuple(particle_reynolds([14.2, 18.6]))),
        ('Results: shear stress at the edge of a centred cell (142 to 397 Pa)',
         (loads.sigma_centred_Pa.min(), loads.sigma_centred_Pa.max())),
        ('Results: centre at the stream boundary (293 to 604 Pa)',
         (loads.sigma_stream_edge_Pa.min(), loads.sigma_stream_edge_Pa.max())),
        ('Results: Ca (about 0.3 to 4.0)', (loads.Ca_min.min(), loads.Ca_max.max())),
        ('Results: focusing length (25 to 380 um)', (loads.L_f_min_um.min(), loads.L_f_max_um.max())),
    ]

    def fmt(v):
        if isinstance(v, tuple):
            return ', '.join(f'{float(x):.4g}' for x in v)
        return f'{float(v):.4g}'
    return '\n'.join(f'{k:<68s} {fmt(v)}' for k, v in lines)


def run(save=True):
    """Compute all tables. With save=True, write them and checks.txt to RESULTS_DIR."""
    out = dict(operating_points=table_operating_points(), hydraulic_resistance=table_resistances(),
               stream_width=table_stream_width())
    out['stream_width_by_ratio'] = table_stream_by_ratio(out['stream_width'])
    out['cell_loads'] = pd.concat([table_loads(q) for q in Q_TOTAL], ignore_index=True)
    out['motion_blur'] = table_motion_blur()
    loads300 = out['cell_loads'][out['cell_loads'].Q_total_ul_min == 300].reset_index(drop=True)
    out['checks'] = checks(out['operating_points'], out['hydraulic_resistance'], out['stream_width'],
                           out['stream_width_by_ratio'], loads300)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in out.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(RESULTS_DIR / f'{name}.csv', index=False, float_format='%.6g')
        (RESULTS_DIR / 'checks.txt').write_text(out['checks'] + '\n', encoding='utf-8')
    return out


if __name__ == '__main__':
    res = run()
    print(res['checks'])
    print('\nwritten to', RESULTS_DIR)
