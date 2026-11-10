"""Summary of the tracked cells per cell line and total flow rate, bright field (BF) and Ph2 pooled.

Input: cells_primary.csv (or .csv.gz) of the cell-tracking run (one row per retained track, track medians).
Each retained track contributes its median. BF and Ph2 tracks of the same cell line and flow rate are pooled,
and the medians of BF and Ph2 alone are kept as extra columns. 600 µL/min is excluded (chip failure).

The output is read by comsol_analysis.py: the median diameter at 300 µL/min selects the simulated particle
diameter, and the median diameter and velocity of every condition enter the stress estimate and the passage
time from the channel entrance to the imaging position (Results sec:res-particles, Discussion).

Run
    python measured_summary.py --tables "<run folder>/tables"
or open 02_measured_summary.ipynb.

Output: results/measured/measured_summary.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get('FLOW_RESULTS', HERE / 'results')) / 'measured'
SUMMARY_CSV = RESULTS_DIR / 'measured_summary.csv'

# Folder with cells_primary.csv of the tracking run (environment variable CELL_TABLES or --tables)
TABLE_DIR = Path(os.environ.get('CELL_TABLES', 'tables'))

MODES = ['BF', 'Ph2']
EXCLUDED_FLOW_RATES = [600.0]                   # chip failure
METRICS = ['speed_m_s', 'equivalent_diameter_um', 'deformation']


def load_tracks(table_dir: Path = TABLE_DIR) -> pd.DataFrame:
    """Retained tracks (cells_primary.csv or .csv.gz) without the excluded flow rates."""
    for suffix in ('.csv', '.csv.gz'):
        path = Path(table_dir) / ('cells_primary' + suffix)
        if path.is_file():
            cells = pd.read_csv(path, low_memory=False)
            return cells.loc[~cells.flow_rate_ul_min.isin(EXCLUDED_FLOW_RATES)].copy()
    raise FileNotFoundError(f'cells_primary.csv not found in {table_dir}')


def _stats(v):
    v = pd.Series(v).dropna()
    if v.empty:
        return np.nan, np.nan, np.nan
    return v.median(), v.quantile(.25), v.quantile(.75)


def pooled_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per cell line and flow rate: number of tracks (all, BF, Ph2), and median, 25th and 75th
    percentile of the track medians (BF and Ph2 pooled), plus the median of each imaging mode alone."""
    rows = []
    for (ct, q), g in cells.groupby(['cell_type', 'flow_rate_ul_min']):
        row = dict(cell_type=ct, flow_rate_ul_min=q, n_tracks=len(g))
        for mode in MODES:
            row[f'n_{mode}'] = int(g.imaging_mode.eq(mode).sum())
        for m in METRICS:
            row[f'{m}_median'], row[f'{m}_q25'], row[f'{m}_q75'] = _stats(g[m])
            for mode in MODES:
                row[f'{m}_median_{mode}'] = _stats(g.loc[g.imaging_mode.eq(mode), m])[0]
        rows.append(row)
    return pd.DataFrame(rows)


def run(table_dir: Path = TABLE_DIR, save=True) -> pd.DataFrame:
    summary = pooled_summary(load_tracks(table_dir))
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(SUMMARY_CSV, index=False)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tables', type=Path, default=TABLE_DIR, help='folder with cells_primary.csv')
    a = p.parse_args()
    s = run(a.tables)
    cols = ['cell_type', 'flow_rate_ul_min', 'n_tracks', 'n_BF', 'n_Ph2', 'speed_m_s_median',
            'equivalent_diameter_um_median', 'deformation_median']
    print(s[cols].round(3).to_string(index=False))
    print('written to', SUMMARY_CSV)


if __name__ == '__main__':
    main()
