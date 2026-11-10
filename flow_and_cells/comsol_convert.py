"""Convert a COMSOL particle export (Spreadsheet, one row per particle, columns per time) into a
compact long table: one row per particle and time, SI units except positions in mm.
Usage: python comsol_convert.py input.txt output.csv.gz
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UNIT = {'N': 1.0, 'mN': 1e-3, 'µN': 1e-6, 'uN': 1e-6, 'nN': 1e-9, 'pN': 1e-12, 'm/s': 1.0, 'mm/s': 1e-3,
        '1/s': 1.0, 'Pa': 1.0, 'kPa': 1e3, 'bar': 1e5, 'mm': 1.0, 'm': 1e3, 'um': 1e-3, 'µm': 1e-3, '': 1.0}
NAMES = {'qx': 'qx', 'qy': 'qy', 'qz': 'qz', 'fpt.V': 'v_p', 'spf.U': 'u_f', 'spf.sr': 'sr', 'vy': 'vy', 'p': 'p',
         'fpt.df1.FDx': 'FDx', 'fpt.df1.FDy': 'FDy', 'fpt.df1.FDz': 'FDz', 'fpt.Ftx': 'Ftx', 'fpt.Fty': 'Fty',
         'fpt.Ftz': 'Ftz', 'fpt.lf1.FLx': 'FLx', 'fpt.lf1.FLy': 'FLy', 'fpt.lf1.FLz': 'FLz'}


def convert(path):
    header = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.startswith('%'):
                break
            header = line
    names = re.findall(r'[^\s(]+\s*\([^)]*\)\s*@\s*t=[-+0-9.eE]+|Index', header.lstrip('%'))
    data = pd.read_csv(path, sep=r'\s+', comment='%', header=None, engine='c').to_numpy(float)
    assert data.shape[1] == len(names), (data.shape, len(names))
    idx = data[:, 0].astype(int)
    pat = re.compile(r'([^\s(]+)\s*\(([^)]*)\)\s*@\s*t=([-+0-9.eE]+)')
    cols = {}
    for j, n in enumerate(names[1:], start=1):
        m = pat.match(n)
        if not m:
            continue
        expr, unit, t = m.group(1), m.group(2), float(m.group(3))
        key = NAMES.get(expr)
        if key is None:
            continue
        cols.setdefault(t, {})
        if key in cols[t]:          # duplicated column (e.g. qy exported twice)
            continue
        cols[t][key] = data[:, j] * UNIT.get(unit, 1.0)
    frames = []
    for t in sorted(cols):
        df = pd.DataFrame(cols[t])
        df.insert(0, 't', t)
        df.insert(0, 'particle', idx)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=['qx', 'qy']).sort_values(['particle', 't']).reset_index(drop=True)
    return out


if __name__ == '__main__':
    out = convert(sys.argv[1])
    out.to_csv(sys.argv[2], index=False, float_format='%.7g')
    print(out.shape, 'particles:', out.particle.nunique(), 'columns:', list(out.columns))
