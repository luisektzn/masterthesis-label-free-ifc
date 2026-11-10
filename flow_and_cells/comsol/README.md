# COMSOL exports

Put the COMSOL exports of the 300 µL/min half model here (or set the environment variable `COMSOL_DIR`):

| File | How it is made |
|---|---|
| `axis_300.txt` | COMSOL cut line on the channel axis at mid-height (symmetry plane), exported as text: x, y, z (mm), p (Pa), spf.U (m/s) |
| `velocity_cross_300.txt` | COMSOL cut line across the channel at the imaging position (s = 1.5 mm), mid-height: x, y, z (mm), spf.U (m/s), spf.sr (1/s) |
| `path_300_d15_v3.csv.gz`, `path_300_d20_v3.csv.gz` | particle tracking export (Spreadsheet, one row per particle) converted with `python comsol_convert.py <export>.txt path_300_d15_v3.csv.gz` |
