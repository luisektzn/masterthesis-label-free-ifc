# Flow model, COMSOL simulation and tracked cells

Code for the flow and cell-dynamics part of the master thesis *Towards High-Throughput Label-Free Imaging Flow
Cytometry: Hydrodynamic Focusing, Cell Dynamics and ONN-Compatible Classification* (Heidelberg University, 2026).

The code computes the analytical flow quantities of the 40 × 40 µm focusing channel, evaluates the COMSOL simulation
at 300 µL/min (flow field, particle trajectories, forces, particle acceleration), estimates the viscous stress on
HeLa and NIH3T3 cells from the simulated paths and the measured cell diameters and velocities, and draws the figures
of this part of the thesis.

## Notebooks

| Notebook | Content | Thesis |
|---|---|---|
| `01_flow_theory` | Velocity profile, hydraulic resistance, Reynolds number, pressure drop, stream width (Lee et al.), shear stress, capillary number, focusing length, motion blur | Results, Sections `sec:res-stream`, `sec:res-flow`, `sec:res-loads` |
| `02_measured_summary` | Median diameter, velocity and deformation of the tracked cells per cell line and flow rate (bright field and Ph2 pooled) | input for notebook 03 |
| `03_comsol_analysis` | COMSOL cut lines, particle trajectories, forces, particle acceleration, shear stress along the simulated paths | Results, Sections `sec:res-flow`, `sec:res-particles`, `sec:res-loads`, `sec:res-position`; Discussion |
| `04_figures` | All figures of this part, under their thesis file names (thesis and talk version) | Results (chip, flow and cell sections); Appendix |

Notebook 01 is independent. Notebooks 03 and 04 need the output of notebook 02 and the COMSOL exports (see
`comsol/README.md`). The deformation, rotation and lateral-position figures also need the tables of the cell-tracking
run (`cells_primary.csv`, `detection_morphology.csv`) and `lateral_position_tracks.csv`.

Modules:

* `flow_theory.py`: analytical model (equations of the Background chapter, estimates of Methods) and the tables of
  notebook 01. `checks.txt` lists every value of the thesis text that the model reproduces.
* `measured_summary.py`: per-condition summary of the tracked cells (`cells_primary.csv` of the tracking run).
* `comsol_analysis.py`: evaluation of the COMSOL exports.
* `flow_figures.py`: all figures of notebook 04 (one function per figure, registered with its thesis file name,
  caption and inputs).
* `thesis_style.py`: figure style (fonts, sizes, colours).
* `comsol_convert.py`: converts a COMSOL particle export (one row per particle, columns per time) into a long table
  (one row per particle and time step).

## Models in short

* Fully developed flow in the rectangular channel: series solution of Mortensen et al. (2005) with 200 odd terms.
  The hydraulic resistance is computed from the full series. The approximation with the factor 0.630 is only
  reported for comparison.
* Stream width: model of Lee et al. (2006), Eqs. `hydrodynamic_focusing_width` and `velocity_ratio_gamma`, solved by
  bisection.
* Shear stress in the mid-plane: Gerum et al. (2022), Eq. 4. For a cell, the stress is evaluated at its outer edge,
  at the distance |x| + d/2 from the axis, in the cell-free flow.
* Extensional stress along the simulated paths: 2η|∂v/∂y| at the particle centre.
* Capillary number Ca = ηγ̇/G with G = E_app/3 and E_app = 0.3 to 1.5 kPa (values for HL60 and SW480 cells).

All estimates assume an undisturbed, cell-free flow field and are approximate for cells with a diameter of about half
the channel width.

## Settings

| Setting | Where | Default |
|---|---|---|
| results folder | environment variable `FLOW_RESULTS` | `results/` |
| COMSOL exports | environment variable `COMSOL_DIR` | `comsol/` |
| tracking tables (`cells_primary.csv`, `detection_morphology.csv`) | environment variable `CELL_TABLES`, `TABLE_DIR` in notebook 02 or `F.CFG.v4_tables` in notebook 04 | `tables/` |
| lateral position per track | environment variable `LATERAL_CSV` or `F.CFG.lateral_csv` | not set |
| static deformation reference | environment variable `STATIC_CSV` or `F.CFG.static_csv` | not set |
| assumed imaging position | `comsol_analysis.S_IMG_MM` | 1.5 mm downstream of the channel entrance |

## Installation

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```
