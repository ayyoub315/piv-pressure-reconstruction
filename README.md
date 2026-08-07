# PIV Pressure Reconstruction: PPE-MNLS vs. RPR-ODI

**Ayyoub Tabti — Turbulence Research Lab, University of Toronto**
**Supervisor: Prof. Pierre Sullivan**

Code accompanying the internship report comparing two PIV pressure-field
reconstruction methods — the Pressure Poisson Equation (PPE-MNLS) and
Regularized Parallel-Ray Omnidirectional Integration (RPR-ODI / OS-MODI) —
on JHTDB isotropic turbulence data (`isotropic1024coarse`).

The report itself lives on Overleaf and is **not** part of this repository.

## What this code does

1. Extracts velocity, pressure, and pressure-gradient fields from the JHTDB
   `isotropic1024coarse` dataset via the `givernylocal` client.
2. Solves the discrete pressure Poisson equation two ways — a sparse
   least-squares solve (`lsqr`) and an exact fast DCT solve — and compares
   both against the reference RPR-ODI (OS-MODI) solver.
3. Injects synthetic correlated noise (at the pressure-gradient level, and
   separately at the velocity level) to study how each method degrades.
4. Reproduces every figure and quantitative result reported in Sections 5–8.

## Solver history — why two PPE solvers coexist in this codebase

This is not an accident or leftover code — it reflects how the project
actually progressed, and the repo structure preserves that history rather
than hiding it:

| Report section | Script | PPE solver | Why |
|---|---|---|---|
| **Section 5** (deterministic results) | `experiments/section5_deterministic_lsqr.py` | `lsqr` (sparse) | The original solver used from the start of the internship. |
| **Section 6** (gradient noise) | `experiments/section6_gradient_noise_dct.py` | DCT (exact) | DCT solver adopted here onward — same accuracy, ~1000–6000× faster. |
| **Section 7** (velocity noise) | `experiments/section7_velocity_noise_dct.py` | DCT (exact) | — |
| **Section 8** (regularization) | `experiments/section8_regularization_and_solver_comparison.py` | `lsqr` **and** DCT, compared directly | Formal runtime/accuracy comparison that justifies the Section 6 switch, plus Tikhonov regularization study. |

Both solvers live side by side in `src/solvers.py`, fully documented, with
neither marked as "deprecated" — which one is authoritative for a given
figure depends on which section you're reproducing.

## Repository structure

```
src/                  Core, reusable code (imported by every experiment script)
  solvers.py            build_LN_bN, solve_PPE_MNLS (lsqr), solve_poisson_DCT,
                         solve_poisson_DCT_tikhonov, solve_OS_MODI
  noise.py               generate_correlated_noise (Section 6),
                         generate_correlated_noise_centered (Section 7)
  jhtdb_io.py            JHTDB connection + field-extraction helpers
  metrics.py             R_and_err, laplacian_nonperiodic, gradients_from_velocity

experiments/          One script per report section — run independently
  section5_deterministic_lsqr.py
  section6_gradient_noise_dct.py
  section7_velocity_noise_dct.py
  section8_regularization_and_solver_comparison.py

results/
  figures/              All generated .png figures (same filenames used by
                         \includegraphics in the report)
  data/                 Raw numerical results (.json) backing each figure/table

run_all.py             Runs every experiment script, in report order
environment.yml        Conda environment
.env.example            Template for the JHTDB auth token (copy to .env)
```

## Setup

```bash
conda env create -f environment.yml
conda activate piv-pressure-reconstruction
cp .env.example .env
# edit .env and paste your own JHTDB token
# (get one at https://turbulence.pha.jhu.edu/)
```

## Reproducing the results

Run everything, in report order:

```bash
python run_all.py
```

Or reproduce a single section (e.g. just the velocity-noise results of
Section 7):

```bash
python experiments/section7_velocity_noise_dct.py
```

Figures land in `results/figures/`, raw numbers in `results/data/`.

## Key findings this code supports

- **Theoretical equivalence, discrete divergence**: PPE-MNLS and RPR-ODI are
  continuously equivalent (Pryce et al., 2025); the two solvers agree to
  machine precision on synthetic MMS tests and closely on JHTDB data, with
  small differences traceable to discretization.
- **Computational efficiency**: the exact DCT solver is ~1000–6000× faster
  than `lsqr`, and RPR-ODI is separately measured at ~50× faster than the
  matrix-inversion approach at N=128 by bypassing the global L_N assembly.
- **Gradient noise**: both solvers degrade slowly and comparably up to 50%
  correlated gradient noise.
- **Velocity noise & differentiation amplification**: temporal
  differentiation (du/dt) amplifies velocity noise by ~1/(√2·dt) ≈ 353.55×;
  at just 2% velocity noise this yields a ~41% noise-to-signal ratio in
  du/dt, and spatial correlation with the DNS reference drops from ~0.994
  to ~0.45.
- **Core insight**: the pressure-reconstruction bottleneck in PIV traces to
  temporal differentiation of the velocity field, not to the choice of
  pressure solver.

## Requirements

- A JHTDB auth token (`JHTDB_AUTH_TOKEN`, see Setup above).
- The `osmodi` package for the RPR-ODI reference solver. If unavailable,
  scripts fall back gracefully and skip RPR-ODI-specific outputs.
