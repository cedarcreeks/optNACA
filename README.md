# Aerodynamic optimization of NACA airfoils with Surrogate Modeling (SMT)

[![CI](https://github.com/cedarcreeks/optNACA/actions/workflows/ci.yml/badge.svg)](https://github.com/cedarcreeks/optNACA/actions/workflows/ci.yml)

A project that demonstrates the power of **surrogate modeling** to optimize
airfoils **without running CFD at every iteration**. It uses the
[SMT (Surrogate Modeling Toolbox)](https://smt.readthedocs.io/) library from
ISAE-SUPAERO together with XFOIL as the aerodynamic evaluator.

## Goal

Find the **4-digit NACA** airfoil parameters that **minimize Cd** for
**Cl ≈ 0.5** at **Re = 1e6**, using **EGO (Efficient Global Optimization)**
with **Kriging** models.

## Why surrogate modeling?

Evaluating an airfoil with CFD (here XFOIL) is expensive. Instead of sweeping
the design space with thousands of evaluations, a Kriging model is trained to
**approximate** the aerodynamic response and quantify its own uncertainty.
EGO exploits that model: at each iteration it proposes the point with the
highest *Expected Improvement* (balancing exploitation of good regions and
exploration of uncertain ones) and only then spends one real CFD evaluation.
Result: the optimum is reached with **tens** of evaluations instead of
**thousands**.

## Design space

| Variable | Meaning                              | Range          |
|----------|--------------------------------------|----------------|
| `m`      | Maximum camber (1st NACA digit/100)  | [0.01, 0.09]   |
| `p`      | Camber position (2nd digit/10)       | [0.10, 0.60]   |
| `t`      | Maximum thickness (last 2 digits/100)| [0.08, 0.20]   |
| `alpha`  | Angle of attack [degrees]            | [0.0, 8.0]     |

## Structure

```
airfoil_smt_optimization/
├── airfoil_utils.py          # NACA geometry + XFOIL evaluation (shared)
├── _xfoil_worker.py          # Single-evaluation XFOIL worker (run in a subprocess)
├── 01_generate_dataset.py    # Initial DOE with LHS + XFOIL -> data/airfoil_dataset.csv
├── 02_train_surrogate.py     # Trains Kriging (Cl, Cd) + LOO cross-validation
├── 03_ego_optimization.py    # Optimization with EGO (30 infill points)
├── 04_visualization.py       # Generates the 4 figures in figures/
├── notebook.ipynb            # Integrated narrative notebook (main deliverable)
├── data/
│   └── airfoil_dataset.csv   # Dataset generated in step 1
├── figures/                  # Figures generated in step 4
└── requirements.txt
```

## Installation

Runs on **macOS, Linux and Windows** with Python 3.10–3.14. XFOIL compiles a
Fortran binary, so you first need a toolchain (`cmake` + `gfortran`):

| OS | Install the toolchain |
|----|-----------------------|
| **macOS** | `brew install cmake gcc` (gcc provides gfortran) |
| **Linux (Debian/Ubuntu)** | `sudo apt-get install cmake gfortran python3-venv` |
| **Linux (Fedora)** | `sudo dnf install cmake gcc-gfortran` |
| **Windows** | `conda install -c conda-forge cmake fortran-compiler` — or use **WSL2** and follow the Linux steps |

Then create the environment and install everything with the helper script:

```bash
./setup.sh           # macOS / Linux
```
```powershell
.\setup.ps1          # Windows (PowerShell)
```

The script creates a virtual environment, installs the dependencies (building
XFOIL from source) and verifies the import. To do it by hand instead:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note on the `xfoil` package.** `requirements.txt` pulls `xfoil` from its
> GitHub source, **not** PyPI. The PyPI sdist is broken (it ships without
> `CMakeLists.txt`), so `pip install xfoil` fails on every platform and Python
> version. The GitHub source builds on **Python 3.10–3.14**. The rest of the
> pipeline (SMT: LHS, Kriging, EGO) is pure Python; only the XFOIL evaluations
> need the `xfoil` package.
>
> **Windows note.** Getting a native Fortran compiler on Windows is the only
> fiddly part; conda-forge's `fortran-compiler` is the smoothest route. If you
> hit a wall, **WSL2** lets you use the Linux path with zero friction.

## Usage

Run the scripts in order:

```bash
python 01_generate_dataset.py     # ~30 s: generates 80 points with XFOIL
python 02_train_surrogate.py      # trains Kriging and validates (LOO)
python 03_ego_optimization.py     # EGO: 30 additional CFD evaluations
python 04_visualization.py        # generates the 4 figures
```

Or open the integrated notebook directly:

```bash
jupyter notebook notebook.ipynb
```

## Tests

The pure-Python parts (NACA geometry, objective/penalty math, designation, and
the `eval_xfoil` subprocess orchestration) are covered by a test suite that runs
**without XFOIL installed**:

```bash
pip install -r requirements-dev.txt
pytest
```

## Results

A representative run finds:

| Quantity | Value |
|----------|-------|
| Optimal airfoil | **NACA 2508** (`m=0.025`, `p=0.53`, `t=0.080`, `alpha≈1.4°`) |
| Lift coefficient | `Cl ≈ 0.495` (inside the target band 0.5 ± 0.05) |
| Drag coefficient | `Cd ≈ 0.00425` |
| Real CFD evaluations | **110** = 80 (initial DOE) + 30 (EGO infill) |
| Equivalent grid search | 10⁴ evaluations (10 points per dimension, 4 dims) |
| Speed-up | **~90×** fewer XFOIL evaluations |

> The exact optimum can shift by a digit (e.g. NACA 2508 vs 3508) between XFOIL
> builds / platforms, because XFOIL's viscous coefficients differ slightly across
> versions and EGO settles on near-equivalent designs. The takeaway — a
> low-drag airfoil at `Cl ≈ 0.5` found with ~110 CFD calls instead of thousands —
> is reproducible.

## Generated figures

1. **Response surface** of the surrogate `Cd(t, alpha)` with the Kriging
   uncertainty as contours and the training points.
2. **Convergence curve** of EGO compared against an equivalent random search.
3. **Optimal airfoil geometry** with its parameters and coefficients.
4. **Surrogate validation**: predicted vs real (Leave-One-Out) with R² and RMSE.

## Robust XFOIL evaluation

XFOIL wraps a Fortran core with global state, so consecutive evaluations in the
same process can interfere and make borderline geometries fail to converge. To
avoid this, **each evaluation runs in a fresh subprocess** (`_xfoil_worker.py`),
which keeps the solver state clean and the results order-independent. If the
default settings do not converge a given airfoil, `eval_xfoil` escalates through
several increasingly robust **convergence recipes** (finer panelling, more
iterations, alpha continuation) until one succeeds. In practice the full 80-point
DOE converges **80/80**. Should a geometry ever still fail, it is handled
gracefully (dropped from the dataset, penalized with `Cd=9999` during EGO) so the
pipeline never raises.

## Reproducibility

The whole chain uses a fixed seed (`random_state=42`): LHS sampling, Kriging
training and the EGO optimizer. Because every XFOIL call is isolated in its own
process, results do not depend on evaluation order.

## Design notes & gotchas

Non-obvious things learned while building this, collected so you don't have to
rediscover them.

### SMT (Surrogate Modeling Toolbox) 2.x API

- The random-seed option is **`seed`**, not `random_state` — for `LHS`, `KRG`
  and `EGO` alike. Passing `random_state` raises `Option ... has not been declared`.
- **`EGO` has no `xlimits` argument.** It reads the design-space bounds from its
  surrogate. You must build the surrogate with a design space:
  `KRG(..., design_space=DesignSpace(XLIMITS))`; EGO then picks the bounds up
  from `surrogate.design_space`.
- Kriging kernel is selected with `corr="matern52"` (Matérn 5/2). Valid values:
  `pow_exp, abs_exp, squar_exp, squar_sin_exp, matern52, matern32`.
- `EGO` reuses the initial DOE: pass it as `xdoe`/`ydoe` and it is **not**
  re-evaluated. EGO only spends new XFOIL calls on its infill points (30 here),
  so the total CFD budget is `len(DOE) + n_iter`, not more.

### XFOIL

- **The PyPI `xfoil` package is broken** — its source distribution ships without
  `CMakeLists.txt`, so `pip install xfoil` fails on *every* Python version. Install
  from the GitHub source instead (already pinned in `requirements.txt`). The source
  build works on Python 3.10–3.14.
- **XFOIL keeps global state in Fortran COMMON blocks.** Creating a new `XFoil()`
  object does not fully reset it, so the boundary layer from one airfoil can break
  the next solve. That is why each evaluation runs in its own subprocess (see the
  *Robust XFOIL evaluation* section). Without isolation, which points fail depends
  on evaluation order — a nasty, irreproducible trap.
- The Python class is **`XFoil`** (capital F lower), not `XFOIL`. `xf.a(alpha)`
  returns `(Cl, Cd, Cm, Cp)` and yields `NaN` on non-convergence.
- Viscous convergence is genuinely fragile for thin / highly-cambered sections.
  No single setting converges everything; the recipe escalation (panel count,
  iteration cap, alpha continuation) is what gets the DOE to 80/80.
- Flow conditions are fixed: `Re = 1e6`, `Ma = 0`, `n_crit` at the default (9).

### NACA geometry

- The classic 4-digit thickness formula uses the `-0.1015` coefficient, which
  leaves a **finite-thickness (open) trailing edge**. Use `-0.1036` if you need a
  closed TE.
- Because the thickness is offset *perpendicular* to a sloped camber line, the
  `x` coordinate can exceed 1.0 by a tiny amount (`<1e-3`) right at the trailing
  edge. This is geometrically correct, not a bug; XFOIL repanels anyway. (A unit
  test pins this tolerance.)

### Optimization formulation

- Two **independent** Kriging models are trained, one for `Cl` and one for `Cd`,
  rather than a single multi-output model — simpler, and each fits its own
  hyperparameters.
- The `Cl ≈ 0.5` requirement is a **soft constraint folded into the objective**:
  `f = Cd + 1000 · max(0, |Cl − 0.5| − 0.05)`. The `max()` introduces a kink at
  the band edges (not perfectly smooth for Kriging), but it works well in practice
  and keeps the problem single-objective for EGO.

## License

This project's source code is released under the **MIT License** (see `LICENSE`).

It relies on external tools that are installed separately and keep their own
licenses — notably **XFOIL, which is GPL v2**. This repository does not bundle
or redistribute XFOIL; it is compiled on your machine and called as an external
program. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the full
breakdown.
