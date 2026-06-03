# Aerodynamic optimization of NACA airfoils with Surrogate Modeling (SMT)

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

XFOIL compiles a Fortran binary, so you need `cmake` and `gfortran` first:

```bash
brew install cmake gcc        # macOS; gcc provides gfortran
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

Or use the helper script, which creates a virtual environment and installs
everything:

```bash
./setup.sh
```

> **Note on the `xfoil` package.** `requirements.txt` pulls `xfoil` from its
> GitHub source, **not** PyPI. The PyPI sdist is broken (it ships without
> `CMakeLists.txt`), so `pip install xfoil` fails on every Python version. The
> GitHub source builds fine on **Python 3.10–3.14** (verified). The rest of the
> pipeline (SMT: LHS, Kriging, EGO) is pure Python; only the XFOIL evaluations
> need the `xfoil` package.

## Usage

Run the scripts in order:

```bash
python 01_generate_dataset.py     # ~5 min: generates 80 points with XFOIL
python 02_train_surrogate.py      # trains Kriging and validates (LOO)
python 03_ego_optimization.py     # EGO: 30 additional CFD evaluations
python 04_visualization.py        # generates the 4 figures
```

Or open the integrated notebook directly:

```bash
jupyter notebook notebook.ipynb
```

## Generated figures

1. **Response surface** of the surrogate `Cd(t, alpha)` with the Kriging
   uncertainty as contours and the training points.
2. **Convergence curve** of EGO compared against an equivalent random search.
3. **Optimal airfoil geometry** with its parameters and coefficients.
4. **Surrogate validation**: predicted vs real (Leave-One-Out) with R² and RMSE.

## Reproducibility

The whole chain uses a fixed seed (`random_state=42`): LHS sampling, Kriging
training and the EGO optimizer. XFOIL convergence failures are handled
robustly (dropped from the dataset and penalized with `Cd=9999` during EGO).

## License

MIT.
