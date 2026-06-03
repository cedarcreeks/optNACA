"""
Step 3 - Optimization with EGO (Efficient Global Optimization).

EGO uses the Kriging surrogate to decide, at each iteration, which point of the
design space to evaluate next (Expected Improvement criterion). XFOIL is called
only there. This finds the optimum with very few real "CFD" evaluations.

Objective: minimize Cd for Cl ~ 0.5.
Soft constraint handled as a penalty:
    f = Cd + 1000 * max(0, |Cl - 0.5| - 0.05)

Budget: 30 infill points (each = 1 XFOIL evaluation).
"""

import os
import pickle

import numpy as np
import pandas as pd
from smt.applications import EGO
from smt.surrogate_models import KRG
from smt.design_space import DesignSpace

from airfoil_utils import (
    XLIMITS, RANDOM_STATE, CL_TARGET, CL_TOL, PENALTY, CD_FAIL,
    eval_xfoil, penalized_objective, naca_designation,
)

DATASET_CSV = os.path.join("data", "airfoil_dataset.csv")
HISTORY_NPZ = os.path.join("data", "ego_history.npz")
RESULT_PKL = os.path.join("data", "ego_result.pkl")

N_INFILL = 30          # extra XFOIL evaluations spent by EGO

# Log of every evaluation done by the objective (in call order).
# Lets us reconstruct the convergence curve with the real Cd of each point.
_log = {"x": [], "cl": [], "cd": [], "f": []}


def objective_value(m, p, t, alpha):
    """Evaluate a point with XFOIL and return (Cl, Cd, f)."""
    result = eval_xfoil(m, p, t, alpha)
    if result is None:
        # Robust handling: if XFOIL fails, penalize hard and keep going.
        cl, cd = 0.0, CD_FAIL
    else:
        cl, cd, _ = result
    f = penalized_objective(cl, cd)
    return cl, cd, f


def objective(x):
    """
    Objective function consumed by EGO. Takes an (n, 4) array and returns (n, 1).
    Records each evaluation in _log for the convergence curve.
    """
    n = x.shape[0]
    y = np.zeros((n, 1))
    for i in range(n):
        m, p, t, alpha = x[i]
        cl, cd, f = objective_value(m, p, t, alpha)
        y[i, 0] = f
        _log["x"].append([m, p, t, alpha])
        _log["cl"].append(cl)
        _log["cd"].append(cd)
        _log["f"].append(f)
    return y


def main():
    if not os.path.exists(DATASET_CSV):
        raise SystemExit(
            f"Dataset not found: {DATASET_CSV}. Run 01_generate_dataset.py first."
        )
    # Initial DOE = dataset from step 1. EGO reuses it without re-evaluating it.
    df = pd.read_csv(DATASET_CSV)
    if len(df) < 10:
        raise SystemExit(
            f"Dataset has only {len(df)} points (need >= 10). "
            "Re-run 01_generate_dataset.py."
        )
    x_doe = df[["m", "p", "t", "alpha"]].values
    cl_doe = df["Cl"].values
    cd_doe = df["Cd"].values
    f_doe = cd_doe + PENALTY * np.maximum(
        0.0, np.abs(cl_doe - CL_TARGET) - CL_TOL
    )
    y_doe = f_doe.reshape(-1, 1)
    print(f"Initial DOE: {x_doe.shape[0]} points. Best initial f = "
          f"{f_doe.min():.5f}")

    # EGO with a Matern 5/2 Kriging surrogate and Expected Improvement criterion.
    # EGO reads the design bounds from the surrogate's design space.
    design_space = DesignSpace(XLIMITS)
    ego = EGO(
        n_iter=N_INFILL,
        criterion="EI",
        xdoe=x_doe,
        ydoe=y_doe,
        surrogate=KRG(corr="matern52", print_global=False,
                      design_space=design_space, seed=RANDOM_STATE),
        seed=RANDOM_STATE,
    )

    print(f"Running EGO with {N_INFILL} infill points (XFOIL on each)...")
    x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=objective)

    # Real coefficients of the optimum (last chunk of _log corresponds to EGO).
    m_o, p_o, t_o, a_o = x_opt
    cl_o, cd_o, f_o = objective_value(m_o, p_o, t_o, a_o)
    desig = naca_designation(m_o, p_o, t_o)

    # --- EGO convergence curve: best feasible Cd vs number of evaluations ---
    # Use the actual number of logged infill evaluations (robust if EGO
    # evaluates a different count than requested).
    infill_f = np.array(_log["f"])
    infill_cd = np.array(_log["cd"])
    n_eval = len(infill_f)

    best_f = f_doe.min()
    best_cd = cd_doe[np.argmin(f_doe)]
    conv_cd = []
    for k in range(n_eval):
        if infill_f[k] < best_f:
            best_f = infill_f[k]
            best_cd = infill_cd[k]
        conv_cd.append(best_cd)
    conv_cd = np.array(conv_cd)

    # --- Random-search baseline: same points, shuffled order ---
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(n_eval)
    rand_f = infill_f[perm]
    rand_cd = infill_cd[perm]
    best_f_r = f_doe.min()
    best_cd_r = cd_doe[np.argmin(f_doe)]
    conv_cd_rand = []
    for k in range(n_eval):
        if rand_f[k] < best_f_r:
            best_f_r = rand_f[k]
            best_cd_r = rand_cd[k]
        conv_cd_rand.append(best_cd_r)
    conv_cd_rand = np.array(conv_cd_rand)

    print("=" * 60)
    print(f"Optimal airfoil: {desig}")
    print(f"  m = {m_o:.4f}   p = {p_o:.4f}   t = {t_o:.4f}   "
          f"alpha = {a_o:.3f} deg")
    print(f"  Cl = {cl_o:+.4f}   Cd = {cd_o:.6f}   f = {f_o:.6f}")
    print(f"XFOIL evaluations: {x_doe.shape[0]} (DOE) + {N_INFILL} (EGO) "
          f"= {x_doe.shape[0] + N_INFILL}")
    print("=" * 60)

    # Save everything for visualization.
    np.savez(
        HISTORY_NPZ,
        conv_cd=conv_cd,
        conv_cd_rand=conv_cd_rand,
        infill_x=np.array(_log["x"][:N_INFILL]),
        infill_cl=np.array(_log["cl"][:N_INFILL]),
        infill_cd=infill_cd,
        infill_f=infill_f,
        x_doe=x_doe,
        n_doe=x_doe.shape[0],
    )
    with open(RESULT_PKL, "wb") as f:
        pickle.dump(
            {
                "x_opt": x_opt, "designation": desig,
                "m": m_o, "p": p_o, "t": t_o, "alpha": a_o,
                "Cl": cl_o, "Cd": cd_o, "f": f_o,
                "n_doe": x_doe.shape[0], "n_infill": N_INFILL,
            },
            f,
        )
    print(f"History saved to : {HISTORY_NPZ}")
    print(f"Result saved to  : {RESULT_PKL}")


if __name__ == "__main__":
    main()
