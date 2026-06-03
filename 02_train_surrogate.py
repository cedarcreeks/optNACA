"""
Step 2 - Surrogate model training (Kriging).

Trains two independent Kriging models with SMT (Matern 5/2 kernel):
  - surrogate_Cl: Cl(m, p, t, alpha)
  - surrogate_Cd: Cd(m, p, t, alpha)

Both are validated with Leave-One-Out cross-validation (RMSE and R^2) and
serialized to data/ so steps 3 and 4 can reuse them.
"""

import os
import pickle

import numpy as np
import pandas as pd
from smt.surrogate_models import KRG

from airfoil_utils import RANDOM_STATE

DATASET_CSV = os.path.join("data", "airfoil_dataset.csv")
MODELS_PKL = os.path.join("data", "surrogates.pkl")
LOO_NPZ = os.path.join("data", "loo_predictions.npz")


def build_krg():
    """Create a Kriging (KRG) model with a Matern 5/2 kernel."""
    return KRG(
        corr="matern52",
        poly="constant",
        theta0=[1e-2],
        print_global=False,
        random_state=RANDOM_STATE,
    )


def train_model(x, y):
    """Train a KRG on (x, y) and return it."""
    sm = build_krg()
    sm.set_training_values(x, y)
    sm.train()
    return sm


def leave_one_out(x, y):
    """
    Leave-One-Out validation: for each point, train on the remaining n-1 and
    predict the held-out one. Returns the vector of predictions.
    """
    n = x.shape[0]
    y_pred = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sm = train_model(x[mask], y[mask])
        y_pred[i] = sm.predict_values(x[i : i + 1])[0, 0]
    return y_pred


def metrics(y_true, y_pred):
    """Return (RMSE, R^2)."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return rmse, r2


def main():
    df = pd.read_csv(DATASET_CSV)
    x = df[["m", "p", "t", "alpha"]].values
    y_cl = df["Cl"].values
    y_cd = df["Cd"].values
    print(f"Dataset loaded: {x.shape[0]} points.")

    # Final models trained on all the data.
    print("Training surrogate_Cl ...")
    sm_cl = train_model(x, y_cl.reshape(-1, 1))
    print("Training surrogate_Cd ...")
    sm_cd = train_model(x, y_cd.reshape(-1, 1))

    # Leave-One-Out validation of both models.
    print("Leave-One-Out CV (Cl) ...")
    cl_loo = leave_one_out(x, y_cl.reshape(-1, 1))
    print("Leave-One-Out CV (Cd) ...")
    cd_loo = leave_one_out(x, y_cd.reshape(-1, 1))

    rmse_cl, r2_cl = metrics(y_cl, cl_loo)
    rmse_cd, r2_cd = metrics(y_cd, cd_loo)

    print("-" * 60)
    print(f"Cl : RMSE = {rmse_cl:.5f}   R2 = {r2_cl:.4f}")
    print(f"Cd : RMSE = {rmse_cd:.6f}   R2 = {r2_cd:.4f}")

    # Save models and LOO predictions for the visualization step.
    with open(MODELS_PKL, "wb") as f:
        pickle.dump({"Cl": sm_cl, "Cd": sm_cd}, f)
    np.savez(
        LOO_NPZ,
        cl_true=y_cl, cl_pred=cl_loo,
        cd_true=y_cd, cd_pred=cd_loo,
        rmse_cl=rmse_cl, r2_cl=r2_cl,
        rmse_cd=rmse_cd, r2_cd=r2_cd,
    )
    print(f"Models saved to          : {MODELS_PKL}")
    print(f"LOO predictions saved to : {LOO_NPZ}")


if __name__ == "__main__":
    main()
