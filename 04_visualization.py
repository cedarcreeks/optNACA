"""
Step 4 - Results visualization.

Generates and saves the four study figures into figures/:
  1. Surrogate response surface Cd(t, alpha) with uncertainty.
  2. EGO convergence curve vs random search.
  3. Geometry of the optimal airfoil found.
  4. Surrogate validation (Leave-One-Out, R^2 and RMSE).
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from airfoil_utils import naca4

DATASET_CSV = os.path.join("data", "airfoil_dataset.csv")
MODELS_PKL = os.path.join("data", "surrogates.pkl")
LOO_NPZ = os.path.join("data", "loo_predictions.npz")
HISTORY_NPZ = os.path.join("data", "ego_history.npz")
RESULT_PKL = os.path.join("data", "ego_result.pkl")
FIG_DIR = "figures"


def load_all():
    df = pd.read_csv(DATASET_CSV)
    with open(MODELS_PKL, "rb") as f:
        models = pickle.load(f)
    loo = dict(np.load(LOO_NPZ))
    hist = dict(np.load(HISTORY_NPZ))
    with open(RESULT_PKL, "rb") as f:
        result = pickle.load(f)
    return df, models, loo, hist, result


def plot_response_surface(df, models, result):
    """
    Response surface Cd(t, alpha) with m and p fixed at the optimum.
    Color is the predicted Cd; the semi-transparent contours show the Kriging
    standard deviation (model uncertainty).
    """
    sm_cd = models["Cd"]
    m_o, p_o = result["m"], result["p"]

    t_grid = np.linspace(0.08, 0.20, 60)
    a_grid = np.linspace(0.0, 8.0, 60)
    T, A = np.meshgrid(t_grid, a_grid)
    pts = np.column_stack([
        np.full(T.size, m_o), np.full(T.size, p_o), T.ravel(), A.ravel()
    ])

    cd_pred = sm_cd.predict_values(pts).reshape(T.shape)
    cd_std = np.sqrt(np.abs(sm_cd.predict_variances(pts))).reshape(T.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(T, A, cd_pred, levels=30, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Predicted Cd (Kriging)")

    # Uncertainty as semi-transparent overlaid contours.
    cu = ax.contour(T, A, cd_std, levels=6, colors="white", alpha=0.5,
                    linewidths=1.0)
    ax.clabel(cu, inline=True, fontsize=7, fmt="std=%.4f")

    # Training points projected onto (t, alpha).
    ax.scatter(df["t"], df["alpha"], c="red", s=18, edgecolors="k",
               linewidths=0.4, label="Training points")
    ax.scatter([result["t"]], [result["alpha"]], marker="*", s=320,
               c="gold", edgecolors="k", linewidths=0.8, label="EGO optimum")

    ax.set_xlabel("Thickness t")
    ax.set_ylabel("Angle of attack alpha [deg]")
    ax.set_title(f"Cd response surface  (m={m_o:.3f}, p={p_o:.3f} fixed)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_response_surface.png"), dpi=150)
    plt.close(fig)


def plot_convergence(hist):
    """Best Cd found vs number of CFD evaluations: EGO vs random."""
    conv = hist["conv_cd"]
    conv_rand = hist["conv_cd_rand"]
    evals = np.arange(1, len(conv) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(evals, conv, "-o", color="C0", label="EGO (Expected Improvement)")
    ax.plot(evals, conv_rand, "--s", color="C3", markersize=4,
            label="Random search (same points)")
    ax.set_xlabel("Number of CFD evaluations (infill)")
    ax.set_ylabel("Best feasible Cd found")
    ax.set_title("Convergence: EGO vs random search")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_convergence.png"), dpi=150)
    plt.close(fig)


def plot_optimal_airfoil(result):
    """Optimal airfoil geometry annotated with its parameters and coefficients."""
    m_o, p_o, t_o = result["m"], result["p"], result["t"]
    x, y = naca4(m_o, p_o, t_o)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(x, y, "-", color="C0", linewidth=1.8)
    ax.fill(x, y, color="C0", alpha=0.10)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    text = (
        f"{result['designation']}\n"
        f"m = {m_o:.4f}\n"
        f"p = {p_o:.4f}\n"
        f"t = {t_o:.4f}\n"
        f"alpha = {result['alpha']:.2f} deg\n"
        f"Cl = {result['Cl']:+.4f}\n"
        f"Cd = {result['Cd']:.5f}"
    )
    ax.text(0.55, -0.12, text, fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(f"Optimal airfoil: {result['designation']}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_optimal_airfoil.png"), dpi=150)
    plt.close(fig)


def plot_validation(loo):
    """Predicted vs real scatter (Leave-One-Out) for Cl and Cd."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, key, label in zip(
        axes, ["cl", "cd"], ["Cl", "Cd"]
    ):
        y_true = loo[f"{key}_true"]
        y_pred = loo[f"{key}_pred"]
        rmse = float(loo[f"rmse_{key}"])
        r2 = float(loo[f"r2_{key}"])

        ax.scatter(y_true, y_pred, s=22, c="C0", edgecolors="k",
                   linewidths=0.3, alpha=0.8)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", label="Ideal (y = x)")
        ax.set_xlabel(f"Real {label} (XFOIL)")
        ax.set_ylabel(f"Predicted {label} (Kriging)")
        ax.set_title(f"{label}:  R2 = {r2:.4f}   RMSE = {rmse:.5f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Surrogate validation (Leave-One-Out)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_surrogate_validation.png"), dpi=150)
    plt.close(fig)


def main():
    required = [DATASET_CSV, MODELS_PKL, LOO_NPZ, HISTORY_NPZ, RESULT_PKL]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise SystemExit(
            "Missing input files: " + ", ".join(missing) + ".\n"
            "Run 01_generate_dataset.py, 02_train_surrogate.py and "
            "03_ego_optimization.py first."
        )

    os.makedirs(FIG_DIR, exist_ok=True)
    df, models, loo, hist, result = load_all()

    plot_response_surface(df, models, result)
    plot_convergence(hist)
    plot_optimal_airfoil(result)
    plot_validation(loo)

    print(f"4 figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
