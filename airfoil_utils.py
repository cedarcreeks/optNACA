"""
Shared utilities for the aerodynamic optimization with surrogate modeling.

Contains:
  - naca4(): generates the analytical geometry of a 4-digit NACA airfoil.
  - eval_xfoil(): evaluates an airfoil with XFOIL and returns (Cl, Cd, Cm).

These two blocks are the foundation of the whole project: the geometry is built
with the standard NACA formula and the "CFD" (actually a panel solver with a
viscous correction) is provided by XFOIL.
"""

import numpy as np

# Global seed for reproducibility across the whole project.
RANDOM_STATE = 42

# Fixed flight conditions for the study.
REYNOLDS = 1e6
MACH = 0.0

# Design space bounds [m, p, t, alpha].
# m: maximum camber (1st NACA digit / 100)
# p: position of maximum camber (2nd NACA digit / 10)
# t: maximum thickness (last two NACA digits / 100)
# alpha: angle of attack in degrees
XLIMITS = np.array(
    [
        [0.01, 0.09],  # m
        [0.10, 0.60],  # p
        [0.08, 0.20],  # t
        [0.0, 8.0],    # alpha
    ]
)
VAR_NAMES = ["m", "p", "t", "alpha"]

# Optimization target: minimize Cd while keeping Cl inside a band around CL_TARGET.
CL_TARGET = 0.5
CL_TOL = 0.05
PENALTY = 1000.0
# Drag value assigned when XFOIL does not converge (large, so EGO avoids it).
CD_FAIL = 9999.0


def penalized_objective(cl, cd):
    """
    Soft-constrained objective: minimize Cd, penalizing departures from the
    target lift band |Cl - CL_TARGET| > CL_TOL.

        f = Cd + PENALTY * max(0, |Cl - CL_TARGET| - CL_TOL)

    Scalar in, scalar out. The vectorized form is inlined where a whole DOE is
    scored at once.
    """
    return cd + PENALTY * max(0.0, abs(cl - CL_TARGET) - CL_TOL)


def naca_designation(m, p, t):
    """Convert (m, p, t) fractions into the 4-digit NACA designation string."""
    d1 = int(round(m * 100))
    d2 = int(round(p * 10))
    d34 = int(round(t * 100))
    return f"NACA {d1}{d2}{d34:02d}"


def naca4(m, p, t, n_points=120):
    """
    Generate the (x, y) coordinates of a 4-digit NACA airfoil.

    Parameters
    ----------
    m : float   maximum camber as a fraction of the chord (e.g. 0.02 -> 2%)
    p : float   chordwise position of maximum camber as a fraction of the chord
    t : float   maximum thickness as a fraction of the chord
    n_points : int  number of points per surface

    Returns
    -------
    x, y : np.ndarray  coordinates ordered for XFOIL
           (trailing edge -> upper surface -> leading edge -> lower surface ->
            trailing edge)
    """
    # Validate inputs so a bad call fails loudly here instead of producing a
    # garbage geometry that silently breaks XFOIL downstream.
    if t <= 0:
        raise ValueError(f"thickness t must be > 0, got {t}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"camber position p must be in (0, 1), got {p}")
    if n_points < 3:
        raise ValueError(f"n_points must be >= 3, got {n_points}")

    # Cosine point distribution: clusters nodes near the leading and trailing
    # edges, where the geometry changes quickly.
    beta = np.linspace(0.0, np.pi, n_points)
    x = (1.0 - np.cos(beta)) / 2.0

    # Thickness distribution (standard 4-digit NACA formula).
    # The -0.1015 coefficient leaves a finite-thickness trailing edge; use
    # -0.1036 to close it exactly. We keep the classic value.
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )

    # Mean camber line and its slope, defined piecewise before and after the
    # position of maximum camber p.
    yc = np.where(
        x < p,
        m / max(p ** 2, 1e-12) * (2 * p * x - x ** 2),
        m / max((1 - p) ** 2, 1e-12) * ((1 - 2 * p) + 2 * p * x - x ** 2),
    )
    dyc = np.where(
        x < p,
        2 * m / max(p ** 2, 1e-12) * (p - x),
        2 * m / max((1 - p) ** 2, 1e-12) * (p - x),
    )
    theta = np.arctan(dyc)

    # Offset the thickness perpendicular to the camber line.
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Order required by XFOIL: upper surface from the trailing edge to the
    # leading edge, then lower surface back to the trailing edge.
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return x_coords, y_coords


def eval_xfoil(m, p, t, alpha, reynolds=REYNOLDS, mach=MACH, max_iter=100):
    """
    Evaluate a NACA airfoil with XFOIL at a given angle of attack.

    Returns (Cl, Cd, Cm) if it converges, or None if XFOIL does not converge or
    fails. The xfoil import is done inside the function so the rest of the
    project can be imported even when XFOIL is not installed.
    """
    try:
        from xfoil import XFoil
        from xfoil.model import Airfoil
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'xfoil' package is not installed. Install it from source:\n"
            "  pip install git+https://github.com/DARcorporation/xfoil-python.git"
        ) from exc

    x, y = naca4(m, p, t)

    # The whole solve is guarded: a degenerate geometry can make XFOIL raise
    # instead of just returning NaN. We treat any such failure as "did not
    # converge" so callers never see an exception, only None.
    try:
        xf = XFoil()
        xf.print = False              # silence the solver output
        xf.airfoil = Airfoil(x, y)
        xf.repanel()                  # clean panel distribution -> better convergence
        xf.Re = reynolds
        xf.M = mach
        xf.max_iter = max_iter
        # xf.a(alpha) returns (cl, cd, cm, cp); NaN if it does not converge.
        cl, cd, cm, _ = xf.a(alpha)
    except Exception:
        return None

    if not np.isfinite(cl) or not np.isfinite(cd) or cd <= 0:
        return None
    return float(cl), float(cd), float(cm)
