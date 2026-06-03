"""
Single-evaluation XFOIL worker, run as a subprocess by airfoil_utils.eval_xfoil.

Why a subprocess: the `xfoil` package wraps a Fortran core that keeps global
state in COMMON blocks. Creating a new ``XFoil()`` object does NOT fully reset
it, so the boundary layer left by one airfoil can break the convergence of the
next. Running every evaluation in a fresh process guarantees a clean solver
state, which makes results order-independent and reproducible.

Each invocation runs ONE convergence "recipe" and prints a JSON result
(``[Cl, Cd, Cm]`` or ``null``) on stdout. eval_xfoil escalates through the
recipes until one converges, so borderline geometries that the default recipe
cannot solve are caught by a stronger one.

Usage:
    python _xfoil_worker.py <m> <p> <t> <alpha> <reynolds> <mach> <recipe> <dir>
"""

import json
import sys

# Make airfoil_utils importable regardless of the caller's working directory.
sys.path.insert(0, sys.argv[-1])

import numpy as np
from airfoil_utils import naca4


def _is_valid(cl, cd):
    return np.isfinite(cl) and np.isfinite(cd) and cd > 0


def solve(m, p, t, alpha, reynolds, mach, recipe):
    """Run one recipe. Returns [Cl, Cd, Cm] or None."""
    from xfoil import XFoil
    from xfoil.model import Airfoil

    xf = XFoil()
    xf.print = False
    # A finer geometry (160 points/surface) already improves convergence.
    xf.airfoil = Airfoil(*naca4(m, p, t, n_points=160))

    # Recipe table: progressively more robust panelling / iteration / alpha ramp.
    ramp = None
    if recipe == 1:
        xf.repanel(); max_iter = 200
    elif recipe == 2:
        xf.repanel(n_nodes=240); max_iter = 300
    elif recipe == 3:
        max_iter = 300                       # no repanel
    elif recipe == 4:
        xf.repanel(); max_iter = 300; ramp = 0.5
    elif recipe == 5:
        xf.repanel(); max_iter = 500; ramp = 0.25
    else:
        return None

    xf.Re = reynolds
    xf.M = mach
    xf.max_iter = max_iter

    cl = cd = cm = float("nan")
    # Optional alpha continuation: walk up from 0 reusing the boundary layer.
    if ramp and alpha > 0:
        for ai in np.arange(ramp, alpha, ramp):
            xf.a(float(ai))
    cl, cd, cm, _ = xf.a(alpha)

    if not _is_valid(cl, cd):
        return None
    return [float(cl), float(cd), float(cm)]


def main():
    m, p, t, alpha, reynolds, mach = (float(v) for v in sys.argv[1:7])
    recipe = int(sys.argv[7])
    try:
        result = solve(m, p, t, alpha, reynolds, mach, recipe)
    except Exception:
        # Any solver-level failure on a degenerate geometry -> "did not converge".
        result = None
    print(json.dumps(result))


if __name__ == "__main__":
    main()
