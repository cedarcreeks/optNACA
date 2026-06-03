"""
Step 1 - Initial dataset generation.

Samples 80 4-digit NACA airfoils in the design space using Latin Hypercube
Sampling (LHS) from SMT, evaluates them with XFOIL at Re=1e6 and stores the
aerodynamic coefficients in data/airfoil_dataset.csv.

This dataset is the "DOE" (Design of Experiments) that feeds the surrogate and
the EGO optimizer in the following steps.
"""

import os
import time

import numpy as np
import pandas as pd
from smt.sampling_methods import LHS

from airfoil_utils import XLIMITS, RANDOM_STATE, REYNOLDS, eval_xfoil

# Number of initial DOE points.
N_SAMPLES = 80
OUTPUT_CSV = os.path.join("data", "airfoil_dataset.csv")


def main():
    t0 = time.time()

    # LHS spreads the 80 points uniformly across the 4 dimensions, covering the
    # space better than purely random sampling.
    sampling = LHS(xlimits=XLIMITS, seed=RANDOM_STATE)
    x_doe = sampling(N_SAMPLES)

    records = []
    failures = 0

    for i, (m, p, t, alpha) in enumerate(x_doe):
        result = eval_xfoil(m, p, t, alpha, reynolds=REYNOLDS)
        if result is None:
            # XFOIL did not converge for this airfoil/angle: discard it.
            failures += 1
            print(f"[{i + 1:3d}/{N_SAMPLES}] did not converge "
                  f"(m={m:.3f} p={p:.3f} t={t:.3f} a={alpha:.2f})")
            continue

        cl, cd, cm = result
        records.append([m, p, t, alpha, cl, cd, cm])
        print(f"[{i + 1:3d}/{N_SAMPLES}] Cl={cl:+.4f} Cd={cd:.5f} Cm={cm:+.4f}")

    # A surrogate needs enough samples to fit. Fail loudly rather than write a
    # dataset that would make the next steps crash on a tiny/empty design table.
    if len(records) < 10:
        raise SystemExit(
            f"Only {len(records)} XFOIL evaluations converged (need >= 10). "
            "Check the XFOIL installation or widen the design space."
        )

    # Save the cleaned dataset.
    columns = ["m", "p", "t", "alpha", "Cl", "Cd", "Cm"]
    df = pd.DataFrame(records, columns=columns)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    dt = time.time() - t0
    print("-" * 60)
    print(f"Points requested   : {N_SAMPLES}")
    print(f"Converged          : {len(records)}")
    print(f"Failures (dropped) : {failures}")
    print(f"Dataset saved to   : {OUTPUT_CSV}")
    print(f"Total time         : {dt:.1f} s")


if __name__ == "__main__":
    main()
