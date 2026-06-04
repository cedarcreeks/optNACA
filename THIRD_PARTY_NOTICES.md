# Third-party notices

This project's own source code is released under the MIT License (see `LICENSE`).
It depends on external packages that are **installed separately** (via `pip` /
the system package manager) and are **not redistributed** as part of this
repository. Each retains its own license, summarized below.

| Component | Role | License |
|-----------|------|---------|
| **XFOIL** (Mark Drela, MIT/MITLL) | Aerodynamic solver called by `eval_xfoil` | **GPL v2** |
| `xfoil` (DARcorporation) | Python bindings that build/wrap XFOIL | GPL v2 (wraps XFOIL) |
| SMT — Surrogate Modeling Toolbox | LHS, Kriging, EGO | BSD-3-Clause |
| NumPy | Numerics | BSD-3-Clause |
| SciPy | Numerics | BSD-3-Clause |
| pandas | Dataset handling | BSD-3-Clause |
| Matplotlib | Plotting | Matplotlib (BSD-style / PSF-based) |

## Important note on XFOIL and the GPL

**XFOIL is licensed under the GPL v2.** This repository does not include or
redistribute XFOIL or its Python bindings; they are downloaded and compiled on
the user's machine at install time, and XFOIL is invoked as an external program
(in a subprocess). The MIT-licensed code in this repository therefore stands on
its own.

However, if you **redistribute a bundle that includes XFOIL** (for example a
Docker image or an archive shipping the compiled `xfoil` package), that combined
distribution is subject to the GPL for the XFOIL portion. Keep XFOIL as a
separately-installed dependency to avoid this.

Nothing here is legal advice; consult each project's full license text for the
authoritative terms.
