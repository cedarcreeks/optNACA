"""
Unit tests that run WITHOUT XFOIL installed.

They cover the deterministic, pure-Python parts of the project: NACA geometry,
the objective/penalty math, the designation helper, and the failure handling of
eval_xfoil (XFOIL is replaced by a fake module). Run with:  pytest
"""

import sys
import types

import numpy as np
import pytest

import airfoil_utils as au
from airfoil_utils import naca4, naca_designation, penalized_objective


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_naca4_shapes_and_bounds():
    x, y = naca4(0.02, 0.4, 0.12, n_points=120)
    assert x.shape == y.shape
    assert len(x) == 2 * 120 - 1            # surfaces share the leading-edge node
    # x stays in [0, 1] apart from a tiny (<1e-3) trailing-edge overshoot caused
    # by offsetting the thickness perpendicular to a sloped camber line.
    assert np.all(x >= -1e-3) and np.all(x <= 1.0 + 1e-3)
    # XFOIL ordering: starts and ends at the trailing edge (x ~ 1).
    assert x[0] == pytest.approx(1.0, abs=1e-3)
    assert x[-1] == pytest.approx(1.0, abs=1e-3)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))


def test_naca4_max_thickness_matches_t():
    # For a symmetric section the full thickness peaks at ~t.
    t = 0.12
    x, y = naca4(0.0 + 1e-9, 0.3, t, n_points=200)
    n = len(x) // 2
    upper = y[:n + 1][::-1]                 # leading -> trailing
    lower = y[n:]
    thickness = upper - lower
    assert thickness.max() == pytest.approx(t, abs=0.01)


def test_naca4_symmetric_is_antisymmetric():
    # m ~ 0 => camber line ~ 0 => upper surface mirrors lower.
    x, y = naca4(1e-9, 0.3, 0.10, n_points=120)
    n = len(x) // 2
    upper = y[:n]
    lower = y[n + 1:][::-1]
    assert np.allclose(upper, -lower, atol=1e-6)


@pytest.mark.parametrize("kwargs", [
    dict(m=0.02, p=0.4, t=0.0),            # zero thickness
    dict(m=0.02, p=0.4, t=-0.1),           # negative thickness
    dict(m=0.02, p=0.0, t=0.12),           # p at boundary
    dict(m=0.02, p=1.0, t=0.12),           # p at boundary
])
def test_naca4_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        naca4(**kwargs)


def test_naca4_rejects_too_few_points():
    with pytest.raises(ValueError):
        naca4(0.02, 0.4, 0.12, n_points=2)


# --------------------------------------------------------------------------- #
# Designation + objective math
# --------------------------------------------------------------------------- #
def test_naca_designation():
    assert naca_designation(0.02, 0.4, 0.12) == "NACA 2412"
    assert naca_designation(0.04, 0.2, 0.10) == "NACA 4210"
    assert naca_designation(0.0, 0.3, 0.08) == "NACA 0308"


def test_penalized_objective_inside_band_is_just_cd():
    assert penalized_objective(au.CL_TARGET, 0.01) == pytest.approx(0.01)
    # within tolerance -> no penalty
    assert penalized_objective(au.CL_TARGET + au.CL_TOL, 0.01) == pytest.approx(0.01)


def test_penalized_objective_outside_band_is_penalized():
    cl = au.CL_TARGET + 2 * au.CL_TOL      # excess = CL_TOL
    expected = 0.01 + au.PENALTY * au.CL_TOL
    assert penalized_objective(cl, 0.01) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# eval_xfoil failure handling (XFOIL faked out)
# --------------------------------------------------------------------------- #
def _install_fake_xfoil(monkeypatch, a_return=None, raise_on_solve=False):
    """Inject a fake `xfoil` package into sys.modules."""
    class FakeXFoil:
        def __init__(self):
            self.print = True
            self.airfoil = None
            self.Re = self.M = self.max_iter = None

        def repanel(self):
            pass

        def a(self, alpha):
            if raise_on_solve:
                raise RuntimeError("solver blew up")
            return a_return

    pkg = types.ModuleType("xfoil")
    pkg.XFoil = FakeXFoil
    model = types.ModuleType("xfoil.model")
    model.Airfoil = lambda x, y: object()
    pkg.model = model
    monkeypatch.setitem(sys.modules, "xfoil", pkg)
    monkeypatch.setitem(sys.modules, "xfoil.model", model)


def test_eval_xfoil_returns_tuple_on_success(monkeypatch):
    _install_fake_xfoil(monkeypatch, a_return=(0.5, 0.006, -0.05, 0.0))
    out = au.eval_xfoil(0.02, 0.4, 0.12, 2.0)
    assert out == pytest.approx((0.5, 0.006, -0.05))


def test_eval_xfoil_returns_none_on_nan(monkeypatch):
    _install_fake_xfoil(monkeypatch, a_return=(np.nan, np.nan, np.nan, np.nan))
    assert au.eval_xfoil(0.02, 0.4, 0.12, 2.0) is None


def test_eval_xfoil_returns_none_on_nonpositive_cd(monkeypatch):
    _install_fake_xfoil(monkeypatch, a_return=(0.5, 0.0, -0.05, 0.0))
    assert au.eval_xfoil(0.02, 0.4, 0.12, 2.0) is None


def test_eval_xfoil_returns_none_when_solver_raises(monkeypatch):
    _install_fake_xfoil(monkeypatch, raise_on_solve=True)
    assert au.eval_xfoil(0.02, 0.4, 0.12, 2.0) is None
