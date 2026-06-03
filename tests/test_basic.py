"""
Unit tests that run WITHOUT XFOIL installed.

They cover the deterministic, pure-Python parts of the project: NACA geometry,
the objective/penalty math, the designation helper, and the failure handling of
eval_xfoil (XFOIL is replaced by a fake module). Run with:  pytest
"""

import subprocess
import types

import numpy as np
import pytest

import airfoil_utils as au
from airfoil_utils import naca4, naca_designation, penalized_objective


def _completed(stdout, returncode=0):
    """Build a fake subprocess.CompletedProcess-like object."""
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


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
# eval_xfoil orchestration (subprocess worker mocked out)
# --------------------------------------------------------------------------- #
def test_eval_xfoil_parses_worker_result(monkeypatch):
    # First recipe already converges -> only one subprocess call.
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed("[0.5, 0.006, -0.05]\n")

    monkeypatch.setattr(au.subprocess, "run", fake_run)
    out = au.eval_xfoil(0.02, 0.4, 0.12, 2.0)
    assert out == pytest.approx((0.5, 0.006, -0.05))
    assert len(calls) == 1


def test_eval_xfoil_returns_none_when_all_recipes_fail(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed("null\n")

    monkeypatch.setattr(au.subprocess, "run", fake_run)
    assert au.eval_xfoil(0.02, 0.4, 0.12, 2.0) is None
    # It must have escalated through every recipe before giving up.
    assert len(calls) == len(au._XFOIL_RECIPES)


def test_eval_xfoil_escalates_to_next_recipe(monkeypatch):
    # First recipe fails (null), second converges.
    outputs = iter(["null\n", "[0.7, 0.009, -0.04]\n"])

    def fake_run(cmd, **kwargs):
        return _completed(next(outputs))

    monkeypatch.setattr(au.subprocess, "run", fake_run)
    assert au.eval_xfoil(0.02, 0.4, 0.12, 2.0) == pytest.approx((0.7, 0.009, -0.04))


def test_eval_xfoil_survives_timeout(monkeypatch):
    # First recipe times out, second converges -> no exception propagates.
    state = {"first": True}

    def fake_run(cmd, **kwargs):
        if state["first"]:
            state["first"] = False
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 90))
        return _completed("[0.5, 0.006, -0.05]\n")

    monkeypatch.setattr(au.subprocess, "run", fake_run)
    assert au.eval_xfoil(0.02, 0.4, 0.12, 2.0) == pytest.approx((0.5, 0.006, -0.05))


def test_eval_xfoil_validates_geometry_before_subprocess(monkeypatch):
    # A bad geometry must raise ValueError without ever spawning a worker.
    def fake_run(cmd, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("subprocess should not run for invalid geometry")

    monkeypatch.setattr(au.subprocess, "run", fake_run)
    with pytest.raises(ValueError):
        au.eval_xfoil(0.02, 0.4, -0.1, 2.0)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("xfoil") is None,
    reason="xfoil not installed",
)
def test_eval_xfoil_real_roundtrip():
    # End-to-end through the real worker subprocess (only if XFOIL is present).
    out = au.eval_xfoil(0.02, 0.4, 0.12, 2.0)
    assert out is not None
    cl, cd, cm = out
    assert 0.2 < cl < 0.8 and 0.0 < cd < 0.05
