"""Unit tests for European Black-Scholes pricing and Greeks."""

from __future__ import annotations

import numpy as np
import pytest
from options_app.bsm_core import bsm_greeks, bsm_price, theo_curve, years_from_dte
from options_app.math_core import TradeError


def test_years_from_dte() -> None:
    assert years_from_dte(None) == 0.0
    assert years_from_dte(365) == pytest.approx(1.0)
    assert years_from_dte(0) == 0.0


def test_t_zero_is_intrinsic() -> None:
    assert bsm_price("call", 110.0, 100.0, 0.0, 0.05, 0.0, 0.2) == pytest.approx(10.0)
    assert bsm_price("put", 90.0, 100.0, 0.0, 0.05, 0.0, 0.2) == pytest.approx(10.0)
    assert bsm_price("call", 90.0, 100.0, 0.0, 0.05, 0.0, 0.2) == pytest.approx(0.0)


def test_known_atm_call_rough_range() -> None:
    # ATM call with σ=20%, T=1y, r=q=0 should be near ~7.965 for S=K=100
    price = bsm_price("call", 100.0, 100.0, 1.0, 0.0, 0.0, 0.2)
    assert 7.5 < price < 8.5


def test_put_call_parity() -> None:
    s, k, t, r, q, vol = 100.0, 100.0, 0.5, 0.05, 0.02, 0.25
    call = bsm_price("call", s, k, t, r, q, vol)
    put = bsm_price("put", s, k, t, r, q, vol)
    lhs = call - put
    rhs = s * np.exp(-q * t) - k * np.exp(-r * t)
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_greeks_signs_and_edge() -> None:
    g = bsm_greeks("call", 100.0, 100.0, 0.5, 0.05, 0.0, 0.2, market_premium=5.0)
    assert 0 < g.delta < 1
    assert g.gamma > 0
    assert g.vega > 0
    assert g.price > 0
    assert g.edge == pytest.approx(g.price - 5.0)


def test_put_delta_negative() -> None:
    g = bsm_greeks("put", 100.0, 100.0, 0.5, 0.05, 0.0, 0.2)
    assert -1 < g.delta < 0


def test_vol_rejected_when_t_positive() -> None:
    with pytest.raises(TradeError):
        bsm_price("call", 100.0, 100.0, 0.5, 0.05, 0.0, 0.0)


def test_gamma_finite_diff_smoke() -> None:
    s, k, t, r, q, vol = 100.0, 100.0, 0.5, 0.05, 0.0, 0.2
    h = 0.05
    g = bsm_greeks("call", s, k, t, r, q, vol)
    d_up = bsm_greeks("call", s + h, k, t, r, q, vol).delta
    d_dn = bsm_greeks("call", s - h, k, t, r, q, vol).delta
    gamma_fd = (d_up - d_dn) / (2 * h)
    assert gamma_fd == pytest.approx(g.gamma, rel=1e-3, abs=1e-4)


def test_theo_curve_shape() -> None:
    spots = np.linspace(80, 120, 21)
    vals = theo_curve("call", spots, 100.0, 0.5, 0.05, 0.0, 0.2)
    assert vals.shape == spots.shape
    assert vals[-1] > vals[0]
