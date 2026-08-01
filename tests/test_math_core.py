"""Unit tests for expiration payoff math."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from options_app.math_core import (
    TradeError,
    default_price_range,
    dte,
    extrinsic_value,
    long_call_breakeven,
    long_call_pnl,
    long_put_breakeven,
    long_put_pnl,
    max_loss,
    max_profit_call,
    max_profit_put,
    moneyness,
    payoff_curve,
    return_pct,
    spot_vs_strike_pct,
    stock_breakeven,
    stock_pnl,
    total_debit,
)
from options_app.models import TradeInput, TradeResult
from pydantic import ValidationError


def test_total_debit_and_max_loss() -> None:
    assert total_debit(3.5, 1, 1.30) == pytest.approx(3.5 * 100 + 1.30)
    assert max_loss(3.5, 1, 1.30) == pytest.approx(351.30)


def test_total_debit_rejects_zero() -> None:
    with pytest.raises(TradeError):
        total_debit(0.0, 1, 0.0)


def test_long_call_otm_and_breakeven() -> None:
    premium, fees, k = 3.5, 1.30, 100.0
    debit = total_debit(premium, 1, fees)
    assert long_call_pnl(90.0, k, premium, 1, fees) == pytest.approx(-debit)
    assert long_call_pnl(100.0, k, premium, 1, fees) == pytest.approx(-debit)
    be = long_call_breakeven(k, premium, 1, fees)
    assert be == pytest.approx(k + premium + fees / 100)
    assert long_call_pnl(be, k, premium, 1, fees) == pytest.approx(0.0, abs=1e-9)
    # Deep ITM: slope ≈ shares
    s = 150.0
    assert long_call_pnl(s, k, premium, 1, fees) == pytest.approx((s - k) * 100 - debit)


def test_long_put_otm_max_profit_breakeven() -> None:
    premium, fees, k = 3.25, 1.30, 100.0
    debit = total_debit(premium, 1, fees)
    assert long_put_pnl(110.0, k, premium, 1, fees) == pytest.approx(-debit)
    be = long_put_breakeven(k, premium, 1, fees)
    assert be == pytest.approx(k - premium - fees / 100)
    assert long_put_pnl(be, k, premium, 1, fees) == pytest.approx(0.0, abs=1e-9)
    assert max_profit_put(k, premium, 1, fees) == pytest.approx(k * 100 - debit)
    assert long_put_pnl(0.0, k, premium, 1, fees) == pytest.approx(k * 100 - debit)
    assert max_profit_call() is None


def test_fees_shift_breakeven() -> None:
    be0 = long_call_breakeven(100.0, 2.0, 1, 0.0)
    be_fees = long_call_breakeven(100.0, 2.0, 1, 10.0)
    assert be_fees - be0 == pytest.approx(0.10)


def test_return_pct() -> None:
    assert return_pct(100.0, 200.0) == pytest.approx(50.0)
    with pytest.raises(TradeError):
        return_pct(1.0, 0.0)


def test_dte() -> None:
    assert dte(None) is None
    assert dte(date(2026, 8, 10), date(2026, 8, 1)) == 9
    assert dte(date(2026, 7, 1), date(2026, 8, 1)) == 0


def test_moneyness_and_extrinsic() -> None:
    assert moneyness(100.0, 100.0, "call") == "ATM"
    assert moneyness(110.0, 100.0, "call") == "ITM"
    assert moneyness(90.0, 100.0, "call") == "OTM"
    assert moneyness(90.0, 100.0, "put") == "ITM"
    assert extrinsic_value(3.5, 0.0) == 3.5
    assert extrinsic_value(3.5, 5.0) == 0.0


def test_spot_vs_strike_pct() -> None:
    assert spot_vs_strike_pct(110.0, 100.0) == pytest.approx(10.0)
    assert spot_vs_strike_pct(90.0, 100.0) == pytest.approx(-10.0)
    assert spot_vs_strike_pct(100.0, 100.0) == pytest.approx(0.0)


def test_stock_pnl_and_breakeven() -> None:
    assert stock_pnl(110.0, 100.0, 1, 1.30) == pytest.approx(10 * 100 - 1.30)
    assert stock_breakeven(100.0, 1, 1.30) == pytest.approx(100.013)


def test_payoff_curve_vectorized() -> None:
    series = payoff_curve(
        "call",
        strike=100.0,
        spot=100.0,
        premium=3.5,
        contracts=1,
        fees=1.30,
        include_stock=True,
        n=50,
    )
    assert len(series.prices) == 50
    assert series.option_pnl.shape == (50,)
    assert series.stock_pnl is not None
    assert series.debit > 0
    # Below strike: flat at -debit
    below = series.prices < 100.0
    assert np.allclose(series.option_pnl[below], -series.debit)


def test_default_price_range() -> None:
    lo, hi = default_price_range("call", 100.0, 100.0, 103.5)
    assert lo >= 0
    assert hi > lo


def test_trade_input_validation() -> None:
    with pytest.raises(ValidationError):
        TradeInput(
            option_type="call",
            strike=100,
            spot=100,
            premium=0,
            contracts=1,
            fees=0,
            scenario_price=110,
        )


def test_trade_result_call() -> None:
    trade = TradeInput(
        option_type="call",
        strike=100,
        spot=100,
        premium=3.5,
        contracts=1,
        fees=1.30,
        scenario_price=110,
        expiration_date=date(2026, 9, 1),
        as_of_date=date(2026, 8, 1),
    )
    result = TradeResult.from_trade(trade)
    assert result.dte == 31
    assert result.moneyness == "ATM"
    assert result.spot_vs_strike_pct == pytest.approx(0.0)
    assert result.max_profit is None
    assert result.breakeven == pytest.approx(103.513)
    assert result.scenario_pnl == pytest.approx((10 * 100) - result.debit)


def test_trade_result_put_warning() -> None:
    trade = TradeInput(
        option_type="put",
        strike=10,
        spot=10,
        premium=12.0,
        contracts=1,
        fees=0,
        scenario_price=5,
    )
    result = TradeResult.from_trade(trade)
    assert result.breakeven < 0
    assert result.put_breakeven_warning is not None
