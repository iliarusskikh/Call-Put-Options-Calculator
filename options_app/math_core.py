"""Pure expiration-payoff calculations with no UI dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
from numpy.typing import NDArray

OptionType = Literal["call", "put"]
Moneyness = Literal["ITM", "ATM", "OTM"]
FloatOrArray = float | NDArray[np.floating]

MULTIPLIER = 100
ATM_REL_TOL = 0.005  # |S - K| / K <= 0.5% counts as ATM


class TradeError(ValueError):
    """Raised when a trade input is invalid for payoff math."""


def _as_float_array(s: FloatOrArray) -> NDArray[np.floating]:
    return np.asarray(s, dtype=float)


def shares(contracts: int, multiplier: int = MULTIPLIER) -> int:
    """Return share count for the position."""
    if contracts < 1:
        raise TradeError(f"contracts must be >= 1, got {contracts}")
    if multiplier < 1:
        raise TradeError(f"multiplier must be >= 1, got {multiplier}")
    return contracts * multiplier


def total_debit(
    premium: float,
    contracts: int,
    fees: float = 0.0,
    *,
    multiplier: int = MULTIPLIER,
) -> float:
    """Total capital at risk: premium × shares + fees."""
    if premium < 0:
        raise TradeError(f"premium must be >= 0, got {premium}")
    if fees < 0:
        raise TradeError(f"fees must be >= 0, got {fees}")
    m = shares(contracts, multiplier)
    debit = float(premium) * m + float(fees)
    if debit <= 0:
        raise TradeError("total debit must be > 0 (set premium and/or fees)")
    return debit


def long_call_intrinsic(s: FloatOrArray, strike: float) -> FloatOrArray:
    """Intrinsic value per share for a long call."""
    arr = np.maximum(_as_float_array(s) - strike, 0.0)
    return float(arr) if np.ndim(arr) == 0 else arr


def long_put_intrinsic(s: FloatOrArray, strike: float) -> FloatOrArray:
    """Intrinsic value per share for a long put."""
    arr = np.maximum(strike - _as_float_array(s), 0.0)
    return float(arr) if np.ndim(arr) == 0 else arr


def long_call_pnl(
    s: FloatOrArray,
    strike: float,
    premium: float,
    contracts: int,
    fees: float = 0.0,
    *,
    multiplier: int = MULTIPLIER,
) -> FloatOrArray:
    """Expiration P/L for a long call across one or many underlying prices."""
    debit = total_debit(premium, contracts, fees, multiplier=multiplier)
    m = shares(contracts, multiplier)
    intrinsic = long_call_intrinsic(s, strike)
    pnl = np.asarray(intrinsic, dtype=float) * m - debit
    return float(pnl) if np.ndim(pnl) == 0 else pnl


def long_put_pnl(
    s: FloatOrArray,
    strike: float,
    premium: float,
    contracts: int,
    fees: float = 0.0,
    *,
    multiplier: int = MULTIPLIER,
) -> FloatOrArray:
    """Expiration P/L for a long put across one or many underlying prices."""
    debit = total_debit(premium, contracts, fees, multiplier=multiplier)
    m = shares(contracts, multiplier)
    intrinsic = long_put_intrinsic(s, strike)
    pnl = np.asarray(intrinsic, dtype=float) * m - debit
    return float(pnl) if np.ndim(pnl) == 0 else pnl


def long_call_breakeven(
    strike: float,
    premium: float,
    contracts: int,
    fees: float = 0.0,
    *,
    multiplier: int = MULTIPLIER,
) -> float:
    """Stock price at expiration where long-call P/L is zero."""
    debit = total_debit(premium, contracts, fees, multiplier=multiplier)
    m = shares(contracts, multiplier)
    return strike + debit / m


def long_put_breakeven(
    strike: float,
    premium: float,
    contracts: int,
    fees: float = 0.0,
    *,
    multiplier: int = MULTIPLIER,
) -> float:
    """Stock price at expiration where long-put P/L is zero."""
    debit = total_debit(premium, contracts, fees, multiplier=multiplier)
    m = shares(contracts, multiplier)
    return strike - debit / m


def max_loss(
    premium: float,
    contracts: int,
    fees: float = 0.0,
    *,
    multiplier: int = MULTIPLIER,
) -> float:
    """Maximum loss for a long option equals the total debit."""
    return total_debit(premium, contracts, fees, multiplier=multiplier)


def max_profit_call() -> float | None:
    """Long call max profit is uncapped; return ``None`` to signal unlimited."""
    return None


def max_profit_put(
    strike: float,
    premium: float,
    contracts: int,
    fees: float = 0.0,
    *,
    multiplier: int = MULTIPLIER,
) -> float:
    """Long put max profit at S = 0: strike × shares − debit."""
    debit = total_debit(premium, contracts, fees, multiplier=multiplier)
    m = shares(contracts, multiplier)
    return strike * m - debit


def return_pct(pnl: FloatOrArray, debit: float) -> FloatOrArray:
    """Return percentage on capital at risk (total debit)."""
    if debit <= 0:
        raise TradeError(f"debit must be > 0 for return%, got {debit}")
    arr = _as_float_array(pnl) / debit * 100.0
    return float(arr) if np.ndim(arr) == 0 else arr


def dte(expiration: date | None, as_of: date | None = None) -> int | None:
    """Calendar days remaining until expiration (floored at 0)."""
    if expiration is None:
        return None
    ref = as_of if as_of is not None else date.today()
    return max((expiration - ref).days, 0)


def moneyness(spot: float, strike: float, option_type: OptionType) -> Moneyness:
    """Classify ITM / ATM / OTM relative to current spot."""
    if spot <= 0 or strike <= 0:
        raise TradeError("spot and strike must be > 0 for moneyness")
    if abs(spot - strike) / strike <= ATM_REL_TOL:
        return "ATM"
    if option_type == "call":
        return "ITM" if spot > strike else "OTM"
    return "ITM" if spot < strike else "OTM"


def spot_vs_strike_pct(spot: float, strike: float) -> float:
    """Percent distance of spot from strike: (spot − strike) / strike × 100.

    Positive ⇒ spot above strike; negative ⇒ spot below strike.
    """
    if spot <= 0 or strike <= 0:
        raise TradeError("spot and strike must be > 0 for spot_vs_strike_pct")
    return (spot - strike) / strike * 100.0


def extrinsic_value(premium: float, intrinsic_per_share: float) -> float:
    """Time/extrinsic value per share (premium − intrinsic, floored at 0)."""
    return max(float(premium) - float(intrinsic_per_share), 0.0)


def stock_pnl(
    s: FloatOrArray,
    entry_spot: float,
    contracts: int,
    fees: float = 0.0,
    *,
    multiplier: int = MULTIPLIER,
) -> FloatOrArray:
    """P/L from buying the same share count at ``entry_spot`` (plus fees)."""
    if entry_spot <= 0:
        raise TradeError(f"entry_spot must be > 0, got {entry_spot}")
    if fees < 0:
        raise TradeError(f"fees must be >= 0, got {fees}")
    m = shares(contracts, multiplier)
    pnl = (_as_float_array(s) - entry_spot) * m - fees
    return float(pnl) if np.ndim(pnl) == 0 else pnl


def stock_breakeven(entry_spot: float, contracts: int, fees: float = 0.0) -> float:
    """Underlying price where long-stock P/L is zero after fees."""
    m = shares(contracts)
    return entry_spot + fees / m


@dataclass(frozen=True, slots=True)
class PayoffSeries:
    """Price grid and corresponding option (and optional stock) P/L series."""

    prices: NDArray[np.floating]
    option_pnl: NDArray[np.floating]
    option_return_pct: NDArray[np.floating]
    stock_pnl: NDArray[np.floating] | None
    breakeven: float
    debit: float


def default_price_range(
    option_type: OptionType,
    spot: float,
    strike: float,
    breakeven: float,
    *,
    pad_frac: float = 0.08,
) -> tuple[float, float]:
    """Choose a sensible chart x-range from trade context."""
    anchors = [spot, strike, breakeven]
    lo_hint = 0.5 * min(spot, strike)
    x_min = max(0.0, min(lo_hint, breakeven) * (1.0 - pad_frac))
    x_max = max(anchors) * 1.5
    if option_type == "put":
        # Ensure room below break-even toward zero for the profitable wing.
        x_min = 0.0
        x_max = max(spot, strike, breakeven, strike * 1.2) * 1.35
    if x_max <= x_min:
        x_max = x_min + max(strike * 0.5, 1.0)
    return x_min, x_max


def payoff_curve(
    option_type: OptionType,
    *,
    strike: float,
    spot: float,
    premium: float,
    contracts: int,
    fees: float = 0.0,
    x_min: float | None = None,
    x_max: float | None = None,
    n: int = 300,
    include_stock: bool = False,
    multiplier: int = MULTIPLIER,
) -> PayoffSeries:
    """Build a vectorized expiration P/L curve for charting."""
    if n < 2:
        raise TradeError(f"n must be >= 2, got {n}")
    debit = total_debit(premium, contracts, fees, multiplier=multiplier)
    if option_type == "call":
        be = long_call_breakeven(strike, premium, contracts, fees, multiplier=multiplier)
        pnl_fn = long_call_pnl
    else:
        be = long_put_breakeven(strike, premium, contracts, fees, multiplier=multiplier)
        pnl_fn = long_put_pnl

    if x_min is None or x_max is None:
        auto_min, auto_max = default_price_range(option_type, spot, strike, be)
        x_min = auto_min if x_min is None else x_min
        x_max = auto_max if x_max is None else x_max
    if x_min < 0 or x_max <= x_min:
        raise TradeError(f"invalid price range [{x_min}, {x_max}]")

    prices = np.linspace(x_min, x_max, n)
    option_pnl = np.asarray(
        pnl_fn(prices, strike, premium, contracts, fees, multiplier=multiplier),
        dtype=float,
    )
    option_ret = np.asarray(return_pct(option_pnl, debit), dtype=float)
    stock_series = (
        np.asarray(stock_pnl(prices, spot, contracts, fees, multiplier=multiplier), dtype=float)
        if include_stock
        else None
    )
    return PayoffSeries(
        prices=prices,
        option_pnl=option_pnl,
        option_return_pct=option_ret,
        stock_pnl=stock_series,
        breakeven=be,
        debit=debit,
    )
