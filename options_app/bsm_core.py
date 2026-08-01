"""European Black-Scholes-Merton pricing and Greeks (v1.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from options_app.math_core import TradeError, long_call_intrinsic, long_put_intrinsic

OptionType = Literal["call", "put"]
FloatOrArray = float | NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class BsmResult:
    """Theoretical price and Greeks for a European option.

    Greeks conventions:
        delta — dV/dS
        gamma — d²V/dS²
        theta — dV/dt in value per calendar day (divide annual theta by 365)
        vega — dV/dσ per 1 percentage point of volatility (÷ 100)
        rho — dV/dr per 1 percentage point of rate (÷ 100)
    """

    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    d1: float
    d2: float
    intrinsic: float
    edge: float  # theo − market premium (per share)


def _validate_bsm(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> None:
    if spot <= 0:
        raise TradeError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise TradeError(f"strike must be > 0, got {strike}")
    if time_years < 0:
        raise TradeError(f"time_years must be >= 0, got {time_years}")
    if rate < 0:
        raise TradeError(f"rate must be >= 0, got {rate}")
    if dividend_yield < 0:
        raise TradeError(f"dividend_yield must be >= 0, got {dividend_yield}")
    if volatility <= 0 and time_years > 0:
        raise TradeError(f"volatility must be > 0 when T > 0, got {volatility}")


def _d1_d2(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> tuple[float, float]:
    sqrt_t = np.sqrt(time_years)
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * time_years) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    return float(d1), float(d2)


def bsm_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """European BSM price per share. At T=0 returns intrinsic."""
    _validate_bsm(spot, strike, time_years, rate, dividend_yield, volatility)
    if time_years == 0:
        if option_type == "call":
            return float(long_call_intrinsic(spot, strike))
        return float(long_put_intrinsic(spot, strike))

    d1, d2 = _d1_d2(spot, strike, time_years, rate, dividend_yield, volatility)
    df_r = np.exp(-rate * time_years)
    df_q = np.exp(-dividend_yield * time_years)
    if option_type == "call":
        return float(spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2))
    return float(strike * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1))


def bsm_greeks(
    option_type: OptionType,
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    market_premium: float | None = None,
) -> BsmResult:
    """Price + Greeks. Theta is per calendar day; vega/rho per 1% point."""
    _validate_bsm(spot, strike, time_years, rate, dividend_yield, volatility)
    intrinsic = (
        float(long_call_intrinsic(spot, strike))
        if option_type == "call"
        else float(long_put_intrinsic(spot, strike))
    )

    if time_years == 0:
        if option_type == "call":
            delta = 1.0 if spot > strike else (0.5 if spot == strike else 0.0)
        else:
            delta = -1.0 if spot < strike else (-0.5 if spot == strike else 0.0)
        price = intrinsic
        edge = price - market_premium if market_premium is not None else 0.0
        return BsmResult(
            price=price,
            delta=delta,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            d1=0.0,
            d2=0.0,
            intrinsic=intrinsic,
            edge=edge,
        )

    d1, d2 = _d1_d2(spot, strike, time_years, rate, dividend_yield, volatility)
    df_r = np.exp(-rate * time_years)
    df_q = np.exp(-dividend_yield * time_years)
    pdf_d1 = float(norm.pdf(d1))
    sqrt_t = np.sqrt(time_years)

    price = bsm_price(option_type, spot, strike, time_years, rate, dividend_yield, volatility)
    gamma = float(df_q * pdf_d1 / (spot * volatility * sqrt_t))
    vega = float(spot * df_q * pdf_d1 * sqrt_t / 100.0)  # per 1 vol point

    if option_type == "call":
        delta = float(df_q * norm.cdf(d1))
        theta_annual = (
            -(spot * df_q * pdf_d1 * volatility) / (2 * sqrt_t)
            - rate * strike * df_r * norm.cdf(d2)
            + dividend_yield * spot * df_q * norm.cdf(d1)
        )
        rho = float(strike * time_years * df_r * norm.cdf(d2) / 100.0)
    else:
        delta = float(-df_q * norm.cdf(-d1))
        theta_annual = (
            -(spot * df_q * pdf_d1 * volatility) / (2 * sqrt_t)
            + rate * strike * df_r * norm.cdf(-d2)
            - dividend_yield * spot * df_q * norm.cdf(-d1)
        )
        rho = float(-strike * time_years * df_r * norm.cdf(-d2) / 100.0)

    theta = float(theta_annual / 365.0)
    edge = price - market_premium if market_premium is not None else 0.0
    return BsmResult(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        d1=d1,
        d2=d2,
        intrinsic=intrinsic,
        edge=edge,
    )


def theo_curve(
    option_type: OptionType,
    spots: FloatOrArray,
    strike: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> NDArray[np.floating]:
    """Vectorized theo prices across a spot grid (same T, σ, r, q)."""
    arr = np.asarray(spots, dtype=float)
    return np.array(
        [
            bsm_price(option_type, float(s), strike, time_years, rate, dividend_yield, volatility)
            for s in arr.ravel()
        ],
        dtype=float,
    ).reshape(arr.shape)


def years_from_dte(dte_days: int | None) -> float:
    """Convert calendar DTE to year fraction (ACT/365). Missing → 0."""
    if dte_days is None:
        return 0.0
    return max(dte_days, 0) / 365.0
