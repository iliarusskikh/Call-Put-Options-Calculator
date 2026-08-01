"""Pydantic models for trade inputs and computed results."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from options_app.math_core import (
    MULTIPLIER,
    extrinsic_value,
    long_call_breakeven,
    long_call_intrinsic,
    long_call_pnl,
    long_put_breakeven,
    long_put_intrinsic,
    long_put_pnl,
    max_loss,
    max_profit_call,
    max_profit_put,
    moneyness,
    return_pct,
    shares,
    spot_vs_strike_pct,
    total_debit,
)
from options_app.math_core import (
    dte as calc_dte,
)

OptionType = Literal["call", "put"]
ChartMode = Literal["dollars", "percent"]


class TradeInput(BaseModel):
    """Validated long-option trade inputs (buy to open).

    Pricing fields (``rate``, ``dividend_yield``, ``volatility``) are optional
    extension points for v1.1 BSM without breaking the payoff API.
    """

    option_type: OptionType
    strike: float = Field(..., gt=0)
    spot: float = Field(..., gt=0)
    premium: float = Field(..., ge=0)
    contracts: int = Field(..., ge=1)
    fees: float = Field(default=0.0, ge=0)
    scenario_price: float = Field(..., ge=0)
    expiration_date: date | None = None
    as_of_date: date = Field(default_factory=date.today)
    multiplier: int = Field(default=MULTIPLIER, ge=1)

    # v1.1 optional pricing inputs (unused by payoff path)
    rate: float | None = Field(default=None, ge=0)
    dividend_yield: float | None = Field(default=None, ge=0)
    volatility: float | None = Field(default=None, gt=0)

    @field_validator(
        "strike",
        "spot",
        "premium",
        "fees",
        "scenario_price",
        "rate",
        "dividend_yield",
        "volatility",
        mode="before",
    )
    @classmethod
    def _coerce_finite(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a real number")
        value_f = float(value)
        if value_f != value_f or value_f in (float("inf"), float("-inf")):
            raise ValueError("must be finite")
        return value_f

    @model_validator(mode="after")
    def _debit_positive(self) -> TradeInput:
        debit = float(self.premium) * self.contracts * self.multiplier + float(self.fees)
        if debit <= 0:
            raise ValueError("total debit must be > 0 (set premium and/or fees)")
        return self

    @property
    def share_count(self) -> int:
        return shares(self.contracts, self.multiplier)

    @property
    def debit(self) -> float:
        return total_debit(self.premium, self.contracts, self.fees, multiplier=self.multiplier)


class TradeResult(BaseModel):
    """Computed summary metrics for a validated trade."""

    model_config = {"arbitrary_types_allowed": True}

    option_type: OptionType
    debit: float
    shares: int
    breakeven: float
    max_loss: float
    max_profit: float | None  # None => unlimited
    dte: int | None
    moneyness: str
    spot_vs_strike_pct: float
    intrinsic_per_share: float
    extrinsic_per_share: float
    scenario_pnl: float
    scenario_return_pct: float
    put_breakeven_warning: str | None = None

    @classmethod
    def from_trade(cls, trade: TradeInput) -> TradeResult:
        """Compute all summary metrics from a trade input."""
        debit = trade.debit
        m = trade.share_count
        mult = trade.multiplier
        if trade.option_type == "call":
            be = long_call_breakeven(
                trade.strike,
                trade.premium,
                trade.contracts,
                trade.fees,
                multiplier=mult,
            )
            intrinsic = float(long_call_intrinsic(trade.spot, trade.strike))
            scenario_pnl = float(
                long_call_pnl(
                    trade.scenario_price,
                    trade.strike,
                    trade.premium,
                    trade.contracts,
                    trade.fees,
                    multiplier=mult,
                )
            )
            max_p = max_profit_call()
            warning = None
        else:
            be = long_put_breakeven(
                trade.strike,
                trade.premium,
                trade.contracts,
                trade.fees,
                multiplier=mult,
            )
            intrinsic = float(long_put_intrinsic(trade.spot, trade.strike))
            scenario_pnl = float(
                long_put_pnl(
                    trade.scenario_price,
                    trade.strike,
                    trade.premium,
                    trade.contracts,
                    trade.fees,
                    multiplier=mult,
                )
            )
            max_p = max_profit_put(
                trade.strike,
                trade.premium,
                trade.contracts,
                trade.fees,
                multiplier=mult,
            )
            warning = (
                "Debit is at or above the strike; break-even is at or below zero."
                if be <= 0
                else None
            )

        return cls(
            option_type=trade.option_type,
            debit=debit,
            shares=m,
            breakeven=be,
            max_loss=max_loss(
                trade.premium, trade.contracts, trade.fees, multiplier=trade.multiplier
            ),
            max_profit=max_p,
            dte=calc_dte(trade.expiration_date, trade.as_of_date),
            moneyness=moneyness(trade.spot, trade.strike, trade.option_type),
            spot_vs_strike_pct=spot_vs_strike_pct(trade.spot, trade.strike),
            intrinsic_per_share=intrinsic,
            extrinsic_per_share=extrinsic_value(trade.premium, intrinsic),
            scenario_pnl=scenario_pnl,
            scenario_return_pct=float(return_pct(scenario_pnl, debit)),
            put_breakeven_warning=warning,
        )


class ChartConfig(BaseModel):
    """Display options for the payoff chart."""

    mode: ChartMode = "dollars"
    compare_stock: bool = False
    show_breakeven: bool = True
    show_strike: bool = True
    show_spot: bool = True
    show_scenario: bool = True
    x_min: float | None = Field(default=None, ge=0)
    x_max: float | None = Field(default=None, gt=0)
    n_points: int = Field(default=300, ge=50, le=2000)

    @model_validator(mode="after")
    def _range_ok(self) -> ChartConfig:
        if self.x_min is not None and self.x_max is not None and self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        return self


# Sensible ATM-ish starter presets for demos / reset.
CALL_DEFAULTS: dict[str, object] = {
    "strike": 100.0,
    "spot": 100.0,
    "premium": 3.50,
    "contracts": 1,
    "fees": 1.30,
    "scenario_price": 110.0,
}

PUT_DEFAULTS: dict[str, object] = {
    "strike": 100.0,
    "spot": 100.0,
    "premium": 3.25,
    "contracts": 1,
    "fees": 1.30,
    "scenario_price": 90.0,
}
