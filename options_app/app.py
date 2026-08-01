"""Streamlit UI for the Options Desk profit calculator."""

from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from options_app.bsm_core import bsm_greeks, theo_curve, years_from_dte
from options_app.math_core import payoff_curve
from options_app.models import (
    CALL_DEFAULTS,
    PUT_DEFAULTS,
    ChartConfig,
    OptionType,
    TradeInput,
    TradeResult,
)
from options_app.viz import payoff_figure, theo_vs_spot_figure

_PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}
.stApp {
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(14, 165, 233, 0.12), transparent 55%),
    radial-gradient(900px 480px at 92% 0%, rgba(217, 119, 6, 0.08), transparent 50%),
    linear-gradient(180deg, #070B14 0%, #0B1220 40%, #0F172A 100%);
  color: #E2E8F0;
}
div[data-testid="stMetricValue"] {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-weight: 500;
}
.opt-brand {
  font-size: 1.65rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: #F8FAFC;
  margin-bottom: 0.15rem;
}
.opt-sub {
  color: #94A3B8;
  font-size: 0.95rem;
  margin-bottom: 1rem;
}
.opt-note {
  color: #94A3B8;
  font-size: 0.85rem;
  border-top: 1px solid rgba(148, 163, 184, 0.2);
  padding-top: 0.75rem;
  margin-top: 1rem;
}
.opt-warn {
  color: #FCD34D;
  background: rgba(120, 53, 15, 0.35);
  border: 1px solid rgba(251, 191, 36, 0.35);
  padding: 0.65rem 0.85rem;
  border-radius: 6px;
  margin: 0.5rem 0 1rem 0;
}
.opt-err {
  color: #FCA5A5;
  background: rgba(127, 29, 29, 0.35);
  border: 1px solid rgba(248, 113, 113, 0.35);
  padding: 0.65rem 0.85rem;
  border-radius: 6px;
}
.net-profit {
  border-radius: 10px;
  padding: 1rem 1.25rem;
  margin: 0.35rem 0 1rem 0;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: rgba(30, 41, 59, 0.55);
}
.net-profit.win {
  border-color: rgba(45, 212, 191, 0.45);
  background: linear-gradient(90deg, rgba(15, 118, 110, 0.28), rgba(15, 23, 42, 0.2));
}
.net-profit.lose {
  border-color: rgba(251, 113, 133, 0.45);
  background: linear-gradient(90deg, rgba(159, 18, 57, 0.28), rgba(15, 23, 42, 0.2));
}
.net-profit .label {
  color: #94A3B8;
  font-size: 0.85rem;
  margin-bottom: 0.25rem;
}
.net-profit .amount {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 1.85rem;
  font-weight: 600;
  color: #F8FAFC;
  line-height: 1.2;
}
.net-profit.win .amount { color: #5EEAD4; }
.net-profit.lose .amount { color: #FDA4AF; }
.net-profit .breakdown {
  color: #CBD5E1;
  font-size: 0.9rem;
  margin-top: 0.55rem;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
}
</style>
"""


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:,.2f}%"


def _fmt_max_profit(value: float | None) -> str:
    return "Unlimited" if value is None else _fmt_money(value)


def _key(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def _defaults_for(option_type: OptionType) -> dict[str, object]:
    return dict(CALL_DEFAULTS if option_type == "call" else PUT_DEFAULTS)


def _ensure_tab_state(prefix: str, option_type: OptionType) -> None:
    defaults = _defaults_for(option_type)
    for field, value in defaults.items():
        k = _key(prefix, field)
        if k not in st.session_state:
            st.session_state[k] = value
    exp_k = _key(prefix, "expiration_date")
    if exp_k not in st.session_state:
        st.session_state[exp_k] = date.today() + timedelta(days=30)
    for field, value in {
        "rate": 0.05,
        "dividend_yield": 0.0,
        "volatility": 0.25,
        "show_pricing": False,
        "chart_mode": "dollars",
        "compare_stock": False,
        "show_markers": True,
        "custom_range": False,
        "x_min": 0.0,
        "x_max": 200.0,
    }.items():
        k = _key(prefix, field)
        if k not in st.session_state:
            st.session_state[k] = value


def _reset_tab(prefix: str, option_type: OptionType) -> None:
    defaults = _defaults_for(option_type)
    for field, value in defaults.items():
        st.session_state[_key(prefix, field)] = value
    st.session_state[_key(prefix, "expiration_date")] = date.today() + timedelta(days=30)
    st.session_state[_key(prefix, "rate")] = 0.05
    st.session_state[_key(prefix, "dividend_yield")] = 0.0
    st.session_state[_key(prefix, "volatility")] = 0.25


def _randomize_tab(prefix: str, option_type: OptionType) -> None:
    spot = round(random.uniform(40.0, 200.0), 2)
    strike = round(spot * random.uniform(0.9, 1.1), 2)
    premium = round(random.uniform(0.5, spot * 0.08), 2)
    st.session_state[_key(prefix, "spot")] = spot
    st.session_state[_key(prefix, "strike")] = strike
    st.session_state[_key(prefix, "premium")] = premium
    st.session_state[_key(prefix, "contracts")] = random.choice([1, 2, 5])
    st.session_state[_key(prefix, "fees")] = round(random.uniform(0.0, 5.0), 2)
    if option_type == "call":
        scenario = spot * random.uniform(1.0, 1.25)
    else:
        scenario = spot * random.uniform(0.75, 1.0)
    st.session_state[_key(prefix, "scenario_price")] = round(scenario, 2)
    st.session_state[_key(prefix, "expiration_date")] = date.today() + timedelta(
        days=random.choice([14, 30, 45, 60, 90])
    )
    st.session_state[_key(prefix, "volatility")] = round(random.uniform(0.15, 0.45), 2)


def _build_trade(prefix: str, option_type: OptionType) -> TradeInput:
    return TradeInput(
        option_type=option_type,
        strike=float(st.session_state[_key(prefix, "strike")]),
        spot=float(st.session_state[_key(prefix, "spot")]),
        premium=float(st.session_state[_key(prefix, "premium")]),
        contracts=int(st.session_state[_key(prefix, "contracts")]),
        fees=float(st.session_state[_key(prefix, "fees")]),
        scenario_price=float(st.session_state[_key(prefix, "scenario_price")]),
        expiration_date=st.session_state[_key(prefix, "expiration_date")],
        as_of_date=date.today(),
        rate=float(st.session_state[_key(prefix, "rate")]),
        dividend_yield=float(st.session_state[_key(prefix, "dividend_yield")]),
        volatility=float(st.session_state[_key(prefix, "volatility")]),
    )


def _render_net_profit(trade: TradeInput, result: TradeResult) -> None:
    """Prominent net profit at the scenario stock price (payoff − debit paid)."""
    gross = result.scenario_pnl + result.debit  # option value at expiration
    net = result.scenario_pnl
    tone = "win" if net >= 0 else "lose"
    sign = "+" if net > 0 else ""
    st.markdown(
        f"""
<div class="net-profit {tone}">
  <div class="label">Net profit at scenario stock price {_fmt_money(trade.scenario_price)}</div>
  <div class="amount">{sign}{_fmt_money(net)}</div>
  <div class="breakdown">
    Option value {_fmt_money(gross)}
    − debit paid {_fmt_money(result.debit)}
    = net {_fmt_money(net)}
    &nbsp;·&nbsp; return {_fmt_pct(result.scenario_return_pct)}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_metrics(trade: TradeInput, result: TradeResult) -> None:
    _render_net_profit(trade, result)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total debit (what you pay)", _fmt_money(result.debit))
    c2.metric("Break-even", _fmt_money(result.breakeven))
    c3.metric("Max loss", _fmt_money(result.max_loss))
    c4.metric("Max profit", _fmt_max_profit(result.max_profit))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Net profit (scenario)", _fmt_money(result.scenario_pnl))
    c6.metric("Return on debit", _fmt_pct(result.scenario_return_pct))
    c7.metric("Moneyness", result.moneyness)
    c8.metric("DTE", "—" if result.dte is None else str(result.dte))

    spot_strike = result.spot_vs_strike_pct
    spot_strike_label = f"{'+' if spot_strike > 0 else ''}{_fmt_pct(spot_strike)}"
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Spot vs strike", spot_strike_label)
    c10.metric("Intrinsic / share", _fmt_money(result.intrinsic_per_share))
    c11.metric("Extrinsic / share", _fmt_money(result.extrinsic_per_share))
    c12.metric("Shares (contracts × 100)", f"{result.shares:,}")


def _render_pricing_panel(trade: TradeInput, result: TradeResult) -> None:
    t = years_from_dte(result.dte)
    rate = float(trade.rate or 0.0)
    q = float(trade.dividend_yield or 0.0)
    vol = float(trade.volatility or 0.25)
    greeks = bsm_greeks(
        trade.option_type,
        trade.spot,
        trade.strike,
        t,
        rate,
        q,
        vol,
        market_premium=trade.premium,
    )

    st.subheader("Pricing (Black-Scholes)")
    if result.dte is None or result.dte == 0:
        st.caption("T = 0 (expired or no DTE) — theo equals intrinsic.")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Theo / share", _fmt_money(greeks.price))
    p2.metric("Edge (theo − premium)", _fmt_money(greeks.edge))
    p3.metric("Delta", f"{greeks.delta:.4f}")
    p4.metric("Gamma", f"{greeks.gamma:.6f}")

    p5, p6, p7 = st.columns(3)
    p5.metric("Theta / day", _fmt_money(greeks.theta))
    p6.metric("Vega / vol pt", _fmt_money(greeks.vega))
    p7.metric("Rho / rate pt", _fmt_money(greeks.rho))

    spots = np.linspace(max(trade.spot * 0.7, 0.01), trade.spot * 1.3, 80)
    theo_vals = theo_curve(trade.option_type, spots, trade.strike, t, rate, q, vol)
    st.plotly_chart(
        theo_vs_spot_figure(
            spots,
            theo_vals,
            market_premium=trade.premium,
            strike=trade.strike,
            spot=trade.spot,
        ),
        width="stretch",
    )


def _render_tab(option_type: OptionType) -> None:
    prefix = option_type
    _ensure_tab_state(prefix, option_type)

    left, right = st.columns([0.95, 1.35], gap="large")

    with left:
        st.markdown(f"#### Long {option_type.title()} inputs")
        b1, b2 = st.columns(2)
        if b1.button("Reset", key=_key(prefix, "reset"), width="stretch"):
            _reset_tab(prefix, option_type)
            st.rerun()
        if b2.button("Randomize", key=_key(prefix, "random"), width="stretch"):
            _randomize_tab(prefix, option_type)
            st.rerun()

        st.number_input("Strike ($)", min_value=0.01, step=0.5, key=_key(prefix, "strike"))
        st.number_input(
            "Current stock price ($)",
            min_value=0.01,
            step=0.5,
            key=_key(prefix, "spot"),
        )
        st.number_input(
            "Option cost / share ($)",
            min_value=0.0,
            step=0.05,
            key=_key(prefix, "premium"),
        )
        st.number_input("Contracts", min_value=1, step=1, key=_key(prefix, "contracts"))
        st.number_input("Fees / commissions ($)", min_value=0.0, step=0.1, key=_key(prefix, "fees"))
        st.date_input("Expiration date", key=_key(prefix, "expiration_date"))
        st.number_input(
            "Scenario (what-if) stock price ($)",
            min_value=0.0,
            step=0.5,
            key=_key(prefix, "scenario_price"),
        )
        st.caption(
            "Each contract = 100 shares. Total debit = option cost × shares + fees "
            "(computed automatically)."
        )

        with st.expander("Chart range override"):
            use_custom = st.checkbox("Custom x-range", key=_key(prefix, "custom_range"))
            x_min_w = st.number_input("Min stock price", min_value=0.0, key=_key(prefix, "x_min"))
            x_max_w = st.number_input("Max stock price", min_value=0.01, key=_key(prefix, "x_max"))

        with st.expander("Pricing inputs (v1.1)"):
            st.number_input(
                "Risk-free rate (decimal)",
                min_value=0.0,
                max_value=1.0,
                step=0.005,
                format="%.4f",
                key=_key(prefix, "rate"),
            )
            st.number_input(
                "Dividend yield (decimal)",
                min_value=0.0,
                max_value=1.0,
                step=0.005,
                format="%.4f",
                key=_key(prefix, "dividend_yield"),
            )
            st.number_input(
                "Volatility σ (decimal)",
                min_value=0.01,
                max_value=3.0,
                step=0.01,
                format="%.2f",
                key=_key(prefix, "volatility"),
            )
            st.checkbox("Show pricing panel", key=_key(prefix, "show_pricing"))

    try:
        trade = _build_trade(prefix, option_type)
        result = TradeResult.from_trade(trade)
    except (ValidationError, ValueError) as exc:
        with right:
            st.markdown(f'<div class="opt-err">{exc}</div>', unsafe_allow_html=True)
        return

    x_min = float(x_min_w) if use_custom else None
    x_max = float(x_max_w) if use_custom else None

    with right:
        if result.put_breakeven_warning:
            st.markdown(
                f'<div class="opt-warn">{result.put_breakeven_warning}</div>',
                unsafe_allow_html=True,
            )

        _render_metrics(trade, result)

        ctrl1, ctrl2, ctrl3 = st.columns(3)
        ctrl1.selectbox(
            "Chart scale",
            options=["dollars", "percent"],
            format_func=lambda m: "$ Profit/Loss" if m == "dollars" else "% Return",
            key=_key(prefix, "chart_mode"),
        )
        ctrl2.checkbox("Compare vs buying stock", key=_key(prefix, "compare_stock"))
        ctrl3.checkbox("Show markers", key=_key(prefix, "show_markers"))

        try:
            show_markers = st.session_state[_key(prefix, "show_markers")]
            config = ChartConfig(
                mode=st.session_state[_key(prefix, "chart_mode")],
                compare_stock=st.session_state[_key(prefix, "compare_stock")],
                show_breakeven=show_markers,
                show_strike=show_markers,
                show_spot=show_markers,
                show_scenario=show_markers,
                x_min=x_min,
                x_max=x_max,
            )
            series = payoff_curve(
                option_type,
                strike=trade.strike,
                spot=trade.spot,
                premium=trade.premium,
                contracts=trade.contracts,
                fees=trade.fees,
                x_min=config.x_min,
                x_max=config.x_max,
                n=config.n_points,
                include_stock=config.compare_stock,
            )
        except (ValidationError, ValueError) as exc:
            st.markdown(f'<div class="opt-err">{exc}</div>', unsafe_allow_html=True)
            return

        fig = payoff_figure(
            series,
            option_type=option_type,
            strike=trade.strike,
            spot=trade.spot,
            scenario_price=trade.scenario_price,
            config=config,
            entry_spot=trade.spot,
            contracts=trade.contracts,
            fees=trade.fees,
        )
        event = st.plotly_chart(
            fig,
            width="stretch",
            key=_key(prefix, "chart"),
            on_select="rerun",
            selection_mode="points",
        )

        # Click-to-set scenario price from chart selection (only when value changes).
        try:
            points = event.selection.points  # type: ignore[attr-defined]
            if points:
                clicked_x = points[0].get("x")
                if clicked_x is not None:
                    new_price = round(float(clicked_x), 2)
                    current = round(float(st.session_state[_key(prefix, "scenario_price")]), 2)
                    if new_price != current:
                        st.session_state[_key(prefix, "scenario_price")] = new_price
                        st.rerun()
        except (AttributeError, TypeError, KeyError, IndexError):
            pass

        export_df = pd.DataFrame(
            {
                "stock_price": series.prices,
                "option_pnl": series.option_pnl,
                "option_return_pct": series.option_return_pct,
            }
        )
        if series.stock_pnl is not None:
            export_df["stock_pnl"] = series.stock_pnl
        st.download_button(
            "Download P/L table (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"long_{option_type}_payoff.csv",
            mime="text/csv",
            key=_key(prefix, "csv"),
        )

        if st.session_state[_key(prefix, "show_pricing")]:
            _render_pricing_panel(trade, result)

    st.markdown(
        '<p class="opt-note">Payoff chart is at expiration. Enable '
        "<strong>Show pricing panel</strong> under Pricing inputs for Black-Scholes "
        "theo value and Greeks (v1.1).</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Options Desk",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)
    st.markdown('<div class="opt-brand">Options Desk</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="opt-sub">Long call &amp; put profit calculator — plug in the trade, '
        "see where you break even.</p>",
        unsafe_allow_html=True,
    )

    call_tab, put_tab = st.tabs(["Call", "Put"])
    with call_tab:
        _render_tab("call")
    with put_tab:
        _render_tab("put")


if __name__ == "__main__":
    main()
