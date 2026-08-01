"""Plotly visualization builders for options payoff charts."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from options_app.math_core import PayoffSeries, return_pct, stock_breakeven
from options_app.models import ChartConfig, OptionType

_COLOR = {
    "pnl": "#0EA5E9",
    "stock": "#94A3B8",
    "win": "rgba(15, 118, 110, 0.22)",
    "loss": "rgba(194, 65, 12, 0.20)",
    "zero": "#64748B",
    "breakeven": "#D97706",
    "strike": "#F472B6",
    "spot": "#A78BFA",
    "scenario": "#F8FAFC",
    "grid": "#1E293B",
    "paper": "#0B1220",
    "plot": "#111827",
    "font": "#E2E8F0",
    "muted": "#94A3B8",
    "theo": "#34D399",
}

_LAYOUT_BASE: dict[str, object] = {
    "paper_bgcolor": _COLOR["paper"],
    "plot_bgcolor": _COLOR["plot"],
    "font": {"color": _COLOR["font"], "family": "IBM Plex Sans, Segoe UI, sans-serif"},
    "margin": {"l": 64, "r": 28, "t": 52, "b": 56},
    "hoverlabel": {"bgcolor": "#1E293B", "font_size": 12},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0,
        "bgcolor": "rgba(0,0,0,0)",
    },
}


def _layout(**overrides: object) -> dict[str, object]:
    return {**_LAYOUT_BASE, **overrides}


def _y_series(series: PayoffSeries, mode: str) -> np.ndarray:
    if mode == "percent":
        return series.option_return_pct
    return series.option_pnl


def _stock_y(series: PayoffSeries, mode: str) -> np.ndarray | None:
    if series.stock_pnl is None:
        return None
    if mode == "percent":
        # Stock return on its own capital outlay is not series.debit; show $ only
        # when percent mode is selected for the option, still overlay stock in $.
        return series.stock_pnl
    return series.stock_pnl


def payoff_figure(
    series: PayoffSeries,
    *,
    option_type: OptionType,
    strike: float,
    spot: float,
    scenario_price: float,
    config: ChartConfig,
    entry_spot: float | None = None,
    contracts: int = 1,
    fees: float = 0.0,
) -> go.Figure:
    """Build the main expiration P/L chart with win/loss shading and markers."""
    x = series.prices
    y = _y_series(series, config.mode)
    y_label = "Return (%)" if config.mode == "percent" else "Profit / Loss ($)"
    title = f"Long {option_type.title()} — Payoff at Expiration"

    fig = go.Figure()

    # Shade by economic win/loss (sign of dollar PnL), mapped onto the displayed y-axis.
    y_win = np.where(series.option_pnl >= 0, y, 0.0)
    y_loss = np.where(series.option_pnl < 0, y, 0.0)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_win,
            line={"width": 0},
            fillcolor=_COLOR["win"],
            fill="tozeroy",
            name="Profit zone",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_loss,
            line={"width": 0},
            fillcolor=_COLOR["loss"],
            fill="tozeroy",
            name="Loss zone",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Option P/L" if config.mode == "dollars" else "Option return %",
            line={"color": _COLOR["pnl"], "width": 2.5},
            hovertemplate="Stock %{x:.2f}<br>"
            + ("P/L $%{y:,.2f}" if config.mode == "dollars" else "Return %{y:.2f}%")
            + "<extra></extra>",
        )
    )

    stock_y = _stock_y(series, config.mode)
    if config.compare_stock and stock_y is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=stock_y,
                mode="lines",
                name="Buy stock P/L ($)",
                line={"color": _COLOR["stock"], "width": 2, "dash": "dot"},
                hovertemplate="Stock %{x:.2f}<br>Stock P/L $%{y:,.2f}<extra></extra>",
            )
        )
        if entry_spot is not None:
            sbe = stock_breakeven(entry_spot, contracts, fees)
            fig.add_vline(
                x=sbe,
                line_width=1,
                line_dash="dot",
                line_color=_COLOR["muted"],
                annotation_text=f"Stock BE {sbe:.2f}",
                annotation_position="top left",
                annotation_font_color=_COLOR["muted"],
            )

    # Zero line
    fig.add_hline(y=0, line_width=1, line_color=_COLOR["zero"])

    if config.show_breakeven:
        be_y = float(return_pct(0.0, series.debit)) if config.mode == "percent" else 0.0
        fig.add_vline(
            x=series.breakeven,
            line_width=2,
            line_dash="dash",
            line_color=_COLOR["breakeven"],
            annotation_text=f"Break-even {series.breakeven:.2f}",
            annotation_position="top right",
            annotation_font_color=_COLOR["breakeven"],
        )
        fig.add_trace(
            go.Scatter(
                x=[series.breakeven],
                y=[be_y],
                mode="markers",
                name="Break-even",
                marker={"size": 10, "color": _COLOR["breakeven"], "symbol": "diamond"},
                hovertemplate=f"Break-even {series.breakeven:.2f}<extra></extra>",
            )
        )

    if config.show_strike:
        fig.add_vline(
            x=strike,
            line_width=1.5,
            line_dash="dashdot",
            line_color=_COLOR["strike"],
            annotation_text=f"Strike {strike:.2f}",
            annotation_position="bottom left",
            annotation_font_color=_COLOR["strike"],
        )

    if config.show_spot:
        fig.add_vline(
            x=spot,
            line_width=1.5,
            line_dash="dot",
            line_color=_COLOR["spot"],
            annotation_text=f"Spot {spot:.2f}",
            annotation_position="bottom right",
            annotation_font_color=_COLOR["spot"],
        )

    if config.show_scenario:
        # Interpolate y at scenario for marker
        scen_y = float(np.interp(scenario_price, x, y))
        fig.add_vline(
            x=scenario_price,
            line_width=2,
            line_color=_COLOR["scenario"],
            annotation_text=f"Scenario {scenario_price:.2f}",
            annotation_position="top left",
            annotation_font_color=_COLOR["scenario"],
        )
        fig.add_trace(
            go.Scatter(
                x=[scenario_price],
                y=[scen_y],
                mode="markers",
                name="Scenario",
                marker={
                    "size": 12,
                    "color": _COLOR["scenario"],
                    "line": {"width": 2, "color": _COLOR["pnl"]},
                },
                hovertemplate="Scenario %{x:.2f}<br>Y %{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        **_layout(
            title=title,
            xaxis={
                "title": "Stock price at expiration ($)",
                "gridcolor": _COLOR["grid"],
                "zeroline": False,
            },
            yaxis={
                "title": y_label,
                "gridcolor": _COLOR["grid"],
                "zeroline": False,
                "tickformat": ",.1f" if config.mode == "percent" else "$,.0f",
            },
        )
    )
    return fig


def theo_vs_spot_figure(
    spots: np.ndarray,
    theo_values: np.ndarray,
    *,
    market_premium: float,
    strike: float,
    spot: float,
) -> go.Figure:
    """v1.1 helper: theoretical option value vs spot at fixed T, σ."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spots,
            y=theo_values,
            mode="lines",
            name="Theo value / share",
            line={"color": _COLOR["theo"], "width": 2.5},
            hovertemplate="Spot %{x:.2f}<br>Theo $%{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=market_premium,
        line_dash="dash",
        line_color=_COLOR["breakeven"],
        annotation_text=f"Market premium {market_premium:.2f}",
        annotation_font_color=_COLOR["breakeven"],
    )
    fig.add_vline(
        x=spot,
        line_dash="dot",
        line_color=_COLOR["spot"],
        annotation_text=f"Spot {spot:.2f}",
        annotation_font_color=_COLOR["spot"],
    )
    fig.add_vline(
        x=strike,
        line_dash="dot",
        line_color=_COLOR["muted"],
        annotation_text=f"Strike {strike:.2f}",
        annotation_font_color=_COLOR["muted"],
    )
    fig.update_layout(
        **_layout(
            title="Theoretical value vs spot (fixed T, σ)",
            xaxis={"title": "Underlying spot ($)", "gridcolor": _COLOR["grid"]},
            yaxis={"title": "Option value ($ / share)", "gridcolor": _COLOR["grid"]},
        )
    )
    return fig
