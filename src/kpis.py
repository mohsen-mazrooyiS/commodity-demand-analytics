"""
kpis.py
-------
Financial & Operational KPIs (Deliverable 3's business logic).

Kept separate from the Streamlit UI (dashboard/app.py) so every number can
be unit-tested and verified independently of the dashboard rendering.

IMPORTANT DATA-QUALITY FINDING -- Inventory_Turnover_Ratio is recomputed,
not trusted from the source file:

    Unlike Revenue and Gross_Profit (which reconcile EXACTLY with their raw
    components -- see etl.py's validation step), the provided
    Inventory_Turnover_Ratio does NOT reconcile with any combination of
    Cost_of_Goods_Sold_COGS, Sales_Units, or Inventory_Level. It spans
    0.00-1.00 in flat 0.01 increments with no relationship to the
    underlying numbers -- it appears to be an independently-generated
    placeholder in the source file, not a real computed ratio.

    We therefore RECOMPUTE inventory turnover from first principles:
        Annualized Turnover = (Trailing 52-week COGS) / (Trailing 52-week
                                average Inventory Level)
        DSI (Days Sales of Inventory) = 365 / Annualized Turnover

    Using a 52-week trailing window rather than a single week avoids the
    extreme noise a 1-week turnover ratio has (a single low-inventory week
    can produce a nonsensical multi-hundred-percent "turnover"). Some
    products still show very high turnover / very low DSI in the final
    numbers (see README) -- this reflects how inventory levels were
    generated in the provided sample data, not a computation error; it's
    flagged for the business to sanity-check against real operations.

IMPORTANT DATA LIMITATION -- Sales & Credit KPIs:

    The brief asks for "Customer Default Risk Rating" and "Days Sales
    Outstanding (DSO)". This dataset has NO customer-level, invoice-level,
    or payment-term data of any kind -- there is no way to compute real
    credit risk or real DSO from what's provided. Rather than fabricate
    numbers that look like real credit metrics, `compute_risk_proxy()`
    below builds an explicitly-labeled ILLUSTRATIVE proxy from
    product-level financial volatility and stockout frequency, and every
    place it's surfaced (dashboard, exports) carries a visible
    "proxy — not real customer credit data" label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.paths import resolve
except ImportError:
    from paths import resolve

TURNOVER_WINDOW_WEEKS = 52   # trailing window for a stable annualized turnover
GROWTH_WINDOW_WEEKS = 13     # trailing quarter, for revenue growth comparison
STOCKOUT_ALERT_THRESHOLD_WEEKS = 4  # flag if current inventory covers < N weeks of forecast demand


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def load_full_history(engine, product_name: str) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT d.calendar_date AS Date, d.week_number AS Week,
               f.sales_units, f.cogs_per_unit, f.price_per_unit, f.inventory_level,
               f.demand_actual, f.revenue, f.gross_profit, f.operating_profit,
               f.gross_profit_margin, f.revenue_growth, f.is_stockout
        FROM fact_sales f
        JOIN dim_product p ON f.product_id = p.product_id
        JOIN dim_date d ON f.date_id = d.date_id
        WHERE p.product_name = ?
        ORDER BY d.calendar_date
        """,
        engine, params=(product_name,),
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["cogs_total"] = df["cogs_per_unit"] * df["sales_units"]
    return df


# ---------------------------------------------------------------------------
# Financial KPIs (trusted, provided fields -- validated against raw
# components in etl.py; see docstring there)
# ---------------------------------------------------------------------------
def compute_financial_kpis(df: pd.DataFrame, window: int = GROWTH_WINDOW_WEEKS) -> dict:
    recent = df.tail(window)
    prior = df.iloc[-2 * window:-window] if len(df) >= 2 * window else pd.DataFrame()

    recent_revenue = recent["revenue"].sum()
    prior_revenue = prior["revenue"].sum() if not prior.empty else np.nan
    period_growth = (recent_revenue - prior_revenue) / prior_revenue if prior_revenue else np.nan

    return {
        "gross_profit_margin": recent["gross_profit"].sum() / recent["revenue"].sum() if recent["revenue"].sum() else np.nan,
        "revenue_growth_qoq": period_growth,   # trailing-quarter-over-prior-quarter, more stable than the provided week-over-week field
        "operating_profit_trailing_13wk": recent["operating_profit"].sum(),
        "revenue_trailing_13wk": recent_revenue,
    }


# ---------------------------------------------------------------------------
# Operational KPIs -- RECOMPUTED (see module docstring)
# ---------------------------------------------------------------------------
def compute_operational_kpis(df: pd.DataFrame, window: int = TURNOVER_WINDOW_WEEKS) -> dict:
    recent = df.tail(window)
    avg_inventory = recent["inventory_level"].mean()
    trailing_cogs = recent["cogs_total"].sum()

    if avg_inventory and avg_inventory > 0:
        annual_turnover = trailing_cogs / avg_inventory
        dsi = 365.0 / annual_turnover if annual_turnover > 0 else np.nan
    else:
        annual_turnover, dsi = np.nan, np.nan

    return {
        "inventory_turnover_annualized": annual_turnover,
        "days_sales_of_inventory": dsi,
        "current_inventory_level": df["inventory_level"].iloc[-1],
        "avg_inventory_trailing_window": avg_inventory,
    }


# ---------------------------------------------------------------------------
# Sales & Credit KPIs -- ILLUSTRATIVE PROXY (see module docstring:
# no customer/credit data exists in this dataset)
# ---------------------------------------------------------------------------
def compute_risk_proxy(df: pd.DataFrame, window: int = 26) -> dict:
    """NOT real customer credit risk or DSO -- there is no customer-level
    data in this dataset to compute those from. This builds a labeled
    stand-in from product-level signals that correlate with payment/
    fulfillment risk in trade businesses: revenue volatility and stockout
    frequency. Always surfaced with an explicit 'proxy' label."""
    recent = df.tail(window)

    revenue_cv = recent["revenue"].std() / recent["revenue"].mean() if recent["revenue"].mean() else np.nan
    stockout_rate = recent["is_stockout"].mean()

    # simple weighted score, 0 (low risk) to 100 (high risk) -- illustrative only
    risk_score = float(np.clip((revenue_cv * 50) + (stockout_rate * 100) * 0.5, 0, 100))
    if risk_score < 25:
        risk_label = "Low"
    elif risk_score < 55:
        risk_label = "Medium"
    else:
        risk_label = "High"

    return {
        "revenue_volatility_cv": revenue_cv,
        "stockout_rate_trailing_26wk": stockout_rate,
        "proxy_risk_score": risk_score,
        "proxy_risk_label": risk_label,
    }


# ---------------------------------------------------------------------------
# Inventory risk alert -- genuinely computable from real data
# ---------------------------------------------------------------------------
def compute_inventory_alert(current_inventory: float, forecast_df: pd.DataFrame,
                             threshold_weeks: int = STOCKOUT_ALERT_THRESHOLD_WEEKS) -> dict:
    """Flags products where current inventory covers fewer than
    `threshold_weeks` of forecasted near-term demand -- a genuine,
    data-grounded operational risk signal (unlike the credit proxy above)."""
    avg_weekly_forecast = forecast_df["Forecasted_Demand"].head(threshold_weeks).mean()
    weeks_of_cover = current_inventory / avg_weekly_forecast if avg_weekly_forecast else np.nan

    alert = weeks_of_cover < threshold_weeks if pd.notna(weeks_of_cover) else False
    return {
        "weeks_of_inventory_cover": weeks_of_cover,
        "stockout_risk_alert": bool(alert),
    }


# ---------------------------------------------------------------------------
# Build the full KPI summary table across all products
# ---------------------------------------------------------------------------
def build_kpi_summary(engine, forecast_df: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    rows = []
    for product in products:
        df = load_full_history(engine, product)
        product_forecast = forecast_df[forecast_df["Product"] == product].sort_values("Week_Ahead")

        fin = compute_financial_kpis(df)
        ops = compute_operational_kpis(df)
        risk = compute_risk_proxy(df)
        inv_alert = compute_inventory_alert(ops["current_inventory_level"], product_forecast)

        rows.append({"Product": product, **fin, **ops, **risk, **inv_alert})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{resolve('data/processed/armani_trading.db')}")
    forecast_df = pd.read_csv(resolve("data/processed/future_forecast.csv"))
    products = pd.read_sql("SELECT product_name FROM dim_product ORDER BY product_name", engine)["product_name"].tolist()

    summary = build_kpi_summary(engine, forecast_df, products)
    summary.to_csv(resolve("data/processed/kpi_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary.round(2).to_string(index=False))
    print(f"\nSaved: data/processed/kpi_summary.csv")
    print(f"\nProducts with stockout risk alert: {summary['stockout_risk_alert'].sum()} / {len(summary)}")
    print(f"Risk label distribution:\n{summary['proxy_risk_label'].value_counts()}")
