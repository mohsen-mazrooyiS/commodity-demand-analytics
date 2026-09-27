"""
dashboard/app.py
-----------------
Financial & Operational KPI Dashboard (Deliverable 3).

Run with:
    streamlit run dashboard/app.py

Shows, per the assessment brief:
    - Financial KPIs: Gross Profit Margin, Revenue Growth, Operating Profit
    - Operational KPIs: Inventory Turnover Ratio, Days Sales of Inventory (DSI)
    - Sales & Credit KPIs: Customer Default Risk Rating, DSO
      (see the "Data limitations" panel -- this dataset has no customer or
      credit data, so these are an explicitly-labeled illustrative proxy,
      not real credit metrics)
    - Historical trends, 6-month forecasted demand, and inventory/risk alerts

All KPI math lives in src/kpis.py, not here -- this file is presentation only.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import resolve
from src.kpis import (
    load_full_history, compute_financial_kpis, compute_operational_kpis,
    compute_risk_proxy, compute_inventory_alert, build_kpi_summary,
)

st.set_page_config(page_title="Armani Trading — Demand & KPI Dashboard", layout="wide")

TEAL = "#1F6F63"
AMBER = "#C98A2C"
RED = "#B23A48"
NEUTRAL = "#5B6B6A"

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .limitation-box {
        background-color: #FBF2ED; border-left: 4px solid #C98A2C;
        padding: 0.9rem 1.1rem; border-radius: 4px; font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    return create_engine(f"sqlite:///{resolve('data/processed/armani_trading.db')}")


@st.cache_data
def get_products(_engine):
    return pd.read_sql("SELECT product_name FROM dim_product ORDER BY product_name", _engine)["product_name"].tolist()


@st.cache_data
def get_forecast():
    return pd.read_csv(resolve("data/processed/future_forecast.csv"), parse_dates=["Date"])


@st.cache_data
def get_kpi_summary(_engine, products, forecast_df):
    return build_kpi_summary(_engine, forecast_df, products)


engine = get_engine()
products = get_products(engine)
forecast_df = get_forecast()

st.title("Armani Middle East Trading")
st.caption("Demand, financial, and inventory-risk overview across 30 commodities")

tab_overview, tab_product, tab_limitations = st.tabs(["Portfolio overview", "Product deep-dive", "Data limitations"])


# =============================================================================
# TAB 1 — PORTFOLIO OVERVIEW
# =============================================================================
with tab_overview:
    with st.spinner("Computing KPIs across all products..."):
        summary = get_kpi_summary(engine, products, forecast_df)

    n_alerts = int(summary["stockout_risk_alert"].sum())
    n_high_risk = int((summary["proxy_risk_label"] == "High").sum())
    avg_margin = summary["gross_profit_margin"].mean()
    total_op_profit = summary["operating_profit_trailing_13wk"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg. gross profit margin", f"{avg_margin:.1%}")
    c2.metric("Operating profit (trailing 13wk, all products)", f"${total_op_profit:,.0f}")
    c3.metric("Products with stockout risk", f"{n_alerts} / {len(summary)}")
    c4.metric("Products flagged high risk (proxy)", f"{n_high_risk} / {len(summary)}")

    st.subheader("Inventory risk alerts")
    alerts = summary[summary["stockout_risk_alert"]].sort_values("weeks_of_inventory_cover")
    if alerts.empty:
        st.success("No products currently show a near-term stockout risk.")
    else:
        st.warning(f"{len(alerts)} products have less than 4 weeks of inventory cover against forecasted demand.")
        st.dataframe(
            alerts[["Product", "current_inventory_level", "weeks_of_inventory_cover", "days_sales_of_inventory"]]
            .rename(columns={
                "current_inventory_level": "Current inventory",
                "weeks_of_inventory_cover": "Weeks of cover",
                "days_sales_of_inventory": "DSI (days)",
            }).round(2),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Full KPI table")
    display_cols = {
        "Product": "Product",
        "gross_profit_margin": "Gross margin",
        "revenue_growth_qoq": "Revenue growth (QoQ)",
        "operating_profit_trailing_13wk": "Operating profit (13wk)",
        "inventory_turnover_annualized": "Inventory turnover (annualized)",
        "days_sales_of_inventory": "DSI (days)",
        "proxy_risk_label": "Risk proxy",
        "stockout_risk_alert": "Stockout risk",
    }
    st.dataframe(
        summary[list(display_cols.keys())].rename(columns=display_cols)
        .style.format({
            "Gross margin": "{:.1%}", "Revenue growth (QoQ)": "{:.1%}",
            "Operating profit (13wk)": "${:,.0f}", "Inventory turnover (annualized)": "{:.1f}x",
            "DSI (days)": "{:.1f}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption("'Risk proxy' is illustrative — see the Data limitations tab.")


# =============================================================================
# TAB 2 — PRODUCT DEEP-DIVE
# =============================================================================
with tab_product:
    selected_product = st.selectbox("Select a product", products)

    history = load_full_history(engine, selected_product)
    product_forecast = forecast_df[forecast_df["Product"] == selected_product].sort_values("Week_Ahead")

    fin = compute_financial_kpis(history)
    ops = compute_operational_kpis(history)
    risk = compute_risk_proxy(history)
    inv_alert = compute_inventory_alert(ops["current_inventory_level"], product_forecast)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross profit margin (13wk)", f"{fin['gross_profit_margin']:.1%}")
    c2.metric("Revenue growth (QoQ)", f"{fin['revenue_growth_qoq']:.1%}" if pd.notna(fin['revenue_growth_qoq']) else "n/a")
    c3.metric("Operating profit (13wk)", f"${fin['operating_profit_trailing_13wk']:,.0f}")
    c4.metric("Days Sales of Inventory", f"{ops['days_sales_of_inventory']:.1f} days" if pd.notna(ops['days_sales_of_inventory']) else "n/a")

    if inv_alert["stockout_risk_alert"]:
        st.warning(
            f"Stockout risk: current inventory ({ops['current_inventory_level']:,.0f} units) covers only "
            f"{inv_alert['weeks_of_inventory_cover']:.1f} weeks of forecasted demand."
        )

    st.markdown(f"**Sales & credit risk proxy:** {risk['proxy_risk_label']} "
                f"(score {risk['proxy_risk_score']:.0f}/100 — illustrative, see Data limitations tab)")

    st.subheader("Demand: history and 6-month forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["Date"], y=history["demand_actual"],
                              name="Historical demand", line=dict(color=TEAL, width=1.5)))
    fig.add_trace(go.Scatter(x=product_forecast["Date"], y=product_forecast["Forecasted_Demand"],
                              name="Forecast (26 weeks)", line=dict(color=AMBER, width=2, dash="dash")))
    stockout_weeks = history[history["is_stockout"] == 1]
    fig.add_trace(go.Scatter(x=stockout_weeks["Date"], y=stockout_weeks["demand_actual"],
                              mode="markers", name="Stockout week", marker=dict(color=RED, size=6, symbol="x")))
    fig.update_layout(height=400, margin=dict(t=20, b=20), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Revenue & gross profit")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=history["Date"], y=history["revenue"], name="Revenue", line=dict(color=TEAL)))
        fig2.add_trace(go.Scatter(x=history["Date"], y=history["gross_profit"], name="Gross profit", line=dict(color=AMBER)))
        fig2.update_layout(height=320, margin=dict(t=20, b=20), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("Inventory level over time")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=history["Date"], y=history["inventory_level"], name="Inventory",
                                   line=dict(color=NEUTRAL), fill="tozeroy"))
        fig3.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
# TAB 3 — DATA LIMITATIONS (transparency, not buried in a footnote)
# =============================================================================
with tab_limitations:
    st.subheader("What this dashboard can and can't show, and why")

    st.markdown("""
<div class="limitation-box">
<b>Sales & Credit KPIs (Customer Default Risk Rating, Days Sales Outstanding)</b><br>
The source dataset has no customer identity, invoice, or payment-term data of any kind —
only product-level weekly sales, inventory, and pricing. It is not possible to compute
real customer credit risk or real DSO from this data.<br><br>
The "risk proxy" shown elsewhere in this dashboard is a stand-in built from
<i>product-level</i> financial volatility and stockout frequency, not customer payment
behavior. It's a reasonable operational signal (volatile, frequently-stocked-out products
carry more commercial risk) but it is <b>not</b> a customer credit score, and shouldn't be
presented to stakeholders as one.
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
<div class="limitation-box">
<b>Inventory Turnover Ratio</b><br>
The source file's Inventory_Turnover_Ratio column doesn't reconcile with any combination
of the raw components (COGS, Sales_Units, Inventory_Level) — it appears to be an
independently-generated placeholder. This dashboard recomputes turnover and Days Sales
of Inventory from first principles (trailing 52-week COGS ÷ average inventory), and
flags this recomputation explicitly rather than silently overriding the source file.
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
<div class="limitation-box">
<b>Sales_Units vs. Demand_Forecast</b><br>
~17% of weeks in this dataset are stockouts (Sales_Units = 0 while Inventory_Level = 0).
In these weeks, Sales_Units understates true demand — Demand_Forecast remains the
uncensored estimate and is what the forecasting model (Stage 2) predicts. Stockout weeks
are marked with an × on the demand chart in the Product deep-dive tab.
</div>
""", unsafe_allow_html=True)

    st.caption("See the project README for the full write-up of every cleaning and modeling decision.")
