"""
pricing.py
----------
Dynamic Pricing & Risk Strategy (Deliverable 4).

A rule-based pricing engine: takes a product's current price, recent demand
history, Stage 2's forecast, and Stage 3's inventory/risk KPIs, and returns
a recommended price adjustment with a full audit trail of WHY (each
multiplier applied, in order), not just a number.

THE FRAMEWORK, IN ONE FORMULA:

    recommended_price = current_price
                         x demand_multiplier      (peaks push price up, lulls push it down)
                         x inventory_multiplier   (scarcity pushes price up, excess pushes it down)
                         , then:
                         -> credit-risk guardrail applied (caps/removes discounts for volatile products)
                         -> competitive guardrail applied (hard cap on total swing either direction)

Each piece below is deliberately simple and auditable (linear, a handful of
named parameters) rather than a black-box model -- a pricing engine that a
category manager can't explain to a customer or a regulator is a liability,
not an asset, in a commodity trading business.

DEMAND-BASED ADJUSTMENT
    demand_index = (avg forecasted demand, next 4 weeks) / (trailing 52-week
                    average actual demand)
    A value above 1 means we're heading into a demand peak relative to the
    product's normal run-rate; below 1, a lull. This already reflects
    seasonality, because the Stage 2 forecast itself was trained on lag-52
    and calendar features -- there's no need for a second, separate
    seasonal adjustment on top of it.

    demand_multiplier = 1 + PRICE_ELASTICITY_DEMAND * (demand_index - 1)

INVENTORY & SEASONALITY SENSITIVITY
    weeks_of_cover = current inventory / avg forecasted demand (next 4 weeks)
    inventory_gap = (TARGET_WEEKS_OF_COVER - weeks_of_cover) / TARGET_WEEKS_OF_COVER
    A positive gap means we're under-stocked relative to a healthy buffer
    (raise price to ration scarce stock and protect margin); negative means
    we're overstocked (lower price to accelerate turnover and cut holding
    cost). TARGET_WEEKS_OF_COVER is the same 4-week policy threshold used
    for Stage 3's stockout alerts, so the two systems agree with each other.

    inventory_multiplier = 1 + PRICE_ELASTICITY_INVENTORY * inventory_gap

    Because demand_index already captures season (peak/off-peak) and
    inventory_gap captures stock position, the four qualitative cases the
    brief asks for fall out of the combination automatically:
        Peak demand   + Low stock  -> both multipliers > 1 -> scarcity premium
        Peak demand   + High stock -> demand up, inventory down -> partially offsetting, usually a modest net increase
        Off-peak      + Low stock  -> demand down, inventory up -> partially offsetting, usually close to flat
        Off-peak      + High stock -> both multipliers < 1 -> clearance markdown

COMPETITIVE & CREDIT RISK GUARDRAILS
    Credit risk guardrail: this dataset has no real customer-level credit
    data (see Stage 3's "Data limitations"), so we reuse the same
    product-level risk proxy (revenue volatility + stockout frequency).
    High-risk products are not discounted at all -- a volatile, frequently
    -out-of-stock product is exactly the wrong place to chase volume with a
    markdown; Medium-risk products have any discount halved.

    Competitive guardrail: we have no real competitor price feed either, so
    this is a documented, symmetric self-referential cap -- the final price
    is never allowed to move more than MAX_PRICE_SWING (default 15%) from
    the current price in either direction, regardless of what the demand/
    inventory math alone would suggest. A production system would replace
    this with a live competitor price band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.paths import resolve
except ImportError:
    from paths import resolve

# ---- tunable policy parameters (all exposed as sliders in the dashboard) ----
# Calibration note: initial defaults of 0.20/0.25 caused the competitive
# guardrail to bind on 27/30 products -- meaning the "guardrail" was
# functioning as the actual policy, not a rare safety net. Diagnosed via
# the raw (pre-guardrail) multiplier distribution: most products in this
# dataset sit well below their 4-week inventory target (many mid-stockout),
# so inventory_gap sits near its upper bound (~0.77 median) far more often
# than a "normal" business would see. Recalibrated elasticities down to
# 0.10/0.10, which brings the guardrail down to firing on 5/30 products
# (~17%) -- a real safety net for genuine extremes, not a constant clamp.
PRICE_ELASTICITY_DEMAND = 0.10      # how aggressively price responds to demand swings
PRICE_ELASTICITY_INVENTORY = 0.10   # how aggressively price responds to stock position
TARGET_WEEKS_OF_COVER = 4.0         # "healthy" inventory buffer, matches Stage 3's alert threshold
MAX_PRICE_SWING = 0.15              # competitive guardrail: max +/-15% from current price
FORECAST_WINDOW_WEEKS = 4           # near-term forecast window used for demand_index and weeks_of_cover
DEMAND_BASELINE_WINDOW_WEEKS = 52   # trailing window defining a product's "normal" demand level


def compute_price_recommendation(
    current_price: float,
    trailing_demand: pd.Series,
    forecast_demand: pd.Series,
    current_inventory: float,
    risk_label: str,
    *,
    elasticity_demand: float = PRICE_ELASTICITY_DEMAND,
    elasticity_inventory: float = PRICE_ELASTICITY_INVENTORY,
    target_weeks_cover: float = TARGET_WEEKS_OF_COVER,
    max_swing: float = MAX_PRICE_SWING,
) -> dict:
    """Returns the recommended price plus a full breakdown of every rule applied."""

    baseline_demand = trailing_demand.tail(DEMAND_BASELINE_WINDOW_WEEKS).mean()
    near_term_forecast = forecast_demand.head(FORECAST_WINDOW_WEEKS).mean()

    demand_index = near_term_forecast / baseline_demand if baseline_demand else 1.0
    demand_index = float(np.clip(demand_index, 0.3, 3.0))  # sanity bound before it hits the multiplier
    demand_multiplier = 1 + elasticity_demand * (demand_index - 1)

    weeks_of_cover = current_inventory / near_term_forecast if near_term_forecast else np.inf
    inventory_gap = (target_weeks_cover - weeks_of_cover) / target_weeks_cover
    inventory_gap = float(np.clip(inventory_gap, -3.0, 3.0))
    inventory_multiplier = 1 + elasticity_inventory * inventory_gap

    raw_multiplier = demand_multiplier * inventory_multiplier

    # --- credit-risk guardrail (product-level proxy -- see Stage 3 limitations) ---
    risk_note = "no adjustment"
    guarded_multiplier = raw_multiplier
    if risk_label == "High":
        if guarded_multiplier < 1.0:
            guarded_multiplier = 1.0
            risk_note = "High-risk product: discount removed entirely"
    elif risk_label == "Medium":
        if guarded_multiplier < 1.0:
            guarded_multiplier = 1.0 - (1.0 - guarded_multiplier) * 0.5
            risk_note = "Medium-risk product: discount halved"

    # --- competitive guardrail: hard cap on total swing, independent of the above ---
    final_multiplier = float(np.clip(guarded_multiplier, 1 - max_swing, 1 + max_swing))
    competitive_capped = abs(final_multiplier - guarded_multiplier) > 1e-9

    recommended_price = round(current_price * final_multiplier, 2)

    if demand_index > 1.05 and inventory_gap > 0.05:
        quadrant = "Peak demand + Low stock -> scarcity premium"
    elif demand_index > 1.05 and inventory_gap <= 0.05:
        quadrant = "Peak demand + High stock -> modest premium, no urgency"
    elif demand_index <= 1.05 and inventory_gap > 0.05:
        quadrant = "Off-peak + Low stock -> hold price, protect future supply"
    else:
        quadrant = "Off-peak + High stock -> clearance markdown"

    return {
        "current_price": current_price,
        "recommended_price": recommended_price,
        "price_change_pct": final_multiplier - 1,
        "demand_index": demand_index,
        "demand_multiplier": demand_multiplier,
        "weeks_of_cover": weeks_of_cover,
        "inventory_gap": inventory_gap,
        "inventory_multiplier": inventory_multiplier,
        "raw_multiplier": raw_multiplier,
        "risk_label": risk_label,
        "risk_guardrail_note": risk_note,
        "competitive_cap_applied": competitive_capped,
        "quadrant": quadrant,
    }


def build_pricing_table(engine, forecast_df: pd.DataFrame, kpi_summary: pd.DataFrame, products: list[str],
                         **policy_kwargs) -> pd.DataFrame:
    """Runs the pricing engine for every product using the latest known price
    and current KPI/risk data, returning one row per product."""
    rows = []
    for product in products:
        history = pd.read_sql(
            """
            SELECT d.calendar_date AS Date, f.demand_actual, f.price_per_unit
            FROM fact_sales f
            JOIN dim_product p ON f.product_id = p.product_id
            JOIN dim_date d ON f.date_id = d.date_id
            WHERE p.product_name = ?
            ORDER BY d.calendar_date
            """,
            engine, params=(product,),
        )
        product_forecast = forecast_df[forecast_df["Product"] == product].sort_values("Week_Ahead")
        kpi_row = kpi_summary[kpi_summary["Product"] == product].iloc[0]

        rec = compute_price_recommendation(
            current_price=history["price_per_unit"].iloc[-1],
            trailing_demand=history["demand_actual"],
            forecast_demand=product_forecast["Forecasted_Demand"],
            current_inventory=kpi_row["current_inventory_level"],
            risk_label=kpi_row["proxy_risk_label"],
            **policy_kwargs,
        )
        rows.append({"Product": product, **rec})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{resolve('data/processed/armani_trading.db')}")
    forecast_df = pd.read_csv(resolve("data/processed/future_forecast.csv"))
    kpi_summary = pd.read_csv(resolve("data/processed/kpi_summary.csv"))
    products = pd.read_sql("SELECT product_name FROM dim_product ORDER BY product_name", engine)["product_name"].tolist()

    pricing_table = build_pricing_table(engine, forecast_df, kpi_summary, products)
    pricing_table.to_csv(resolve("data/processed/pricing_recommendations.csv"), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    display_cols = ["Product", "current_price", "recommended_price", "price_change_pct",
                     "demand_index", "weeks_of_cover", "risk_label", "quadrant"]
    print(pricing_table[display_cols].round(3).to_string(index=False))
    print(f"\nSaved: data/processed/pricing_recommendations.csv")
    print(f"\nQuadrant distribution:\n{pricing_table['quadrant'].value_counts()}")
    print(f"\nProducts with competitive cap engaged: {pricing_table['competitive_cap_applied'].sum()} / {len(pricing_table)}")
