# Dynamic Pricing & Risk Strategy

**Deliverable 4** — a rule-based, auditable pricing engine (`src/pricing.py`),
documented here with the exact mathematical logic, calibrated and tested
against Armani's real 30-product dataset.

## Why rule-based, not a black-box model

A pricing engine that a category manager can't explain to a customer, or
justify to a regulator, is a liability in a commodity trading business —
this is deliberately a small set of linear, named, inspectable rules
rather than an ML model. Every recommendation comes with a full breakdown
of which factor moved it and by how much (see "Audit trail" below).

## The framework

```
recommended_price = current_price
                     × demand_multiplier       (peaks push price up, lulls push it down)
                     × inventory_multiplier    (scarcity pushes price up, excess pushes it down)
                     → credit-risk guardrail applied
                     → competitive guardrail applied (hard cap on total swing)
```

### 1. Demand-based adjustment

```
demand_index = (avg forecasted demand, next 4 weeks) / (trailing 52-week average actual demand)
demand_multiplier = 1 + α_demand × (demand_index − 1)
```

`demand_index > 1` means the near-term forecast (Stage 2's model) sits
above the product's normal run-rate — a demand peak. Below 1, a lull.
This already reflects seasonality: the forecasting model itself was
trained on `lag_52` and calendar (sin/cos week-of-year) features, so a
seasonal December spike is already baked into the forecast that feeds
this formula — there's no separate seasonal adjustment layered on top,
because that would double-count the same signal.

### 2. Inventory & seasonality sensitivity

```
weeks_of_cover = current_inventory / (avg forecasted demand, next 4 weeks)
inventory_gap = (TARGET_WEEKS_OF_COVER − weeks_of_cover) / TARGET_WEEKS_OF_COVER
inventory_multiplier = 1 + α_inventory × inventory_gap
```

`TARGET_WEEKS_OF_COVER = 4` — deliberately the same policy threshold used
for Stage 3's stockout alerts, so the two systems agree with each other
rather than defining "healthy stock" two different ways.

A positive gap = under-stocked relative to the 4-week buffer → raise
price (ration scarce stock, protect margin while supply catches up). A
negative gap = overstocked → lower price (accelerate turnover, cut
holding cost).

**Why this covers "high-stock vs. low-stock during peak and off-peak"
without a separate rule for each combination:** the four cases in the
brief fall out automatically from multiplying these two independent
signals together:

| | Low stock | High stock |
|---|---|---|
| **Peak demand** | Both multipliers > 1 → **scarcity premium** (raise price, ration supply) | Demand up, inventory down → partially offsetting → **modest premium, no urgency** |
| **Off-peak** | Demand down, inventory up → partially offsetting → **hold price, protect future supply** | Both multipliers < 1 → **clearance markdown** |

### 3. Competitive & credit risk guardrails

**Credit risk guardrail.** This dataset has no real customer-level credit
or payment data (see Stage 3's "Data limitations" — same constraint
applies here), so this reuses the same product-level risk proxy (revenue
volatility + stockout frequency) rather than fabricating a customer-level
rule that isn't backed by real data:

- **High risk:** any discount is removed entirely (floored at 0% change).
  A volatile, frequently-out-of-stock product is exactly the wrong place
  to chase volume with a markdown.
- **Medium risk:** any discount is halved.
- **Low risk:** no adjustment — full demand/inventory-driven discount applies.

**Competitive guardrail.** No live competitor price feed exists in this
project, so this is a documented, symmetric, self-referential cap: the
final recommended price is never allowed to move more than
`MAX_PRICE_SWING` (default **±15%**) from the current price, regardless of
what the demand/inventory math alone would suggest. **In a production
deployment, this constant would be replaced by a live competitor price
band** (e.g., stay within ±X% of the median tracked competitor price for
that commodity) — flagged here as a data limitation, not treated as a
solved problem.

**What triggers "High risk" in the first place**

This isn't customer credit risk (this dataset has no real customer data — see Stage 3's limitations). It's a product-level proxy, computed earlier in kpis.py from two signals:

Revenue volatility — how much a product's weekly revenue swings around relative to its own average (coefficient of variation)
Stockout frequency — how often it's run out of stock in the last 26 weeks

A product that's both volatile and frequently out of stock gets labeled "High" risk.

**What the guardrail does with that label**

The demand/inventory math (the earlier steps) computes a recommended price change — for example "lower the price by 13.1%" because inventory is high and demand is soft. Before that recommendation goes out, the code checks:

if risk_label == "High" and the recommendation is a discount:
    cancel the discount entirely → 0% change (keep price as-is)

The reasoning in plain terms: if a product is unpredictable (revenue swings wildly) and keeps running out of stock, that's usually a sign of an underlying supply or demand problem — not something a price cut fixes. Discounting it to "chase volume" would:

Erode margin on a product that's already financially unstable
Encourage even more demand right as supply is already unreliable, making the next stockout worse, not better

So the rule is essentially: don't put your least-stable products on sale — that's the opposite of what they need. Medium-risk products get a softer version (discount cut in half, not removed); Low-risk products get whatever the demand/inventory math recommends, unchanged.

## Calibration — what we tried, what broke, and the fix

The first version used `α_demand = 0.20`, `α_inventory = 0.25`. Tested
against all 30 products, the competitive guardrail (±15%) engaged on
**27 of 30 products (90%)** — meaning the "guardrail" had become the
actual pricing policy, not a rare safety net for extremes. That defeats
the purpose of having a guardrail at all.

Root cause: most products in this dataset sit well below their 4-week
inventory target at the most recent observed week (many mid-stockout —
see Stage 1's stockout finding), so `inventory_gap` sits near its upper
bound (median ≈ 0.77 of the way to its clipped max) far more often than a
typical business would see. With `α_inventory = 0.25`, that alone produces
a ~19% swing before the demand term or guardrail even get involved.

**Fix:** recalibrated to `α_demand = 0.10`, `α_inventory = 0.10`. Re-tested:
the guardrail now engages on **5 of 30 products (17%)** — a real safety
net, catching genuine extremes (e.g. Spices - Turmeric and Wheat - Hard
Red, both carrying 300+ weeks of inventory cover, correctly floored at the
maximum -15% markdown) rather than binding on nearly everything.

## Verified guardrail behavior (synthetic test — see `src/pricing.py` docstring)

The current real dataset happens to contain only one "High"-risk product,
and it isn't in a discount-eligible situation — so the credit-risk
guardrail's discount-blocking branch never visibly fires on real data.
Rather than leave that unverified, it was tested directly with a
constructed scenario (overstocked, demand lull — a case that would
otherwise trigger a markdown):

| Risk label | Raw (unguarded) price change | After guardrail |
|---|---|---|
| Low | −13.1% | −13.1% (unchanged) |
| Medium | −13.1% | **−6.6%** (halved) |
| High | −13.1% | **0.0%** (fully blocked) |

## Audit trail

Every call to `compute_price_recommendation()` returns not just the
recommended price but every intermediate value: `demand_index`,
`weeks_of_cover`, `inventory_gap`, each multiplier, which guardrail (if
any) fired and why, and a plain-language `quadrant` label. This is
deliberately verbose — a pricing decision that affects revenue should
never be a single opaque number.

## Running it

```bash
python src/pricing.py
```

Produces `data/processed/pricing_recommendations.csv` (one row per
product) and prints the same table plus a quadrant-distribution summary
to the console. All four policy parameters
(`elasticity_demand`, `elasticity_inventory`, `target_weeks_cover`,
`max_swing`) are keyword arguments to `compute_price_recommendation()` /
`build_pricing_table()`, and are exposed as live sliders in the dashboard's
**Pricing engine** tab (`streamlit run dashboard/app.py`) so a category
manager can see how a recommendation changes as policy assumptions change,
without touching code.

## What a production version would add

- A real competitor price feed, replacing the symmetric self-referential
  competitive guardrail with an actual market-position constraint.
- Real customer-level credit/payment data, replacing the product-level
  risk proxy with genuine per-customer credit terms and default risk.
- Price-elasticity estimation per product (from historical price/volume
  variation) rather than a single global `α_demand` — some commodities
  (staples like Rice, Wheat) are far less price-elastic than discretionary
  ones (Spices, Cocoa), and the engine currently treats them identically.
