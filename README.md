# Armani Middle East Trading — Demand, KPI & Pricing Platform

Technical assessment submission: end-to-end pipeline covering data
engineering, 6-month demand forecasting, financial/operational KPI
reporting, and a dynamic pricing framework.

## Status

- [x] **Stage 1 — Data Engineering, ETL & Database**
- [x] **Stage 2 — Time-series demand forecasting & factor analysis** (this commit)
- [ ] Stage 3 — Financial & operational KPI dashboard
- [ ] Stage 4 — Dynamic pricing & risk strategy

## Architecture

```
Raw data (Excel/CSV)
        |
        v
   [ etl.py ]  --clean--> data/processed/cleaned_sales_data.parquet
        |
        v
  [ SQLite star schema ]  data/processed/armani_trading.db
   dim_product / dim_region / dim_customer_segment / dim_date
                    |
              fact_sales (grain: product x week x region x segment)
        |
        v
  [ Stage 2: forecasting ] -> [ Stage 3: KPI dashboard ] -> [ Stage 4: pricing engine ]
```

## The real dataset: schema, findings, and limitations

`data/raw/Sample_Data.xlsx` (the real file provided for this assessment)
contains 7,839 rows: 30 products x ~260 weeks, with `Week`, `Product`,
`Sales_Units`, `Cost_of_Goods_Sold_COGS`, `Inventory_Level`,
`Demand_Forecast`, `Price_Per_Unit`, and pre-computed `Revenue`,
`Gross_Profit`, `Operating_Profit`, `Inventory_Turnover_Ratio`,
`Gross_Profit_Margin`, and `Revenue_Growth`.

**Key finding — Sales_Units is inventory-censored.** ~17% of rows (429
after cleaning) have `Sales_Units = 0` coinciding with `Inventory_Level =
0`. These are stockout weeks: `Demand_Forecast` remains populated and
non-zero in these weeks (customers wanted the product, it wasn't
available), while `Sales_Units` only reflects what was actually sold. This
is why `Demand_Forecast`, not `Sales_Units`, is the correct forecasting
target, and why `Sales_Units` needs care as a model feature. We flag these
rows with `is_stockout` rather than silently treating 0 as a real demand
signal.

**Data-quality issues found and how they were handled:**

| Issue | Approach | Rationale |
|---|---|---|
| 39 exact duplicate (Week, Product) rows | Dropped, kept first | No legitimate reason for identical rows at this grain |
| `Demand_Forecast` missing in 606 rows (7.8%) | Linear interpolation, per product, time-ordered | This is an ordered weekly series — interpolation preserves trend/seasonality far better than a flat median fill |
| `Sales_Units` missing in 16 rows | Same, linear interpolation per product | Same reasoning; these are the rows where `Revenue`/`Gross_Profit` will no longer exactly back-calculate (expected, documented in code) |
| `Revenue_Growth` = `inf` in 895 rows | Replaced with NaN, then capped at +/-500%, then filled | These are legitimate divide-by-zero results (prior week's revenue was genuinely $0 during a stockout), not data errors — capping preserves the "just recovered from stockout" signal without breaking downstream models |
| Provided `Revenue`/`Gross_Profit` vs. raw components | Cross-validated (`Revenue = Sales_Units x Price_Per_Unit`, `Gross_Profit = Revenue - COGS x Sales_Units`) | 0 mismatches among non-null values — the provided derived fields are trustworthy, not recomputed |
| No `Date` column, only sequential `Week` (1-260) | Synthesized a calendar date, anchoring Week 1 = 2021-01-06 | **Documented assumption**, not a fact recovered from data — used only to derive month/quarter/year seasonal features |

**Data limitations vs. the assessment brief.** The brief's KPI table asks
for `Customer Default Risk Rating` and `Days Sales Outstanding` under
"Sales & Credit" — this requires customer-level or transaction-level credit
data (payment terms, invoice aging, customer identity), none of which
exists in this file. There's no `Region` or `Customer_Segment` either.
Stage 3 will compute every KPI that *is* supportable by this data (Gross
Margin, Revenue Growth, Operating Profit, Inventory Turnover, Days Sales
of Inventory) and will clearly label the credit-risk metrics as
**illustrative/proxy logic** built from product-level financial patterns
(e.g., revenue volatility as a rough proxy for demand-driven payment risk)
rather than real customer credit behavior — flagged as such in the
dashboard, not presented as if it were measured.

An earlier draft of this pipeline (`src/generate_synthetic_data.py`) used
placeholder synthetic data before the real file was available. It's kept
in the repo for reference but is no longer part of the active pipeline.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**macOS users:** LightGBM's macOS wheel depends on the OpenMP runtime
(`libomp`), which pip cannot install itself. If `import lightgbm` fails
with an `OSError: ... Library not loaded: @rpath/libomp.dylib` error, run:

```bash
brew install libomp
```

then restart your Python kernel/session (a simple re-run of the import
won't pick up the newly installed library).

**Running notebooks:** all scripts resolve `data/`, `sql/`, etc. relative
to the project root via `src/paths.py`, regardless of the notebook's
working directory — so `notebooks/*.ipynb` can `from src.etl import ...`
and `from src.forecasting import ...` directly without needing an `os.chdir()`
workaround first.

## Running Stage 1 (ETL)

```bash
# (optional) regenerate the placeholder dataset
python src/generate_synthetic_data.py

# run the ETL pipeline: clean raw data -> load into SQLite star schema
python src/etl.py
```

This produces:
- `data/processed/cleaned_sales_data.parquet` — cleaned flat file
- `data/processed/armani_trading.db` — SQLite star schema (see `sql/schema.sql`)

Cleaning decisions (imputation method, outlier handling, dedup counts) are
logged to stdout via `CleaningReport` — see `etl.py` for the full report
generated on the most recent run.

## Stage 2 — Demand Forecasting

**Approach: 30 separate per-product LightGBM models** (per-product chosen
over a single global model — these 30 commodities have genuinely different
demand dynamics, e.g. Rice vs. Spices vs. Dairy, and per-product data volume
(~260 weeks) is enough to support a dedicated small model per product
without the complexity of a global multi-product architecture).

### Model iteration: what we tried first, and why it changed

The first design followed the "textbook" hybrid: MSTL-decompose each
product's demand into trend + seasonal(52) + seasonal(4), then train
LightGBM on the leftover remainder only. On walk-forward CV, **this hybrid
lost to a naive seasonal baseline** (this week's demand = demand 52 weeks
ago) on every product tested.

Root cause, confirmed via autocorrelation analysis: demand shows strong
signal at lag 1 (~0.70, short-term momentum) and lag 52 (~0.69, genuine
annual seasonality) — real, usable signal. But with only ~5 years of
weekly history, MSTL sees just 2.5–5 annual cycles per training window.
Its smoothed seasonal component blurs the exact product-specific
week-to-week pattern that the raw lag-52 value captures precisely, and the
extra seasonal(4) term mostly fit noise as if it were a real monthly
cycle. Decomposing first and extrapolating the pieces separately threw
away exactly the sharp signal a feature-based model could use directly.

**Final design:** one LightGBM regressor per product, trained directly on
`Demand_Forecast` using lag features (1, 2, 3, 4, 8, 12, 26, **52** weeks),
rolling means/stds, calendar (sin/cos of week-of-year, month, quarter),
lagged price/COGS/inventory, and a recent-stockout flag. MSTL is kept as a
**descriptive factor-analysis tool** (`decompose_for_analysis()`) to
visualize and explain trend/seasonal/residual structure — satisfying the
brief's "factor analysis" ask — but it's no longer inside the prediction
path itself.

### Evaluation

Walk-forward (rolling-origin), **fully recursive** cross-validation: each
fold predicts week 1, appends that prediction, recomputes lag/rolling
features from the now-extended series, predicts week 2, and so on — the
same mechanism used for the real 26-week-ahead forecast, so CV honestly
reflects production performance rather than leaking future actuals into
lag features.

| Model | Mean MAE | Mean RMSE | Mean MAPE |
|---|---|---|---|
| Baseline (seasonal-naive, demand 52 weeks ago) | 254.2 | 312.8 | 14.30% |
| **LightGBM (direct, per-product)** | **219.1** | **270.9** | **12.61%** |

The model beats the seasonal-naive baseline on **26 of 30 products**
(87%), improving average MAPE by ~12%. The four products where the
baseline still wins (Cocoa Powder, Milk Powder - Whole, Spices - Turmeric,
Sugar - White Refined) are flagged in `data/processed/cv_metrics.csv` for
follow-up — likely candidates for a longer lookback window or
product-specific hyperparameter tuning in a future iteration, rather than
a sign the overall approach is wrong.

### Leakage safeguards

- `Sales_Units` is excluded from features entirely — it's inventory-censored
  (see ETL findings: ~17% of weeks are stockouts where Sales_Units=0 but
  true demand wasn't) and would teach the model to under-forecast during
  stockout recovery.
- `Price_Per_Unit`, `Cost_of_Goods_Sold_COGS`, `Inventory_Level` are used
  only in **lagged** form (1-week and 4-week lags), since it's unclear
  whether price is set in advance (safe) or reacts to demand (leakage) —
  lagging is the conservative default.
- Future exogenous drivers (price, COGS, inventory) are unknown at
  forecast time, so the recursive forecaster carries the last observed
  values forward — a documented assumption, easily replaced with a real
  pricing/inventory plan if the business has one.

### Running it

```bash
python src/forecasting.py
```

Produces:
- `data/processed/cv_metrics.csv` — per-product, per-fold MAE/RMSE/MAPE for both the model and the baseline
- `data/processed/future_forecast.csv` — the 26-week-ahead forecast for all 30 products

## Repository structure

```
armani-demand-forecast/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # Sample_Data.xlsx (real data, provided by Armani)
│   └── processed/      # cleaned parquet + SQLite DB (generated, git-ignored)
├── sql/
│   └── schema.sql      # star schema DDL (dim_product, dim_date, fact_sales)
├── src/
│   ├── paths.py                       # project-root-relative path resolution (works from notebooks too)
│   ├── generate_synthetic_data.py   # superseded — kept for reference only
│   ├── etl.py                        # ingestion, cleaning, validation, DB load
│   └── forecasting.py                # per-product LightGBM demand forecasting + walk-forward CV
├── notebooks/           # exploratory analysis (Stage 2+)
└── dashboard/           # Streamlit app (Stage 3+)
```
