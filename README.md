# Armani Middle East Trading — Demand, KPI & Pricing Platform

Technical assessment submission: end-to-end pipeline covering data
engineering, 6-month demand forecasting, financial/operational KPI
reporting, and a dynamic pricing framework.

## Status

- [x] **Stage 1 — Data Engineering, ETL & Database** (this commit)
- [ ] Stage 2 — Time-series demand forecasting & factor analysis
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
│   ├── generate_synthetic_data.py   # superseded — kept for reference only
│   └── etl.py                        # ingestion, cleaning, validation, DB load
├── notebooks/           # exploratory analysis (Stage 2+)
└── dashboard/           # Streamlit app (Stage 3+)
```
