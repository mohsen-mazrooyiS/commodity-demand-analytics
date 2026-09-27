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

## Key assumption: the provided data vs. this build

The assessment brief references `Sample_Data.xlsx` with (at minimum) `Week`,
`Product`, `Sales_Units`, and `Cost_of_Goods_Sold_COGS`. The KPI/pricing
requirements (Gross Margin, DSO, Customer Default Risk, Inventory Turnover)
need price, inventory, and customer/credit fields that weren't visible in
the schema excerpt I received, so **`src/generate_synthetic_data.py`
generates a placeholder dataset with the full schema I'd expect**, with
realistic data-quality issues (nulls, duplicates, outliers, bad casing)
deliberately injected so the ETL step has real work to do.

**To use the real dataset:** drop `Sample_Data.xlsx` into `data/raw/` and
run `etl.py` with that path instead — no other code changes needed as long
as the column names match `REQUIRED_COLUMNS` in `etl.py` (the pipeline
will raise a clear error listing any missing columns otherwise).

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

## Data cleaning decisions

| Issue | Approach | Rationale |
|---|---|---|
| Missing numeric values | Median imputation, grouped by `Product` | Preserves product-level scale (a missing Rice price shouldn't be filled with a Coffee median) |
| Exact duplicate rows | Dropped | No legitimate reason for identical rows at this grain |
| Extreme outliers (e.g. 100x sales spike) | Capped via IQR (k=3, conservative) | Caps data-entry errors without discarding legitimate high-demand weeks |
| Negative cost/price values | Took absolute value | Treated as sign-entry errors, not genuine negative costs |
| Inconsistent casing/whitespace (`"RICE  "` vs `"Rice"`) | Standardized via `.str.strip().str.title()` | Prevents the same product being split into multiple dimension rows |
| Rows missing both `Week`/`Date`/`Product` and any demand signal | Dropped | Unrecoverable — no reasonable imputation for a missing primary key or missing target |

## Repository structure

```
armani-demand-forecast/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # source files (place Sample_Data.xlsx here)
│   └── processed/      # cleaned parquet + SQLite DB (generated, git-ignored)
├── sql/
│   └── schema.sql      # star schema DDL
├── src/
│   ├── generate_synthetic_data.py   # placeholder data (remove once real file provided)
│   └── etl.py                        # ingestion, cleaning, DB load
├── notebooks/           # exploratory analysis (Stage 2+)
└── dashboard/           # Streamlit app (Stage 3+)
```
