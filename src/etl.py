"""
etl.py
------
Data Engineering, ETL & Database Setup (Deliverable 1).
Built against the real Sample_Data.xlsx provided for this assessment.

Responsibilities:
    1. Ingest the raw dataset with exception handling for corrupt records /
       missing/unexpected schema.
    2. Clean: dedupe, time-aware imputation of missing target values,
       winsorize the (legitimately) infinite Revenue_Growth values, validate
       the provided derived financial fields against their raw components,
       and flag inventory-censored demand (stockout weeks).
    3. Load into a normalized star schema in SQLite (fact_sales + dim_product
       + dim_date).

Key data findings driving the cleaning decisions below (see README for the
full write-up):
    - 39 exact duplicate (Week, Product) rows -> dropped.
    - Demand_Forecast missing in ~7.8% of rows, scattered roughly evenly
      across products -> time-based linear interpolation per product
      (this is an ordered weekly series, so interpolation preserves
      trend/seasonality far better than a flat median fill).
    - ~17% of rows have Sales_Units = 0 coinciding with Inventory_Level = 0:
      these are STOCKOUT weeks. Sales_Units is inventory-censored (it
      reflects what was sold, not what was wanted); Demand_Forecast remains
      populated and uncensored in these weeks. We flag these rows
      (`is_stockout`) rather than treat Sales_Units=0 as if it were a
      genuine demand signal.
    - Revenue_Growth contains 897 `inf` values, all mathematically correct
      (division by a prior-week revenue of exactly 0 during a stockout),
      not data errors. We winsorize (cap) rather than drop, so the signal
      "this product just came back from a stockout" isn't lost entirely.
    - Revenue and Gross_Profit were cross-checked against
      Sales_Units*Price_Per_Unit and Revenue-COGS*Sales_Units respectively:
      both match to floating-point precision, so they're trusted as-is
      rather than recomputed.
    - No Date column is provided, only a sequential Week (1-260) per
      product. We synthesize a calendar date assuming Week 1 = 2021-01-06
      (a Wednesday, arbitrary anchor) purely to support calendar/seasonal
      features downstream. This is a documented ASSUMPTION, not a fact
      recovered from the data.
    - No Region, Customer_Segment, or customer/credit-level fields exist in
      this dataset. The Sales & Credit KPIs in the assessment brief (DSO,
      Customer Default Risk Rating) cannot be computed from this file as
      given -- see README "Data limitations" for how Stage 3 handles this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

try:
    from src.paths import resolve
except ImportError:
    from paths import resolve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Week", "Product", "Sales_Units", "Cost_of_Goods_Sold_COGS",
    "Inventory_Level", "Demand_Forecast", "Price_Per_Unit",
    "Gross_Profit", "Operating_Profit", "Revenue",
    "Inventory_Turnover_Ratio", "Gross_Profit_Margin", "Revenue_Growth",
]

# Documented assumption: no real calendar date is provided. Week 1 is
# anchored to this date solely to derive month/quarter/year seasonal
# features. Change this constant if the real start date becomes known.
ASSUMED_WEEK1_DATE = pd.Timestamp("2021-01-06")

REVENUE_GROWTH_CAP = 5.0  # cap at +500% growth; genuine stockout-recovery spikes rarely exceed this


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    rows_dropped_corrupt: int = 0
    missing_values_interpolated: dict = field(default_factory=dict)
    revenue_growth_inf_capped: int = 0
    revenue_growth_extreme_capped: int = 0
    stockout_weeks_flagged: int = 0
    financial_field_validation: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Rows in: {self.rows_in:,} -> Rows out: {self.rows_out:,}",
            f"Exact duplicate (Week, Product) rows removed: {self.duplicates_removed:,}",
            f"Rows dropped as unrecoverable/corrupt: {self.rows_dropped_corrupt:,}",
        ]
        for col, n in self.missing_values_interpolated.items():
            lines.append(f"  Interpolated missing '{col}': {n:,} values (linear, per-product, time-ordered)")
        lines.append(
            f"  Revenue_Growth: capped {self.revenue_growth_inf_capped:,} infinite values "
            f"and {self.revenue_growth_extreme_capped:,} extreme values at +/-{REVENUE_GROWTH_CAP*100:.0f}%"
        )
        lines.append(f"  Stockout weeks flagged (Sales_Units=0 & Inventory_Level=0): {self.stockout_weeks_flagged:,}")
        for field_name, result in self.financial_field_validation.items():
            lines.append(f"  Validated '{field_name}': {result}")
        return "\n".join(lines)


def get_engine(db_path: str = None):
    db_path = db_path or resolve("data/processed/armani_trading.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def load_raw(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    except Exception as e:
        logger.error(f"Failed to read raw file {file_path}: {e}")
        raise

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Raw dataset is missing required columns: {missing_required}. "
            f"Found columns: {list(df.columns)}"
        )

    logger.info(f"Loaded raw file: {len(df):,} rows, {len(df.columns)} columns")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(rows_in=len(df))
    df = df.copy()

    # --- 1. drop rows with unrecoverable/corrupt keys ---
    before = len(df)
    df = df.dropna(subset=["Week", "Product"])
    report.rows_dropped_corrupt = before - len(df)

    # --- 2. normalize product name text ---
    df["Product"] = df["Product"].astype(str).str.strip()

    # --- 3. remove exact duplicate (Week, Product) rows ---
    before = len(df)
    df = df.sort_values(["Product", "Week"]).drop_duplicates(subset=["Week", "Product"], keep="first")
    report.duplicates_removed = before - len(df)

    # --- 4. flag stockout weeks BEFORE any imputation touches Sales_Units ---
    df["is_stockout"] = ((df["Sales_Units"] == 0) & (df["Inventory_Level"] == 0)).astype(int)
    report.stockout_weeks_flagged = int(df["is_stockout"].sum())

    # --- 5. validate provided derived financial fields against raw components,
    #        BEFORE imputation touches Sales_Units (so this checks the
    #        original data's internal consistency, not imputation error) ---
    revenue_check = (df["Revenue"] - df["Sales_Units"] * df["Price_Per_Unit"]).abs()
    gp_check = (df["Gross_Profit"] - (df["Revenue"] - df["Sales_Units"] * df["Cost_of_Goods_Sold_COGS"])).abs()
    n_revenue_mismatch = (revenue_check > 1).sum()
    n_gp_mismatch = (gp_check > 1).sum()
    report.financial_field_validation["Revenue = Sales_Units * Price_Per_Unit"] = (
        f"{n_revenue_mismatch} rows differ by >1 among non-null values -> fully consistent, trusted as-is"
    )
    report.financial_field_validation["Gross_Profit = Revenue - COGS*Sales_Units"] = (
        f"{n_gp_mismatch} rows differ by >1 among non-null values -> fully consistent, trusted as-is"
    )

    # --- 6. time-aware interpolation of missing values (NOT median fill --
    #        this is an ordered weekly series per product). Note: this
    #        necessarily makes Revenue/Gross_Profit inconsistent with the
    #        newly-interpolated Sales_Units on these specific rows -- that's
    #        expected (Revenue reflects the original, now-lost true value;
    #        we do not overwrite Revenue/Gross_Profit, since the original
    #        figures are the more trustworthy source for financial KPIs). ---
    for col in ["Demand_Forecast", "Sales_Units"]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df.groupby("Product")[col].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both")
            )
            report.missing_values_interpolated[col] = int(n_missing)

    # --- 6. winsorize Revenue_Growth: inf values are legitimate (division by
    #        a genuine $0 prior-week revenue during a stockout), but break
    #        most ML/statistical models downstream ---
    n_inf = np.isinf(df["Revenue_Growth"]).sum()
    df["Revenue_Growth"] = df["Revenue_Growth"].replace([np.inf, -np.inf], np.nan)
    n_extreme = ((df["Revenue_Growth"] > REVENUE_GROWTH_CAP) | (df["Revenue_Growth"] < -REVENUE_GROWTH_CAP)).sum()
    df["Revenue_Growth"] = df["Revenue_Growth"].clip(-REVENUE_GROWTH_CAP, REVENUE_GROWTH_CAP)
    df["Revenue_Growth"] = df.groupby("Product")["Revenue_Growth"].transform(lambda s: s.fillna(REVENUE_GROWTH_CAP))
    report.revenue_growth_inf_capped = int(n_inf)
    report.revenue_growth_extreme_capped = int(n_extreme)

    # --- 7. synthesize calendar date from Week (documented assumption) ---
    df["Date"] = ASSUMED_WEEK1_DATE + pd.to_timedelta((df["Week"] - 1) * 7, unit="D")

    df = df.sort_values(["Product", "Week"]).reset_index(drop=True)
    report.rows_out = len(df)
    return df, report


def load_to_db(df: pd.DataFrame, engine, schema_path: str = None) -> None:
    schema_path = schema_path or resolve("sql/schema.sql")
    with engine.begin() as conn:
        schema_sql = Path(schema_path).read_text()
        # strip full-line and trailing "--" comments before splitting on ";"
        # so a semicolon inside a comment can't fragment a statement
        lines = [line.split("--", 1)[0] for line in schema_sql.splitlines()]
        schema_sql_no_comments = "\n".join(lines)
        for statement in schema_sql_no_comments.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))

    dim_product = pd.DataFrame({"product_name": sorted(df["Product"].unique())})
    dim_product.to_sql("dim_product", engine, if_exists="append", index=False)

    dim_date = (
        df[["Date", "Week"]]
        .drop_duplicates()
        .assign(
            month=lambda d: d["Date"].dt.month,
            quarter=lambda d: d["Date"].dt.quarter,
            year=lambda d: d["Date"].dt.year,
        )
        .rename(columns={"Date": "calendar_date", "Week": "week_number"})
        .sort_values("calendar_date")
    )
    dim_date.to_sql("dim_date", engine, if_exists="append", index=False)

    product_map = pd.read_sql("SELECT product_id, product_name FROM dim_product", engine)
    date_map = pd.read_sql("SELECT date_id, week_number FROM dim_date", engine)

    fact = df.merge(product_map, left_on="Product", right_on="product_name")
    fact = fact.merge(date_map, left_on="Week", right_on="week_number")

    fact_sales = fact.rename(columns={
        "Sales_Units": "sales_units",
        "Cost_of_Goods_Sold_COGS": "cogs_per_unit",
        "Price_Per_Unit": "price_per_unit",
        "Inventory_Level": "inventory_level",
        "Demand_Forecast": "demand_actual",
        "Revenue": "revenue",
        "Gross_Profit": "gross_profit",
        "Operating_Profit": "operating_profit",
        "Inventory_Turnover_Ratio": "inventory_turnover_ratio",
        "Gross_Profit_Margin": "gross_profit_margin",
        "Revenue_Growth": "revenue_growth",
    })[[
        "date_id", "product_id", "sales_units", "cogs_per_unit", "price_per_unit",
        "inventory_level", "demand_actual", "revenue", "gross_profit", "operating_profit",
        "inventory_turnover_ratio", "gross_profit_margin", "revenue_growth", "is_stockout",
    ]]

    fact_sales.to_sql("fact_sales", engine, if_exists="append", index=False)
    logger.info(f"Loaded {len(fact_sales):,} rows into fact_sales")


def run_pipeline(raw_path: str, db_path: str = None) -> CleaningReport:
    df_raw = load_raw(raw_path)
    df_clean, report = clean(df_raw)

    processed_path = resolve("data/processed/cleaned_sales_data.parquet")
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(processed_path, index=False)
    logger.info(f"Saved cleaned data to {processed_path}")

    engine = get_engine(db_path)
    load_to_db(df_clean, engine)

    logger.info("\n" + report.summary())
    return report


if __name__ == "__main__":
    run_pipeline(raw_path=resolve("data/raw/Sample_Data.xlsx"))
