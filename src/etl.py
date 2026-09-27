"""
etl.py
------
Data Engineering, ETL & Database Setup (Deliverable 1).

Responsibilities:
    1. Ingest the raw dataset (Excel/CSV) with exception handling for corrupt
       records / missing fields.
    2. Clean: standardize schema, fix casing/whitespace, dedupe, impute
       missing values, cap/flag anomalous outliers.
    3. Load the cleaned data into a normalized star schema in SQLite
       (fact_sales + dimension tables).

Design notes:
    - SQLite is used for portability (zero-setup, single file, runs anywhere
      a reviewer clones the repo). Swapping to Postgres/SQL Server only
      requires changing the SQLAlchemy connection string in `get_engine()`
      — the schema.sql and load logic are otherwise vendor-agnostic.
    - All cleaning decisions are logged, not silent, so the README's
      "key assumptions" section can be generated from `cleaning_report`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Week", "Date", "Product", "Region", "Customer_Segment",
    "Sales_Units", "Cost_of_Goods_Sold_COGS", "Price_Per_Unit",
    "Inventory_Level", "Demand_Forecast",
]

OPTIONAL_COLUMNS = [
    "Days_Sales_Outstanding_Input", "Customer_Credit_Limit",
    "Customer_Outstanding_Balance",
]


@dataclass
class CleaningReport:
    """Tracks every cleaning decision so it can be dropped straight into the README."""
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    missing_values_imputed: dict = field(default_factory=dict)
    outliers_capped: dict = field(default_factory=dict)
    negative_values_fixed: dict = field(default_factory=dict)
    categorical_values_normalized: dict = field(default_factory=dict)
    rows_dropped_corrupt: int = 0

    def summary(self) -> str:
        lines = [
            f"Rows in: {self.rows_in:,} -> Rows out: {self.rows_out:,}",
            f"Duplicates removed: {self.duplicates_removed:,}",
            f"Rows dropped as unrecoverable/corrupt: {self.rows_dropped_corrupt:,}",
        ]
        for col, n in self.missing_values_imputed.items():
            lines.append(f"  Imputed missing '{col}': {n:,} values (median-by-product)")
        for col, n in self.outliers_capped.items():
            lines.append(f"  Capped outliers in '{col}': {n:,} values (IQR method)")
        for col, n in self.negative_values_fixed.items():
            lines.append(f"  Fixed negative '{col}': {n:,} values (took abs value)")
        for col in self.categorical_values_normalized:
            lines.append(f"  Normalized casing/whitespace in '{col}'")
        return "\n".join(lines)


def get_engine(db_path: str = "data/processed/armani_trading.db"):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def load_raw(file_path: str) -> pd.DataFrame:
    """Ingest raw file with exception handling for corrupt records / bad schema."""
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


def _cap_outliers_iqr(series: pd.Series, k: float = 3.0) -> tuple[pd.Series, int]:
    """Cap outliers using IQR method (k=3 -> conservative, only extreme values)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    n_capped = ((series < lower) | (series > upper)).sum()
    return series.clip(lower, upper), int(n_capped)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(rows_in=len(df))
    df = df.copy()

    # --- 1. drop rows with corrupt/unusable keys (can't recover these) ---
    before = len(df)
    df = df.dropna(subset=["Week", "Date", "Product"])
    df = df[df["Sales_Units"].notna() | df["Demand_Forecast"].notna()]  # need at least one demand signal
    report.rows_dropped_corrupt = before - len(df)

    # --- 2. normalize categorical text fields ---
    for col in ["Product", "Region", "Customer_Segment"]:
        if col in df.columns:
            original = df[col].copy()
            df[col] = df[col].astype(str).str.strip().str.title()
            if not original.astype(str).equals(df[col]):
                report.categorical_values_normalized[col] = True

    # --- 3. remove exact duplicate rows ---
    before = len(df)
    df = df.drop_duplicates()
    report.duplicates_removed = before - len(df)

    # --- 4. fix negative values that are physically impossible ---
    for col in ["Cost_of_Goods_Sold_COGS", "Price_Per_Unit", "Sales_Units", "Inventory_Level"]:
        if col in df.columns:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                df[col] = df[col].abs()
                report.negative_values_fixed[col] = int(n_negative)

    # --- 5. impute missing numeric values (median by product -> preserves
    #        product-level scale differences rather than a global median) ---
    numeric_cols = ["Sales_Units", "Cost_of_Goods_Sold_COGS", "Price_Per_Unit", "Inventory_Level"]
    for col in numeric_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df.groupby("Product")[col].transform(lambda s: s.fillna(s.median()))
                df[col] = df[col].fillna(df[col].median())  # fallback for products with all-NaN
                report.missing_values_imputed[col] = int(n_missing)

    # --- 6. cap extreme outliers (data entry errors), don't drop rows ---
    for col in ["Sales_Units", "Cost_of_Goods_Sold_COGS", "Price_Per_Unit"]:
        if col in df.columns:
            df[col], n_capped = _cap_outliers_iqr(df[col])
            if n_capped > 0:
                report.outliers_capped[col] = n_capped

    # --- 7. ensure correct dtypes ---
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week"] = df["Week"].astype(int)

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values(["Product", "Date"]).reset_index(drop=True)
    report.rows_out = len(df)
    return df, report


def load_to_db(df: pd.DataFrame, engine, schema_path: str = "sql/schema.sql") -> None:
    """Build dimension + fact tables from the cleaned dataframe."""
    with engine.begin() as conn:
        schema_sql = Path(schema_path).read_text()
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))

    # --- dimensions ---
    dim_product = pd.DataFrame({"product_name": sorted(df["Product"].unique())})
    dim_product.to_sql("dim_product", engine, if_exists="append", index=False)

    dim_region = pd.DataFrame({"region_name": sorted(df["Region"].unique())})
    dim_region.to_sql("dim_region", engine, if_exists="append", index=False)

    dim_segment = pd.DataFrame({"segment_name": sorted(df["Customer_Segment"].unique())})
    dim_segment.to_sql("dim_customer_segment", engine, if_exists="append", index=False)

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

    # --- read back dimension keys for FK mapping ---
    product_map = pd.read_sql("SELECT product_id, product_name FROM dim_product", engine)
    region_map = pd.read_sql("SELECT region_id, region_name FROM dim_region", engine)
    segment_map = pd.read_sql("SELECT segment_id, segment_name FROM dim_customer_segment", engine)
    date_map = pd.read_sql("SELECT date_id, calendar_date FROM dim_date", engine)
    date_map["calendar_date"] = pd.to_datetime(date_map["calendar_date"])

    fact = df.merge(product_map, left_on="Product", right_on="product_name")
    fact = fact.merge(region_map, left_on="Region", right_on="region_name")
    fact = fact.merge(segment_map, left_on="Customer_Segment", right_on="segment_name")
    fact = fact.merge(date_map, left_on="Date", right_on="calendar_date")

    fact["revenue"] = fact["Sales_Units"] * fact["Price_Per_Unit"]
    fact["gross_profit"] = fact["revenue"] - (fact["Sales_Units"] * fact["Cost_of_Goods_Sold_COGS"])

    fact_sales = fact.rename(columns={
        "Sales_Units": "sales_units",
        "Cost_of_Goods_Sold_COGS": "cogs_per_unit",
        "Price_Per_Unit": "price_per_unit",
        "Inventory_Level": "inventory_level",
        "Days_Sales_Outstanding_Input": "days_sales_outstanding_input",
        "Customer_Credit_Limit": "customer_credit_limit",
        "Customer_Outstanding_Balance": "customer_outstanding_balance",
        "Demand_Forecast": "demand_actual",
    })[[
        "date_id", "product_id", "region_id", "segment_id",
        "sales_units", "cogs_per_unit", "price_per_unit", "inventory_level",
        "days_sales_outstanding_input", "customer_credit_limit",
        "customer_outstanding_balance", "demand_actual", "revenue", "gross_profit",
    ]]

    fact_sales.to_sql("fact_sales", engine, if_exists="append", index=False)
    logger.info(f"Loaded {len(fact_sales):,} rows into fact_sales")


def run_pipeline(raw_path: str, db_path: str = "data/processed/armani_trading.db") -> CleaningReport:
    df_raw = load_raw(raw_path)
    df_clean, report = clean(df_raw)

    processed_path = "data/processed/cleaned_sales_data.parquet"
    df_clean.to_parquet(processed_path, index=False)
    logger.info(f"Saved cleaned data to {processed_path}")

    engine = get_engine(db_path)
    load_to_db(df_clean, engine)

    logger.info("\n" + report.summary())
    return report


if __name__ == "__main__":
    run_pipeline(raw_path="data/raw/Sample_Data_synthetic.xlsx")
