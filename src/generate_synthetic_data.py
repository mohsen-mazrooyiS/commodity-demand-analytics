"""
generate_synthetic_data.py
---------------------------
PLACEHOLDER DATA GENERATOR — swap out for the real Sample_Data.xlsx.

The assessment brief confirms the real file has at minimum:
    Week (1-260), Product (30 commodities), Sales_Units, Cost_of_Goods_Sold_COGS

The KPI/pricing requirements (Gross Margin, DSO, Customer Default Risk,
Inventory Turnover) cannot be computed without price, inventory, and
customer/credit data, so this generator adds those fields with realistic
distributions and intentional data-quality issues (nulls, duplicates,
outliers, inconsistent casing) so the ETL step in etl.py has real work to do.

USAGE:
    Once Sample_Data.xlsx is available, drop it in data/raw/ and point
    etl.py at it directly -- this script becomes unnecessary. Nothing
    downstream (database schema, forecasting, dashboard, pricing) depends
    on this file; they all depend on the *cleaned* output of etl.py.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

PRODUCTS = [
    "Rice", "Wheat", "Sugar", "Milk", "Tea", "Coffee", "Flour", "Barley",
    "Corn", "Soybean Oil", "Palm Oil", "Lentils", "Chickpeas", "Salt",
    "Cocoa", "Honey", "Butter", "Cheese", "Yogurt", "Pasta",
    "Canned Tomatoes", "Dates", "Almonds", "Cashews", "Raisins",
    "Black Pepper", "Cinnamon", "Baking Soda", "Vinegar", "Ghee",
]

CUSTOMER_SEGMENTS = ["Retail Chain", "Wholesaler", "Hospitality", "Distributor", "Government"]
REGIONS = ["UAE", "Saudi Arabia", "Qatar", "Oman", "Bahrain", "Kuwait"]

N_WEEKS = 260


def generate_raw_dataset() -> pd.DataFrame:
    rows = []
    start_date = pd.Timestamp("2021-01-04")

    for p_idx, product in enumerate(PRODUCTS):
        base_demand = RNG.uniform(800, 5000)
        trend_slope = RNG.uniform(-1.5, 3.0)
        annual_amp = base_demand * RNG.uniform(0.05, 0.25)
        base_cost = RNG.uniform(1.5, 40)
        base_price = base_cost * RNG.uniform(1.25, 1.8)
        segment = RNG.choice(CUSTOMER_SEGMENTS)
        region = RNG.choice(REGIONS)

        for week in range(1, N_WEEKS + 1):
            date = start_date + pd.Timedelta(weeks=week - 1)
            seasonal = annual_amp * np.sin(2 * np.pi * week / 52.18)
            monthly = (annual_amp * 0.3) * np.sin(2 * np.pi * week / 4.33)
            noise = RNG.normal(0, base_demand * 0.06)
            demand = max(base_demand + trend_slope * week + seasonal + monthly + noise, 10)

            sales_units = demand * RNG.uniform(0.93, 1.07)
            cogs_unit = base_cost * (1 + RNG.normal(0, 0.03))
            price_unit = base_price * (1 + RNG.normal(0, 0.04))
            inventory_level = max(demand * RNG.uniform(1.5, 3.0) + RNG.normal(0, base_demand * 0.1), 0)
            days_payment = max(RNG.normal(35, 15), 0)
            credit_limit = RNG.uniform(20000, 500000)
            outstanding_balance = credit_limit * RNG.uniform(0, 0.9)

            rows.append({
                "Week": week,
                "Date": date,
                "Product": product,
                "Region": region,
                "Customer_Segment": segment,
                "Sales_Units": round(sales_units, 1),
                "Cost_of_Goods_Sold_COGS": round(cogs_unit, 2),
                "Price_Per_Unit": round(price_unit, 2),
                "Inventory_Level": round(inventory_level, 1),
                "Days_Sales_Outstanding_Input": round(days_payment, 1),
                "Customer_Credit_Limit": round(credit_limit, 2),
                "Customer_Outstanding_Balance": round(outstanding_balance, 2),
                "Demand_Forecast": round(demand, 1),  # historical actual demand estimate
            })

    df = pd.DataFrame(rows)

    # ---- inject realistic data-quality problems for the ETL step to handle ----
    n = len(df)

    # 1. missing values scattered across a few columns
    for col in ["Sales_Units", "Cost_of_Goods_Sold_COGS", "Price_Per_Unit", "Inventory_Level"]:
        missing_idx = RNG.choice(n, size=int(n * 0.01), replace=False)
        df.loc[missing_idx, col] = np.nan

    # 2. inconsistent casing / whitespace in categorical fields
    dirty_idx = RNG.choice(n, size=int(n * 0.05), replace=False)
    df.loc[dirty_idx, "Product"] = df.loc[dirty_idx, "Product"].str.upper() + "  "

    # 3. exact duplicate rows
    dup_rows = df.sample(n=30, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 4. extreme outliers (data entry errors, e.g. missing decimal point)
    outlier_idx = RNG.choice(len(df), size=15, replace=False)
    df.loc[outlier_idx, "Sales_Units"] = df.loc[outlier_idx, "Sales_Units"] * 100

    # 5. negative values that shouldn't exist (sensor/entry error)
    neg_idx = RNG.choice(len(df), size=10, replace=False)
    df.loc[neg_idx, "Cost_of_Goods_Sold_COGS"] = -df.loc[neg_idx, "Cost_of_Goods_Sold_COGS"]

    return df


if __name__ == "__main__":
    df = generate_raw_dataset()
    out_path = "data/raw/Sample_Data_synthetic.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Generated {len(df):,} rows across {df['Product'].str.strip().str.title().nunique()} products")
    print(f"Saved to {out_path}")
    print(df.head())
