"""
forecasting.py
--------------
Time-Series Demand Forecasting & Factor Analysis (Deliverable 2).

MODEL DESIGN -- and why it changed during development:

We initially tried the "textbook" hybrid: MSTL decomposition (trend +
seasonal(52) + seasonal(4)) with LightGBM modeling only the leftover
remainder. On walk-forward CV, this hybrid LOST to a naive seasonal
baseline (demand 52 weeks ago) by a wide margin on every product tested.
Root-causing this (see README "Model iteration" section):
  - Autocorrelation analysis showed strong signal at lag 1 (~0.70, momentum)
    and lag 52 (~0.69, genuine annual seasonality) -- real, usable signal.
  - But with only ~5 years of weekly history, MSTL sees ~2.5-5 annual
    cycles per training window. Its smoothed seasonal component blurs
    product-specific week-to-week idiosyncrasies that the raw lag-52
    value captures exactly. The spurious seasonal(4) term compounded
    this by fitting noise as if it were a real monthly cycle.
  - Net effect: decomposing first and extrapolating trend/seasonal
    separately threw away exactly the precise signal (lag_52, lag_1) that
    a feature-based model could otherwise use directly.

FINAL DESIGN: per PRODUCT (30 separate models), a single LightGBM
regressor trained DIRECTLY on Demand_Forecast using lag features
(including lag_1 and lag_52 explicitly), rolling stats, calendar
features, and lagged exogenous drivers. MSTL is kept as a descriptive/
factor-analysis tool (see `decompose_for_analysis`) to visualize and
explain trend/seasonality/residual structure -- exactly the "factor
analysis" the brief asks for -- but it is no longer inside the
prediction path.

Evaluated with walk-forward (rolling-origin), RECURSIVE cross-validation
(each fold predicts week-by-week, feeding its own prior predictions back
in as lag inputs -- exactly like production) against the seasonal-naive
baseline, per the brief's requirement to "compare your model against a
baseline and justify your choice."

Feature-leakage note: Price_Per_Unit and Inventory_Level are used only in
LAGGED form. We don't know whether this business sets price in advance
(in which case same-period price would be legitimate) or reacts to demand
(in which case it's leakage) -- lagging is the conservative, safe default.
Sales_Units is excluded entirely as a feature: it's inventory-censored
(see ETL findings) and would teach the model to under-forecast during
recovery-from-stockout periods.

Run as a script: iterates over all 30 products, runs CV, trains a final
model per product, and writes:
    data/processed/cv_metrics.csv       -- per-product, per-fold metrics
    data/processed/future_forecast.csv  -- 26-week-ahead forecast per product
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sqlalchemy import create_engine
from statsmodels.tsa.seasonal import MSTL
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

HORIZON = 26          # 6 months of weekly data
MSTL_PERIODS = (52, 4)
N_CV_SPLITS = 3
MIN_TRAIN_SIZE = 130  # ~2.5 years before the first fold


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def get_engine(db_path: str = "data/processed/armani_trading.db"):
    return create_engine(f"sqlite:///{db_path}")


def load_product_series(engine, product_name: str) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT d.calendar_date AS Date, d.week_number AS Week,
               f.cogs_per_unit, f.price_per_unit, f.inventory_level,
               f.demand_actual AS Demand_Forecast, f.is_stockout
        FROM fact_sales f
        JOIN dim_product p ON f.product_id = p.product_id
        JOIN dim_date d ON f.date_id = d.date_id
        WHERE p.product_name = ?
        ORDER BY d.calendar_date
        """,
        engine, params=(product_name,),
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def list_products(engine) -> list[str]:
    return pd.read_sql("SELECT product_name FROM dim_product ORDER BY product_name", engine)["product_name"].tolist()


# ---------------------------------------------------------------------------
# Feature engineering (leakage-safe)
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, target_col: str = "Demand_Forecast") -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)

    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    for window in [4, 12, 26]:
        df[f"roll_mean_{window}"] = df[target_col].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"] = df[target_col].shift(1).rolling(window).std()

    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["sin_woy"] = np.sin(2 * np.pi * df["week_of_year"] / 52.18)
    df["cos_woy"] = np.cos(2 * np.pi * df["week_of_year"] / 52.18)

    # exogenous drivers: LAGGED only (see module docstring on leakage)
    for col in ["price_per_unit", "cogs_per_unit", "inventory_level"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag4"] = df[col].shift(4)

    # was the product out of stock recently? (recent stockouts often precede a demand rebound)
    df["stockout_recent_4wk"] = df["is_stockout"].shift(1).rolling(4).sum()

    return df


LEAKY_RAW_COLS = ["price_per_unit", "cogs_per_unit", "inventory_level", "is_stockout"]


def get_feature_cols(feat_df: pd.DataFrame, target_col: str, extra_drop: set = frozenset()) -> list[str]:
    drop_cols = {"Date", "Week", target_col, "remainder"} | set(LEAKY_RAW_COLS) | set(extra_drop)
    return [c for c in feat_df.columns if c not in drop_cols]


# ---------------------------------------------------------------------------
# MSTL decomposition -- DESCRIPTIVE / FACTOR-ANALYSIS USE ONLY.
# Not used in the prediction path; see module docstring for why.
# ---------------------------------------------------------------------------
def decompose_for_analysis(series: pd.Series, periods=(52,)) -> pd.DataFrame:
    """Decompose a product's full demand history into trend/seasonal/residual
    for the Stage 2 'factor analysis' deliverable (interpretation, plots,
    explaining WHY demand moves the way it does) -- not fed into the
    forecasting model itself."""
    mstl = MSTL(series, periods=periods, stl_kwargs={"seasonal_deg": 0})
    res = mstl.fit()
    out = pd.DataFrame({"trend": res.trend, "seasonal": res.seasonal, "resid": res.resid}, index=series.index)
    return out


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------
def seasonal_naive_forecast(train_series: pd.Series, horizon: int) -> np.ndarray:
    """Baseline: this week's demand = demand 52 weeks ago. Falls back to the
    last observed value if fewer than 52 weeks of history exist."""
    if len(train_series) >= 52:
        last_cycle = train_series.values[-52:]
        reps = int(np.ceil(horizon / 52))
        return np.tile(last_cycle, reps)[:horizon]
    return np.full(horizon, train_series.values[-1])


# ---------------------------------------------------------------------------
# Walk-forward CV for one product
# ---------------------------------------------------------------------------
def train_lgbm(X_train, y_train) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=250, learning_rate=0.04, max_depth=4, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        min_child_samples=10, random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def recursive_forecast(
    history_df: pd.DataFrame, model, feature_cols: list[str], target_col: str, horizon: int,
) -> pd.DataFrame:
    """Shared recursive forecasting logic used by BOTH walk-forward CV and
    the final production forecast, so CV honestly measures what production
    will actually do: predict week 1, append it, recompute lags/rolling
    stats from the now-extended series, predict week 2, ...

    Future exogenous drivers (price, COGS, inventory) are unknown, so we
    carry the last observed values forward (documented assumption -- swap
    in a real planning calendar if the business has one).
    """
    history_df = history_df.sort_values("Date").reset_index(drop=True)
    last_row = history_df.iloc[-1]
    future_dates = pd.date_range(last_row["Date"] + pd.Timedelta(weeks=1), periods=horizon, freq="7D")

    extended = history_df[["Date", target_col, "price_per_unit", "cogs_per_unit",
                            "inventory_level", "is_stockout"]].copy()
    predictions = []

    for future_date in future_dates:
        new_row = pd.DataFrame([{
            "Date": future_date,
            target_col: np.nan,
            "price_per_unit": last_row["price_per_unit"],
            "cogs_per_unit": last_row["cogs_per_unit"],
            "inventory_level": last_row["inventory_level"],
            "is_stockout": 0,  # assume no future stockouts for the demand forecast itself
        }])
        extended = pd.concat([extended, new_row], ignore_index=True)

        feat_step = build_features(extended, target_col=target_col)
        x_step = feat_step[feature_cols].iloc[[-1]].ffill(axis=0).fillna(0)
        pred = max(model.predict(x_step)[0], 0)  # demand can't be negative

        extended.loc[extended.index[-1], target_col] = pred
        predictions.append(pred)

    return pd.DataFrame({"Date": future_dates, "Week_Ahead": range(1, horizon + 1), "Forecasted_Demand": predictions})


def walk_forward_cv_product(
    df: pd.DataFrame, product_name: str, target_col: str = "Demand_Forecast",
    horizon: int = HORIZON, n_splits: int = N_CV_SPLITS, min_train_size: int = MIN_TRAIN_SIZE,
) -> pd.DataFrame:
    n = len(df)
    max_origin = n - horizon
    if max_origin <= min_train_size:
        return pd.DataFrame()

    origins = np.linspace(min_train_size, max_origin, n_splits, dtype=int)
    results = []

    for fold_idx, origin in enumerate(origins, start=1):
        train_df = df.iloc[:origin].copy()
        test_df = df.iloc[origin:origin + horizon].copy()
        if len(test_df) < horizon:
            continue

        train_feat = build_features(train_df, target_col=target_col)
        feature_cols = get_feature_cols(train_feat, target_col)
        train_feat_clean = train_feat.dropna(subset=feature_cols + [target_col])
        if len(train_feat_clean) < 20:
            continue

        model = train_lgbm(train_feat_clean[feature_cols], train_feat_clean[target_col])
        model_fc = recursive_forecast(train_df, model, feature_cols, target_col, horizon)["Forecasted_Demand"].values

        train_series = train_df.set_index("Date")[target_col]
        baseline_fc = seasonal_naive_forecast(train_series, horizon)
        actual = test_df[target_col].values

        for name, forecast in [("lightgbm_direct", model_fc), ("baseline_seasonal_naive", baseline_fc)]:
            mae = mean_absolute_error(actual, forecast)
            rmse = np.sqrt(mean_squared_error(actual, forecast))
            mape = float(np.mean(np.abs((actual - forecast) / np.clip(actual, 1e-6, None))) * 100)
            results.append({
                "product": product_name, "fold": fold_idx, "model": name,
                "MAE": mae, "RMSE": rmse, "MAPE_%": mape,
            })

    return pd.DataFrame(results)


def forecast_future_product(df: pd.DataFrame, target_col: str = "Demand_Forecast",
                             horizon: int = HORIZON) -> pd.DataFrame:
    """Train on ALL available history and forecast the next `horizon` weeks."""
    df = df.sort_values("Date").reset_index(drop=True)
    feat = build_features(df.copy(), target_col=target_col)
    feature_cols = get_feature_cols(feat, target_col)
    train_feat = feat.dropna(subset=feature_cols + [target_col])
    model = train_lgbm(train_feat[feature_cols], train_feat[target_col])
    return recursive_forecast(df, model, feature_cols, target_col, horizon)


# ---------------------------------------------------------------------------
# MAIN: run for all products
# ---------------------------------------------------------------------------
def run_all_products(db_path: str = "data/processed/armani_trading.db"):
    engine = get_engine(db_path)
    products = list_products(engine)
    print(f"Running forecasting pipeline for {len(products)} products...")

    all_cv_results = []
    all_forecasts = []

    for i, product in enumerate(products, start=1):
        print(f"  [{i}/{len(products)}] {product}...", end=" ")
        df = load_product_series(engine, product)

        cv_result = walk_forward_cv_product(df, product)
        if not cv_result.empty:
            all_cv_results.append(cv_result)

        future_fc = forecast_future_product(df)
        future_fc.insert(0, "Product", product)
        all_forecasts.append(future_fc)
        print("done")

    cv_df = pd.concat(all_cv_results, ignore_index=True)
    forecast_df = pd.concat(all_forecasts, ignore_index=True)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    cv_df.to_csv("data/processed/cv_metrics.csv", index=False)
    forecast_df.to_csv("data/processed/future_forecast.csv", index=False)

    return cv_df, forecast_df


if __name__ == "__main__":
    cv_df, forecast_df = run_all_products()

    print("\n" + "=" * 70)
    print("MODEL vs BASELINE -- AVERAGED ACROSS ALL PRODUCTS & FOLDS")
    print("=" * 70)
    summary = cv_df.groupby("model")[["MAE", "RMSE", "MAPE_%"]].mean().round(2)
    print(summary)

    model_mape = summary.loc["lightgbm_direct", "MAPE_%"]
    baseline_mape = summary.loc["baseline_seasonal_naive", "MAPE_%"]
    improvement = (baseline_mape - model_mape) / baseline_mape * 100
    print(f"\nLightGBM model improves MAPE by {improvement:.1f}% over the seasonal-naive baseline.")

    print("\nSaved: data/processed/cv_metrics.csv, data/processed/future_forecast.csv")
    print(f"\nSample future forecast (first product, first 5 weeks):")
    print(forecast_df.head(5).to_string(index=False))
