"""
ablation.py
-----------
Feature-group ablation for the Stage 2 LightGBM model.

Question this answers: WHY does the direct LightGBM model beat the
seasonal-naive baseline? Feature-importance alone can't answer that
(gain importance is split unevenly across correlated features), so this
re-runs the exact same walk-forward, recursive CV as forecasting.py with
feature groups removed and compares MAPE:

    full            all 26 features (the production model)
    calendar_only   only week_of_year / month / quarter / sin / cos
    no_calendar     everything EXCEPT the calendar features
    no_lag52        full model minus lag_52 (does the "copy last year" signal matter?)

Run:  python src/ablation.py      (~2-3 minutes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    from src.forecasting import (
        get_engine, list_products, load_product_series, build_features,
        get_feature_cols, train_lgbm, recursive_forecast, seasonal_naive_forecast,
        HORIZON, N_CV_SPLITS, MIN_TRAIN_SIZE,
    )
except ImportError:
    from forecasting import (
        get_engine, list_products, load_product_series, build_features,
        get_feature_cols, train_lgbm, recursive_forecast, seasonal_naive_forecast,
        HORIZON, N_CV_SPLITS, MIN_TRAIN_SIZE,
    )

TARGET = "Demand_Forecast"
CALENDAR = ["week_of_year", "month", "quarter", "sin_woy", "cos_woy"]


def select_features(all_cols: list[str], variant: str) -> list[str]:
    if variant == "full":
        return all_cols
    if variant == "calendar_only":
        return [c for c in all_cols if c in CALENDAR]
    if variant == "no_calendar":
        return [c for c in all_cols if c not in CALENDAR]
    if variant == "no_lag52":
        return [c for c in all_cols if c != "lag_52"]
    raise ValueError(variant)


def mape(actual, forecast) -> float:
    return float(np.mean(np.abs((actual - forecast) / np.clip(actual, 1e-6, None))) * 100)


def run_ablation() -> pd.DataFrame:
    engine = get_engine()
    variants = ["full", "calendar_only", "no_calendar", "no_lag52"]
    rows = []

    for product in list_products(engine):
        df = load_product_series(engine, product)
        n = len(df)
        origins = np.linspace(MIN_TRAIN_SIZE, n - HORIZON, N_CV_SPLITS, dtype=int)

        for fold, origin in enumerate(origins, start=1):
            train_df = df.iloc[:origin].copy()
            actual = df.iloc[origin:origin + HORIZON][TARGET].values

            feat = build_features(train_df, target_col=TARGET)
            all_cols = get_feature_cols(feat, TARGET)

            baseline = seasonal_naive_forecast(train_df.set_index("Date")[TARGET], HORIZON)
            rows.append({"product": product, "fold": fold, "variant": "baseline",
                         "MAPE_%": mape(actual, baseline)})

            for variant in variants:
                cols = select_features(all_cols, variant)
                clean = feat.dropna(subset=cols + [TARGET])
                model = train_lgbm(clean[cols], clean[TARGET])
                fc = recursive_forecast(train_df, model, cols, TARGET, HORIZON)["Forecasted_Demand"].values
                rows.append({"product": product, "fold": fold, "variant": variant,
                             "MAPE_%": mape(actual, fc)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    try:
        from src.paths import resolve
    except ImportError:
        from paths import resolve

    results = run_ablation()
    out_path = resolve("data/processed/ablation_results.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    summary = results.groupby("variant")["MAPE_%"].mean().sort_values().round(2)
    print("Mean MAPE by feature set (all products x 3 folds):\n")
    print(summary.to_string())

    wide = results.groupby(["product", "variant"])["MAPE_%"].mean().unstack()
    print("\nProducts where each variant beats the baseline (out of 30):")
    for v in ["full", "calendar_only", "no_calendar", "no_lag52"]:
        print(f"  {v:15s} {(wide[v] < wide['baseline']).sum()}")

    # paired comparison at the finest grain (product x fold = 90 pairs), so a
    # ~1pt average gap can be judged against fold-to-fold noise
    pair = results.pivot_table(index=["product", "fold"], columns="variant", values="MAPE_%")
    print(f"\nPaired comparison over {len(pair)} product-fold pairs:")
    print(f"  calendar_only beats full      in {(pair['calendar_only'] < pair['full']).mean():.0%} of pairs")
    print(f"  no_lag52 beats full           in {(pair['no_lag52'] < pair['full']).mean():.0%} of pairs")
    print(f"  full beats baseline           in {(pair['full'] < pair['baseline']).mean():.0%} of pairs")
    print(f"  calendar_only beats baseline  in {(pair['calendar_only'] < pair['baseline']).mean():.0%} of pairs")
    print(f"\nSaved: {out_path}")
