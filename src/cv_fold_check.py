"""
cv_fold_check.py
----------------
Documents HOW the walk-forward CV folds are built, and stress-tests them.

Background: forecasting.py builds N_CV_SPLITS=3 folds with
    origins = np.linspace(MIN_TRAIN_SIZE, n - HORIZON, N_CV_SPLITS)
                = np.linspace(130, 234, 3) = [130, 182, 234]
Those origins are exactly 52 weeks apart, so all three 26-week test windows
fall on the SAME half of the annual cycle (cycle weeks 27-52), while the
production forecast (weeks 261-286) covers cycle weeks 1-26. The reported
CV numbers therefore never tested the seasonal phase that is being forecast.

This script:
  1. prints the production fold schedule and the forecast window's position;
  2. re-runs the comparison with DENSE folds (origin every 13 weeks, 9 folds,
     test windows overlapping) so every seasonal phase is covered, including
     folds whose test window starts at cycle week 1 exactly like production;
  3. reports MAPE for baseline / full model / calendar-only model overall and
     split by seasonal phase.

Run:  python src/cv_fold_check.py     (~3-5 minutes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.forecasting import (
        get_engine, list_products, load_product_series, build_features, get_feature_cols,
        train_lgbm, recursive_forecast, seasonal_naive_forecast, HORIZON,
    )
    from src.ablation import select_features, mape, TARGET
    from src.paths import resolve
except ImportError:
    from forecasting import (
        get_engine, list_products, load_product_series, build_features, get_feature_cols,
        train_lgbm, recursive_forecast, seasonal_naive_forecast, HORIZON,
    )
    from ablation import select_features, mape, TARGET
    from paths import resolve

DENSE_ORIGINS = list(range(130, 235, 13))  # 130,143,...,234 -> 9 folds


def cycle_start(origin: int) -> int:
    """Position (1-52) in the annual cycle of the first test week."""
    return (origin) % 52 + 1


def run_dense_folds() -> pd.DataFrame:
    engine = get_engine()
    rows = []
    for product in list_products(engine):
        df = load_product_series(engine, product)
        for origin in DENSE_ORIGINS:
            train_df = df.iloc[:origin].copy()
            actual = df.iloc[origin:origin + HORIZON][TARGET].values
            if len(actual) < HORIZON:
                continue

            feat = build_features(train_df, target_col=TARGET)
            all_cols = get_feature_cols(feat, TARGET)
            phase = cycle_start(origin)

            baseline = seasonal_naive_forecast(train_df.set_index("Date")[TARGET], HORIZON)
            rows.append({"product": product, "origin": origin, "phase_start_week": phase,
                         "variant": "baseline", "MAPE_%": mape(actual, baseline)})

            for variant in ("full", "calendar_only"):
                cols = select_features(all_cols, variant)
                clean = feat.dropna(subset=cols + [TARGET])
                model = train_lgbm(clean[cols], clean[TARGET])
                fc = recursive_forecast(train_df, model, cols, TARGET, HORIZON)["Forecasted_Demand"].values
                rows.append({"product": product, "origin": origin, "phase_start_week": phase,
                             "variant": variant, "MAPE_%": mape(actual, fc)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n = 260
    print("Production fold schedule (np.linspace(130, 234, 3)):")
    for i, o in enumerate(np.linspace(130, n - HORIZON, 3, dtype=int), 1):
        print(f"  fold {i}: train weeks 1-{o}, test weeks {o+1}-{o+HORIZON}, "
              f"cycle weeks {cycle_start(o)}-{(o + HORIZON - 1) % 52 + 1}")
    print(f"  production forecast: weeks {n+1}-{n+HORIZON}, cycle weeks {cycle_start(n)}-{(n + HORIZON - 1) % 52 + 1}\n")

    res = run_dense_folds()
    res.to_csv(resolve("data/processed/cv_dense_folds.csv"), index=False)

    print(f"Dense folds: origins {DENSE_ORIGINS} ({len(DENSE_ORIGINS)} folds x 30 products)\n")
    overall = res.groupby("variant")["MAPE_%"].mean().round(2)
    print("Mean MAPE, all dense folds:")
    print(overall.to_string(), "\n")

    piv = res.pivot_table(index=["product", "origin", "phase_start_week"], columns="variant", values="MAPE_%").reset_index()
    print("Mean MAPE by seasonal phase of the test window (cycle week where it starts):")
    by_phase = piv.groupby("phase_start_week")[["baseline", "full", "calendar_only"]].mean().round(2)
    by_phase["origins"] = piv.groupby("phase_start_week")["origin"].unique().apply(lambda x: sorted(int(v) for v in x))
    print(by_phase.to_string(), "\n")

    p1 = piv[piv["phase_start_week"] == 1]
    print("Folds that start at cycle week 1 (same phase as the production forecast):")
    print(f"  baseline {p1['baseline'].mean():.2f}% | full {p1['full'].mean():.2f}% | calendar_only {p1['calendar_only'].mean():.2f}%")
    print(f"  full beats baseline in {(p1['full'] < p1['baseline']).mean():.0%} of {len(p1)} product-folds; "
          f"calendar_only beats baseline in {(p1['calendar_only'] < p1['baseline']).mean():.0%}; "
          f"calendar_only beats full in {(p1['calendar_only'] < p1['full']).mean():.0%}")
    print(f"\nAll {len(piv)} product-folds: full beats baseline {(piv['full'] < piv['baseline']).mean():.0%}, "
          f"calendar_only beats baseline {(piv['calendar_only'] < piv['baseline']).mean():.0%}, "
          f"calendar_only beats full {(piv['calendar_only'] < piv['full']).mean():.0%}")
    print("\nSaved: data/processed/cv_dense_folds.csv")
