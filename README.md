# Armani Middle East Trading — Demand, KPI & Pricing Platform

Technical assessment submission: end-to-end pipeline covering data
engineering, 6-month demand forecasting, financial/operational KPI
reporting, and a dynamic pricing framework.

## Status

- [x] **Stage 1 — Data Engineering, ETL & Database**
- [x] **Stage 2 — Time-series demand forecasting & factor analysis**
- [x] **Stage 3 — Financial & operational KPI dashboard**
- [x] **Stage 4 — Dynamic pricing & risk strategy**

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

### The two models being compared

Everything in the results below is a comparison of two forecasting methods
on identical data. This section explains exactly how each one turns
historical demand into a 26-week forecast.

#### Model A — Baseline: seasonal-naive

**Rule:** the forecast for any future week is the demand recorded 52 weeks
earlier.

```
forecast(week t) = actual demand(week t − 52)
```

There are no parameters and nothing is trained. In code
(`seasonal_naive_forecast` in `src/forecasting.py`) it takes the last 52
weeks of training data as a "template year" and reads the first 26 values
off it.

**Worked example** — Rice - Basmati, CV fold 2 (trained on weeks 1–182,
forecasting weeks 183–208):

- Forecast for week 183 = actual demand in week 131 = **1,013 units**.
- Actual demand in week 183 was 1,061, so the absolute error is 48 units.
- Forecast for week 184 = actual demand in week 132 = 1,426 units. Actual
  was 1,207, so the error is 219 units.

**Why it's the standard benchmark.** Demand here is strongly seasonal
(for Rice - Basmati, autocorrelation at lag 52 is 0.69 and at lag 26 is
−0.68, the signature of an annual cycle), so "repeat last year" is the
usual first thing to beat. It costs nothing to run and can't overfit.
**It is not a hard benchmark, though:** it uses a single year, so a
one-line improvement, averaging the same week across *all* earlier years,
scores 12.01% MAPE against its 14.30% (see the ablation below). Beating
this baseline is a lower bar than it may sound.

**Its weakness.** It copies last year's random noise along with last
year's signal. If week 131 happened to be an unusually high week for
reasons that won't repeat, the baseline forecasts that spike again. It also
ignores trend and everything that has happened in the most recent weeks.

#### Model B — LightGBM, direct

"Direct" means the model predicts demand *itself* from a table of features.
(The alternative we tried first, described under "Model iteration" below,
decomposed demand into trend and seasonal parts and modelled only the
leftover.) One separate model is trained per product.

**Step 1 — Build 26 features for every week.** Each is computed using
only information available *before* that week:

| Group | Features | How each is computed |
|---|---|---|
| Lags (8) | `lag_1, 2, 3, 4, 8, 12, 26, 52` | Demand *k* weeks earlier |
| Rolling stats (6) | `roll_mean_4/12/26`, `roll_std_4/12/26` | Mean / standard deviation of the previous 4, 12 or 26 weeks. Shifted by one week so the current week is never included |
| Calendar (5) | `week_of_year`, `month`, `quarter`, `sin_woy`, `cos_woy` | `sin_woy = sin(2π · week / 52.18)`, `cos_woy = cos(2π · week / 52.18)`. Sin/cos put week 52 and week 1 next to each other on a circle, which a plain week number can't |
| Lagged drivers (6) | `price_per_unit_lag1/4`, `cogs_per_unit_lag1/4`, `inventory_level_lag1/4` | Price, unit cost and inventory 1 and 4 weeks earlier (lagged only, see "Leakage safeguards") |
| Stockout (1) | `stockout_recent_4wk` | Number of stockout weeks among the previous 4 |

**Step 2 — Build the training table.** One row per historical week: the 26
feature values, and that week's actual demand as the target. Any row
missing a feature is dropped. The first 52 weeks have no `lag_52`, so they
are lost. That leaves **78, 130 and 182 usable rows** in CV folds 1, 2 and
3 (trained on 130, 182 and 234 weeks), and 208 rows for the final
production model. This is a small dataset for 26 features, which matters
for the ablation results further down.

**Step 3 — Fit a gradient-boosted tree ensemble.** LightGBM builds 250
small decision trees one after another. Each new tree is fitted to the
errors the ensemble still makes, and its contribution is scaled down by
the learning rate (0.04) so no single tree dominates. The final
prediction is the sum of all trees' contributions. Trees can represent
curved, non-linear patterns (such as a seasonal cycle) without anyone
specifying the shape in advance.

| Setting | Value | Meaning |
|---|---|---|
| `n_estimators` | 250 | Number of trees |
| `learning_rate` | 0.04 | Shrinks each tree's contribution |
| `max_depth`, `num_leaves` | 4, 15 | Caps how complex any one tree can be |
| `min_child_samples` | 10 | Minimum rows a leaf must contain |
| `colsample_bytree` | 0.8 | Each tree sees a random 80% of the features |
| `reg_alpha`, `reg_lambda` | 0.1, 0.1 | L1 / L2 penalties that discourage extreme leaf values |
| `subsample` | 0.8 | **Set but inactive.** LightGBM only subsamples rows when `subsample_freq` is also non-zero, and it defaults to 0. Verified: predictions are identical with `subsample` at 0.8 and 1.0. Left unchanged so published results stay reproducible |
| `random_state` | 42 | Reproducibility |

These values were chosen once, by judgment, and **not tuned**.

**Step 4 — Forecast 26 weeks ahead, recursively.** A 26-week forecast
needs features for weeks whose demand isn't known yet, so the model feeds
its own output back in:

1. Build the feature row for forecast week 1 from real history and predict.
2. Append that prediction to the history as if it were an actual.
3. Rebuild the features for week 2 (so `lag_1` is now the week-1
   *prediction*), predict, append, and repeat through week 26.

By week 26, `lag_1` to `lag_12` and most rolling windows are built from
the model's own earlier predictions, while `lag_26` and `lag_52` still
point at real history. Two consequences: errors can compound, and
forecasts come out smoother than reality, because the model can't predict
noise and so feeds smooth values back in. Future price, cost and inventory
are unknown, so they are held at their last observed values, and no future
stockouts are assumed.

**Worked example** — same product and fold as above:

| Forecast week | Actual | Baseline | LightGBM | Baseline error | LightGBM error |
|---|---|---|---|---|---|
| 1 (data week 183) | 1,061 | 1,013 | 1,270 | 48 | 209 |
| 2 | 1,207 | 1,426 | 1,360 | 219 | 153 |
| 3 | 1,295 | 1,139 | 1,274 | 156 | 21 |
| 4 | 1,022 | 1,248 | 1,216 | 226 | 194 |
| 5 | 1,049 | 1,166 | 1,242 | 117 | 193 |
| 6 | 1,025 | 1,083 | 1,199 | 58 | 174 |

Over all 26 weeks: baseline MAE 155.1 and MAPE 14.43%; LightGBM MAE 126.6
and MAPE 12.67%. LightGBM was the more accurate forecast in 15 of the 26
weeks and the baseline in 11, so the win comes from a modest edge that
adds up, and not from beating the baseline every week. The LightGBM
forecast is also visibly smoother: its weekly standard deviation is 96
against 152 for actual demand.

### How the folds are built

The walk-forward cross-validation mimics the real situation: stand at some
point in history, forecast the next 26 weeks using only what was known
then, and score the forecast against what actually happened. Repeating
that at several points in history gives several independent-ish tests.

**Inputs** (constants at the top of `src/forecasting.py`):

| Constant | Value | Meaning |
|---|---|---|
| `n` | 260 | Weeks of history per product |
| `HORIZON` | 26 | Weeks forecast in each fold (same as the deliverable) |
| `MIN_TRAIN_SIZE` | 130 | Fewest weeks the first fold may train on |
| `N_CV_SPLITS` | 3 | Number of folds |

**Derivation:**

```
max_origin = n − HORIZON = 260 − 26 = 234
origins    = linspace(MIN_TRAIN_SIZE, max_origin, N_CV_SPLITS)
           = linspace(130, 234, 3) = [130, 182, 234]

fold k:   train = weeks 1 … origin_k
          test  = weeks origin_k + 1 … origin_k + 26
```

| Fold | Train weeks | Usable training rows | Test weeks | Test dates* | Position in the 52-week cycle |
|---|---|---|---|---|---|
| 1 | 1–130 | 78 | 131–156 | 2023-07-05 → 2023-12-27 | weeks 27–52 |
| 2 | 1–182 | 130 | 183–208 | 2024-07-03 → 2024-12-25 | weeks 27–52 |
| 3 | 1–234 | 182 | 235–260 | 2025-07-02 → 2025-12-24 | weeks 27–52 |
| *Production forecast* | *1–260* | *208* | *261–286* | *2025-12-31 → 2026-06-24* | *weeks 1–26* |

\*Dates come from the anchor assumed in Stage 1 (week 1 = 2021-01-06), since
the source file has no calendar dates. The cycle position, which is what
matters here, does not depend on that assumption.

**Rules every fold follows:**

- **Expanding window.** Each fold trains on *all* history up to its
  origin, so later folds see more data. Nothing is discarded.
- **Refit from scratch.** A fresh model is trained in every fold. Nothing
  learned from a later fold's data can reach an earlier fold.
- **Nothing after the origin is visible.** Neither the training rows nor
  the features used to forecast the test window contain post-origin
  actuals (the forecast is recursive, see Step 4 above).
- **Same windows for both models.** The baseline and LightGBM are trained
  on, and scored against, identical weeks.
- **Test windows do not overlap** in the production folds (they are 52
  weeks apart and 26 weeks long).

**Why these values.** These were judgment calls and were not tuned:

- *`HORIZON = 26`* matches the 6-month deliverable, so CV measures the
  same task production performs.
- *The last origin is `n − HORIZON`* so the final fold's test window ends
  exactly on the last observed week. That makes the last fold the most
  recent, most relevant test and uses every week of data.
- *`MIN_TRAIN_SIZE = 130` (about 2.5 years)* has to exceed 52 weeks
  simply to compute `lag_52` at all, and 130 leaves 78 usable rows to fit
  on. A smaller value would leave even fewer rows to fit 26 features on;
  a larger one would pull the evenly spaced folds closer together (more
  overlap, less variety between them).
- *`N_CV_SPLITS = 3`* is a compromise between run time (each extra fold
  is another 30 model fits plus 30 recursive forecasts per variant) and
  amount of evidence.

**A limitation we found in this design.** `linspace` spaces the three
origins evenly, and (234 − 130) / 2 = 52, exactly one year. So all three
test windows sit in the *same half* of the annual cycle (weeks 27–52,
roughly July–December), while the production forecast covers weeks 1–26
(January–June). **The three production folds never tested the seasonal
phase that is actually being forecast.** The headline CV numbers could
therefore have been unrepresentative of the forecast period.

**Robustness check** (`src/cv_fold_check.py`). We re-ran the comparison with
a denser schedule: an origin every 13 weeks from 130 to 234, giving 9
folds per product whose test windows start at every quarter of the cycle,
including two (origins 156 and 208) that start at cycle week 1 exactly
like the production forecast. Mean MAPE across 30 products:

| Test window starts at cycle week | Fold origins | Baseline | Full model (ships) | Calendar-only |
|---|---|---|---|---|
| **1 (same phase as production forecast)** | 156, 208 | 13.70% | 11.74% | 11.28% |
| 14 | 169, 221 | 13.91% | 11.57% | 11.30% |
| 27 (the original 3 folds) | 130, 182, 234 | 14.30% | 12.61% | 11.38% |
| 40 | 143, 195 | 13.84% | 11.52% | 11.31% |
| **All 9 dense folds** | | **13.98%** | **11.94%** | **11.32%** |

What this shows:

1. **The conclusions hold in the phase we forecast.** For folds starting at
   cycle week 1, the full model has 11.74% MAPE against the baseline's
   13.70% (14% lower), and beats the baseline in 87% of the 60
   product-folds. Calendar-only beats the baseline in 92%.
2. **The original three folds were, if anything, the hardest phase for the
   shipped model** (12.61%, against 11.5–11.7% in the other phases), so
   the headline "about 12% better than baseline" was slightly
   conservative. Across all 9 folds the improvement is about 15%.
3. **The calendar-only ranking is unchanged:** 11.32% against 11.94% for
   the full model, better in 60% of 270 product-folds (64% in the original
   ablation).

**Caveat.** Dense folds overlap (adjacent test windows share up to 13 weeks
and adjacent training sets share almost all their history), so the 270
product-folds are not independent evidence. Read them as a check that the
result isn't an artifact of one seasonal phase, not as a 270-sample
significance test.

### How the comparison is scored

**Metrics**, computed per fold and then averaged:

```
MAE  = mean( |actual − forecast| )                  units of demand
RMSE = sqrt( mean( (actual − forecast)² ) )         units; penalizes large misses more
MAPE = mean( |actual − forecast| / actual ) × 100   percent; scale-free
```

Averages are taken over 30 products × 3 folds = 90 evaluations. MAE and
RMSE are in units, so high-volume products dominate those two averages.
MAPE is scale-free and treats every product equally, which is why it's the
headline number.

### Results

The headline table uses the three production folds described above.

| Model | Mean MAE | Mean RMSE | Mean MAPE |
|---|---|---|---|
| Baseline (seasonal-naive) | 254.2 | 312.8 | 14.30% |
| **LightGBM direct, 26 features** | **219.1** | **270.9** | **12.61%** |

LightGBM beats the baseline on **26 of 30 products** (by average MAPE) and
lowers average MAPE by about 12%. Baseline still wins on Cocoa Powder,
Milk Powder - Whole, Spices - Turmeric and Sugar - White Refined. Per-fold,
per-product numbers are in `data/processed/cv_metrics.csv`. Note the
baseline here is a *single-year* copy; a multi-year seasonal average (no ML)
already reaches 12.01%, so "beats the baseline" overstates what the ML adds
(see the ablation below). The 9-fold
robustness check above puts the improvement closer to 15% and shows it
also holds for the seasonal phase being forecast.

### What actually drives LightGBM's advantage (ablation)

**An earlier draft of this README, and the docstring in `forecasting.py`,
said the model wins mainly because it uses `lag_52` and `lag_1` directly.
Testing showed that is not right.** Averaged gain-based feature importance
across the 30 models attributes about **64% to calendar features**
(`sin_woy` alone is 47%, `week_of_year` 13%), 16% to all lag features
(`lag_52` only 2.6%, `lag_1` 3.4%), 11% to rolling statistics, 9% to
lagged price/cost/inventory, and 0.2% to the stockout flag. Gain
importance is split unevenly across correlated features (the five
calendar features encode the same information), so it is only a hint. To
test it properly, `src/ablation.py` re-runs the identical recursive CV
with feature groups removed. First, what each variant is:

**What each row means.** Every LightGBM row below uses the same recursive
CV, the same folds, and the same settings; only the *input features*
differ:

| Variant | Inputs the model is given |
|---|---|
| **Calendar only** (5 features) | Only *where the week falls in the yearly cycle*: `week_of_year`, `month`, `quarter`, `sin_woy`, `cos_woy`. **It never sees any past demand, price, cost or inventory.** |
| Full minus `lag_52` (25) | Everything in the full model except demand 52 weeks ago |
| **Full** (26), what ships | Calendar + lags + rolling stats + lagged price/cost/inventory + stockout flag |
| Everything except calendar (21) | Lags, rolling stats, lagged price/cost/inventory, stockout flag; no calendar features |

Two no-ML reference forecasts are scored on the same folds:
*Baseline* (demand 52 weeks ago, one year) and *Seasonal average* (the
average of the same week in **every** earlier year: 52, 104, 156 weeks ago
and so on).

**Understanding "calendar only".** A calendar-only model answers a single
question: *"what is demand typically like at this time of year?"* Its only
knowledge of a week is its position in the 52-week cycle, so it learns a
seasonal profile from every training year at once. As a check, its
forecast has a 0.978 correlation with the plain multi-year same-week
average (checked on one product and fold: Rice - Basmati, fold 2). Consequences:

- **It can't react to anything recent.** No trend, no recent spike or
  slump, no price or inventory effects. Each product gets the same
  seasonal shape every year. That's a real limitation for products whose
  level is drifting.
- **It needs no history of its own predictions,** so the recursive
  forecast can't compound errors, and it trains on *every* week (52
  more rows than the full model, which loses its first year for lack of
  `lag_52`; in fold 1 that is 130 rows against 78).
- **"Calendar" doesn't mean real holidays or real months here.** The
  source file has no dates; Stage 1 assumed Week 1 = 2021-01-06. So these
  features really encode *position in the 52-week cycle*. The model can't
  know that a given week is, say, a religious holiday or harvest season.

| Feature set | Features | Mean MAPE | Products beating baseline |
|---|---|---|---|
| Calendar only | 5 | **11.38%** | **30 / 30** |
| Full minus `lag_52` | 25 | 11.99% | 28 / 30 |
| *Seasonal average (no ML)* | none | *12.01%* | *30 / 30* |
| **Full (what ships)** | 26 | 12.61% | 26 / 30 |
| *Baseline (seasonal-naive)* | none | *14.30%* | — |
| Everything except calendar | 21 | 18.02% | 10 / 30 |

Paired at the finest grain (90 product-fold pairs):

| Comparison | Left side wins |
|---|---|
| Calendar only vs baseline | 97% |
| Seasonal average vs baseline | 96% |
| Full vs baseline | 79% |
| Calendar only vs full | 64% |
| Calendar only vs seasonal average | 88% |
| Seasonal average vs full | 48% |
| Full minus `lag_52` vs full | 54% |

The last two rows are coin flips: the shipped model is statistically
indistinguishable from a plain multi-year average, and dropping `lag_52`
makes no reliable difference, so we make no claim that `lag_52` hurts.

**What this means:**

1. The advantage comes from learning each product's *annual seasonal
   shape* pooled across every training year. That averages out the
   one-off noise the single-year baseline copies.
2. **Most of the gain does not need machine learning.** A plain average of
   earlier years captures most of it (12.01% vs 14.30%). The shipped
   26-feature model does no better than that average (48% of pairs).
3. **ML adds a smaller but consistent extra on top:** calendar-only
   LightGBM beats the plain average in 88% of pairs (11.38% vs 12.01%),
   plausibly by smoothing across neighbouring weeks (untested).
4. Lag, rolling, price and inventory features alone are *worse* than the
   baseline. With only 78–208 usable rows per model, the extra features
   most likely add noise more than signal (an overfitting hypothesis we
   have not directly tested).
5. **The shipped 26-feature model is not the best variant we tested.** The
   5-feature calendar-only model scored better on average (11.38% vs
   12.61%) and beat the baseline on all 30 products. We did *not* switch,
   because doing so changes the forecasts and therefore every downstream
   output (stockout alerts, weeks-of-cover, pricing recommendations). It
   is the recommended next iteration. Caveats: 3 folds is limited
   evidence, the 64% pairwise edge is real but modest, and all variants
   share one un-tuned set of hyperparameters. The 9-fold check in "How the
   folds are built" reproduces the ranking of calendar-only over full
   (11.32% vs 11.94%, better in 60% of product-folds), which is
   corroboration, not independent proof, because those folds overlap. (The
   seasonal average was only scored on the 3 production folds.)

### Model iteration: what we tried first

The first design was the "textbook" hybrid: MSTL-decompose each product's
demand into trend + seasonal(52) + seasonal(4), extrapolate those pieces
forward, and train LightGBM on the leftover remainder only. It was tested
on **one product, Rice - Basmati** (not all 30), where it lost to the
baseline in all three folds (MAPE 17.9%, 26.4%, 28.4% against 13.6%,
14.4%, 11.4%). Switching to the direct model brought that product to 12.5%
against the baseline's 13.1%.

We did **not** isolate why the hybrid failed. Plausible contributors:
extrapolating a straight line from the last 26 weeks of the MSTL trend;
the `seasonal(4)` term possibly fitting noise; a seasonal component
estimated from as few as 2.5 annual cycles in fold 1; and stacking a
LightGBM model on the remainder. An earlier version of this README stated
a definite root cause. The evidence supported less than that.

MSTL is kept as a **descriptive factor-analysis tool**
(`decompose_for_analysis()`) to visualize trend, seasonal and residual
structure, but it is not in the prediction path.

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
python src/forecasting.py     # ~1 minute: CV + 26-week forecast for all 30 products
python src/ablation.py        # ~2 minutes: the feature-group ablation described above
python src/cv_fold_check.py   # ~3-5 minutes: fold schedule + 9-fold seasonal-phase robustness check
```

Produces:
- `data/processed/cv_metrics.csv` — per-product, per-fold MAE/RMSE/MAPE for both the model and the baseline
- `data/processed/future_forecast.csv` — the 26-week-ahead forecast for all 30 products
- `data/processed/ablation_results.csv` — per-product, per-fold MAPE for each feature-set variant
- `data/processed/cv_dense_folds.csv` — per-product MAPE for each of the 9 dense folds

## Stage 3 — KPI Dashboard

**Built with Streamlit + Plotly** (chosen over Power BI/Tableau to keep the
whole pipeline in one reproducible, version-controlled Python stack — no
separate paid tool or manual refresh step required).

```bash
streamlit run dashboard/app.py
```

Business logic (`src/kpis.py`) is kept separate from presentation
(`dashboard/app.py`) so every KPI can be unit-tested independently of the
UI (see `python src/kpis.py` for a plain-console version of the same
numbers).

### Financial KPIs (trusted, provided fields)

Gross Profit Margin, Revenue Growth, and Operating Profit come straight
from the source file's `Revenue`, `Gross_Profit`, and `Operating_Profit`
columns, which were cross-validated against their raw components in Stage
1 (`Revenue = Sales_Units × Price_Per_Unit`, etc. — 0 mismatches). Revenue
Growth is recomputed as trailing-13-week-over-prior-13-week rather than
the source file's week-over-week figure, which is far too noisy to read
meaningfully at a glance.

### Operational KPIs — recomputed, not trusted from the source file

**Finding:** unlike Revenue/Gross_Profit, the provided
`Inventory_Turnover_Ratio` does **not** reconcile with any combination of
`Cost_of_Goods_Sold_COGS`, `Sales_Units`, or `Inventory_Level` — it spans
0.00–1.00 in flat 0.01 increments with no relationship to the underlying
numbers, suggesting it's an independently-generated placeholder rather
than a real computed ratio.

The dashboard recomputes it properly:
```
Annualized Inventory Turnover = (Trailing 52-week COGS) / (Trailing 52-week average Inventory)
Days Sales of Inventory (DSI) = 365 / Annualized Turnover
```
Some products still show extreme values (e.g. Rice - White at ~0.15 days,
Spices - Turmeric at ~298 days) — this reflects how inventory levels were
generated in the provided sample data (some products carry inventory
levels that are tiny relative to weekly COGS, others carry very large
buffer stock), not a computation error. Flagged in the dashboard for the
business to sanity-check against real operations.

### Sales & Credit KPIs — data limitation, handled transparently

The brief asks for **Customer Default Risk Rating** and **Days Sales
Outstanding (DSO)**. This dataset has **no customer identity, invoice, or
payment-term data of any kind** — only product-level weekly sales,
inventory, and pricing. These cannot be genuinely computed from what's
provided.

Rather than fabricate numbers that look like real credit metrics, the
dashboard shows an explicitly-labeled **illustrative risk proxy** built
from product-level revenue volatility and stockout frequency (Low /
Medium / High), with a permanent "Data limitations" tab explaining exactly
why and what it isn't. This is presented as an honest placeholder for what
real customer/credit data would enable — not disguised as the real thing.

### Inventory risk alerts (genuinely data-grounded)

Flags any product whose current inventory covers fewer than 4 weeks of
its own forecasted demand (from Stage 2's 26-week forecast). Verified
against underlying numbers — e.g. Barley: 2,500 units on hand vs. ~2,884
units/week forecasted demand → correctly flagged, real signal, not a
false positive from stale data.

### Verification

Tested end-to-end: Streamlit server starts and serves cleanly
(`/_stcore/health` → `ok`), and every KPI function plus every format
string the dashboard uses was exercised directly against all 30 products
with zero errors before this was committed.

## Stage 4 — Dynamic Pricing & Risk Strategy

Full mathematical write-up, calibration story, and worked examples in
[`docs/pricing_strategy.md`](docs/pricing_strategy.md). Summary:

A rule-based pricing engine (`src/pricing.py`) recommends a price
adjustment per product from `demand_index` (near-term forecast vs.
trailing 52-week average) and `inventory_gap` (current stock vs. a 4-week
target — same threshold as Stage 3's stockout alerts), then applies two
guardrails: a **credit-risk guardrail** (blocks/halves discounts on
high/medium-risk products, using the same proxy from Stage 3) and a
**competitive guardrail** (hard-caps total swing at ±15%, standing in for
a real competitor price feed this dataset doesn't have).

**Calibration finding worth flagging:** the first parameter set made the
competitive guardrail fire on 27/30 products (90%) — the "guardrail" had
become the actual policy. Root cause: most products in this dataset sit
well below their 4-week inventory target at the latest observed week
(consistent with Stage 1's stockout finding), so the inventory term
dominated. Recalibrated elasticities down; the guardrail now fires on 5/30
(17%), a real safety net rather than a default clamp.

Interactive: run `streamlit run dashboard/app.py` and open the **Pricing
engine** tab — all four policy parameters are live sliders, and every
recommendation shows its full audit trail (which multiplier moved it, by
how much, which guardrail fired and why).

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
│   ├── forecasting.py                # per-product LightGBM demand forecasting + walk-forward CV
│   ├── ablation.py                   # feature-group ablation: what drives the model's edge over the baseline
│   ├── cv_fold_check.py              # documents fold construction + 9-fold seasonal-phase robustness check
│   ├── kpis.py                       # Financial/Operational/Sales & Credit KPI business logic
│   └── pricing.py                    # Dynamic pricing engine: demand + inventory rules, guardrails
├── docs/
│   └── pricing_strategy.md            # Stage 4 deliverable: full mathematical write-up
├── dashboard/
│   └── app.py                         # Streamlit dashboard (KPIs + live pricing engine)
├── notebooks/           # exploratory analysis (Stage 2+)
└── dashboard/           # Streamlit app (Stage 3+)
```
