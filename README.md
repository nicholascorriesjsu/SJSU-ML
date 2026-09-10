# Coding Exercise — ML Basics

Submission for the ML Basics coding exercise. Each part of the assignment has its own
script. All four scripts run standalone, print their results to the console, and save
plots and CSVs alongside themselves.

```bash
pip install -r requirements.txt

python house_price_prediction.py     # Part 1 — linear regression
python customer_churn_prediction.py  # Part 2 — logistic regression
python customer_segmentation.py      # Part 3 — K-Means clustering
python forecasting_sales.py          # Extra credit — demand forecasting
```

---

## Datasets

The starter code used 10 hard-coded rows per exercise. Every script here uses a real
public dataset of 100+ records, pulled from a URL at runtime and cited in the header
comment of the file. Each script also carries a reproducible synthetic fallback, so it
still runs if the network is unavailable.

| Part | Dataset | Records | Source |
|---|---|---|---|
| 1. House prices | Ames Housing (Kaggle "House Prices" training set) | 1,460 sales | De Cock, D. (2011), *Journal of Statistics Education* 19(3) |
| 2. Customer churn | IBM Telco Customer Churn | 7,032 customers | IBM Cognos Analytics sample data |
| 3. Segmentation | Mall Customer Segmentation | 200 members | Kaggle (Choudhary, 2018) |
| Extra credit | Monthly car sales, Quebec 1960–68 | 108 months | Abraham & Ledolter (1983), via jbrownlee/Datasets |

---

## Part 1 — House Price Prediction

Predicts sale price from above-grade living area and neighborhood, using one-hot
encoding inside a scikit-learn pipeline.

**Results (20% held-out test set)**

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Baseline (mean price) | 0.00 | $74,324 | $57,980 |
| Square footage only | 0.49 | $52,993 | $37,694 |
| Square footage + location | **0.72** | **$39,305** | **$28,372** |

5-fold cross-validated R² = 0.745 (± 0.041).

The model learns **$89 per square foot**, and location moves the price of an identical
2,000 sq ft house by **$163,077** between the most and least expensive neighborhoods —
more than the square footage of a 1,800 sq ft house is worth. That gap is the whole
argument for including location as a feature.

**Improvements over the starter code**

- Real data instead of 10 invented rows.
- Documented outlier removal (the two partial sales over 4,000 sq ft that De Cock flags).
- Actual evaluation — the starter trained a model and never scored it.
- Two baselines, so the value added by the location feature is measurable rather than assumed.
- Coefficient names taken from `get_feature_names_out()` instead of hand-assembled in an
  order that breaks silently if the `ColumnTransformer` changes.
- `handle_unknown='ignore'`, so an unseen neighborhood does not crash prediction.
- Predicted-vs-actual and residual plots.

---

## Part 2 — Customer Churn Prediction

Logistic regression over demographics, usage, billing and support-interaction features.

**Results (20% stratified held-out test set, threshold 0.50)**

| Metric | Value |
|---|---|
| Accuracy | 0.731 |
| Precision | 0.497 |
| Recall | 0.802 |
| F1 | 0.613 |
| ROC-AUC | **0.835** |

**The threshold is a business decision, not a default.** The assignment specifies 0.5,
but 0.5 is just a convention. The script sweeps thresholds from 0.20 to 0.80 and scores
each one on expected net value, assuming a $50 retention offer, $500 of retained margin
per saved customer, and a 35% save rate. **0.65 wins** — it still catches 69% of churners
while cutting wasted offers by roughly a third versus 0.50. Change those three constants
at the top of the file and the recommended threshold moves with them, which is the point.

**What drives churn** (odds ratios, >1 raises churn odds):

- Fiber optic internet — **2.30×**
- Electronic check payment — 1.53×
- Two-year contract — **0.24×** (the strongest retention factor by far)
- Longer tenure — 0.30×
- Tech support — 0.69×

The actionable read: month-to-month fiber customers paying by electronic check are the
retention team's target list, and moving customers onto annual contracts is worth more
than any single service add-on.

**Improvements over the starter code**

- Real data, plus cleaning that the real data actually requires (`TotalCharges` arrives as
  text with blanks for tenure-0 customers).
- Full classification evaluation with a confusion matrix — the starter had none.
- `class_weight='balanced'`. Churn is a 27/73 split, so an unweighted model maximizes
  accuracy by under-predicting churn, which is exactly backwards for a retention use case.
- Business-driven threshold selection instead of a blind 0.5 cutoff.
- Coefficients reported as odds ratios, which can be read in a memo.
- Stratified split, ROC curve, and precision/recall tradeoff plot.

---

## Part 3 — Customer Segmentation

K-Means over income, spending score and age.

**K is chosen from the data, not assumed.** The starter code hard-coded K=3 "based on a
typical elbow curve". This version computes silhouette scores for K=2 through 10 and
selects the best (**K=6**, silhouette 0.428), using the elbow curve only as a visual
cross-check. The elbow is ambiguous by construction; a silhouette score is a number you
can defend.

**Segments found**

| Cluster | Persona | Size | Income | Spending | Age |
|---|---|---|---|---|---|
| 3 | Prime Targets | 19.5% | $86k | 82 | 33 |
| 2 | Untapped Wallets | 16.5% | $89k | 17 | 42 |
| 4 | Enthusiasts | 11.5% | $25k | 78 | 25 |
| 0 | Mainstream Core (older) | 22.5% | $54k | 49 | 56 |
| 1 | Mainstream Core (younger) | 19.5% | $57k | 48 | 27 |
| 5 | Low Engagement | 10.5% | $26k | 19 | 46 |

The interesting one is **Untapped Wallets** — the highest income in the base and nearly
the lowest spending. That is 16.5% of customers with proven capacity and no engagement,
and it is invisible if you segment on income alone.

**Improvements over the starter code**

- Real data, and K selected by silhouette score rather than assumed.
- Fixed a logic bug in the starter's strategy block: its `if/elif` chain could hand the
  same generic message to several clusters and never described the cluster it was
  actually evaluating. Personas here come from a 3×3 income × spending grid scored
  against the population's own terciles, so no two segments collapse onto one label.
- Added cluster size and share of base — a segment holding 3% of customers does not
  deserve the marketing spend of one holding 30%.
- Kept the categorical field out of the distance calculation and used it for profiling.
  K-Means minimizes Euclidean distance, which is meaningless across one-hot dummies.
- Added a segment scatter plot with centroids next to the elbow and silhouette charts.

---

## Extra Credit — Demand Forecasting

Reads sales history from a CSV, backtests two models, forecasts 6 months ahead, plots
the result.

**Backtest (trained on 96 months, tested on the last 12)**

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| Naive seasonal benchmark (same month last year) | 2,291 | 1,960 | 10.8% |
| Linear trend only (the starter approach) | 3,797 | 3,344 | 19.3% |
| **Linear trend + month-of-year seasonality** | **1,723** | **1,522** | **8.7%** |

Adding 11 month dummies cuts RMSE by **55%** against the trend-only model — and, more
importantly, is the only version that beats the naive seasonal benchmark. The starter's
trend-only model loses to "just repeat last year," which is the standard test for whether
a forecasting model is worth deploying at all.

Assumptions, challenges encountered, and potential improvements are documented in full in
the header of `forecasting_sales.py`, per the extra-credit requirement. In short: the model
assumes a locally linear trend and stable additive seasonality, has no features for
promotions, pricing or shocks, and would be improved most by prediction intervals and
exogenous drivers rather than by a fancier algorithm.

---

## Generated outputs

| File | Produced by |
|---|---|
| `house_price_model.png`, `house_price_by_location.csv` | Part 1 |
| `churn_model.png`, `churn_scored_customers.csv` | Part 2 |
| `elbow_plot.png`, `customer_segments.csv` | Part 3 |
| `sales_forecast.png`, `sales_forecast.csv`, `sales_data.csv` | Extra credit |

All scripts use `random_state=42` and are fully reproducible.
