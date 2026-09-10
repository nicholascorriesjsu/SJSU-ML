"""
Coding Exercise - ML Basics
Extra Credit: A simple demand forecasting tool.

Reads historical sales data from a CSV file, trains a regression model on it,
forecasts demand for the next 6 months, and plots the result.

===============================================================================
DATA SOURCE
===============================================================================
Monthly car sales, Quebec, Canada, January 1960 - December 1968 (Abraham &
Ledolter, "Statistical Methods for Forecasting", Wiley 1983). Republished in
Jason Brownlee's public time-series dataset collection.

  Records used : 108 monthly observations
  Raw CSV      : https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv
  Columns      : Month (YYYY-MM), Sales (units)

On first run the script downloads this file and caches it locally as
sales_data.csv, then reads from that CSV - so it satisfies the "reads from a
CSV file" requirement and also runs offline afterwards. Point SALES_CSV at any
other file with a date column and a value column to forecast your own data.

===============================================================================
CHANGES / IMPROVEMENTS OVER THE STARTER CODE
===============================================================================
1. The starter regressed sales on a single month counter, which can only ever
   fit a straight line. This version adds month-of-year dummy variables so the
   model can learn seasonality on top of the trend - the dominant pattern in
   almost all real sales data.
2. Added a backtest: the last 12 months are held out, both models are scored on
   them (RMSE, MAE, MAPE), and the winner is reported. The starter never
   measured whether its forecast was any good.
3. Added a naive seasonal benchmark (same month last year). If a machine
   learning model cannot beat that, it is not worth deploying.
4. Handles real date parsing rather than assuming an integer month column.
5. Writes the forecast to CSV and the chart to PNG instead of calling
   plt.show(), which produces nothing in a headless or Colab batch run.

===============================================================================
ASSUMPTIONS
===============================================================================
- The trend is linear over the forecast horizon. Six months out that is
  reasonable; two years out it is not.
- Seasonality is additive and stable year to year (same absolute lift each
  March, not the same percentage lift).
- History repeats. There is no feature for promotions, pricing, competitor
  entry, supply constraints or recessions, so the model cannot anticipate any
  of them.
- Every month is weighted equally. A structural break in the last year is
  diluted by eight years of older data.
- Monthly totals are complete and comparable; no adjustment is made for months
  having different numbers of selling days.

===============================================================================
CHALLENGES ENCOUNTERED
===============================================================================
- With 108 observations and 12 model terms, the seasonal model has limited data
  per parameter, so the backtest window matters a great deal.
- Time series cannot use a random train/test split - shuffling would let the
  model peek at the future. The split has to be chronological.
- The strong December/January swing dominates the error metrics; a model that
  gets the trend right but the seasonal amplitude wrong still scores badly.
- Linear regression will happily forecast negative demand, so the output is
  clipped at zero.

===============================================================================
POTENTIAL IMPROVEMENTS
===============================================================================
- Use SARIMA, Prophet or gradient boosting with lag features instead of OLS.
- Add exogenous drivers: price, marketing spend, holidays, macro indicators.
- Model multiplicative seasonality by regressing on log(sales).
- Replace the single holdout with rolling-origin cross-validation.
- Produce prediction intervals, not a single point forecast - a range is what a
  planner actually needs for safety stock.
- Retrain on a schedule and monitor forecast error drift in production.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
            "monthly-car-sales.csv")
SALES_CSV = "sales_data.csv"     # local cache / bring your own file
DATE_COL, VALUE_COL = "Month", "Sales"
HOLDOUT_MONTHS = 12
FORECAST_MONTHS = 6


# =============================================================================
# 1. LOAD AND PREPROCESS
# =============================================================================
def load_data():
    """Read sales history from CSV, downloading and caching it if needed."""
    if not os.path.exists(SALES_CSV):
        try:
            pd.read_csv(DATA_URL).to_csv(SALES_CSV, index=False)
            print(f"Downloaded source data and cached it as {SALES_CSV}")
        except Exception as err:
            print(f"[warn] download failed ({err}); generating a sample CSV instead.")
            _simulate().to_csv(SALES_CSV, index=False)

    df = pd.read_csv(SALES_CSV)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df.dropna(subset=[VALUE_COL])

    # Feature engineering: a time index for trend, month number for seasonality.
    df["t"] = np.arange(len(df))
    df["month_of_year"] = df[DATE_COL].dt.month
    return df


def _simulate(n=108):
    """Reproducible stand-in: linear trend + seasonal wave + noise."""
    rng = np.random.default_rng(42)
    t = np.arange(n)
    seasonal = 2200 * np.sin(2 * np.pi * (t % 12) / 12 - 1.2)
    sales = 8000 + 55 * t + seasonal + rng.normal(0, 900, n)
    dates = pd.date_range("1960-01-01", periods=n, freq="MS")
    return pd.DataFrame({DATE_COL: dates.strftime("%Y-%m"),
                         VALUE_COL: sales.clip(0).round().astype(int)})


def make_features(df, seasonal=True):
    """Trend only, or trend plus 11 month-of-year dummies."""
    X = df[["t"]].copy()
    if seasonal:
        dummies = pd.get_dummies(df["month_of_year"], prefix="m", drop_first=True)
        # reindex guarantees all 11 dummy columns exist even if the slice is short
        dummies = dummies.reindex(columns=[f"m_{m}" for m in range(2, 13)], fill_value=0)
        X = pd.concat([X, dummies.astype(int)], axis=1)
    return X


# =============================================================================
# 2. BACKTEST
# =============================================================================
def score(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  {name:<34} RMSE = {rmse:>9,.0f}   MAE = {mae:>9,.0f}   MAPE = {mape:>5.1f}%")
    return rmse


def backtest(df):
    """Chronological holdout - never shuffle a time series."""
    train, test = df.iloc[:-HOLDOUT_MONTHS], df.iloc[-HOLDOUT_MONTHS:]
    y_train, y_test = train[VALUE_COL], test[VALUE_COL]
    print(f"\nBACKTEST - trained on {len(train)} months, tested on the last "
          f"{len(test)} months ({test[DATE_COL].dt.strftime('%Y-%m').iloc[0]} to "
          f"{test[DATE_COL].dt.strftime('%Y-%m').iloc[-1]})")

    # Benchmark: repeat the same month one year earlier.
    naive = df[VALUE_COL].shift(12).iloc[-HOLDOUT_MONTHS:]
    score("Naive seasonal benchmark", y_test.values, naive.values)

    results = {}
    for label, seasonal in [("Linear trend only (starter code)", False),
                            ("Linear trend + seasonality", True)]:
        model = LinearRegression().fit(make_features(train, seasonal), y_train)
        predictions = model.predict(make_features(test, seasonal))
        results[seasonal] = score(label, y_test.values, predictions)

    use_seasonal = results[True] <= results[False]
    print(f"\n  Winner: {'trend + seasonality' if use_seasonal else 'trend only'} "
          f"({(1 - min(results.values()) / max(results.values())) * 100:.1f}% "
          f"lower RMSE than the alternative)")
    return use_seasonal


# =============================================================================
# 3. FORECAST
# =============================================================================
def forecast(df, seasonal):
    """Refit on all history, then project FORECAST_MONTHS ahead."""
    model = LinearRegression().fit(make_features(df, seasonal), df[VALUE_COL])

    last_date = df[DATE_COL].iloc[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                                 periods=FORECAST_MONTHS, freq="MS")
    future = pd.DataFrame({
        DATE_COL: future_dates,
        "t": np.arange(len(df), len(df) + FORECAST_MONTHS),
        "month_of_year": future_dates.month,
    })
    # Demand cannot be negative; a linear model does not know that.
    future["forecast"] = model.predict(make_features(future, seasonal)).clip(0).round()
    return model, future


def main():
    print("=" * 78)
    print("EXTRA CREDIT: 6-MONTH DEMAND FORECAST")
    print("=" * 78)
    df = load_data()
    print(f"Records     : {len(df)} monthly observations")
    print(f"Date range  : {df[DATE_COL].min():%Y-%m} to {df[DATE_COL].max():%Y-%m}")
    print(f"Sales range : {df[VALUE_COL].min():,.0f} to {df[VALUE_COL].max():,.0f} units "
          f"(mean {df[VALUE_COL].mean():,.0f})")

    use_seasonal = backtest(df)
    model, future = forecast(df, use_seasonal)

    print(f"\nFORECAST FOR THE NEXT {FORECAST_MONTHS} MONTHS")
    recent_avg = df[VALUE_COL].tail(12).mean()
    for _, row in future.iterrows():
        delta = (row["forecast"] - recent_avg) / recent_avg * 100
        print(f"  {row[DATE_COL]:%Y-%m}   {row['forecast']:>9,.0f} units   "
              f"({delta:+5.1f}% vs. trailing 12-month average)")
    print(f"\n  Total forecast demand: {future['forecast'].sum():,.0f} units")

    if use_seasonal:
        coefs = pd.Series(model.coef_, index=make_features(df, True).columns)
        print(f"\n  Monthly trend: {coefs['t']:+,.0f} units per month")
        seasonal_terms = coefs.drop("t")
        peak, trough = seasonal_terms.idxmax(), seasonal_terms.idxmin()
        print(f"  Strongest month: {peak.replace('m_', 'month ')} "
              f"({seasonal_terms.max():+,.0f} units vs. January)")
        print(f"  Weakest month  : {trough.replace('m_', 'month ')} "
              f"({seasonal_terms.min():+,.0f} units vs. January)")

    # ---- plot --------------------------------------------------------------
    fitted = model.predict(make_features(df, use_seasonal))
    plt.figure(figsize=(13, 6))
    plt.plot(df[DATE_COL], df[VALUE_COL], label="Historical sales",
             color="#1f77b4", linewidth=1.6)
    plt.plot(df[DATE_COL], fitted, label="Model fit", color="#ff7f0e",
             linewidth=1.1, alpha=0.8)
    plt.plot(future[DATE_COL], future["forecast"], "o--", color="#2ca02c",
             linewidth=2, label=f"{FORECAST_MONTHS}-month forecast")
    plt.axvline(df[DATE_COL].iloc[-1], color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Month")
    plt.ylabel("Sales (units)")
    plt.title("Demand forecast: linear regression with trend and seasonality")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("sales_forecast.png", dpi=120)
    plt.close()
    print("\nSaved forecast chart to sales_forecast.png")

    future[[DATE_COL, "forecast"]].to_csv("sales_forecast.csv", index=False)
    print("Saved forecast values to sales_forecast.csv")


if __name__ == "__main__":
    main()
