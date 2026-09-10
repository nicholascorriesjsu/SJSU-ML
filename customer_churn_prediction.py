"""
Coding Exercise - ML Basics
Part 2: Predict Customer Churn.

===============================================================================
DATA SOURCE
===============================================================================
IBM Telco Customer Churn sample dataset - a widely used telecom retention
dataset describing a fictional-but-realistic California telco's customer base.

  Records used : 7,043 customers (1,869 churned = 26.5% churn rate)
  Raw CSV      : https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
  Publisher    : IBM Cognos Analytics / IBM Community sample data

Feature groups map onto the four categories named in the assignment:
  demographics       -> gender, SeniorCitizen, Partner, Dependents
  usage patterns     -> tenure, InternetService, StreamingTV
  purchase history   -> MonthlyCharges, TotalCharges, Contract, PaymentMethod
  service interaction-> TechSupport, OnlineSecurity

===============================================================================
CHANGES / IMPROVEMENTS OVER THE STARTER CODE
===============================================================================
1. Replaced the 10-row hard-coded dictionary with 7,043 real customer records
   (with a reproducible synthetic fallback so the script runs offline).
2. Added real data cleaning: TotalCharges arrives as text with blanks for
   brand-new customers; those are coerced to numeric and dropped.
3. Added a full classification evaluation - the starter code trained a model
   and never scored it. Reports accuracy, precision, recall, F1, ROC-AUC and a
   confusion matrix on a stratified held-out test set.
4. Used class_weight='balanced'. Churn is a 27/73 split, so an unweighted model
   maximizes accuracy by under-predicting churn, which is the opposite of what
   a retention team needs.
5. Replaced the blind 0.5 threshold with a threshold sweep plus a simple
   expected-value calculation (cost of an offer vs. value of a saved customer),
   so the cutoff is chosen for business impact rather than by convention.
6. Reported coefficients as odds ratios, which are interpretable ("a two-year
   contract multiplies the odds of churn by X") rather than raw log-odds.
7. Added ROC curve and threshold-tradeoff plots saved to PNG.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

RANDOM_STATE = 42
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents",
                        "InternetService", "StreamingTV", "TechSupport",
                        "OnlineSecurity", "Contract", "PaymentMethod",
                        "PaperlessBilling"]

# Business assumptions used to pick the classification threshold.
OFFER_COST = 50.0          # $ cost of a retention offer sent to one customer
CUSTOMER_VALUE = 500.0     # $ margin retained if an at-risk customer stays
SAVE_RATE = 0.35           # share of contacted at-risk customers actually saved


# =============================================================================
# 1. LOAD AND CLEAN DATA
# =============================================================================
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        # TotalCharges is stored as text; blanks are customers with tenure = 0.
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        blanks = df["TotalCharges"].isna().sum()
        df = df.dropna(subset=["TotalCharges"])
        df["churn"] = (df["Churn"] == "Yes").astype(int)
        print(f"Cleaned {blanks} blank TotalCharges values (tenure-0 customers)")
        return df, "IBM Telco Customer Churn (real data)"
    except Exception as err:
        print(f"[warn] could not download Telco data ({err}); using simulated data.")
        return _simulate(n=2000), "Simulated fallback (2000 rows)"


def _simulate(n=2000):
    """Reproducible stand-in with the same column names and similar structure."""
    rng = np.random.default_rng(RANDOM_STATE)
    tenure = rng.integers(0, 72, n)
    monthly = rng.normal(65, 30, n).clip(18, 120).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .21, .24])
    support = rng.choice(["Yes", "No", "No internet service"], n, p=[.29, .49, .22])
    logit = (-1.2 - 0.045 * tenure + 0.018 * monthly
             + np.where(contract == "Month-to-month", 1.4, -0.6)
             + np.where(support == "No", 0.6, -0.2))
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({
        "tenure": tenure, "MonthlyCharges": monthly,
        "TotalCharges": (tenure * monthly).round(2),
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[.84, .16]),
        "Partner": rng.choice(["Yes", "No"], n), "Dependents": rng.choice(["Yes", "No"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[.34, .44, .22]),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n, p=[.38, .40, .22]),
        "TechSupport": support, "OnlineSecurity": rng.choice(["Yes", "No"], n),
        "Contract": contract,
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)",
                                     "Credit card (automatic)"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n), "churn": churn,
    }), "Simulated fallback"


# =============================================================================
# 2. BUILD AND TRAIN THE MODEL
# =============================================================================
def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first",
                                  sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        # class_weight='balanced' stops the model from ignoring the 27% minority
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=RANDOM_STATE)),
    ])


def choose_threshold(y_true, probabilities):
    """Sweep thresholds and pick the one with the best expected net value."""
    rows = []
    for t in np.arange(0.20, 0.81, 0.05):
        flagged = probabilities >= t
        if flagged.sum() == 0:
            continue
        true_positives = ((flagged == 1) & (y_true == 1)).sum()
        net = (true_positives * SAVE_RATE * CUSTOMER_VALUE) - (flagged.sum() * OFFER_COST)
        rows.append({
            "threshold": t,
            "flagged": int(flagged.sum()),
            "precision": precision_score(y_true, flagged, zero_division=0),
            "recall": recall_score(y_true, flagged, zero_division=0),
            "net_value": net,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("PART 2: CUSTOMER CHURN PREDICTION")
    print("=" * 78)
    df, source = load_data()
    print(f"Data source : {source}")
    print(f"Records     : {len(df):,}")
    print(f"Churn rate  : {df['churn'].mean():.1%} "
          f"({df['churn'].sum():,} churned / {len(df):,} customers)")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churn"]
    # stratify keeps the churn rate identical in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # ---- evaluation --------------------------------------------------------
    print("\nMODEL PERFORMANCE ON THE HELD-OUT TEST SET (20%, threshold = 0.50)")
    print(f"  Accuracy  : {accuracy_score(y_test, predictions):.3f}")
    print(f"  Precision : {precision_score(y_test, predictions):.3f}  "
          f"(of customers we flag, this share really churn)")
    print(f"  Recall    : {recall_score(y_test, predictions):.3f}  "
          f"(of customers who churn, this share we catch)")
    print(f"  F1 score  : {f1_score(y_test, predictions):.3f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, probabilities):.3f}")

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print("\n  Confusion matrix:")
    print(f"    True negatives  {tn:>5}   False positives {fp:>5}")
    print(f"    False negatives {fn:>5}   True positives  {tp:>5}")
    print("\n" + classification_report(y_test, predictions,
                                       target_names=["Stayed", "Churned"]))

    # ---- threshold selection ----------------------------------------------
    sweep = choose_threshold(y_test.values, probabilities)
    best = sweep.loc[sweep["net_value"].idxmax()]
    print("THRESHOLD SELECTION")
    print(f"  Assumptions: ${OFFER_COST:.0f} per retention offer, "
          f"${CUSTOMER_VALUE:.0f} margin per saved customer, "
          f"{SAVE_RATE:.0%} of contacted at-risk customers are saved.")
    print(f"  {'thresh':>7} {'flagged':>8} {'precision':>10} {'recall':>8} {'net value':>12}")
    for _, r in sweep.iterrows():
        marker = "  <-- best" if r["threshold"] == best["threshold"] else ""
        print(f"  {r['threshold']:>7.2f} {r['flagged']:>8.0f} {r['precision']:>10.3f} "
              f"{r['recall']:>8.3f} {r['net_value']:>12,.0f}{marker}")
    print(f"\n  Recommended threshold: {best['threshold']:.2f} "
          f"(not the default 0.50) - catches {best['recall']:.0%} of churners "
          f"for an estimated ${best['net_value']:,.0f} net on this test sample.")

    # ---- coefficients as odds ratios ---------------------------------------
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = pd.Series(model.named_steps["classifier"].coef_[0], index=feature_names)
    odds = np.exp(coefs).sort_values(ascending=False)
    print("\nWHAT DRIVES CHURN (odds ratios; >1 raises churn odds, <1 lowers them)")
    print("  Strongest churn drivers:")
    for feature, ratio in odds.head(5).items():
        print(f"    {feature:<32} {ratio:>6.2f}x")
    print("  Strongest retention factors:")
    for feature, ratio in odds.tail(5).items():
        print(f"    {feature:<32} {ratio:>6.2f}x")

    # ---- predict for a new customer ---------------------------------------
    new_customer = X_test.iloc[[0]].copy()
    new_customer.loc[:, ["tenure", "MonthlyCharges", "TotalCharges"]] = [3, 95.0, 285.0]
    if "Contract" in new_customer:
        new_customer.loc[:, "Contract"] = "Month-to-month"
        new_customer.loc[:, "TechSupport"] = "No"
    probability = model.predict_proba(new_customer)[0][1]
    print(f"\nNEW CUSTOMER SCORING (3 months tenure, $95/mo, month-to-month, no tech support)")
    print(f"  Churn probability            : {probability:.2f}")
    print(f"  Classified at 0.50 threshold : {int(probability >= 0.5)}")
    print(f"  Classified at {best['threshold']:.2f} threshold : "
          f"{int(probability >= best['threshold'])}  (1 = flag for a retention offer)")

    # ---- plots -------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, probabilities):.3f}")
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=1, label="random guessing")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC curve")
    axes[0].legend()

    axes[1].plot(sweep["threshold"], sweep["precision"], "o-", label="precision")
    axes[1].plot(sweep["threshold"], sweep["recall"], "s-", label="recall")
    axes[1].axvline(best["threshold"], color="green", linestyle="--",
                    label=f"chosen = {best['threshold']:.2f}")
    axes[1].set(xlabel="Classification threshold", ylabel="Score",
                title="Precision / recall tradeoff")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("churn_model.png", dpi=120)
    plt.close()
    print("\nSaved evaluation plots to churn_model.png")

    scored = X_test.copy()
    scored["actual_churn"] = y_test.values
    scored["churn_probability"] = probabilities.round(4)
    scored["flag_for_retention"] = (probabilities >= best["threshold"]).astype(int)
    scored.to_csv("churn_scored_customers.csv", index=False)
    print("Saved scored test customers to churn_scored_customers.csv")


if __name__ == "__main__":
    main()
