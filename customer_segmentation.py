"""
Coding Exercise - ML Basics
Part 3: Customer Segmentation with K-Means.

===============================================================================
DATA SOURCE
===============================================================================
Mall Customer Segmentation data - a membership-card dataset recording each
customer's age, annual income and a 1-100 spending score assigned by the mall
based on purchase behaviour and frequency.

  Records used : 200 mall members
  Raw CSV      : https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv
  Origin       : "Mall Customer Segmentation Data", Kaggle (Choudhary, 2018)

Column mapping to the assignment's feature list:
  Annual Income (k$)     -> annual_income   (spending capacity)
  Spending Score (1-100) -> spending_score  (purchase frequency / behaviour)
  Age                    -> age             (demographics)
  Genre                  -> gender          (categorical profiling dimension)

===============================================================================
CHANGES / IMPROVEMENTS OVER THE STARTER CODE
===============================================================================
1. Replaced the 10-row hard-coded dictionary with 200 real customer records
   (with a reproducible synthetic fallback so the script runs offline).
2. The starter code assumed K=3 "based on a typical elbow curve". This version
   actually chooses K from the data: it computes silhouette scores for K=2..10
   and selects the K with the best score, using the elbow only as a visual
   cross-check. The elbow is ambiguous by nature; silhouette is a number you
   can defend in a writeup.
3. Fixed a logic bug in the starter's strategy block: its if/elif chain could
   hand the same generic message to several clusters and never described the
   cluster it was actually looking at. Personas here are generated from each
   cluster's own position relative to the overall median.
4. Added cluster size and share-of-base, which the starter omitted - a segment
   holding 3% of customers does not deserve the same marketing spend as one
   holding 30%.
5. Left the categorical field out of the distance calculation and used it for
   profiling instead. K-Means minimizes Euclidean distance, which has no
   meaningful interpretation across one-hot dummy columns.
6. Added a segment scatter plot with cluster centroids alongside the elbow and
   silhouette charts.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
DATA_URL = (
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/"
    "Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/"
    "Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"
)
FEATURES = ["annual_income", "spending_score", "age"]
K_RANGE = range(2, 11)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data():
    try:
        raw = pd.read_csv(DATA_URL)
        df = raw.rename(columns={
            "Annual Income (k$)": "annual_income",
            "Spending Score (1-100)": "spending_score",
            "Age": "age",
            "Genre": "gender",
        })[["annual_income", "spending_score", "age", "gender"]]
        return df.dropna(), "Mall Customer Segmentation (real data, 200 members)"
    except Exception as err:
        print(f"[warn] could not download mall data ({err}); using simulated data.")
        return _simulate(n=300), "Simulated fallback (300 rows)"


def _simulate(n=300):
    """Reproducible stand-in with four latent shopper groups."""
    rng = np.random.default_rng(RANDOM_STATE)
    centers = [(25, 20, 45), (30, 80, 26), (85, 18, 42), (78, 82, 33)]
    rows = []
    for income_c, spend_c, age_c in centers:
        k = n // len(centers)
        rows.append(pd.DataFrame({
            "annual_income": rng.normal(income_c, 9, k).clip(15, 140),
            "spending_score": rng.normal(spend_c, 10, k).clip(1, 100),
            "age": rng.normal(age_c, 8, k).clip(18, 70),
        }))
    df = pd.concat(rows, ignore_index=True).round(0)
    df["gender"] = rng.choice(["Male", "Female"], len(df))
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# =============================================================================
# 2. CHOOSE K
# =============================================================================
def choose_k(X_scaled):
    """Return (best_k, inertias, silhouettes). Elbow is visual; silhouette decides."""
    inertias, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    return best_k, inertias, silhouettes


# =============================================================================
# 3. NAME THE SEGMENTS
# =============================================================================
def band(value, low_cut, high_cut):
    """Bucket a cluster mean into low / mid / high against the customer base."""
    if value <= low_cut:
        return "low"
    if value >= high_cut:
        return "high"
    return "mid"


# Persona for every income x spending combination, so two different clusters
# never collapse onto the same generic message.
PERSONAS = {
    ("high", "high"): ("Prime Targets",
                       "Highest lifetime value. Give them the top loyalty tier, early "
                       "access to new lines and a dedicated concierge. Protect first."),
    ("high", "mid"):  ("Growth Headroom",
                       "Real capacity, moderate engagement. Test premium bundles and "
                       "upsell at checkout - the ceiling here is the highest of any segment."),
    ("high", "low"):  ("Untapped Wallets",
                       "They can afford to spend and do not. Diagnose why before "
                       "discounting - survey them, then try styling invites or private events."),
    ("mid", "high"):  ("Loyal Regulars",
                       "Reliable engagement on an average budget. Reward consistency "
                       "with tiered points and restock reminders rather than deep discounts."),
    ("mid", "mid"):   ("Mainstream Core",
                       "The volume of the base. Serve efficiently with broad seasonal "
                       "campaigns and use A/B tests here before rolling out anywhere else."),
    ("mid", "low"):   ("Drifting Middle",
                       "Average means, below-average engagement. Winback offers and "
                       "category recommendations based on their last purchase."),
    ("low", "high"):  ("Enthusiasts",
                       "Highly engaged but budget-constrained. Push frequency, not basket "
                       "size: punch cards, installment plans, members-only sale nights."),
    ("low", "mid"):   ("Value Seekers",
                       "Price-sensitive and occasional. Lead with clearance, bundles and "
                       "loyalty points that convert into real discounts."),
    ("low", "low"):   ("Low Engagement",
                       "Lowest expected return. Keep acquisition cost near zero - "
                       "automated email reactivation only, no paid media spend."),
}


def describe(row, cuts, median_age):
    """Turn one cluster's mean profile into a persona label and a strategy."""
    income = band(row["annual_income"], *cuts["annual_income"])
    spend = band(row["spending_score"], *cuts["spending_score"])
    age = "younger" if row["age"] < median_age else "older"

    persona, strategy = PERSONAS[(income, spend)]
    # Two clusters can legitimately share an income/spending profile and differ
    # only by age, so the channel recommendation is age-driven.
    channel = ("Reach them on social and mobile push." if age == "younger"
               else "Reach them by email, direct mail and in-store staff.")
    return f"{persona} - {age} ({income} income, {spend} spending)", f"{strategy} {channel}"


def main():
    print("=" * 78)
    print("PART 3: CUSTOMER SEGMENTATION (K-MEANS)")
    print("=" * 78)
    df, source = load_data()
    print(f"Data source : {source}")
    print(f"Records     : {len(df):,}")
    print(f"Features    : {', '.join(FEATURES)}")

    # Scale first: income runs 15-137 while spending score runs 1-99, and
    # K-Means would otherwise let the larger-range feature dominate distance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    best_k, inertias, silhouettes = choose_k(X_scaled)
    print("\nCHOOSING K")
    print(f"  {'K':>3} {'inertia':>10} {'silhouette':>12}")
    for k, inertia, sil in zip(K_RANGE, inertias, silhouettes):
        marker = "  <-- best" if k == best_k else ""
        print(f"  {k:>3} {inertia:>10.1f} {sil:>12.3f}{marker}")
    print(f"\n  Selected K = {best_k} by highest silhouette score "
          f"({max(silhouettes):.3f}), cross-checked against the elbow curve.")

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # ---- profile the clusters ---------------------------------------------
    summary = df.groupby("cluster")[FEATURES].mean().round(1)
    summary["size"] = df["cluster"].value_counts().sort_index()
    summary["share"] = (summary["size"] / len(df) * 100).round(1)
    if "gender" in df.columns:
        summary["pct_female"] = (
            df.groupby("cluster")["gender"].apply(lambda s: (s == "Female").mean() * 100)
        ).round(1)

    print("\nSEGMENT PROFILES")
    print(summary.to_string())

    # Terciles of the actual customer base define "low / mid / high" bands,
    # so the labels are relative to this population rather than hard-coded.
    cuts = {f: (df[f].quantile(0.33), df[f].quantile(0.67))
            for f in ["annual_income", "spending_score"]}
    median_age = df["age"].median()

    print("\nSEGMENT PERSONAS AND TARGETED STRATEGIES")
    labels = {}
    for cluster, row in summary.iterrows():
        persona, strategy = describe(row, cuts, median_age)
        labels[cluster] = persona
        print(f"\n  Cluster {cluster} - {persona}")
        print(f"    {int(row['size'])} customers ({row['share']:.1f}% of base) | "
              f"income ${row['annual_income']:.0f}k | "
              f"spending score {row['spending_score']:.0f} | "
              f"avg age {row['age']:.0f}")
        print(f"    Strategy: {strategy}")

    df["segment"] = df["cluster"].map(labels)

    # ---- plots -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(list(K_RANGE), inertias, "bo-")
    axes[0].axvline(best_k, color="green", linestyle="--")
    axes[0].set(xlabel="Number of clusters (K)", ylabel="Inertia",
                title="Elbow method")

    axes[1].plot(list(K_RANGE), silhouettes, "mo-")
    axes[1].axvline(best_k, color="green", linestyle="--", label=f"chosen K = {best_k}")
    axes[1].set(xlabel="Number of clusters (K)", ylabel="Silhouette score",
                title="Silhouette score (higher is better)")
    axes[1].legend()

    scatter = axes[2].scatter(df["annual_income"], df["spending_score"],
                              c=df["cluster"], cmap="viridis", s=45,
                              edgecolor="white", linewidth=0.5)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    axes[2].scatter(centroids[:, 0], centroids[:, 1], marker="X", s=260,
                    c="red", edgecolor="black", label="centroids")
    axes[2].set(xlabel="Annual income (k$)", ylabel="Spending score (1-100)",
                title=f"Customer segments (K = {best_k})")
    axes[2].legend()
    plt.colorbar(scatter, ax=axes[2], label="cluster")
    plt.tight_layout()
    plt.savefig("elbow_plot.png", dpi=120)
    plt.close()
    print("\nSaved elbow, silhouette and segment plots to elbow_plot.png")

    df.to_csv("customer_segments.csv", index=False)
    print("Saved cluster assignments to customer_segments.csv")


if __name__ == "__main__":
    main()
