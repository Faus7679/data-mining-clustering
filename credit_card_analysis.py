from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DATASET_URL = (
    "https://raw.githubusercontent.com/leonswl/credit-card-clustering/main/data/CC_GENERAL.csv"
)
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run K-means clustering and association rules on the credit-card dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional absolute path to CreditCardDataSetforClustering.csv or CC_GENERAL.csv.",
    )
    return parser.parse_args()


def locate_dataset(dataset_arg: Path | None) -> Path:
    candidates = []
    if dataset_arg:
        candidates.append(dataset_arg.expanduser())

    candidates.extend(
        [
            Path.home() / "Downloads" / "CreditCardDataSetforClustering.csv",
            Path.home() / "Downloads" / "CC_GENERAL.csv",
            ROOT / "data" / "CC_GENERAL.csv",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    downloaded_path = data_dir / "CC_GENERAL.csv"
    urlretrieve(DATASET_URL, downloaded_path)
    return downloaded_path


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_df = df.drop(columns=["CUST_ID"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    missing_counts = numeric_df.isna().sum()
    clean_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    return clean_df, missing_counts[missing_counts > 0].sort_values(ascending=False).to_frame("missing_values")


def choose_cluster_count(scaled_features) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled_features)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled_features, labels)),
            }
        )

    metrics = pd.DataFrame(rows)
    best_k = int(metrics.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    return best_k, metrics


def build_kmeans_outputs(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clean_df)
    best_k, metrics = choose_cluster_count(scaled_features)

    model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    cluster_labels = model.fit_predict(scaled_features)

    clustered_df = clean_df.copy()
    clustered_df["cluster"] = cluster_labels

    summary_columns = [
        "BALANCE",
        "PURCHASES",
        "CASH_ADVANCE",
        "CREDIT_LIMIT",
        "PAYMENTS",
        "PRC_FULL_PAYMENT",
    ]
    cluster_summary = clustered_df.groupby("cluster")[summary_columns].mean().round(2)
    cluster_summary.insert(0, "size", clustered_df["cluster"].value_counts().sort_index())
    return clustered_df, metrics, cluster_summary


def build_association_rules(clean_df: pd.DataFrame) -> pd.DataFrame:
    behavior_columns = [
        "BALANCE",
        "PURCHASES",
        "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES",
        "CASH_ADVANCE",
        "PAYMENTS",
        "PRC_FULL_PAYMENT",
    ]

    encoded = pd.DataFrame(index=clean_df.index)
    labels = ["low", "medium", "high"]

    # Rank before qcut so tied values still split into equal-frequency bins.
    for column in behavior_columns:
        binned = pd.qcut(clean_df[column].rank(method="first"), q=3, labels=labels)
        for label in labels:
            encoded[f"{column}_{label}"] = binned.eq(label)

    frequent_itemsets = apriori(encoded, min_support=0.15, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.70)
    rules = rules[(rules["lift"] > 1.0) & (rules["consequents"].apply(len) == 1)].copy()

    readable_rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    readable_rules["antecedents"] = readable_rules["antecedents"].apply(lambda items: ", ".join(sorted(items)))
    readable_rules["consequents"] = readable_rules["consequents"].apply(lambda items: ", ".join(sorted(items)))
    return readable_rules.sort_values(["lift", "confidence", "support"], ascending=False).head(10)


def save_cluster_selection_plot(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=metrics, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("Elbow curve")
    axes[0].set_xlabel("Clusters (k)")
    axes[0].set_ylabel("Inertia")

    sns.lineplot(data=metrics, x="k", y="silhouette", marker="o", ax=axes[1], color="darkgreen")
    axes[1].set_title("Silhouette score by k")
    axes[1].set_xlabel("Clusters (k)")
    axes[1].set_ylabel("Silhouette")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_selection.png", dpi=200)
    plt.close(fig)


def save_cluster_pca_plot(clustered_df: pd.DataFrame) -> None:
    features_only = clustered_df.drop(columns=["cluster"])
    scaled_features = StandardScaler().fit_transform(features_only)
    components = PCA(n_components=2, random_state=42).fit_transform(scaled_features)
    plot_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    plot_df["cluster"] = clustered_df["cluster"].astype(str)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster", palette="Set2", s=35, ax=ax)
    ax.set_title("K-means clusters projected with PCA")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_pca.png", dpi=200)
    plt.close(fig)


def save_cluster_heatmap(cluster_summary: pd.DataFrame) -> None:
    plot_df = cluster_summary.drop(columns=["size"])
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Average cluster behavior")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_summary_heatmap.png", dpi=200)
    plt.close(fig)


def save_rules_plot(rules_df: pd.DataFrame) -> None:
    plot_df = rules_df.iloc[:6].copy()
    plot_df["rule"] = plot_df["antecedents"] + " → " + plot_df["consequents"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="lift",
        y="rule",
        hue="rule",
        palette="Blues_r",
        legend=False,
        ax=ax,
    )
    ax.set_title("Top association rules by lift")
    ax.set_xlabel("Lift")
    ax.set_ylabel("Rule")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "association_rules.png", dpi=200)
    plt.close(fig)


def write_summary(
    dataset_path: Path,
    missing_df: pd.DataFrame,
    metrics: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    rules_df: pd.DataFrame,
) -> str:
    best_row = metrics.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    lines = [
        f"Dataset: {dataset_path}",
        f"Selected k: {int(best_row['k'])}",
        f"Best silhouette: {best_row['silhouette']:.4f}",
        "",
        "Missing values handled with median imputation:",
        missing_df.to_string() if not missing_df.empty else "No missing values detected.",
        "",
        "Cluster summary:",
        cluster_summary.to_string(),
        "",
        "Top association rules:",
        rules_df.to_string(index=False),
    ]
    summary = "\n".join(lines)
    (OUTPUT_DIR / "analysis_summary.txt").write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    args = parse_args()
    dataset_path = locate_dataset(args.dataset)
    raw_df = pd.read_csv(dataset_path)

    clean_df, missing_df = preprocess(raw_df)
    clustered_df, metrics_df, cluster_summary_df = build_kmeans_outputs(clean_df)
    rules_df = build_association_rules(clean_df)

    metrics_df.to_csv(OUTPUT_DIR / "cluster_selection_metrics.csv", index=False)
    cluster_summary_df.to_csv(OUTPUT_DIR / "cluster_summary.csv")
    rules_df.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)

    save_cluster_selection_plot(metrics_df)
    save_cluster_pca_plot(clustered_df)
    save_cluster_heatmap(cluster_summary_df)
    save_rules_plot(rules_df)

    summary = write_summary(dataset_path, missing_df, metrics_df, cluster_summary_df, rules_df)
    print(summary)


if __name__ == "__main__":
    main()
