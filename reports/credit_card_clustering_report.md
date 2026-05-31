# Technical Report: Credit Card Customer Segmentation

## Problem statement

This project explains three data-mining techniques and applies unsupervised learning to the credit-card clustering dataset in order to identify meaningful customer segments. The implementation uses K-means clustering for segmentation and association rules to uncover recurring behavior patterns that complement the clusters.

## Part I: Data-mining techniques

- **Clustering** groups unlabeled records by similarity. Algorithms compare features such as spending, balances, and payment behavior, then place nearby observations into the same segment. Its strength is exploratory power on unlabeled data; its weakness is that results depend on feature scaling and parameter choices such as the number of clusters.
- **Association** discovers co-occurring events, often with support, confidence, and lift. It is strong for market-basket style insights and behavioral combinations, but weak when the data is not transactional or when too many trivial rules appear.
- **Correlation analysis** measures how strongly variables move together, commonly with Pearson or Spearman coefficients. It helps detect redundant variables and relationships, but correlation does not imply causation and can miss nonlinear structure.

Real-life examples include customer segmentation in a bank database (clustering), product bundle discovery in a supermarket transaction database (association), and studying how credit limits and payments move together in a finance dataset (correlation).

## Part II: Algorithm of the solution

The Python workflow is implemented in `/tmp/workspace/Faus7679/data-mining-clustering/credit_card_analysis.py`.

```python
# Core execution flow
raw_df = pd.read_csv(dataset_path)
clean_df, missing_df = preprocess(raw_df)
clustered_df, metrics_df, cluster_summary_df = build_kmeans_outputs(clean_df)
rules_df = build_association_rules(clean_df)
```

The script first loads pandas, scikit-learn, matplotlib, seaborn, and mlxtend. It then reads `CreditCardDataSetforClustering.csv` from a supplied path or falls back to a public mirror of the same credit-card dataset. Preprocessing removes the `CUST_ID` identifier, converts fields to numeric form, imputes missing values with medians (`MINIMUM_PAYMENTS` had 313 missing values and `CREDIT_LIMIT` had 1), and standardizes all numeric columns. Standardization is essential because variables such as `BALANCE` and `PURCHASES_TRX` are on very different scales.

K-means was evaluated for `k = 2..8` using inertia and silhouette score. The best silhouette score was obtained at **k = 3 (0.2506)**, so the final model used three clusters. The model then predicted a cluster label for every customer. For visualization, the standardized data was projected to two principal components. To satisfy the assignment’s association-rule requirement, the same dataset was discretized into low/medium/high behavior bands for balance, purchases, cash advance, payments, and full-payment rate, then mined with Apriori.

## Quantitative outputs

```text
Selected k: 3
Cluster sizes: 6119, 1235, 1596

Cluster 0 -> lower-balance, lower-spend customers
Cluster 1 -> high-purchase, high-payment customers
Cluster 2 -> cash-advance-heavy revolving customers
```

| Cluster | Size | Avg Balance | Avg Purchases | Avg Cash Advance | Avg Credit Limit | Avg Payments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6119 | 799.75 | 505.53 | 330.82 | 3271.51 | 909.68 |
| 1 | 1235 | 2220.00 | 4268.52 | 458.42 | 7733.97 | 4151.28 |
| 2 | 1596 | 3989.14 | 384.53 | 3866.21 | 6675.44 | 3019.11 |

The strongest association rules showed that high one-off purchases strongly imply high total purchases, while low one-off and low installment spending imply low total purchases. These rules reinforce the cluster interpretation by exposing common combinations of behavior inside the customer base.

## Plots

Files are stored in `/tmp/workspace/Faus7679/data-mining-clustering/outputs/` and embedded below with repository-relative paths for Markdown rendering.

![Cluster selection](../outputs/cluster_selection.png)
![PCA clusters](../outputs/cluster_pca.png)
![Cluster heatmap](../outputs/cluster_summary_heatmap.png)
![Association rules](../outputs/association_rules.png)

## Analysis of findings

The results indicate three actionable customer segments. Cluster 0 contains the majority of customers and represents moderate or inactive card usage. Cluster 1 captures valuable transactors who purchase heavily and make large payments, suggesting higher engagement and stronger repayment behavior. Cluster 2 is the risk-oriented segment because balances and cash advances are both high while purchases are comparatively low and `PRC_FULL_PAYMENT` is near zero. From a business perspective, the bank could target Cluster 1 with premium rewards, Cluster 0 with activation campaigns, and Cluster 2 with repayment or risk-management interventions.

The clustering was adjusted by comparing several `k` values rather than assuming a fixed answer. Although inertia steadily improved as `k` increased, silhouette peaked at `k = 3`, which balanced separation and interpretability. Association rules were also filtered with minimum support and confidence thresholds so that only high-value, non-trivial patterns were retained.

## References

1. Han, J., Kamber, M., & Pei, J. *Data Mining: Concepts and Techniques*.
2. Pedregosa, F. et al. “Scikit-learn: Machine Learning in Python.”
3. Raschka, S. *mlxtend* documentation for Apriori and association rules.
4. Public mirror of the Credit Card Dataset for Clustering: `https://raw.githubusercontent.com/leonswl/credit-card-clustering/main/data/CC_GENERAL.csv`
