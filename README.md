# Data Mining Clustering Project

This repository contains a reproducible credit-card customer segmentation exercise that satisfies the assignment requirements in the issue description.

## Repository contents

- `/tmp/workspace/Faus7679/data-mining-clustering/credit_card_analysis.py` - end-to-end Python workflow for loading data, preprocessing, K-means clustering, association-rule mining, and saving plots/results.
- `/tmp/workspace/Faus7679/data-mining-clustering/reports/credit_card_clustering_report.md` - 500-750 word technical report with findings, outputs, and references.
- `/tmp/workspace/Faus7679/data-mining-clustering/outputs/` - generated charts and CSV summaries used in the report.
- `/tmp/workspace/Faus7679/data-mining-clustering/requirements.txt` - Python dependencies for the analysis.

## How to run the project

1. Install dependencies:

   ```bash
   python -m pip install -r /tmp/workspace/Faus7679/data-mining-clustering/requirements.txt
   ```

2. Run the analysis with a local dataset:

   ```bash
   python /tmp/workspace/Faus7679/data-mining-clustering/credit_card_analysis.py \
     --dataset /absolute/path/to/CreditCardDataSetforClustering.csv
   ```

3. If you omit `--dataset`, the script searches common local paths (including `~/Downloads`) and then downloads a public copy of the credit-card clustering dataset to `/tmp/workspace/Faus7679/data-mining-clustering/data/CC_GENERAL.csv`.

4. Review the generated deliverables:

   - `/tmp/workspace/Faus7679/data-mining-clustering/outputs/cluster_selection_metrics.csv`
   - `/tmp/workspace/Faus7679/data-mining-clustering/outputs/cluster_summary.csv`
   - `/tmp/workspace/Faus7679/data-mining-clustering/outputs/association_rules.csv`
   - `/tmp/workspace/Faus7679/data-mining-clustering/outputs/*.png`

## Explanation of each step

1. **Load packages** - imports pandas, scikit-learn, matplotlib, seaborn, and mlxtend to perform preprocessing, clustering, plotting, and association mining.
2. **Load data** - reads the credit-card dataset from a user-supplied file, common local locations, or a public fallback URL.
3. **Preprocess** - removes the customer identifier, converts fields to numeric values, imputes missing values with medians, and standardizes features so large-dollar columns do not dominate the model.
4. **Build models** - evaluates K-means for several values of `k`, selects the best silhouette score, and transforms behavioral measures into categorical transactions for Apriori association rules.
5. **Run predictions** - assigns each customer to a cluster and generates association rules that summarize recurring behavioral combinations.
6. **Display results** - writes summary tables, model metrics, and PNG charts to the `outputs` directory.
7. **Interpret and adjust** - the report explains how the selected `k` and rule thresholds affect the final segmentation and what the resulting customer groups mean.

## Notes

- No pre-existing test or lint infrastructure was present in the repository, so validation is performed by running the analysis script and checking that the outputs are generated successfully.
- The association-rules section is included because it was explicitly requested in the assignment, even though association-rule mining is a pattern-discovery technique rather than a clustering algorithm.
