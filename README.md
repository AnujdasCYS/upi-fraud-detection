# HELF: Hybrid Ensemble Learning Framework for UPI Fraud Detection

A single-notebook implementation of the **Hybrid Ensemble Learning Framework (HELF)** for real-time fraud detection and risk scoring in UPI transactions, based on:

> Abhilasha, Vijayakarthik & Pol, *"A Hybrid Ensemble Learning Framework for Real-Time Fraud Detection and Risk Scoring in UPI Transactions"*, IJRESM 9(8):36-42, 2026.

The idea in one line: **run four models on every transaction, blend their outputs into one risk score, then turn that score into one of three verdicts.**

---

## How it works

Every transaction is scored by four models:

| Model | Type | Role |
|---|---|---|
| Logistic Regression | Supervised | Linear baseline |
| Random Forest | Supervised | Main tree ensemble |
| XGBoost (or scikit-learn HistGradientBoosting if XGBoost is not installed) | Supervised | Gradient boosting |
| Isolation Forest | Unsupervised | Flags unusual transactions without using labels |

The outputs are fused into a single risk score `S` between 0 and 1:

```
S = 0.70 * (0.20*LR + 0.45*RF + 0.35*XGB)  +  0.30 * IsolationForest
```

`S` is then mapped to a three-tier verdict using two thresholds calibrated on the validation set:

| Verdict | Condition | Action |
|---|---|---|
| Legitimate | `S` below the approve line | Approve |
| Suspicious | `S` between the two lines | Send to a human reviewer |
| Fraudulent | `S` above the block line | Block |

- **Block line:** the highest cut that keeps false alarms on good customers at or below 4%.
- **Approve line:** set low enough that at most about 1% of validation fraud falls below it.

## Pipeline

The notebook runs in this order:

1. **Make the dataset:** generate 150,000 synthetic UPI transactions (about 8.6% fraud) with two fraud patterns (large-amount and social-engineering style), plus injected missing values, invalid rows, and 600 unreported frauds.
2. **Clean:** drop rows with non-positive amounts or impossible hours.
3. **Split:** stratified 70 / 15 / 15 train / validation / test.
4. **Prepare features:** median imputation, one-hot encoding of `txn_type`, standardisation, and a from-scratch **SMOTE** applied to the training set only.
5. **Train** the four models (Isolation Forest is trained on the original, un-balanced training data).
6. **Fuse** the scores into `S`.
7. **Calibrate thresholds** on the validation set only.
8. **Evaluate** on the test set, touched once.
9. **Save charts** and the results table.

## Features

`amount`, `txn_type`, `hour`, `day_of_week`, `txn_freq_24h`, `amount_to_avg_ratio`, `account_age_days`, `distinct_payees_7d`, `recipient_reputation`, `second_degree_links`, `device_match`, `vpa_verified`, `new_payee`, `collect_request`.

## Results (test set: 22,108 transactions, 1,810 fraud)

| Model | Acc % | Prec % | Recall % | F1 % | AUC | Missed frauds |
|---|---|---|---|---|---|---|
| Logistic Regression | 94.6 | 74.5 | 51.7 | 61.1 | 0.900 | 874 |
| Random Forest | 99.2 | 92.3 | 98.3 | 95.2 | 0.997 | 30 |
| XGBoost | 97.4 | 84.6 | 83.9 | 84.2 | 0.987 | 292 |
| Isolation Forest | 93.4 | 69.3 | 34.7 | 46.2 | 0.769 | 1182 |
| Majority Vote | 97.5 | 80.2 | 91.9 | 85.7 | 0.978 | 146 |
| **HELF** | 98.5 | 90.3 | 91.9 | 91.1 | 0.994 | 146 |

At a 3% false-alarm budget, HELF catches **97.8%** of fraud.

**Three-tier verdicts**

| Verdict | Transactions | Share | Of which fraud |
|---|---|---|---|
| Legitimate | 18,611 | 84.2% | 10 |
| Suspicious | 866 | 3.9% | 20 |
| Fraudulent | 2,631 | 11.9% | 1,780 |

Only 10 of 1,810 frauds slip through as Legitimate (99.45% caught). The trade-off is that **851 honest customers are blocked** by mistake.

**Most important features (random forest):** `second_degree_links` (0.272), `amount_to_avg_ratio` (0.171), `distinct_payees_7d` (0.133), `txn_freq_24h` (0.124), `amount` (0.103), `recipient_reputation` (0.096).

## Important notes

- **The data is synthetic.** The original paper used a private dataset, so this repo generates a stand-in with a similar shape. The scores above reflect how this generator separates fraud from normal traffic and should not be read as real-world UPI performance.
- Class balancing (SMOTE) is applied to training data only. Thresholds are tuned on validation, never on test.
- In the saved run, the Random Forest scored higher than the fused HELF model on the single-threshold comparison. HELF's advantage in this notebook is the three-tier workflow, which routes uncertain cases to humans and cuts missed fraud.
- The thresholds in the notebook's header docstring (0.48 / 0.72) are the paper's illustrative values. The code calibrates its own thresholds on the validation set (0.263 / 0.311 in the saved run).
- Some log messages in the code mention "200 trees" and "300 rounds, depth 6", but the model constructors use 100 trees (Random Forest) and 100 rounds at depth 4 (XGBoost). The constructors are what actually runs.

## Getting started

### Requirements

- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- `xgboost` (optional; falls back to scikit-learn's gradient boosting)

```bash
pip install numpy pandas scikit-learn matplotlib xgboost
```

### Run in Google Colab

1. Open `upi_project_final.ipynb` in Colab.
2. Run all cells. The first cell generates the dataset, triggers a download of `synthetic_fraud_dataset.csv`, and runs the full pipeline.

### Run locally

The notebook uses Colab-specific lines. Before running locally, edit the first cell:

- Change `csv_path = "/content/synthetic_fraud_dataset.csv"` to a local path such as `"synthetic_fraud_dataset.csv"`.
- Remove `from google.colab import files` and `files.download(...)`.

Then run the notebook with Jupyter, or export the first cell to a `.py` file and run it with `python`.

## Outputs

| File | Description |
|---|---|
| `synthetic_fraud_dataset.csv` | Generated raw dataset (150,000 rows, 15 columns) |
| `results_table.csv` | Model comparison table |
| `chart_roc.png` | ROC curves for LR, RF, XGBoost and HELF |
| `chart_risk_scores.png` | Risk score distributions with the two decision lines |
| `chart_confusion.png` | HELF confusion matrix |

## Repository structure

```
.
├── upi_project_final.ipynb   # full pipeline: data, training, evaluation, charts
└── README.md
```

## Reproducibility

All randomness is seeded with `SEED = 42`.

## Citation

If you use this work, please cite the original paper:

```
Abhilasha, Vijayakarthik, Pol. "A Hybrid Ensemble Learning Framework for Real-Time
Fraud Detection and Risk Scoring in UPI Transactions." IJRESM 9(8):36-42, 2026.
```

## License

Add a license of your choice (for example MIT) before publishing.
