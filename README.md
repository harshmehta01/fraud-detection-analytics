# Fintech Transaction Fraud Detection

An end-to-end fraud detection pipeline — from raw transaction data to a business-facing evaluation of what the model actually catches. Built as a portfolio project to demonstrate SQL and applied ML skills on a realistic, severely imbalanced fintech dataset.

## Problem

Fraud teams need to catch fraudulent transactions in near real-time without overwhelming reviewers with false alarms. The core challenge is severe class imbalance — fraud is rare, so a naive model can look "accurate" while catching almost nothing. This project addresses that properly and translates the result into a business-facing decision: which model to ship, and why.

## Dataset

[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions made by European cardholders over two days, with 492 (0.17%) labeled fraudulent. Features `V1`–`V28` are PCA-anonymized for privacy; `Time` and `Amount` are provided as-is.

`merchant_category` and `location` fields were synthetically generated (see `python/fraud_detection_model.py`, Step 1) to simulate a realistic business schema, since the source data is anonymized and doesn't include these dimensions. This is documented transparently rather than presented as real merchant/location data.

## Approach

**1. SQL (PostgreSQL)** — analyzed transaction volume and fraud rate by hour, merchant category, and location; flagged statistical outliers on transaction amount using z-score and IQR methods with window functions; built a rolling 7-day fraud rate trend.

**2. Python (pandas, scikit-learn, imbalanced-learn)** — addressed the 0.17% fraud class imbalance using SMOTE oversampling and class-weighted logistic regression; trained and compared Logistic Regression and Random Forest classifiers; evaluated using precision-recall curves, PR-AUC, and confusion matrices rather than accuracy, since accuracy is meaningless on this imbalanced a dataset.

**3. Model evaluation (matplotlib)** — visualized the precision-recall trade-off for both models side by side and plotted Random Forest's feature importances, making the model-selection decision (and the reasoning behind it) visually verifiable rather than just a table of numbers.

**4. Fraud Shield app (Streamlit)** — a small interactive app that loads the trained model and gives a live verdict on a transaction: flagged and blocked, or delivered normally — the same behavior a real fraud-screening layer would perform automatically, without the customer ever seeing the fraudulent transaction.

## Fraud Shield App

Instead of a BI dashboard, this project includes a working demo app: pick a real or randomly generated transaction, and the model scores it live, showing a clear "safe" or "flagged" verdict instead of a raw probability number.

```bash
pip install -r app/requirements.txt
streamlit run app/fraud_shield_app.py
```

### Future Work

The current app runs locally as a demo, but the same scoring logic is designed to generalize to a real deployment:

- **Email/SMS gateway integration** — score incoming payment notification messages before they ever reach the customer's inbox, silently routing flagged ones to a spam/review folder instead of alerting the customer to a transaction that never should have gone through.
- **Card network / payment processor API** — expose the model as a lightweight REST API (e.g. FastAPI) that a bank's transaction-authorization system calls in real time, before the transaction is even approved, rather than after the fact.
- **Mobile banking app integration** — surface the same verdict as a background check inside a banking app, so flagged transactions are held for review before the customer's balance is ever affected.
- **Any device, one interface** — because the scoring logic is a plain model + threshold, it isn't tied to Streamlit at all; the same `fraud_model.joblib` file could sit behind a REST endpoint, a browser extension, or an IoT payment terminal with no retraining needed.

## Results

Evaluated on a held-out, stratified test set of 56,962 transactions (98 fraud, 56,864 legitimate):

| Metric | Logistic Regression (class-weighted) | Random Forest (SMOTE) |
|---|---|---|
| PR-AUC | 0.7229 | **0.8386** |
| Precision | 5.60% | **60.56%** |
| Recall | 90.82% | 87.76% |
| ROC-AUC | 0.9718 | **0.9889** |
| False Positives | 1,501 | **56** |
| False Negatives | 9 | 12 |

**Random Forest was selected as the shipped model.** Logistic Regression edges out slightly higher recall (90.8% vs 87.8%), but at a precision of just 5.6% — meaning 94 out of every 100 flagged transactions would be false alarms, an unworkable volume for a fraud review team. Random Forest catches 87.8% of fraud while keeping false positives to 56 (vs 1,501), a **27x reduction in false alarms** for a 3-point drop in recall — a much better trade-off for a real review queue.

Of the 98 fraudulent transactions in the test set, Random Forest correctly flagged 86 (missing 12), while generating only 56 false alarms out of 56,864 legitimate transactions (a 0.098% false positive rate). At an average fraud transaction value of €108.62, this translates to an estimated **€9,341.32 in fraud losses caught** on the test set, against **€1,303.44 in fraud losses missed**. See `PROJECT_GUIDE.md` for the full business impact narrative.

## Model Evaluation Charts

Generated by `fraud_detection_model.py` and saved to `outputs/`:

- `pr_curve_Logistic_Regression_(class_weight).png` — precision-recall curve, Logistic Regression baseline
- `pr_curve_Random_Forest_(SMOTE).png` — precision-recall curve, Random Forest (shipped model)
- `feature_importance.png` — top 10 most predictive features driving the Random Forest's decisions

![Random Forest precision-recall curve](outputs/pr_curve_Random_Forest_(SMOTE).png)
![Feature importance](outputs/feature_importance.png)

## Tools

Python (pandas, scikit-learn, imbalanced-learn, matplotlib) · PostgreSQL · SQL (window functions, CTEs, statistical outlier detection)

## Repo Structure

```
fraud-detection-analytics/
├── README.md
├── sql/
│   └── fraud_analysis_queries.sql       # volume, outlier detection, rolling fraud rate
├── python/
│   └── fraud_detection_model.py         # data prep, SMOTE, model training, evaluation, model export
├── app/
│   ├── fraud_shield_app.py              # Streamlit app: live fraud verdict on a transaction
│   └── requirements.txt
└── outputs/
    ├── scored_transactions.csv          # test set with model predictions attached
    ├── fraud_model.joblib                # saved model the app loads
    ├── pr_curve_Logistic_Regression_(class_weight).png
    ├── pr_curve_Random_Forest_(SMOTE).png
    └── feature_importance.png
```

## How to Run

1. Download `creditcard.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into the `python/` folder.
2. Install dependencies: `pip install pandas scikit-learn imbalanced-learn matplotlib joblib`
3. Run `python fraud_detection_model.py` — outputs `scored_transactions.csv`, precision-recall curve plots, a feature importance chart, and the saved model, all into `outputs/`.
4. Launch the app: `pip install -r app/requirements.txt && streamlit run app/fraud_shield_app.py`
5. (Optional, for the SQL portion) Load `scored_transactions.csv` into a PostgreSQL table named `transactions` and run the queries in `sql/fraud_analysis_queries.sql` — see `SQL_STEPS.md` for exact steps.

## Author

Harsh Mehta — [LinkedIn] · [GitHub]
