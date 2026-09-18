"""
Fintech Transaction Fraud Detection
------------------------------------
End-to-end pipeline: load data -> enrich with business dimensions ->
handle class imbalance -> train models -> evaluate properly (not just
accuracy) -> save the model -> export scored data for SQL/analysis.

Dataset: Kaggle "Credit Card Fraud Detection"
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
284,807 transactions | 492 frauds (0.172%) | features V1-V28 are PCA-
anonymized for privacy, plus Time (seconds since first txn) and Amount.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# Pointing every generated file (charts, scored CSV, saved model) at one
# clean outputs/ folder so the repo structure stays tidy.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# STEP 1 — LOAD DATA + ADD SYNTHETIC BUSINESS DIMENSIONS
# ------------------------------------------------------------------
# Loading the raw CSV, then enriching it with merchant_category and
# location fields, since the source dataset is anonymized for privacy
# and doesn't include real business dimensions. Also converting the
# raw "seconds since first transaction" Time column into a proper
# timestamp and hour-of-day. Renaming Amount/Class to friendlier names.
df = pd.read_csv("creditcard.csv")

rng = np.random.default_rng(RANDOM_STATE)

merchant_categories = [
    "Grocery", "Electronics", "Travel", "Gaming",
    "Fashion", "Utilities", "Restaurants", "Online Services"
]
locations = [
    "Dublin", "Cork", "Galway", "Limerick", "London",
    "Manchester", "New York", "Remote/Online"
]

df["merchant_category"] = rng.choice(merchant_categories, size=len(df))
df["location"] = rng.choice(locations, size=len(df))

start = pd.Timestamp("2026-01-01")
df["txn_time"] = start + pd.to_timedelta(df["Time"], unit="s")
df["hour_of_day"] = df["txn_time"].dt.hour

df = df.rename(columns={"Amount": "amount", "Class": "is_fraud"})
df["transaction_id"] = df.index

print(df["is_fraud"].value_counts(normalize=True))
# ~99.83% legit, ~0.17% fraud — this imbalance is the whole point of Step 3.


# ------------------------------------------------------------------
# STEP 2 — TRAIN / TEST SPLIT
# ------------------------------------------------------------------
# Splitting the data 80/20, stratified so both sets keep the same
# ~0.17% fraud ratio — without stratification, the tiny test set
# could end up with almost no fraud cases to evaluate against.
feature_cols = [c for c in df.columns if c.startswith("V")] + [
    "amount", "hour_of_day"
]
X = df[feature_cols]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


# ------------------------------------------------------------------
# STEP 3 — HANDLE CLASS IMBALANCE
# ------------------------------------------------------------------
# Applying SMOTE to the training set only (never to test data, or
# synthetic patterns leak into evaluation), generating synthetic
# minority-class examples until fraud/legit are balanced 50/50 for
# training purposes.
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE:  {y_train_smote.value_counts().to_dict()}")


# ------------------------------------------------------------------
# STEP 4 — TRAIN MODELS
# ------------------------------------------------------------------
# Training two candidate models to compare: a class-weighted Logistic
# Regression as a simple, interpretable baseline, and a Random Forest
# trained on the SMOTE-resampled data as the stronger candidate.
log_reg = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
)
log_reg.fit(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train_smote, y_train_smote)


# ------------------------------------------------------------------
# STEP 5 — EVALUATE PROPERLY
# ------------------------------------------------------------------
# Evaluating both models on the untouched test set using precision,
# recall, PR-AUC, and a confusion matrix — never plain accuracy, since
# a model predicting "not fraud" every time would score 99.8% accuracy
# while catching zero fraud. Saving a precision-recall curve PNG for
# each model so the trade-off is visible, not just a table of numbers.
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
    print("PR-AUC (Average Precision):",
          round(average_precision_score(y_test, y_proba), 4))

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {name}")
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"pr_curve_{name.replace(' ', '_')}.png"), dpi=150
    )
    plt.close()

    return y_proba


evaluate(log_reg, X_test, y_test, "Logistic Regression (class_weight)")
rf_proba = evaluate(rf, X_test, y_test, "Random Forest (SMOTE)")

# Calculating the average fraud transaction amount on the test set —
# needed later for the € business-impact narrative in the README.
avg_fraud_amount = df.loc[X_test.index][y_test == 1]["amount"].mean()
print(f"Average fraud transaction amount in test set: €{avg_fraud_amount:.2f}")

# Charting the Random Forest's top 10 most influential features, so the
# model-selection story is visually verifiable, not just claimed.
importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances.sort_values(ascending=False).head(10).plot(kind="barh")
plt.title("Top 10 Feature Importances — Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()


# ------------------------------------------------------------------
# STEP 6 — CHOOSE AN OPERATING THRESHOLD AND EXPORT SCORED DATA
# ------------------------------------------------------------------
# Setting the probability cutoff above which a transaction gets
# flagged as fraud — a business decision, not the default 0.5. Tune
# this by inspecting the saved PR-curve PNGs. Exporting the fully
# scored test set to CSV for the SQL analysis step.
THRESHOLD = 0.35

df_test = X_test.copy()
df_test["transaction_id"] = df.loc[X_test.index, "transaction_id"]
df_test["txn_time"] = df.loc[X_test.index, "txn_time"]
df_test["merchant_category"] = df.loc[X_test.index, "merchant_category"]
df_test["location"] = df.loc[X_test.index, "location"]
df_test["is_fraud"] = y_test
df_test["predicted_fraud_probability"] = rf_proba
df_test["flagged_as_fraud"] = (rf_proba >= THRESHOLD).astype(int)

df_test.to_csv(os.path.join(OUTPUT_DIR, "scored_transactions.csv"), index=False)


# ------------------------------------------------------------------
# STEP 7 — SAVE THE MODEL FOR THE APP
# ------------------------------------------------------------------
# Saving the trained Random Forest, the exact feature column order it
# expects, and the chosen threshold together in one file — the
# Streamlit app (app/fraud_shield_app.py) loads this directly rather
# than retraining anything itself.
joblib.dump(
    {"model": rf, "feature_cols": feature_cols, "threshold": THRESHOLD},
    os.path.join(OUTPUT_DIR, "fraud_model.joblib"),
)

print(f"\nExported scored_transactions.csv, charts, and fraud_model.joblib to {OUTPUT_DIR}")
