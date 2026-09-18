"""
Fraud Shield — a small demo app that takes a transaction and tells you
whether it's safe or flagged, using the Random Forest trained in
python/fraud_detection_model.py.

Run with: streamlit run app/fraud_shield_app.py
"""

import os
import random

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "fraud_model.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "scored_transactions.csv")

st.set_page_config(page_title="Fraud Shield", page_icon="🛡️", layout="centered")


# ------------------------------------------------------------------
# Loading the saved model bundle (model + feature order + threshold)
# once, and caching it so it doesn't reload on every interaction.
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ------------------------------------------------------------------
# Loading a handful of real test-set rows to use as example
# transactions in the sidebar — no need to hand-type 28 PCA values.
# ------------------------------------------------------------------
@st.cache_data
def load_sample_transactions():
    df = pd.read_csv(DATA_PATH)
    fraud_examples = df[df["is_fraud"] == True].sample(3, random_state=1)
    legit_examples = df[df["is_fraud"] == False].sample(3, random_state=1)
    return pd.concat([fraud_examples, legit_examples]).reset_index(drop=True)


bundle = load_model()
model = bundle["model"]
feature_cols = bundle["feature_cols"]
threshold = bundle["threshold"]

samples = load_sample_transactions()

st.title("🛡️ Fraud Shield")
st.caption("A live demo of the fraud detection model — checks a transaction and tells you whether it's safe or should be blocked.")

st.divider()

# ------------------------------------------------------------------
# Letting the user either pick a real sample transaction from the
# test set, or generate a fully random one — either way, avoids
# making them manually enter 28 anonymized PCA values by hand.
# ------------------------------------------------------------------
st.subheader("Choose a transaction to check")

option = st.radio(
    "Source",
    ["Pick a sample transaction", "Generate a random transaction"],
    label_visibility="collapsed",
)

if option == "Pick a sample transaction":
    labels = [
        f"{row.merchant_category} — €{row.amount:.2f} — {'known fraud' if row.is_fraud else 'known legit'}"
        for row in samples.itertuples()
    ]
    choice = st.selectbox("Sample transaction", labels)
    selected_row = samples.iloc[labels.index(choice)]
    input_features = selected_row[feature_cols].values.astype(float)
    display_amount = selected_row["amount"]
    display_merchant = selected_row["merchant_category"]
    display_location = selected_row["location"]
    ground_truth = bool(selected_row["is_fraud"])
else:
    rng = np.random.default_rng()
    # Generating plausible random values: PCA features roughly follow
    # a standard normal distribution in this dataset, so sampling from
    # one gives a believable-looking (if not real) transaction.
    random_values = {c: rng.normal(0, 1.5) for c in feature_cols if c.startswith("V")}
    display_amount = round(random.uniform(5, 800), 2)
    display_merchant = random.choice(
        ["Grocery", "Electronics", "Travel", "Gaming", "Fashion", "Utilities", "Restaurants", "Online Services"]
    )
    display_location = random.choice(
        ["Dublin", "Cork", "Galway", "Limerick", "London", "Manchester", "New York", "Remote/Online"]
    )
    random_values["amount"] = display_amount
    random_values["hour_of_day"] = random.randint(0, 23)
    input_features = np.array([random_values[c] for c in feature_cols])
    ground_truth = None

st.write(f"**Merchant:** {display_merchant}  |  **Location:** {display_location}  |  **Amount:** €{display_amount:.2f}")

st.divider()

# ------------------------------------------------------------------
# Scoring the chosen transaction through the model and rendering a
# clear verdict — this is the core "app" behavior: instead of a raw
# probability number, translating it into the action a real system
# would take (deliver normally vs. flag/block).
# ------------------------------------------------------------------
if st.button("Check transaction", type="primary", use_container_width=True):
    probability = model.predict_proba(input_features.reshape(1, -1))[0][1]
    is_flagged = probability >= threshold

    if is_flagged:
        st.error(f"🚫 **Flagged as fraud** — blocked and sent to review queue")
    else:
        st.success(f"✅ **Looks safe** — transaction delivered normally")

    st.metric("Fraud probability", f"{probability:.1%}")
    st.progress(min(float(probability), 1.0))

    if ground_truth is not None:
        correct = is_flagged == ground_truth
        st.caption(
            f"{'✔️ Matches' if correct else '⚠️ Differs from'} the known label for this sample "
            f"({'fraud' if ground_truth else 'legit'})."
        )
    else:
        st.caption("This is a randomly generated transaction — no ground-truth label to compare against.")

st.divider()
st.caption(
    "Model: Random Forest (SMOTE-balanced training). "
    f"Operating threshold: {threshold} — probabilities at or above this are flagged. "
    "See the main project README for full precision/recall figures and the reasoning behind this threshold."
)
