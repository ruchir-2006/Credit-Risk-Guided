"""
Credit Risk Prediction — Streamlit app
Predicts whether a loan applicant is 'good' or 'bad' credit risk
using the German Credit dataset.

Run locally:   streamlit run app.py
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------
# CHOOSE YOUR MODEL HERE: "random_forest" or "logistic_regression"
# (Logistic Regression scored slightly higher in your notebook.)
MODEL_CHOICE = "random_forest"
# ---------------------------------------------------------------

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳")

# Human-readable label -> encoded value (must match your notebook's LabelEncoder)
SEX_MAP      = {"female": 0, "male": 1}
HOUSING_MAP  = {"free": 0, "own": 1, "rent": 2}
SAVING_MAP   = {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3}
CHECKING_MAP = {"little": 0, "moderate": 1, "rich": 2}
PURPOSE_MAP  = {
    "business": 0, "car": 1, "domestic appliances": 2, "education": 3,
    "furniture/equipment": 4, "radio/TV": 5, "repairs": 6, "vacation/others": 7,
}
JOB_MAP = {
    "0 - unskilled & non-resident": 0,
    "1 - unskilled & resident": 1,
    "2 - skilled": 2,
    "3 - highly skilled": 3,
}

# Exact feature order the model was trained on
FEATURE_ORDER = ["Age", "Sex", "Job", "Housing", "Saving accounts",
                 "Checking account", "Credit amount", "Duration", "Purpose"]


@st.cache_resource
def train_model():
    """Load data, preprocess exactly like the notebook, and train.
    Cached so it only runs once. Returns (model, test_accuracy)."""
    df = pd.read_csv("german_credit_data.csv")

    # Preprocessing — pandas-safe versions of your notebook steps
    df["Checking account"] = df["Checking account"].fillna("little")
    df = df.dropna(subset=["Saving accounts"])

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    X = df[FEATURE_ORDER]
    y = df["Risk"]  # bad=0, good=1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    if MODEL_CHOICE == "logistic_regression":
        model = LogisticRegression(random_state=42, solver="liblinear")
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


model, accuracy = train_model()

# ---------------------------- UI ----------------------------
st.title("💳 Credit Risk Predictor")
st.write(
    "Enter an applicant's details to predict whether they are a "
    "**good** or **bad** credit risk."
)

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex      = st.selectbox("Sex", list(SEX_MAP.keys()))
    job      = st.selectbox("Job", list(JOB_MAP.keys()))
    housing  = st.selectbox("Housing", list(HOUSING_MAP.keys()))
    duration = st.number_input("Duration (months)", min_value=1, max_value=100, value=12)

with col2:
    saving   = st.selectbox("Saving account", list(SAVING_MAP.keys()))
    checking = st.selectbox("Checking account", list(CHECKING_MAP.keys()))
    credit   = st.number_input("Credit amount", min_value=100, max_value=200000, value=2000)
    purpose  = st.selectbox("Purpose", list(PURPOSE_MAP.keys()))

if st.button("Predict", type="primary"):
    # Build the row in the exact trained feature order
    row = pd.DataFrame([{
        "Age": age,
        "Sex": SEX_MAP[sex],
        "Job": JOB_MAP[job],
        "Housing": HOUSING_MAP[housing],
        "Saving accounts": SAVING_MAP[saving],
        "Checking account": CHECKING_MAP[checking],
        "Credit amount": credit,
        "Duration": duration,
        "Purpose": PURPOSE_MAP[purpose],
    }])[FEATURE_ORDER]

    pred = model.predict(row)[0]

    if pred == 1:
        st.success(f"✅ Prediction: **GOOD** credit risk")
    else:
        st.error(f"⚠️ Prediction: **BAD** credit risk")

    st.caption(
        f"Model: {MODEL_CHOICE.replace('_', ' ').title()} · "
        f"This prediction is based on a model with **{accuracy*100:.2f}%** "
        f"test accuracy."
    )

st.divider()
st.caption(
    "Note: accuracy reflects overall test-set performance, not the confidence "
    "of this single prediction. Model trained on the German Credit dataset."
)
