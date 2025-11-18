# ================================================================
#               🍷 UCI WINE CLASSIFIER — SINGLE FILE APP
#        DATA PREPROCESSING + MODEL TRAINING + STREAMLIT UI
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# --------------------------------------------------------------
# 1) LOAD & PREPARE DATA
# --------------------------------------------------------------
@st.cache_data
def load_dataset():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data.feature_names, data.target_names

df, feature_names, target_names = load_dataset()


# --------------------------------------------------------------
# 2) TRAIN MODEL (Automatically runs once)
# --------------------------------------------------------------
@st.cache_resource
def train_model():
    X = df[feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Accuracy
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return pipeline, acc, X_test, y_test

model, accuracy, X_test, y_test = train_model()


# --------------------------------------------------------------
# 3) STREAMLIT INTERFACE
# --------------------------------------------------------------

st.set_page_config(page_title="Wine Classifier", layout="wide")
st.title("🍷 UCI Wine Classification App")
st.write("A complete ML pipeline in one file — training, evaluation, and live predictions.")

st.subheader("📊 Model Accuracy")
st.success(f"Model Accuracy: **{accuracy*100:.2f}%**")


# --------------------------------------------------------------
# SIDEBAR INPUT FEATURES
# --------------------------------------------------------------

st.sidebar.header("🔧 Input Wine Features")

user_input = []

for feature in feature_names:
    val = st.sidebar.number_input(
        label=feature,
        min_value=0.0,
        value=float(df[feature].mean())
    )
    user_input.append(val)

input_array = np.array(user_input).reshape(1, -1)


# --------------------------------------------------------------
# 4) MAKE PREDICTION
# --------------------------------------------------------------

if st.sidebar.button("Predict"):
    pred = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]

    st.subheader("🍇 Predicted Wine Class")
    st.success(f"### {target_names[pred].title()}")

    st.subheader("📌 Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Class": target_names,
        "Probability": proba
    })

    st.bar_chart(prob_df.set_index("Class"))

    # Show detailed output
    st.write("### Raw Probability Values")
    st.write(prob_df)


# --------------------------------------------------------------
# 5) OPTIONAL — SHOW SAMPLE DATA
# --------------------------------------------------------------

with st.expander("📁 View Sample Dataset"):
    st.dataframe(df.head())


# --------------------------------------------------------------
# 6) FEATURE IMPORTANCE
# --------------------------------------------------------------

with st.expander("📈 Feature Importance (Random Forest)"):
    importances = model.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(imp_df.set_index("Feature"))


# --------------------------------------------------------------
# 7) PREDICTION ON TEST SAMPLES
# --------------------------------------------------------------

with st.expander("🔍 Test Sample Prediction Demo"):
    idx = st.number_input("Select Test Index", 0, len(X_test)-1, 0)

    sample = X_test.iloc[idx]
    true_label = y_test.iloc[idx]

    st.write("### Input Features")
    st.write(sample)

    pred = model.predict([sample])[0]

    st.write("### Prediction")
    st.info(f"Model Predicted: **{target_names[pred]}**")
    st.write(f"Actual Class: **{target_names[true_label]}**")

