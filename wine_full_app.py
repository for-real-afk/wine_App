# wine_app_full.py
"""
🍷 UCI Wine ML App (Single File)
Features:
 - Load UCI Wine dataset
 - EDA: summary, histograms, correlation, PCA
 - Preprocessing: imputation (not required but included), scaling, log-transform option
 - Feature engineering: polynomial interactions (optional)
 - Train multiple models: RandomForest, SVM, LogisticRegression, KNeighbors
 - Compare models (accuracy, confusion matrix)
 - SHAP explainability for RandomForest (TreeExplainer)
 - PCA 2D visualization
 - CSV upload for batch predictions and download results
 - Save/load models to disk (joblib)
 - Light/Dark theme toggle
All in one file. Run with: streamlit run wine_app_full.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import os
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Optional SHAP import — if not installed, app will continue but SHAP section will show instructions
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -------------------------
# App config + CSS for themes
# -------------------------
st.set_page_config(page_title="Wine Classifier (All-in-One)", layout="wide")
st.title("🍷 UCI Wine — All-in-One ML App")

# simple CSS theme toggle
def local_css_dark():
    st.markdown(
        """
        <style>
        .reportview-container, .main, header, footer {
            background-color: #0e1117;
            color: #e6eef8;
        }
        .stButton>button { background-color:#0b6e4f; color:white }
        .stSidebar { background-color: #0b1220; color: #e6eef8; }
        </style>
        """, unsafe_allow_html=True)

def local_css_light():
    st.markdown(
        """
        <style>
        .reportview-container, .main, header, footer {
            background-color: white;
            color: black;
        }
        .stButton>button { background-color:#0b6e4f; color:white }
        .stSidebar { background-color: #f5f5f5; color: black; }
        </style>
        """, unsafe_allow_html=True)

theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
if theme == "Dark":
    local_css_dark()
else:
    local_css_light()

st.sidebar.markdown("---")

# -------------------------
# Data loading and caching
# -------------------------
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    return df, feature_names, target_names

df, FEATURE_NAMES, TARGET_NAMES = load_data()

# Add optional synthetic categorical "region" for demo of encoding (keeps reproducible)
@st.cache_data
def add_demo_region(dataframe):
    df2 = dataframe.copy()
    # deterministic assignment
    df2["region"] = np.where(np.arange(len(df2)) % 2 == 0, "France", "Italy")
    return df2

df = add_demo_region(df)  # includes 'region' column now

# -------------------------
# Sidebar inputs - preprocessing options
# -------------------------
st.sidebar.header("Preprocessing & Options")
use_log = st.sidebar.checkbox("Apply log(1+x) transform to numeric features", value=False)
use_polynomial = st.sidebar.checkbox("Add polynomial & interaction features (degree=2)", value=False)
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
train_button = st.sidebar.button("Train / Retrain Models")

st.sidebar.markdown("---")
st.sidebar.header("Model Save / Load")
model_save_name = st.sidebar.text_input("Model filename (joblib)", value="wine_models.joblib")
if st.sidebar.button("Save Models"):
    # will be handled after training; placeholder
    st.sidebar.write("Will save after training step.")
st.sidebar.markdown("---")
st.sidebar.header("SHAP")
if not SHAP_AVAILABLE:
    st.sidebar.warning("shap not installed. Install with `pip install shap` to enable SHAP visuals.")
st.sidebar.markdown("---")
st.sidebar.header("Upload CSV for Batch Predictions")
uploaded_file = st.sidebar.file_uploader("Upload CSV (features only or with header)", type=["csv"])

# -------------------------
# EDA Panel
# -------------------------
with st.expander("🔎 Exploratory Data Analysis (EDA)"):
    st.subheader("Dataset Head & Summary")
    st.dataframe(df.head(10))

    st.markdown("**Basic stats**")
    st.write(df.describe().T)

    st.markdown("**Class Distribution**")
    st.bar_chart(df["target"].value_counts().sort_index())

    st.markdown("**Feature histograms**")
    # show histograms in grid
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_NAMES):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(col)
    # empty last axes if any
    for j in range(len(FEATURE_NAMES), len(axes)):
        axes[j].axis("off")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Correlation heatmap**")
    fig2 = plt.figure(figsize=(10,8))
    sns.heatmap(df[FEATURE_NAMES].corr(), cmap="coolwarm", center=0)
    st.pyplot(fig2)
    plt.close(fig2)

# -------------------------
# Preprocessing pipeline builder
# -------------------------
def build_preprocessor(feature_names, use_log=False, use_poly=False):
    numeric_cols = list(feature_names)
    categorical_cols = ["region"]

    numeric_steps = []
    numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    if use_log:
        numeric_steps.append(("log", FunctionTransformer(lambda x: np.log1p(x))))
    if use_poly:
        numeric_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor

# -------------------------
# Model training function (returns dict of trained models + metadata)
# -------------------------
@st.cache_resource
def train_models(df_local, feature_names, test_size=0.2, random_state=42, use_log=False, use_poly=False):
    # Prepare X, y
    X = df_local[feature_names + ["region"]]
    y = df_local["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(feature_names, use_log=use_log, use_poly=use_poly)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=3000, random_state=random_state),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5)
    }

    trained = {}
    performance = {}

    for name, mdl in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", mdl)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        trained[name] = pipe
        performance[name] = {
            "accuracy": acc,
            "classification_report": classification_report(y_test, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, preds)
        }

    # For convenience, return test split as well
    return {
        "models": trained,
        "performance": performance,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": preprocessor
    }

# Initial training (cached)
if "trained_store" not in st.session_state:
    st.session_state.trained_store = train_models(df, list(FEATURE_NAMES), test_size=test_size, random_state=random_state, use_log=use_log, use_poly=use_polynomial)

# Retrain if user clicks the train button or changed options
if train_button:
    st.info("Retraining models with current settings...")
    st.session_state.trained_store = train_models(df, list(FEATURE_NAMES), test_size=test_size, random_state=random_state, use_log=use_log, use_poly=use_polynomial)
    st.success("Retraining completed.")

store = st.session_state.trained_store
models = store["models"]
perf = store["performance"]
X_test = store["X_test"]
y_test = store["y_test"]

# -------------------------
# Model Comparison Panel
# -------------------------
with st.expander("🧾 Model Comparison & Metrics", expanded=True):
    st.write("### Accuracy Summary")
    acc_df = pd.DataFrame({k: {"accuracy": v["accuracy"]} for k, v in perf.items()}).T
    st.dataframe(acc_df.style.format("{:.4f}"))
    st.bar_chart(acc_df["accuracy"])

    st.write("### Confusion Matrices")
    for name, v in perf.items():
        st.write(f"**{name}** — accuracy: {v['accuracy']:.4f}")
        cm = v["confusion_matrix"]
        fig_cm = plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    st.write("### Example Classification Reports (RandomForest)")
    st.json(perf["RandomForest"]["classification_report"])

# -------------------------
# PCA 2D visualization
# -------------------------
with st.expander("📐 PCA (2D) Visualization"):
    st.write("Projecting features onto 2 principal components (for visualization).")
    # preprocess full feature matrix using preprocessor (fitted inside RandomForest pipeline)
    rf_pipe = models["RandomForest"]
    preproc = rf_pipe.named_steps["preprocess"]
    X_all = df[list(FEATURE_NAMES) + ["region"]]
    X_trans = preproc.transform(X_all)
    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    pcs = pca.fit_transform(X_trans)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pca_df["target"] = df["target"]
    fig_pca = plt.figure(figsize=(8,6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="target", palette="tab10")
    st.pyplot(fig_pca)
    plt.close(fig_pca)

# -------------------------
# SHAP explainability for RandomForest
# -------------------------
with st.expander("🔍 SHAP Explainability (RandomForest)"):
    st.write("SHAP values (global & local) for RandomForest model. SHAP must be installed.")
    if not SHAP_AVAILABLE:
        st.warning("SHAP library not installed. Install via `pip install shap` and refresh to enable SHAP plots.")
        st.info("Note: SHAP for non-tree models can be approximated but is slower. This app computes SHAP for RandomForest (TreeExplainer).")
    else:
        rf_pipe = models["RandomForest"]
        preproc = rf_pipe.named_steps["preprocess"]
        rf_model = rf_pipe.named_steps["model"]

        # Get a sample of training data for SHAP background (use X_test to avoid heavy computations)
        X_for_shap = preproc.transform(X_test)
        # If too big, subsample
        if X_for_shap.shape[0] > 200:
            idxs = np.random.choice(np.arange(X_for_shap.shape[0]), 200, replace=False)
            shap_background = X_for_shap[idxs]
        else:
            shap_background = X_for_shap

        # TreeExplainer for forest
        explainer = shap.TreeExplainer(rf_model, data=shap_background, feature_perturbation="tree_path_dependent")

        # compute shap values for X_test (small set)
        shap_values = explainer.shap_values(X_for_shap)

        st.write("### SHAP Summary Plot (feature importance)")
        # shap.summary_plot outputs matplotlib figure if show=False? We'll capture with pyplot
        fig_shap = plt.figure(figsize=(8,6))
        try:
            shap.summary_plot(shap_values, X_for_shap, feature_names=(preproc.transformers_[0][2] + list(preproc.transformers_[1][1].named_steps["onehot"].get_feature_names_out(["region"]))), show=False)
            st.pyplot(fig_shap)
        except Exception as e:
            st.write("SHAP summary_plot error (falling back to simple bar importance).")
            importances = rf_model.feature_importances_
            # we don't have direct names if polynomial used; just show numeric feature importances
            imp_df = pd.DataFrame({"feature_index": np.arange(len(importances)), "importance": importances}).sort_values("importance", ascending=False).head(20)
            st.bar_chart(imp_df.set_index("feature_index"))

        plt.close(fig_shap)

        st.write("### SHAP values for a single sample (select index)")
        idx = st.number_input("Select sample index from test set for local SHAP", min_value=0, max_value=len(X_test)-1, value=0, step=1)
        sample = X_test.iloc[idx:idx+1]
        sample_trans = preproc.transform(sample)
        sv = explainer.shap_values(sample_trans)
        try:
            fig_force = shap.force_plot(explainer.expected_value[0], sv[0], sample_trans, feature_names=(preproc.transformers_[0][2] + list(preproc.transformers_[1][1].named_steps["onehot"].get_feature_names_out(["region"])) ), matplotlib=True, show=False)
            st.pyplot(fig_force)
            plt.close()
        except Exception as e:
            st.write("Unable to render force plot as matplotlib. Showing numeric SHAP values instead.")
            st.write(pd.DataFrame(sv[0], columns=["shap_value"]))

# -------------------------
# Single-sample prediction UI
# -------------------------
st.markdown("---")
st.header("🔮 Single Sample Prediction")

with st.form("single_predict"):
    st.write("Set features in sidebar values, or use random sample button below to load a dataset sample.")
    if st.button("Load random sample into sidebar"):
        # populate session state for sidebar defaults (not persistent across reruns in this simple approach)
        sample_idx = np.random.randint(low=0, high=len(df))
        sample_row = df.iloc[sample_idx]
        # Provide feedback to user
        st.info(f"Loaded row {sample_idx} (actual class: {TARGET_NAMES[int(sample_row['target'])]}) — now click Predict below.")
        # Show sample
        st.write(sample_row[ list(FEATURE_NAMES) + ["region", "target"] ])

    st.write("Click Predict to use the sidebar 'Manual Input' area below to enter feature values.")
    # We will allow manual sidebar inputs; gather them now:
    st.sidebar.markdown("### Manual Input for Single Prediction")
    manual_inputs = {}
    for feature in FEATURE_NAMES:
        # show defaults as mean
        manual_inputs[feature] = st.sidebar.number_input(f"{feature}", value=float(df[feature].mean()))
    manual_inputs["region"] = st.sidebar.selectbox("region", options=df["region"].unique())

    submitted = st.form_submit_button("Predict")
    if submitted:
        # build a single-row dataframe
        single_df = pd.DataFrame([manual_inputs])
        # Use the best model choice
        model_choice = st.selectbox("Choose model for prediction", options=list(models.keys()), index=0)
        pipe = models[model_choice]
        pred = pipe.predict(single_df)[0]
        proba = pipe.predict_proba(single_df)[0]
        st.success(f"Predicted class: **{TARGET_NAMES[int(pred)]}**")
        st.write("Probabilities:")
        prob_df = pd.DataFrame({"class": TARGET_NAMES, "probability": proba})
        st.table(prob_df)

# -------------------------
# Batch prediction via CSV upload
# -------------------------
st.markdown("---")
st.header("📥 Batch Predictions (CSV Upload)")

if uploaded_file:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.write("Uploaded file preview:")
        st.dataframe(uploaded_df.head())

        # Check columns. If 'target' present, drop it for predictions
        if "target" in uploaded_df.columns:
            uploaded_df = uploaded_df.drop(columns=["target"])

        # If the uploaded data lacks 'region', add default
        if "region" not in uploaded_df.columns:
            uploaded_df["region"] = "France"

        # Ensure columns are present
        missing_cols = [c for c in list(FEATURE_NAMES) + ["region"] if c not in uploaded_df.columns]
        if missing_cols:
            st.error(f"Missing required columns for prediction: {missing_cols}")
        else:
            model_choice_batch = st.selectbox("Choose model for batch prediction", options=list(models.keys()), key="batch_model_choice")
            pipe = models[model_choice_batch]
            preds = pipe.predict(uploaded_df[list(FEATURE_NAMES) + ["region"]])
            probs = pipe.predict_proba(uploaded_df[list(FEATURE_NAMES) + ["region"]])
            out_df = uploaded_df.copy()
            out_df["predicted_class"] = [TARGET_NAMES[int(p)] for p in preds]
            # add probability columns
            for i, cls in enumerate(TARGET_NAMES):
                out_df[f"prob_{cls}"] = probs[:, i]
            st.write("Predictions:")
            st.dataframe(out_df.head())

            # download
            csv_buffer = io.StringIO()
            out_df.to_csv(csv_buffer, index=False)
            b = csv_buffer.getvalue().encode()
            st.download_button("Download predictions CSV", data=b, file_name="wine_predictions.csv")
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")

else:
    st.info("Upload a CSV file to run batch predictions. File must contain numeric feature columns matching the dataset feature names, and optionally 'region' (default 'France').")

# -------------------------
# Save / Load models to disk
# -------------------------
st.markdown("---")
st.header("💾 Save / Load Trained Models")

cols = st.columns(2)
with cols[0]:
    if st.button("Save current models to disk"):
        try:
            save_obj = {
                "models": models,
                "feature_names": list(FEATURE_NAMES),
                "target_names": list(TARGET_NAMES)
            }
            joblib.dump(save_obj, model_save_name)
            st.success(f"Saved models to {model_save_name}")
        except Exception as e:
            st.error(f"Failed to save models: {e}")

with cols[1]:
    upload_model_file = st.file_uploader("Load models (joblib)", type=["joblib", "pkl"], key="load_model_uploader")
    if upload_model_file:
        try:
            loaded_obj = joblib.load(upload_model_file)
            # Basic sanity check
            if "models" in loaded_obj:
                st.session_state.trained_store["models"] = loaded_obj["models"]
                st.success("Loaded models into session.")
            else:
                st.error("Uploaded joblib does not contain expected structure.")
        except Exception as e:
            st.error(f"Failed to load model file: {e}")

# -------------------------
# Footer / notes
# -------------------------
st.markdown("---")
st.write("### Notes & Tips")
st.write("""
- This app intentionally trains models on the UCI Wine dataset on demand. Training is done in-memory and results are cached for speed.
- SHAP visuals require the `shap` package. If SHAP is not installed, the app will display a warning and skip SHAP plots.
- The synthetic 'region' column is included as a demonstration of categorical handling; production data may have different categorical fields.
- Polynomial feature generation can blow up dimensionality quickly — use only if you need it.
""")
