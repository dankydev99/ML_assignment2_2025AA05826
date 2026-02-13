import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Obesity Classification", layout="wide")

st.title("🏥 Obesity Classification Model Evaluator")

# ---------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------
st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio(
    "Select Mode",
    ["Evaluation Mode (with labels)", "Prediction Mode (no labels)"]
)

models = {
    "Logistic Regression": "best_logistic_regression_model.pkl",
    "KNN": "best_knn_model.pkl",
    "Naive Bayes": "best_gaussian_nb_model.pkl",
    "Decision Tree": "best_decision_tree_model.pkl",
    "Random Forest": "best_random_forest_model.pkl",
    "XGBoost": "best_xgboost_model.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys())
)

st.success(f"Model Selected: {selected_model_name}")

# ---------------------------------------------------
# Cached Loaders
# ---------------------------------------------------
@st.cache_resource
def load_label_encoder():
    with open("models/label_encoder.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model(model_filename):
    with open(f"models/{model_filename}", "rb") as f:
        return pickle.load(f)

label_encoder = load_label_encoder()
model = load_model(models[selected_model_name])

# ---------------------------------------------------
# Download Test Dataset
# ---------------------------------------------------
st.subheader("📥 Download Test Dataset")

with open("data/obesity_test_data.csv", "rb") as f:
    st.download_button(
        label="Download Test CSV",
        data=f,
        file_name="obesity_test_data.csv",
        mime="text/csv"
    )

# ---------------------------------------------------
# Upload Dataset
# ---------------------------------------------------
st.subheader("📤 Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    st.dataframe(df.head())

# ---------------------------------------------------
# Predict Button
# ---------------------------------------------------
if st.button("🚀 Run"):

    if df is None:
        st.error("Please upload a dataset first.")
    else:
        if mode == "Prediction Mode (no labels)":

            X = df.copy()

            y_pred = model.predict(X)
            y_labels = label_encoder.inverse_transform(y_pred)

            df_results = df.copy()
            df_results["Predicted_Class"] = y_labels

            st.subheader("🔮 Predictions")
            st.dataframe(df_results)

        else:
            # ---------------------------------------------------
            # Evaluation Mode
            # ---------------------------------------------------
            if "NObeyesdad" not in df.columns:
                st.error("Target column 'NObeyesdad' not found.")
            else:
                X = df.drop(columns=["NObeyesdad"])
                y_true_labels = df["NObeyesdad"]

                y_true = label_encoder.transform(y_true_labels)

                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)

                # ---------------------------------------------------
                # Metrics
                # ---------------------------------------------------
                accuracy = accuracy_score(y_true, y_pred)

                precision = precision_score(y_true, y_pred, average="weighted")
                recall = recall_score(y_true, y_pred, average="weighted")
                f1 = f1_score(y_true, y_pred, average="weighted")
                mcc = matthews_corrcoef(y_true, y_pred)

                auc_score = roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="weighted"
                )

                # ---------------------------------------------------
                # Tabs Layout
                # ---------------------------------------------------
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📊 Metrics", "🧩 Confusion Matrix", "📈 ROC Curves", "📄 Report"]
                )

                # ---------------------------------------------------
                # Metrics Tab
                # ---------------------------------------------------
                with tab1:
                    st.subheader("📊 Evaluation Metrics")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Accuracy", f"{accuracy:.3f}")
                    col1.metric("Precision (Weighted)", f"{precision:.3f}")
                    col1.metric("Recall (Weighted)", f"{recall:.3f}")

                    col2.metric("F1 Score (Weighted)", f"{f1:.3f}")
                    col2.metric("MCC", f"{mcc:.3f}")
                    col2.metric("AUC", f"{auc_score:.3f}")

                    # ---------------------------------------------------
                    # Probability Chart
                    # ---------------------------------------------------
                    st.subheader("🔎 Prediction Confidence")

                    sample_idx = st.number_input(
                        "Select sample index",
                        min_value=0,
                        max_value=len(X)-1,
                        value=0
                    )

                    sample_proba = y_proba[sample_idx]

                    fig, ax = plt.subplots()
                    ax.bar(label_encoder.classes_, sample_proba)
                    plt.xticks(rotation=45)
                    plt.ylabel("Probability")
                    plt.title("Class Probabilities")

                    st.pyplot(fig)

                # ---------------------------------------------------
                # Confusion Matrix Tab
                # ---------------------------------------------------
                with tab2:
                    st.subheader("🧩 Confusion Matrix")

                    cm = confusion_matrix(y_true, y_pred)

                    fig, ax = plt.subplots()
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_,
                        ax=ax
                    )

                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")

                    st.pyplot(fig)

                # ---------------------------------------------------
                # ROC Curves Tab
                # ---------------------------------------------------
                with tab3:
                    st.subheader("📈 ROC Curves (One-vs-Rest)")

                    y_true_bin = label_binarize(
                        y_true,
                        classes=np.arange(len(label_encoder.classes_))
                    )

                    fig, ax = plt.subplots()

                    for i, class_name in enumerate(label_encoder.classes_):
                        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                        roc_auc = auc(fpr, tpr)

                        ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

                    ax.plot([0, 1], [0, 1], linestyle="--")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()

                    st.pyplot(fig)

                # ---------------------------------------------------
                # Classification Report Tab
                # ---------------------------------------------------
                with tab4:
                    st.subheader("📄 Classification Report")

                    report = classification_report(
                        y_true,
                        y_pred,
                        target_names=label_encoder.classes_,
                        output_dict=True
                    )

                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)