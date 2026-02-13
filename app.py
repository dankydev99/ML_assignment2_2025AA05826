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
    roc_auc_score
)

st.set_page_config(page_title="Obesity Classification", layout="wide")

st.title("🏥 Obesity Classification Model Evaluator")

# ---------------------------------------------------
# Load Label Encoder
# ---------------------------------------------------
@st.cache_resource
def load_label_encoder():
    with open("models/label_encoder.pkl", "rb") as f:
        return pickle.load(f)

label_encoder = load_label_encoder()

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
@st.cache_resource
def load_model(model_name):
    with open(f"models/{model_name}", "rb") as f:
        return pickle.load(f)

models = {
    "Logistic Regression": "best_logistic_regression_model.pkl",
    "KNN": "best_knn_model.pkl",
    "Naive Bayes": "best_gaussian_nb_model.pkl",
    "Decision Tree": "best_decision_tree_model.pkl",
    "Random Forest": "best_random_forest_model.pkl",
    "XGBoost": "best_xgboost_model.pkl"
}

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
st.subheader("📤 Upload Test Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    st.dataframe(df.head())

# ---------------------------------------------------
# Model Selection
# ---------------------------------------------------
st.subheader("🤖 Select Model")

selected_model_name = st.selectbox("Choose a model", list(models.keys()))

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------
if st.button("🚀 Predict"):

    if df is None:
        st.error("Please upload a dataset first.")
    else:
        model = load_model(models[selected_model_name])

        # ---------------------------------------------------
        # Separate Features & Target
        # ---------------------------------------------------
        if "NObeyesdad" not in df.columns:
            st.error("Target column 'NObeyesdad' not found.")
        else:
            X = df.drop(columns=["NObeyesdad"])
            y_true_labels = df["NObeyesdad"]

            # Encode true labels
            y_true = label_encoder.transform(y_true_labels)

            # ---------------------------------------------------
            # Predictions
            # ---------------------------------------------------
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)

            # ---------------------------------------------------
            # Metrics
            # ---------------------------------------------------
            accuracy = accuracy_score(y_true, y_pred)

            precision = precision_score(
                y_true, y_pred, average="weighted"
            )

            recall = recall_score(
                y_true, y_pred, average="weighted"
            )

            f1 = f1_score(
                y_true, y_pred, average="weighted"
            )

            mcc = matthews_corrcoef(y_true, y_pred)

            auc = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted"
            )

            # ---------------------------------------------------
            # Display Metrics
            # ---------------------------------------------------
            st.subheader("📊 Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.3f}")
            col1.metric("Precision (Weighted)", f"{precision:.3f}")
            col1.metric("Recall (Weighted)", f"{recall:.3f}")

            col2.metric("F1 Score (Weighted)", f"{f1:.3f}")
            col2.metric("MCC", f"{mcc:.3f}")
            col2.metric("AUC", f"{auc:.3f}")

            # ---------------------------------------------------
            # Confusion Matrix
            # ---------------------------------------------------
            st.subheader("🧩 Confusion Matrix")

            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=label_encoder.classes_,
                columns=label_encoder.classes_
            )

            st.dataframe(cm_df)

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            st.pyplot(plt.gcf())

            # ---------------------------------------------------
            # Classification Report
            # ---------------------------------------------------
            st.subheader("📄 Classification Report")

            report = classification_report(
                y_true,
                y_pred,
                target_names=label_encoder.classes_,
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)