# 🏥 Obesity Classification — Machine Learning Evaluation

## 📌 Project Overview

This project focuses on **multiclass classification** of obesity levels using machine learning models.  
The dataset contains demographic, behavioral, and physiological attributes used to predict:

**Target Variable:** `NObeyesdad`  
**Number of Classes:** 7

---

## 📊 Dataset Summary

- **Rows:** 2111  
- **Features:** 16  

### **Categorical Features**
`Gender`, `family_history_with_overweight`, `FAVC`, `CAEC`, `SMOKE`, `SCC`, `CALC`, `MTRANS`

### **Numerical Features**
`Age`, `Height`, `Weight`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`

- **Class Distribution:** Balanced  
- **Encoding Strategy:** One-Hot Encoding  
- **Scaling Strategy:**  
  - StandardScaler for linear & distance-based models  
  - No scaling for tree-based models

---

## 🤖 Models Evaluated

1. Logistic Regression (Multinomial Softmax)  
2. K-Nearest Neighbors (KNN)  
3. Gaussian Naive Bayes  
4. Decision Tree  
5. Random Forest  
6. XGBoost  

---

## 🎯 Evaluation Metrics

Models were evaluated on the **test set** using:

- Accuracy  
- Precision (Weighted)  
- Recall (Weighted)  
- F1 Score (Weighted)  
- Matthews Correlation Coefficient (MCC)  
- AUC Score (OvR, Weighted)  

---

## 📈 Test Set Performance

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|------|-----------|--------|----------|------|
| Logistic Regression | **0.931** | 0.994 | 0.931 | 0.931 | 0.931 | 0.920 |
| KNN | 0.894 | 0.966 | 0.891 | 0.894 | 0.891 | 0.876 |
| Naive Bayes | 0.532 | 0.855 | 0.577 | 0.532 | 0.498 | 0.474 |
| Decision Tree | **0.946** | 0.968 | 0.947 | 0.946 | 0.946 | 0.937 |
| Random Forest | 0.936 | 0.995 | 0.942 | 0.936 | 0.938 | 0.926 |
| **XGBoost** 🏆 | **0.962** | **0.998** | **0.965** | **0.962** | **0.963** | **0.956** |

---

## 🧠 Key Observations

### ✅ Logistic Regression
Despite being a linear classifier, Logistic Regression achieved strong performance:

- Accuracy: 93.1%  
- AUC: 0.994  

This suggests that obesity categories are **reasonably separable in the transformed feature space**.

---

### ⚠️ KNN
KNN showed lower performance compared to other models:

- Accuracy: 89.4%  

Likely due to:

- High dimensionality from One-Hot Encoding  
- Curse of dimensionality affecting distance metrics  

---

### ❌ Naive Bayes
Naive Bayes performed poorly:

- Accuracy: 53.2%  

Reasons:

- Strong violation of feature independence assumption  
- Non-Gaussian numeric feature distributions  
- Correlated physiological attributes  

---

### 🌳 Decision Tree
Decision Tree performed very well:

- Accuracy: 94.6%  
- MCC: 0.937  

Benefits:

- Captures nonlinear relationships  
- Learns feature interactions  

---

### 🌲 Random Forest
Random Forest achieved strong results:

- Accuracy: 93.6%  
- AUC: 0.995  

While slightly below Decision Tree in accuracy, it produced **more stable probability estimates**.

---

### 🚀 XGBoost 🏆
Best-performing model:

- Accuracy: **96.2%**  
- MCC: **0.956**  
- AUC: **0.998**  

Why it excels:

- Boosting reduces bias  
- Sequential error correction  
- Handles nonlinearities effectively  
- Built-in regularization  

---

## 🏆 Final Model Ranking

1️⃣ **XGBoost** 🥇  
2️⃣ Decision Tree 🥈  
3️⃣ Random Forest 🥉  
4️⃣ Logistic Regression  
5️⃣ KNN  
6️⃣ Naive Bayes  

---

## ✅ Conclusion

Among the evaluated models, **XGBoost demonstrated superior predictive performance**, achieving the highest Accuracy, F1 Score, MCC, and AUC.

This indicates:

- Strong nonlinear relationships in the dataset  
- Importance of ensemble boosting methods  

Logistic Regression’s strong results also suggest **partial linear separability**, while Naive Bayes’ poor performance highlights the impact of violated model assumptions.

---

## 🖥 Streamlit Application

A Streamlit app was developed to:

✅ Download test dataset  
✅ Upload CSV for predictions  
✅ Select trained model  
✅ View evaluation metrics  
✅ Visualize confusion matrix & ROC curves  

Run locally:

```bash
streamlit run app.py
```

---

## 📦 Dependencies

See `requirements.txt`

---

## 👨‍💻 Author

**Devashish Verma**
