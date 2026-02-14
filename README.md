# Obesity Level Classification using Machine Learning

## a. Problem Statement

The objective of this project is to build and evaluate **multiclass classification models** to predict an individual's **obesity level** based on demographic, behavioral, and physiological attributes.  
The target variable `NObeyesdad` consists of **seven distinct obesity categories**, making this a supervised multiclass classification problem.

---

## b. Dataset Description

- **Number of rows:** 2111  
- **Number of features:** 16  
- **Target variable:** `NObeyesdad`  
- **Number of classes:** 7  
- **Class distribution:** Balanced  

### **Categorical Features**
Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS

### **Numerical Features**
Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE

### **Preprocessing Strategy**
- One-Hot Encoding for categorical variables  
- StandardScaler applied only to numerical features (for linear & distance-based models)  
- No scaling for tree-based models  

---

## c. Models Used

1. Logistic Regression (Multinomial Softmax)  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## 📊 Comparison of Model Performance (Test Set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.931 | 0.994 | 0.931 | 0.931 | 0.931 | 0.920 |
| Decision Tree | 0.946 | 0.968 | 0.947 | 0.946 | 0.946 | 0.937 |
| KNN | 0.894 | 0.966 | 0.891 | 0.894 | 0.891 | 0.876 |
| Naive Bayes | 0.532 | 0.855 | 0.577 | 0.532 | 0.498 | 0.474 |
| Random Forest (Ensemble) | 0.936 | 0.995 | 0.942 | 0.936 | 0.938 | 0.926 |
| XGBoost (Ensemble) | **0.962** | **0.998** | **0.965** | **0.962** | **0.963** | **0.956** |

---

## 🧠 Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|--------------------------------------|
| **Logistic Regression** | Delivered strong performance despite being a linear classifier, suggesting that the obesity categories exhibit substantial linear separability after one-hot encoding and feature scaling. The high AUC further indicates well-ranked class probabilities. |
| **Decision Tree** | Achieved excellent accuracy by learning nonlinear decision boundaries and capturing feature interactions. Its performance improvement over Logistic Regression highlights the presence of nonlinear relationships among physiological and behavioral attributes. |
| **KNN** | Showed comparatively lower performance, likely due to the curse of dimensionality introduced by one-hot encoding. In high-dimensional sparse spaces, distance metrics become less discriminative, reducing classification effectiveness. |
| **Naive Bayes** | Performed poorly as its core assumptions — conditional independence of features and Gaussian-distributed numeric variables — are strongly violated. The dataset contains correlated features such as Height, Weight, and lifestyle factors, degrading predictive power. |
| **Random Forest (Ensemble)** | Produced stable and robust predictions by averaging multiple trees, reducing variance and overfitting risk. The very high AUC suggests well-calibrated probability estimates and strong class separability. |
| **XGBoost (Ensemble)** | Emerged as the best-performing model by leveraging boosting, sequentially correcting errors from previous trees. Its superior MCC and F1 score indicate balanced multiclass predictions and excellent generalization. |

---

## ✅ Conclusion

Among all evaluated models, **XGBoost achieved the highest predictive performance**, demonstrating the effectiveness of boosting-based ensemble methods for this dataset.  
Tree-based models significantly outperformed simpler probabilistic models, highlighting the presence of nonlinear feature interactions.

---

## 👨‍💻 Author

Devashish Verma
