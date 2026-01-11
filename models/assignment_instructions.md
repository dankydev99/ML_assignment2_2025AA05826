 Assignment Details
Step 1: Dataset choice
Choose ONE classification dataset of your choice from any public repository -
Kaggle or UCI. It may be a binary classification problem or a multi-class
classification problem.
Minimum Feature Size: 12
Minimum Instance Size: 500
Step 2: Machine Learning Classification models and Evaluation metrics
Implement the following classification models using the dataset chosen above. All
the 6 ML models have to be implemented on the same dataset.
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost
For each of the models above, calculate the following evaluation metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeﬃcient (MCC Score)
The assignment has to be performed on BITS Virtual Lab and a (ONE) screenshot
has to be uploaded as a proof of that. [ 1 mark ]
Step 3: Prepare Your GitHub Repository
Your repository must contain:
project-folder/
│-- app.py (or streamlit_app.py)
│-- requirements.txt
│-- README.md
│-- model/ (saved model files for all implemented models - *.py or *.ipynb)
Step 4: Create requirements.txt
Example:
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
Missing dependencies are the #1 cause of deployment failure.
Step 5: README.md with the following structure. This README content should also
be part of the submitted PDF file. Follow the required structure carefully.
a. Problem statement
b. Dataset description [ 1 mark ]
c. Models used: [ 6 marks - 1 marks for all the metrics for each model ]
Make a Comparison Table with the evaluation metrics calculated for all the 6
models as below:
ML Model Name Accuracy AUC Precision Recall F1 MCC
Logistic
Regression
Decision Tree
kNN
Naive Bayes
Random Forest
(Ensemble)
XGBoost
(Ensemble)
- Add your observations on the performance of each model on the chosen
dataset. [ 3 marks ]
ML Model Name Observation about model performance
Logistic
Regression
Decision Tree
kNN
Naive Bayes
Random Forest
(Ensemble)
XGBoost
(Ensemble)
Step 6: Deploy on Streamlit Community Cloud
1. Go to https://streamlit.io/cloud
2. Sign in using GitHub account
3. Click “New App”
4. Select your repository
5. Choose branch (usually main)
6. Select app.py
7. Click Deploy
Within a few minutes, your app will be live.
Your Streamlit app must include at least the following features : -
a. Dataset upload option (CSV) [As streamlit free tier has limited capacity,
upload only test data] [ 1 mark ]
b. Model selection dropdown (if multiple models) [ 1 mark ]
c. Display of evaluation metrics [ 1 mark ]
d. Confusion matrix or classification report [ 1 mark ]