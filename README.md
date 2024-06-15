# Credit_Risk_Assessment
This project involves developing a data science model to predict whether a loan application will be approved or not, and to analyze the associated credit risk as low, moderate, or high. The model will be built using historical loan application data, and the risk assessment will help lenders make informed decisions.

Dataset Available at Kaggle: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

Data Collection The dataset typically includes the following features:

Applicant details: age, gender, marital status, dependents Employment information: employment status, job title, work experience Financial information: income, credit history, loan amount, loan term Property information: property area, property value Loan application status: approved, rejected

Data Preprocessing Handling Missing Values: Identify and fill or remove missing values. Encoding Categorical Variables: Convert categorical features to numerical using techniques like One-Hot Encoding or Label Encoding. Feature Scaling: Normalize or standardize numerical features to ensure they are on a comparable scale.

Exploratory Data Analysis (EDA) Univariate Analysis: Analyze the distribution of individual features. Bivariate Analysis: Study the relationship between features and the target variable (loan approval status). Correlation Analysis: Identify correlations between features to detect multicollinearity.

Model Building Train-Test Split: Split the data into training and testing sets (e.g., 80-20 split). Model Selection: Choose appropriate algorithms, such as: Logistic Regression Decision Trees Random Forest Gradient Boosting Support Vector Machine (SVM)

Model Training: Train the selected models on the training set. Model Evaluation: Evaluate the models using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC score.

Risk Assessment Model Probability Estimation: Use the predicted probabilities from the classification model to assess the risk. Risk Categorization: Define thresholds to categorize the risk into low, moderate, and high.

Implementation details:

1. Create your own virtual environment.
2. In Jupyter or collab run the Credit_Risk.ipynb file so there you create the models.
3. In VS Code install all dependicies needed for this project.
4. Install Flask.
   Run app.py file in terminal -- python app.py
5. Open it on localhost.
