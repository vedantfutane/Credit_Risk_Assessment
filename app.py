from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/xgb_model.pkl', 'rb'))

# Assume we have this information about the model
model_info = {
    "algorithm": "XGBoost (Extreme Gradient Boosting)",
    "description": "Gradient Boosting is a machine learning technique that iteratively builds a strong predictive model. XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient boosting designed for speed and performance. While gradient boosting builds an ensemble of weak learners (typically decision trees) sequentially to minimize a loss function, XGBoost enhances this process with advanced features like regularization, parallel processing, and handling missing values, making it faster and more efficient than standard gradient boosting methods."
}

@app.route('/')
def home():
    return render_template("index.html", prediction_result=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        employed = request.form['employed']
        credit = float(request.form['credit'])
        area = request.form['area']
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])

        # gender
        male = 1 if gender == "Male" else 0
        
        # married
        married_yes = 1 if married == "Yes" else 0

        # dependents
        dependents_1 = 1 if dependents == '1' else 0
        dependents_2 = 1 if dependents == '2' else 0
        dependents_3 = 1 if dependents == "3+" else 0

        # education
        not_graduate = 1 if education == "Not Graduate" else 0

        # employed
        employed_yes = 1 if employed == "Yes" else 0

        # property area
        semiurban = 1 if area == "Semiurban" else 0
        urban = 1 if area == "Urban" else 0

        ApplicantIncomelog = np.log(ApplicantIncome)
        totalincomelog = np.log(ApplicantIncome + CoapplicantIncome)
        LoanAmountlog = np.log(LoanAmount)
        Loan_Amount_Termlog = np.log(Loan_Amount_Term)

        # Prepare the input features (ensure correct order and correct number of features)
        features = np.array([[credit, ApplicantIncomelog, LoanAmountlog, Loan_Amount_Termlog, totalincomelog,
                              male, married_yes, dependents_1, dependents_2, dependents_3,
                              not_graduate, employed_yes]])

        # Predict the probability of loan approval
        prediction_proba = model.predict_proba(features)[0][1]
        
        # Determine loan status
        loan_status = "Yes" if prediction_proba > 0.5 else "No"

        # Determine risk level
        def get_risk_level(probability):
            if probability < 0.3:
                return 'High Risk'
            elif probability < 0.7:
                return 'Moderate Risk'
            else:
                return 'Low Risk'

        risk_level = get_risk_level(prediction_proba)

        # Get feature importances from the model
        feature_importances = model.feature_importances_
        feature_names = ['Credit History', 'Applicant Income Log', 'Loan Amount Log', 'Loan Amount Term Log', 
                         'Total Income Log', 'Gender (Male)', 'Married', 'Dependents (1)', 'Dependents (2)', 
                         'Dependents (3+)', 'Not Graduate', 'Self Employed']  # Note: Remove extra features

        # Pair features with their importance
        feature_contributions = list(zip(feature_names, feature_importances, features[0]))

        # Sort features by importance
        feature_contributions.sort(key=lambda x: x[1], reverse=True)

        # Prepare the input feature values for display
        input_features = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self Employed": employed,
            "Credit History": credit,
            "Property Area": area,
            "Applicant Income": ApplicantIncome,
            "Coapplicant Income": CoapplicantIncome,
            "Loan Amount": LoanAmount,
            "Loan Amount Term": Loan_Amount_Term
        }

        # Add model information to the prediction text
        prediction_text = (
            f"Loan Approval: {loan_status}, Risk Level: {risk_level}.<br>"
            f"Algorithm: {model_info['algorithm']}.<br>"
            f"{model_info['description']}"
        )
        
        prediction_result = (
            f"Loan Approval: {loan_status}, Risk Level: {risk_level}"
        )

        return render_template("index.html", prediction_result=prediction_result, prediction_text=prediction_text, input_features=input_features, feature_contributions=feature_contributions)
    else:
        return render_template("index.html", prediction_result=None)

if __name__ == "__main__":
    app.run(debug=True)
