# Loan Approval Prediction System

A student machine-learning project that generates a synthetic dataset, explores patterns, compares classification algorithms, and provides a Flask web interface for educational loan-outcome predictions.

## Features
- Registration and login with password hashing
- Synthetic dataset with 2,000 applicant records
- Missing-value handling, duplicate removal, categorical encoding and numeric scaling
- EDA charts: employment approval rate, correlation heatmap, income-vs-loan scatter plot, approval distribution
- Logistic Regression, Decision Tree and Random Forest comparison
- Accuracy, precision, recall, F1-score and confusion matrix
- Web form for applicant details and prediction history

## Requirements
Python 3.10 or later.

## Setup (Windows)
1. Extract the project ZIP.
2. Open Command Prompt in the project folder.
3. Create a virtual environment:
   `py -m venv .venv`
4. Activate it:
   `.venv\\Scripts\\activate`
5. Install libraries:
   `pip install -r requirements.txt`
6. Generate dataset, charts and trained model:
   `python train_model.py`
7. Start the website:
   `python app.py`
8. Open `http://127.0.0.1:5000` in your browser.
9. Register a new account, log in, and submit a loan application.

## Files
- `train_model.py`: dataset generation, EDA, model training and evaluation
- `app.py`: Flask routes, authentication, prediction and history
- `loan_dataset.csv`: generated when train_model.py runs
- `loan_model.pkl`: trained model saved when train_model.py runs
- `notebooks/Loan_Approval_Prediction.ipynb`: notebook overview and runnable workflow
- `static/plots/`: generated charts

## Dataset columns
ApplicantID, ApplicantName, Gender, Age, MonthlyIncome, LoanAmount, CreditScore, EmploymentStatus, ExistingLoans, LoanTerm, PropertyArea, LoanApproved.

ApplicantID and ApplicantName are for identification only and are excluded from model training. LoanApproved is the target label and is not entered in the prediction form. Gender is collected in the form for the requested applicant details but is not used by this sample model.

## Important limitation
The dataset and labels are synthetic and designed only for learning. This model is not a real bank underwriting system and must not be used to make actual lending decisions. Model metrics reflect this synthetic dataset and do not establish real-world accuracy. Do not deploy publicly without security review, HTTPS, a strong secret key, database protections and privacy/legal review.
