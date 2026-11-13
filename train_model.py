
"""Loan Approval Prediction - Complete Machine Learning Project"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "static" / "plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==================================================
# 2. GENERATE SYNTHETIC DATASET
# ==================================================

def generate_data(n=2000):

    rng = np.random.default_rng(42)

    applicant_id = np.arange(1, n + 1)

    names = [f"Applicant_{i}" for i in applicant_id]

    gender = rng.choice(
        ["Male", "Female"], size=n
    )

    age = rng.integers(21, 61, size=n)

    income = rng.integers(
        15000, 150001, size=n
    )

    loan_amount = rng.integers(
        50000, 1000001, size=n
    )

    credit_score = rng.integers(
        300, 901, size=n
    )

    employment = rng.choice(
        ["Employed", "Self-Employed", "Unemployed"],
        size=n,
        p=[0.65, 0.25, 0.10]
    )

    existing_loans = rng.integers(
        0, 6, size=n
    )

    loan_term = rng.choice(
        [12, 24, 36, 48, 60],
        size=n
    )

    property_area = rng.choice(
        ["Urban", "Semi-Urban", "Rural"],
        size=n
    )

    # Synthetic approval probability

    logit = (
        -2.0
        + (credit_score - 600) / 105
        + (income - 40000) / 65000
        - loan_amount / np.maximum(income * 15, 1) * 1.2
        - existing_loans * 0.22
        + np.where(
            employment == "Employed",
            0.65,
            np.where(
                employment == "Self-Employed",
                0.25,
                -0.8
            )
        )
        + np.where(loan_term <= 36, 0.15, 0)
    )

    probability = 1 / (1 + np.exp(-logit))

    approved = (
        rng.random(n) < probability
    ).astype(int)

    df = pd.DataFrame({
        "ApplicantID": applicant_id,
        "ApplicantName": names,
        "Gender": gender,
        "Age": age,
        "MonthlyIncome": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "EmploymentStatus": employment,
        "ExistingLoans": existing_loans,
        "LoanTerm": loan_term,
        "PropertyArea": property_area,
        "LoanApproved": approved
    })

    return df


# ==================================================
# 3. GENERATE AND CLEAN DATASET
# ==================================================

print("\nGenerating Loan Dataset...")

df = generate_data()

print("\nOriginal Dataset Shape:")
print(df.shape)

# Remove duplicate records

df = df.drop_duplicates().reset_index(drop=True)

print("\nDuplicate records removed.")

print("Final Dataset Records:", len(df))

# Save dataset

dataset_path = DATA_DIR / "loan_data.csv"

df.to_csv(dataset_path, index=False)

print("\nDataset saved:", dataset_path)

print("\nDataset Preview:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# ==================================================
# 4. EXPLORATORY DATA ANALYSIS
# ==================================================

print("\nGenerating EDA Visualizations...")

# 4.1 Loan Approval Distribution

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="LoanApproved"
)

plt.title("Loan Approval Distribution")
plt.xlabel("Loan Approved (0 = Rejected, 1 = Approved)")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.savefig(PLOT_DIR / "loan_approval_distribution.png")
plt.close()


# 4.2 Credit Score Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="CreditScore",
    hue="LoanApproved",
    bins=30,
    kde=True
)

plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")

plt.tight_layout()
plt.savefig(PLOT_DIR / "credit_score_distribution.png")
plt.close()


# 4.3 Monthly Income vs Loan Amount

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="MonthlyIncome",
    y="LoanAmount",
    hue="LoanApproved"
)

plt.title("Monthly Income vs Loan Amount")
plt.xlabel("Monthly Income")
plt.ylabel("Loan Amount")

plt.tight_layout()
plt.savefig(PLOT_DIR / "income_vs_loan.png")
plt.close()


# 4.4 Correlation Heatmap

plt.figure(figsize=(10, 7))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Loan Dataset Correlation Matrix")

plt.tight_layout()
plt.savefig(PLOT_DIR / "correlation_matrix.png")
plt.close()


# 4.5 Employment Status vs Loan Approval

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="EmploymentStatus",
    hue="LoanApproved"
)

plt.title("Loan Approval by Employment Status")
plt.xlabel("Employment Status")
plt.ylabel("Number of Applicants")

plt.legend(
    title="Loan Approved",
    labels=["Rejected", "Approved"]
)

plt.tight_layout()
plt.savefig(PLOT_DIR / "employment_approval.png")
plt.close()


# 4.6 Approved vs Rejected Pie Chart

approval_counts = df["LoanApproved"].value_counts().reindex(
    [1, 0], fill_value=0
)

plt.figure(figsize=(7, 7))

plt.pie(
    approval_counts,
    labels=["Approved", "Rejected"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Loan Approval Percentage")

plt.tight_layout()
plt.savefig(PLOT_DIR / "approval_pie_chart.png")
plt.close()

print("All EDA charts generated successfully!")
# ==================================================
# 5. FEATURES AND TARGET
# ==================================================
features = [
    "Gender",
    "Age",
    "MonthlyIncome",
    "LoanAmount",
    "CreditScore",
    "EmploymentStatus",
    "ExistingLoans",
    "LoanTerm",
    "PropertyArea"
]
target = "LoanApproved"
X = df[features]
y = df[target]
categorical_features = [
    "Gender",
    "EmploymentStatus",
    "PropertyArea"
]
numeric_features = [
    "Age",
    "MonthlyIncome",
    "LoanAmount",
    "CreditScore",
    "ExistingLoans",
    "LoanTerm"
]
# ==================================================
# 6. DATA PREPROCESSING
# ==================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
# ==================================================
# 7. TRAIN TEST SPLIT
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))
# ==================================================
# 8. MACHINE LEARNING MODELS
# ==================================================
models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}
results = []
trained_models = {}
print("\nTraining Machine Learning Models...")
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })
    trained_models[name] = pipeline
    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
# ==================================================
# 9. MODEL COMPARISON
# ==================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)
print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))
results_df.to_csv(
    DATA_DIR / "model_comparison.csv",
    index=False
)
# ==================================================
# 10. SELECT BEST MODEL
# ==================================================
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
print("\nBest model by F1-score:", best_name)
# ==================================================
# 11. RANDOM FOREST FEATURE IMPORTANCE
# ==================================================
rf_pipeline = trained_models["Random Forest"]
rf_model = rf_pipeline.named_steps["classifier"]
feature_names = rf_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(10)
print("\nTop 10 Important Features:")
print(importance_df.to_string(index=False))
plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)
plt.title("Top 10 Loan Approval Features")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance.png")
plt.close()
# ==================================================
# 12. CONFUSION MATRIX
# ==================================================
best_predictions = best_model.predict(X_test)
cm = confusion_matrix(
    y_test,
    best_predictions
)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"]
)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix.png")
plt.close()
# ==================================================
# 13. CLASSIFICATION REPORT
# ==================================================
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)
# ==================================================
# 14. SAVE BEST MODEL
# ==================================================
model_bundle = {
    "model": best_model,
    "model_name": best_name,
    "features": features
}
model_path = BASE_DIR / "loan_model.pkl"
joblib.dump(
    model_bundle,
    model_path
)
print("\nBest model saved successfully!")
print("Saved model:", model_path)
print("\nEDA charts saved in:", PLOT_DIR)
print("\nLoan Approval Prediction Model Training Completed!")
