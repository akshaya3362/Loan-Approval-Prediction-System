
from pathlib import Path
import sqlite3
import joblib
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"
MODEL_PATH = BASE_DIR / "loan_model.pkl"

app = Flask(__name__)
app.secret_key = "change-this-to-a-long-random-secret-before-deployment"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            applicant_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            monthly_income REAL NOT NULL,
            loan_amount REAL NOT NULL,
            credit_score INTEGER NOT NULL,
            employment_status TEXT NOT NULL,
            existing_loans INTEGER NOT NULL,
            loan_term INTEGER NOT NULL,
            property_area TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or len(password) < 6:
            flash("Enter all details. Password must be at least 6 characters.", "error")
            return render_template("register.html")

        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users(name,email,password_hash) VALUES(?,?,?)",
                    (name, email, generate_password_hash(password))
                )

            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("This email is already registered.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email=?", (email,)
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE user_id=?",
            (session["user_id"],)
        ).fetchone()[0]

        approved = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE user_id=? AND prediction=1",
            (session["user_id"],)
        ).fetchone()[0]

    return render_template("dashboard.html", total=total, approved=approved)


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            name = request.form.get("applicant_name", "").strip()
            gender = request.form["gender"]
            age = int(request.form["age"])
            income = float(request.form["monthly_income"])
            amount = float(request.form["loan_amount"])
            credit = int(request.form["credit_score"])
            employment = request.form["employment_status"]
            existing = int(request.form["existing_loans"])
            term = int(request.form["loan_term"])
            area = request.form["property_area"]

            if not name or not (18 <= age <= 100) or income <= 0 or amount <= 0:
                raise ValueError()

            if not (300 <= credit <= 900) or existing < 0 or term <= 0:
                raise ValueError()

            if employment not in ["Employed", "Self-Employed", "Unemployed"]:
                raise ValueError()

            if area not in ["Urban", "Semi-Urban", "Rural"]:
                raise ValueError()

            bundle = load_model()

            if bundle is None:
                flash("Model not found. Run train_model.py first.", "error")
                return redirect(url_for("predict"))

            sample = pd.DataFrame([{
                "Gender": gender,
                "Age": age,
                "MonthlyIncome": income,
                "LoanAmount": amount,
                "CreditScore": credit,
                "EmploymentStatus": employment,
                "ExistingLoans": existing,
                "LoanTerm": term,
                "PropertyArea": area
            }], columns=bundle["features"])

            prediction = int(bundle["model"].predict(sample)[0])

            # Possible risk factors for a rejected prediction
            reasons = []

            if prediction == 0:
                if credit < 600:
                    reasons.append("Credit score is relatively low.")

                if income < 30000:
                    reasons.append("Monthly income is relatively low.")

                if amount > income * 15:
                    reasons.append("Loan amount is high compared to monthly income.")

                if existing >= 3:
                    reasons.append("Applicant has multiple existing loans.")

                if employment == "Unemployed":
                    reasons.append("Applicant is currently unemployed.")

                if term > 36:
                    reasons.append("Longer loan repayment term may increase risk.")

                if not reasons:
                    reasons.append(
                        "The combination of applicant details resulted in a rejection prediction."
                    )

            with get_db() as conn:
                conn.execute("""INSERT INTO predictions
                    (user_id,applicant_name,age,monthly_income,loan_amount,credit_score,
                     employment_status,existing_loans,loan_term,property_area,prediction)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (session["user_id"], name, age, income, amount, credit,
                     employment, existing, term, area, prediction)
                )

            return render_template(
                "result.html",
                name=name,
                approved=prediction == 1,
                model_name=bundle["model_name"],
                reasons=reasons
            )

        except (ValueError, KeyError):
            flash("Please enter valid values in every field.", "error")

    return render_template("prediction.html")


@app.route("/history")
@login_required
def history():
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM predictions WHERE user_id=?
               ORDER BY created_at DESC""",
            (session["user_id"],)
        ).fetchall()

    return render_template("history.html", rows=rows)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))
@app.context_processor
def inject_user():
    return {"user_name": session.get("user_name")}


init_db()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)