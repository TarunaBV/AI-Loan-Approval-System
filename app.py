from flask import Flask, render_template, request
import joblib
import pandas as pd
import mysql.connector

app = Flask(__name__)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Taru..KA.15",
    database="loan_approval_db"
)

cursor = db.cursor()

model = joblib.load("model/loan_model.pkl")
encoders = joblib.load("model/encoders.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    data = {
        "Gender": request.form["Gender"],
        "Married": request.form["Married"],
        "Dependents": request.form["Dependents"],
        "Education": request.form["Education"],
        "Self_Employed": request.form["Self_Employed"],
        "ApplicantIncome": float(request.form["ApplicantIncome"]),
        "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
        "LoanAmount": float(request.form["LoanAmount"]),
        "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
        "Credit_History": float(request.form["Credit_History"]),
        "Property_Area": request.form["Property_Area"]
    }

    df = pd.DataFrame([data])

    for column in encoders:

        if column == "Loan_Status":
            continue

        df[column] = encoders[column].transform(df[column])

    prediction = model.predict(df)[0]
    prediction_text = "Approved" if prediction == 1 else "Rejected"

    result = (
        "✅ Loan Approved!"
        if prediction == 1
        else "❌ Loan Rejected!"
    )
    
    query = """
    INSERT INTO loan_applications(
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area,
        prediction
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    values = (
        data["Gender"],
        data["Married"],
        data["Dependents"],
        data["Education"],
        data["Self_Employed"],
        data["ApplicantIncome"],
        data["CoapplicantIncome"],
        data["LoanAmount"],
        data["Loan_Amount_Term"],
        data["Credit_History"],
        data["Property_Area"],
        prediction_text
    )

    cursor.execute(query, values)
    db.commit()

    return render_template("result.html", result=result)


if __name__ == '__main__':
    app.run(debug=True)