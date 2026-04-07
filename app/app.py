from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__, template_folder="templates")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

print("Base dir:", BASE_DIR)
print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    default_probability = None
    error = None

    if request.method == "POST":
        try:
            credit_lines_outstanding = float(request.form["credit_lines_outstanding"])
            loan_amt_outstanding = float(request.form["loan_amt_outstanding"])
            total_debt_outstanding = float(request.form["total_debt_outstanding"])
            income = float(request.form["income"])
            years_employed = float(request.form["years_employed"])
            fico_score = float(request.form["fico_score"])

            input_df = pd.DataFrame([{
                "credit_lines_outstanding": credit_lines_outstanding,
                "loan_amt_outstanding": loan_amt_outstanding,
                "total_debt_outstanding": total_debt_outstanding,
                "income": income,
                "years_employed": years_employed,
                "fico_score": fico_score
            }])

            prediction = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)
                if probs.shape[1] > 1:
                    default_probability = round(float(probs[0][1]), 4)
                else:
                    default_probability = round(float(probs[0][0]), 4)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        default_probability=default_probability,
        error=error
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            "credit_lines_outstanding": data["credit_lines_outstanding"],
            "loan_amt_outstanding": data["loan_amt_outstanding"],
            "total_debt_outstanding": data["total_debt_outstanding"],
            "income": data["income"],
            "years_employed": data["years_employed"],
            "fico_score": data["fico_score"]
        }])

        prediction = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)
            if probs.shape[1] > 1:
                default_probability = float(probs[0][1])
            else:
                default_probability = float(probs[0][0])
        else:
            default_probability = None

        return jsonify({
            "prediction": prediction,
            "default_probability": default_probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)