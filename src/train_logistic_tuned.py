import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import load_data, clean_data, split_data
from src.evaluate import evaluate_model

mlflow.set_tracking_uri("file:./mlruns")

def main():
    df = load_data("data/raw/Loan_Data.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.1,
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    mlflow.set_experiment("logistic_regression")

    with mlflow.start_run(run_name="tuned_v1"):
        pipeline.fit(X_train, y_train)

        metrics, cm = evaluate_model(pipeline, X_test, y_test)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", 0.1)
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)

        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value)

        os.makedirs("models", exist_ok=True)
        model_path = "models/logistic_model_tuned.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        print("Tuned Logistic Regression metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        print("Confusion matrix:")
        print(cm)

if __name__ == "__main__":
    main()