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
    # Load and prepare data
    df = load_data("data/raw/Loan_Data.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])

    # MLflow experiment
    mlflow.set_experiment("logistic_regression")

    with mlflow.start_run(run_name="baseline_v1"):
        pipeline.fit(X_train, y_train)

        metrics, cm = evaluate_model(pipeline, X_test, y_test)

        # log params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)

        # log metrics
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value)

        # save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/logistic_model.joblib")

        # log artifact
        mlflow.log_artifact("models/logistic_model.joblib")

        print("Logistic Regression metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        print("Confusion matrix:")
        print(cm)


if __name__ == "__main__":
    main()