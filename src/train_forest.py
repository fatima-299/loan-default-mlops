import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import load_data, clean_data, split_data
from src.evaluate import evaluate_model

mlflow.set_tracking_uri("file:./mlruns")

def main():
    df = load_data("data/raw/Loan_Data.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    mlflow.set_experiment("random_forest")

    with mlflow.start_run(run_name="baseline_v1"):
        model.fit(X_train, y_train)

        metrics, cm = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)

        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value)

        os.makedirs("models", exist_ok=True)
        model_path = "models/forest_model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print("Random Forest metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        print("Confusion matrix:")
        print(cm)

if __name__ == "__main__":
    main()