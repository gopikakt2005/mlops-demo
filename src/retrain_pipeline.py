import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.monitor_drift import check_data_drift


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_classification")

MODEL_NAME = "IrisRandomForest"
ACCURACY_THRESHOLD = 0.90


def automated_continuous_training():

    print("Checking production data for drift...")

    drift_detected = check_data_drift()

    if not drift_detected:
        print("No significant data drift detected.")
        print("Retraining not required.")
        return

    print("Data drift detected!")
    print("Starting model retraining...")

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    with mlflow.start_run(run_name="automated_retrain_run"):

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("accuracy", accuracy)

        print(f"Candidate Model Accuracy: {accuracy:.4f}")

        if accuracy >= ACCURACY_THRESHOLD:

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

            print("Model successfully registered in MLflow!")

        else:
            print("Candidate model failed accuracy threshold.")
            print("Model registration skipped.")

    print("Retraining pipeline completed!")


if __name__ == "__main__":
    automated_continuous_training()
