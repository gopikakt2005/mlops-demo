import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():

    # Use the same MLflow database as the application
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    mlflow.set_experiment("iris_classification")

    with mlflow.start_run(run_name="initial_training"):

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="IrisRandomForest",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print("Training completed!")
        print("Accuracy:", accuracy)
        print("Model registered successfully!")


if __name__ == "__main__":
    train_model()