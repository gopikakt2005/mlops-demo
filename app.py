from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="Iris MLOps API",
    description="Iris Flower Classification using MLflow",
    version="2.0"
)

MODEL_NAME = "IrisRandomForest"

model = None
model_version = None


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.on_event("startup")
def load_model():
    global model, model_version

    try:
        client = mlflow.tracking.MlflowClient()

        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        if not versions:
            raise Exception("No registered model versions found")

        latest = max(versions, key=lambda item: int(item.version))

        model_version = latest.version

        model_uri = f"models:/{MODEL_NAME}/{model_version}"

        model = mlflow.sklearn.load_model(model_uri)

        print(f"MLflow model Version {model_version} loaded successfully!")

    except Exception as e:
        model = None
        print("Model not available:", e)


@app.get("/")
def home():
    return {
        "message": "Iris MLOps API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version
    }


@app.post("/predict")
def predict(data: IrisInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable."
        )

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0]),
        "model_version": model_version
    }