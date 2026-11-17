import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_valid_input(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in [0, 1, 2]
    assert "model_version" in data


def test_predict_invalid_schema(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422