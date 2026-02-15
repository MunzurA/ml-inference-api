from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_success():
    response = client.post(
        "/predict",
        json={"feature1": 5.1, "feature2": 3.5, "feature3": 1.4, "feature4": 0.2},
    )

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)


def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={"feature1": "invalid", "feature2": 3.5, "feature3": 1.4, "feature4": 0.2},
    )

    assert response.status_code == 422
