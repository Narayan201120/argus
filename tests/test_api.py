import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "ARGUS" in response.json()["name"]

def test_health():
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "connectors" in data

def test_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "connectors" in data
    assert "total" in data
    assert data["total"] >= 0

def test_query_no_available_connectors():
    response = client.post("/v1/query", json={
        "query": "test",
        "model_config": {"connectors": ["nonexistent"]}
    })
    assert response.status_code == 503
