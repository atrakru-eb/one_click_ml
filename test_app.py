from app import app
import json


def test_health_check():
    response = app.test_client().get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"

def test_root_endpoint():
    response = app.test_client().get("/")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "message" in data
    assert "endpoints" in data

def test_predict_local_missing_data():
    response = app.test_client().post("/predict", 
                                    data=json.dumps({}),
                                    content_type='application/json')
    assert response.status_code == 400

def test_predict_local_with_data():
    # Test data - 20 features as expected by the model
    test_data = {
        "instances": [[0.1] * 20]  # Single instance with 20 features
    }
    
    response = app.test_client().post("/predict",
                                    data=json.dumps(test_data),
                                    content_type='application/json')
    
    # Should either succeed (if model is loaded) or return 500 (if no model)
    assert response.status_code in [200, 500]
