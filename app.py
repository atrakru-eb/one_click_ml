from flask import Flask, request, jsonify
import boto3
import json
import os
import joblib
import numpy as np

app = Flask(__name__)

# Initialize SageMaker client
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Load local model for testing (optional)
try:
    local_model = joblib.load('model/artifacts/model.pkl')
    print("Local model loaded successfully")
except:
    local_model = None
    print("No local model found - will use SageMaker endpoint")

@app.route("/")
def hello_world():
    return jsonify({
        "message": "ML Model API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_sagemaker": "/predict/sagemaker"
        }
    })

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "ml-model-api"})

@app.route("/predict", methods=['POST'])
def predict_local():
    """Make predictions using local model"""
    if local_model is None:
        return jsonify({"error": "Local model not available"}), 500
    
    try:
        data = request.get_json()
        if 'instances' not in data:
            return jsonify({"error": "Missing 'instances' in request body"}), 400
        
        input_data = np.array(data['instances'])
        predictions = local_model.predict(input_data)
        probabilities = local_model.predict_proba(input_data)
        
        return jsonify({
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/sagemaker", methods=['POST'])
def predict_sagemaker():
    """Make predictions using SageMaker endpoint"""
    endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
    if not endpoint_name:
        return jsonify({"error": "SAGEMAKER_ENDPOINT_NAME environment variable not set"}), 500
    
    try:
        data = request.get_json()
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(data)
        )
        
        result = json.loads(response['Body'].read().decode())
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
