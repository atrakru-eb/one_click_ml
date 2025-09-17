# ML Model Deployment with SageMaker

This project demonstrates how to deploy a machine learning model as a SageMaker endpoint using AWS CodeBuild and GitHub integration.

## Architecture

- **Flask API**: Serves predictions locally and via SageMaker endpoint
- **SageMaker**: Hosts the ML model as a scalable endpoint
- **CodeBuild**: Trains, tests, and deploys the model automatically
- **GitHub Integration**: Triggers builds on code changes

## Project Structure

```
├── app.py                 # Flask API with prediction endpoints
├── model/
│   ├── train.py          # Model training script
│   ├── inference.py      # SageMaker inference handler
│   └── artifacts/        # Trained model artifacts
├── sagemaker/
│   ├── deploy.py         # SageMaker deployment script
│   └── cleanup.py        # Resource cleanup script
├── buildspec.yml         # CodeBuild configuration
└── requirements.txt      # Python dependencies
```

## Setup Instructions

### 1. AWS Prerequisites

- Create a SageMaker execution role with necessary permissions
- Update `SAGEMAKER_ROLE_ARN` in `buildspec.yml` with your role ARN
- Ensure CodeBuild has permissions to access SageMaker

### 2. Environment Variables

Set these in your CodeBuild project or locally:

```bash
MODEL_NAME=ml-model
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole
SAGEMAKER_ENDPOINT_NAME=ml-model-endpoint
```

### 3. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python model/train.py

# Run the Flask app
python app.py

# Test the API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]}'
```

### 4. Deployment

The CodeBuild pipeline will:
1. Install dependencies
2. Run pre-commit hooks and tests
3. Train the ML model
4. Deploy to SageMaker endpoint
5. Create deployment artifacts

### 5. API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Local model predictions
- `POST /predict/sagemaker` - SageMaker endpoint predictions

### 6. Cleanup

To avoid ongoing costs, delete the SageMaker endpoint:

```bash
python sagemaker/cleanup.py --endpoint-name ml-model-endpoint
```

## Model Customization

Replace the sample model in `model/train.py` with your own:

1. Update data loading logic
2. Modify model architecture
3. Adjust feature preprocessing
4. Update inference logic in `model/inference.py`

## Monitoring and Scaling

- Enable SageMaker Model Monitor for data drift detection
- Configure auto-scaling policies for the endpoint
- Set up CloudWatch alarms for monitoring
- Use SageMaker Pipelines for MLOps workflows

## Security Considerations

- Use IAM roles with minimal required permissions
- Enable VPC endpoints for private communication
- Encrypt model artifacts in S3
- Enable endpoint data capture for audit trails