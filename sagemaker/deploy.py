"""
SageMaker deployment script
"""
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import get_execution_role
import os
import tarfile

def create_model_tar():
    """Create a tar.gz file with model artifacts"""
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('model/artifacts/model.pkl', arcname='model.pkl')
        tar.add('model/inference.py', arcname='code/inference.py')
    
    print("Model tar.gz created successfully")
    return 'model.tar.gz'

def deploy_model(
    model_name='ml-model',
    instance_type='ml.t2.medium',
    instance_count=1,
    role_arn=None
):
    """Deploy model to SageMaker endpoint"""
    
    # Get SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Get execution role
    if role_arn is None:
        try:
            role = get_execution_role()
        except:
            print("Please provide a SageMaker execution role ARN")
            return None
    else:
        role = role_arn
    
    # Create model tar
    model_tar_path = create_model_tar()
    
    # Upload model to S3
    bucket = sagemaker_session.default_bucket()
    model_s3_path = sagemaker_session.upload_data(
        path=model_tar_path,
        bucket=bucket,
        key_prefix=f'models/{model_name}'
    )
    
    print(f"Model uploaded to: {model_s3_path}")
    
    # Create SageMaker model
    sklearn_model = SKLearnModel(
        model_data=model_s3_path,
        role=role,
        entry_point='inference.py',
        framework_version='1.2-1',
        py_version='py3',
        name=model_name
    )
    
    # Deploy the model
    predictor = sklearn_model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=f'{model_name}-endpoint'
    )
    
    print(f"Model deployed to endpoint: {predictor.endpoint_name}")
    return predictor

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='ml-model', help='Model name')
    parser.add_argument('--instance-type', default='ml.t2.medium', help='Instance type')
    parser.add_argument('--role-arn', help='SageMaker execution role ARN')
    
    args = parser.parse_args()
    
    predictor = deploy_model(
        model_name=args.model_name,
        instance_type=args.instance_type,
        role_arn=args.role_arn
    )
    
    if predictor:
        print(f"Deployment successful! Endpoint: {predictor.endpoint_name}")