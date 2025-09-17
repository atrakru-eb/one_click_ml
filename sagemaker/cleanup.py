"""
Cleanup SageMaker resources
"""
import boto3
import argparse

def cleanup_endpoint(endpoint_name):
    """Delete SageMaker endpoint and endpoint configuration"""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        # Delete endpoint
        print(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        
        # Delete endpoint configuration
        endpoint_config_name = f"{endpoint_name}-config"
        print(f"Deleting endpoint configuration: {endpoint_config_name}")
        sagemaker_client.delete_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        
        print("Cleanup completed successfully")
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', required=True, help='Endpoint name to delete')
    
    args = parser.parse_args()
    cleanup_endpoint(args.endpoint_name)