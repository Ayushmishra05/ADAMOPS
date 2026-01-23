"""
AdamOps Cloud Deployment Module

AWS, GCP, and Azure deployment helpers.
"""

from typing import Any, Dict, Optional
from pathlib import Path

from adamops.utils.logging import get_logger
from adamops.utils.helpers import ensure_dir

logger = get_logger(__name__)


class CloudDeployer:
    """Base class for cloud deployment."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def deploy(self, model_path: str, name: str) -> Dict:
        raise NotImplementedError


class AWSDeployer(CloudDeployer):
    """AWS SageMaker deployment."""
    
    def upload_to_s3(self, local_path: str, bucket: str, key: str) -> str:
        """Upload file to S3."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required. Install with: pip install boto3")
        
        s3 = boto3.client('s3')
        s3.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"
    
    def deploy_sagemaker(
        self, model_path: str, name: str,
        instance_type: str = "ml.t2.medium",
        role_arn: Optional[str] = None
    ) -> Dict:
        """Deploy to SageMaker endpoint."""
        try:
            import boto3
            import sagemaker
        except ImportError:
            raise ImportError("boto3 and sagemaker required")
        
        logger.info(f"Deploying {name} to SageMaker...")
        
        # This is a simplified example - full implementation would need
        # proper model packaging for SageMaker
        return {
            "status": "pending",
            "message": "SageMaker deployment requires additional setup",
            "model_path": model_path,
            "endpoint_name": name,
        }
    
    def generate_lambda_handler(self, output_path: str, model_s3_path: str) -> str:
        """Generate AWS Lambda handler code."""
        code = f'''"""AWS Lambda handler for model inference."""
import json
import boto3
import joblib
from io import BytesIO

# Download model from S3 on cold start
s3 = boto3.client('s3')
bucket = "{model_s3_path.split('/')[2]}"
key = "/".join("{model_s3_path}".split('/')[3:])

response = s3.get_object(Bucket=bucket, Key=key)
model = joblib.load(BytesIO(response['Body'].read()))

def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body', '{{}}'))
        features = body.get('features', [])
        
        import numpy as np
        predictions = model.predict(np.array(features)).tolist()
        
        return {{
            'statusCode': 200,
            'body': json.dumps({{'predictions': predictions}})
        }}
    except Exception as e:
        return {{
            'statusCode': 500,
            'body': json.dumps({{'error': str(e)}})
        }}
'''
        with open(output_path, 'w') as f:
            f.write(code)
        
        return output_path


class GCPDeployer(CloudDeployer):
    """Google Cloud Platform deployment."""
    
    def upload_to_gcs(self, local_path: str, bucket: str, blob_name: str) -> str:
        """Upload file to Google Cloud Storage."""
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError("google-cloud-storage required")
        
        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(blob_name)
        blob.upload_from_filename(local_path)
        
        return f"gs://{bucket}/{blob_name}"
    
    def generate_cloud_run_config(self, output_dir: str, name: str, port: int = 8080) -> str:
        """Generate Cloud Run configuration."""
        config = f'''# Cloud Run service configuration
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/{name}', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/{name}']
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  entrypoint: gcloud
  args:
  - 'run'
  - 'deploy'
  - '{name}'
  - '--image'
  - 'gcr.io/$PROJECT_ID/{name}'
  - '--region'
  - 'us-central1'
  - '--platform'
  - 'managed'
  - '--port'
  - '{port}'
images:
- 'gcr.io/$PROJECT_ID/{name}'
'''
        output_path = Path(output_dir) / "cloudbuild.yaml"
        with open(output_path, 'w') as f:
            f.write(config)
        
        return str(output_path)


class AzureDeployer(CloudDeployer):
    """Azure ML deployment."""
    
    def upload_to_blob(self, local_path: str, container: str, blob_name: str,
                       connection_string: str) -> str:
        """Upload file to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise ImportError("azure-storage-blob required")
        
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
        
        with open(local_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        
        return f"https://{blob_service.account_name}.blob.core.windows.net/{container}/{blob_name}"


def get_deployer(cloud: str, config: Optional[Dict] = None) -> CloudDeployer:
    """Get cloud deployer by name."""
    deployers = {
        "aws": AWSDeployer,
        "gcp": GCPDeployer,
        "azure": AzureDeployer,
    }
    
    if cloud not in deployers:
        raise ValueError(f"Unknown cloud: {cloud}. Available: {list(deployers.keys())}")
    
    return deployers[cloud](config)


def deploy_to_cloud(
    model_path: str, cloud: str, name: str, config: Optional[Dict] = None
) -> Dict:
    """Deploy model to cloud platform."""
    deployer = get_deployer(cloud, config)
    return deployer.deploy(model_path, name)
