from dotenv import load_dotenv
import os
import boto3

# Load credentials from .env file
load_dotenv()

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

# Initialize Bedrock client
client = boto3.client(
    "bedrock",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
)

# Test call: list available foundation models
response = client.list_foundation_models()
print("✅ Connected successfully! Available models:")
for model in response.get("modelSummaries", []):
    print("-", model["modelId"])

