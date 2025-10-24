from dotenv import load_dotenv
import os
import boto3

# Load credentials from .env file
load_dotenv()

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

# Initialize Bedrock runtime client (for model inference)
client = boto3.client(
    "bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
)

# Initialize Bedrock client (for model listing)
bedrock = boto3.client(
    "bedrock",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
)

# Test call: list available foundation models
try:
    response = bedrock.list_foundation_models()
    print("✅ Connected successfully! Available models:")
    for model in response.get("modelSummaries", []):
        print("-", model["modelId"])
except Exception as e:
    print(f"❌ Bedrock connection error: {e}")

