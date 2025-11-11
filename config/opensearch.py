import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import boto3
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

load_dotenv()

EMBEDDING_DIMENSION = 1536

INDEX_BODY = {
    "settings": {
        "index": {
            "knn": True
        }
    },
    "mappings": {
        "properties": {
            "product_id": {"type": "keyword"},
            "title": {"type": "text"},
            "text": {"type": "text"},
            "price": {"type": "float"},
            "in_stock": {"type": "boolean"},
            "category": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "url": {"type": "keyword"},
            "image": {"type": "keyword"},
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIMENSION,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "faiss"
                }
            }
        }
    }
}


def _parse_host(host_url: str) -> Tuple[str, Optional[str]]:
    """Split the OpenSearch endpoint into hostname and optional path prefix."""
    parsed = urlparse(host_url)
    hostname = parsed.hostname
    if hostname:
        prefix = parsed.path.lstrip("/") if parsed.path else None
        if prefix == "":
            prefix = None
        return hostname, prefix

    cleaned = host_url.replace("https://", "").replace("http://", "")
    host_only, _, path = cleaned.partition("/")
    prefix = path or None
    return host_only, prefix


def _build_aws_auth(region: str) -> AWS4Auth:
    """Create an AWS SigV4 auth helper for OpenSearch Serverless."""
    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials not found for OpenSearch authentication.")

    frozen = credentials.get_frozen_credentials()
    return AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        region,
        "aoss",
        session_token=frozen.token,
    )


def get_opensearch_client() -> OpenSearch:
    """Instantiate an OpenSearch Serverless client using SigV4 auth."""
    host_url = os.getenv("OPENSEARCH_HOST")
    if not host_url:
        raise RuntimeError("OPENSEARCH_HOST is not set in the environment.")

    region = os.getenv("AWS_REGION", "us-east-1")
    hostname, url_prefix = _parse_host(host_url)
    awsauth = _build_aws_auth(region)

    client_params: Dict[str, Any] = {
        "hosts": [{"host": hostname, "port": 443}],
        "http_auth": awsauth,
        "use_ssl": True,
        "verify_certs": True,
        "connection_class": RequestsHttpConnection,
        "pool_maxsize": 20,
    }

    if url_prefix:
        client_params["url_prefix"] = url_prefix

    return OpenSearch(**client_params)


client = get_opensearch_client()
