from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv
from urllib.parse import urlparse
import os

load_dotenv()

# Parse OpenSearch host URL properly
host_url = os.getenv("OPENSEARCH_HOST", "")
username = os.getenv("OPENSEARCH_USER")
password = os.getenv("OPENSEARCH_PASS")

# Parse the URL properly to extract host and port
if not host_url.startswith(('http://', 'https://')):
    host_url = f'https://{host_url}'

parsed = urlparse(host_url)
host = parsed.hostname or parsed.netloc.split(':')[0]
port = parsed.port or 443

# Clean up any trailing slashes or paths
host = host.strip('/')

client = OpenSearch(
    hosts=[{"host": host, "port": port}],
    http_auth=(username, password),
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)

# Ensure index exists with desired mapping
INDEX_NAME = "products"
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
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "faiss"
                }
            }
        }
    }
}

try:
    info = client.info()
    print("✅ Connected to OpenSearch:", info["version"]["number"])

    exists = client.indices.exists(index=INDEX_NAME)
    if exists:
        print(f"✅ Index '{INDEX_NAME}' already exists")
    else:
        client.indices.create(index=INDEX_NAME, body=INDEX_BODY)
        print(f"✅ Index '{INDEX_NAME}' created")
except Exception as e:
    print("❌ OpenSearch error:", e)
