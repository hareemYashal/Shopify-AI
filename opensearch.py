from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv
import os

load_dotenv()

host = os.getenv("OPENSEARCH_HOST").replace("https://", "")
username = os.getenv("OPENSEARCH_USER")
password = os.getenv("OPENSEARCH_PASS")

client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
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
