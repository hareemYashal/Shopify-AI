#!/usr/bin/env python3
"""
OpenSearch Serverless vector pipeline for product embeddings.

Mirrors the ChromaDB workflow but targets AWS OpenSearch Serverless collections
configured for `vector_search`.
"""

import fnmatch
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk, scan
from requests_aws4auth import AWS4Auth

# Load environment variables (.env, system, AWS profiles, etc.)
load_dotenv()

# --------------------------------------------------------------------------- #
# AWS / OpenSearch client setup
# --------------------------------------------------------------------------- #

EMBEDDING_DIMENSION = 1536  # Must match the embedding model output
MAX_WORKERS = 7
DEFAULT_BATCH_SIZE = 1000
DEFAULT_INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "products")


def _parse_host(host_url: str) -> tuple[str, Optional[str]]:
    """
    Break down the full OpenSearch host URL into hostname and optional URL prefix.

    OpenSearch Serverless endpoints often embed the collection name in the path
    (e.g. https://xxx.aoss.amazonaws.com/collections/my-collection). opensearch-py
    needs the hostname and the path separately.
    """
    parsed = urlparse(host_url)

    hostname = parsed.hostname
    if not hostname:
        stripped = host_url.replace("https://", "").replace("http://", "")
        hostname, _, path = stripped.partition("/")
        prefix = path or None
        return hostname, prefix

    prefix = parsed.path.lstrip("/") if parsed.path else None
    if prefix == "":
        prefix = None

    return hostname, prefix


def _build_aws_auth(region: str) -> AWS4Auth:
    """Construct an AWS SigV4 auth object for OpenSearch Serverless."""
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
    """Instantiate an OpenSearch Serverless client."""
    host_url = os.getenv("OPENSEARCH_HOST")
    if not host_url:
        raise RuntimeError("OPENSEARCH_HOST is not set in the environment.")

    region = os.getenv("AWS_REGION", "us-east-1")
    hostname, url_prefix = _parse_host(host_url)
    awsauth = _build_aws_auth(region)

    client_params: Dict[str, Any] = dict(
        hosts=[{"host": hostname, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )

    if url_prefix:
        client_params["url_prefix"] = url_prefix

    return OpenSearch(**client_params)


client = get_opensearch_client()


def ensure_index(index_name: str) -> None:
    """Create the target index with vector mappings if it does not exist."""
    if client.indices.exists(index=index_name):
        return

    index_body = {
        "settings": {
            "index": {"knn": True},
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
                        "engine": "faiss",
                        "space_type": "cosinesimil",
                    },
                },
            }
        },
    }

    client.indices.create(index=index_name, body=index_body)
    print(f"✅ Created OpenSearch index '{index_name}'")

# --------------------------------------------------------------------------- #
# Bedrock embedding helpers
# --------------------------------------------------------------------------- #

bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))


def create_embedding(
    text: str,
    model_id: str = "amazon.titan-embed-text-v1",
    max_retries: int = 3,
) -> Optional[List[float]]:
    """Generate embeddings using Amazon Bedrock with retry + backoff."""
    for attempt in range(max_retries):
        try:
            time.sleep(0.05 + random.uniform(0, 0.05))  # jitter

            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps({"inputText": text}),
            )
            result = json.loads(response["body"].read())
            embedding = result["embedding"]

            if len(embedding) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(embedding)}"
                )

            return embedding
        except Exception as exc:
            if attempt < max_retries - 1:
                wait_time = (2**attempt) + random.uniform(0, 1)
                print(f"⚠️  Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s... ({exc})")
                time.sleep(wait_time)
            else:
                print(f"❌ Error creating embedding after {max_retries} attempts: {exc}")
    return None


def build_text_for_embedding(product: Dict[str, Any]) -> str:
    """Combine product fields into a single embedding prompt string."""
    parts: List[str] = [product.get("title", "")]
    text = product.get("text")
    if text:
        parts.append(text)

    category = product.get("category")
    if category:
        parts.append(f"Category: {category}")

    tags = product.get("tags")
    if isinstance(tags, list) and tags:
        parts.append("Tags: " + ", ".join(tags[:5]))

    return " ".join(part for part in parts if part)


def process_single_product(product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate embedding and return a document ready for OpenSearch."""
    try:
        embedding_text = build_text_for_embedding(product)
        embedding = create_embedding(embedding_text)
        if embedding is None:
            print(f"❌ Failed to generate embedding for {product.get('product_id')}")
            return None

        return {
            "product_id": str(product["product_id"]),
            "title": product.get("title"),
            "text": product.get("text"),
            "price": product.get("price"),
            "url": product.get("url"),
            "image": product.get("image"),
            "in_stock": product.get("in_stock"),
            "category": product.get("category"),
            "tags": product.get("tags", []),
            "embedding": embedding,
        }
    except Exception as exc:
        print(f"❌ Error processing product {product.get('product_id', 'unknown')}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Indexing utilities
# --------------------------------------------------------------------------- #

def bulk_upsert_documents(documents: List[Dict[str, Any]], index_name: str = DEFAULT_INDEX_NAME) -> None:
    """Upsert a batch of documents into OpenSearch using the bulk API."""
    if not documents:
        return

    actions = []
    for doc in documents:
        product_id = doc["product_id"]
        actions.append(
            {
                "_op_type": "index",
                "_index": index_name,
                "_source": doc,
            }
        )

    success, errors = bulk(client, actions, raise_on_error=False)

    if errors:
        first_error = errors[0]
        error_meta = first_error.get("update", first_error.get("index", {}))
        status = error_meta.get("status")
        error_details = error_meta.get("error")
        reason = error_details.get("reason") if isinstance(error_details, dict) else error_details
        print(f"⚠️  Bulk upsert completed with errors ({len(errors)}). Status={status}, reason={reason}")
    else:
        print(f"✅ Upserted {success} documents into '{index_name}'")

    return success


def get_existing_product_ids(index_name: str = DEFAULT_INDEX_NAME) -> set[str]:
    """Fetch existing product IDs from OpenSearch to support incremental loads."""
    existing_ids: set[str] = set()

    try:
        for hit in scan(
            client,
            index=index_name,
            query={"_source": ["product_id"], "query": {"match_all": {}}},
            size=1000,
        ):
            source = hit.get("_source", {})
            product_id = source.get("product_id")
            if product_id is not None:
                existing_ids.add(str(product_id))
    except Exception as exc:
        print(f"⚠️  Could not fetch existing product IDs: {exc}")

    return existing_ids


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def embed_products_to_opensearch(
    catalog_file: str = "data/catalog.jsonl",
    incremental: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    index_name: str = DEFAULT_INDEX_NAME,
    create_index_if_missing: bool = True,
) -> None:
    """Embed all products from a JSONL catalog file and upsert to OpenSearch."""
    print("🔄 Starting OpenSearch embedding process...")

    if create_index_if_missing:
        ensure_index(index_name)
    elif not client.indices.exists(index=index_name):
        raise RuntimeError(f"Index '{index_name}' does not exist. Choose a different index or create it first.")

    if incremental:
        print("🔍 Incremental mode enabled - existing IDs will be skipped.")
        existing_ids = get_existing_product_ids(index_name=index_name)
    else:
        existing_ids = set()

    products: List[Dict[str, Any]] = []
    with open(catalog_file, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            try:
                product = json.loads(line.strip())
                if incremental and str(product["product_id"]) in existing_ids:
                    continue
                products.append(product)
            except json.JSONDecodeError as exc:
                print(f"❌ Error parsing line {line_num}: {exc}")

    print(f"📊 {len(products)} products queued for embedding.")

    total_indexed = 0
    for batch_start in range(0, len(products), batch_size):
        batch_end = min(batch_start + batch_size, len(products))
        batch = products[batch_start:batch_end]
        print(f"\n📦 Processing batch {batch_start // batch_size + 1} ({batch_start + 1}-{batch_end})")

        results: Dict[int, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_single_product, product): idx
                for idx, product in enumerate(batch)
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                idx = futures[future]
                try:
                    doc = future.result()
                    if doc:
                        results[idx] = doc
                except Exception as exc:
                    print(f"❌ Error processing product: {exc}")

                if completed % 100 == 0 or completed == len(batch):
                    print(f"   ⚡ Processed {completed}/{len(batch)} embeddings...")

        ordered_docs = [results[i] for i in sorted(results.keys())]
        successful = bulk_upsert_documents(ordered_docs, index_name=index_name) or 0
        total_indexed += successful

    print(f"\n🎉 Finished embedding. Total products indexed: {total_indexed}")


def embed_products_incremental(
    catalog_file: str = "data/catalog.jsonl",
    index_name: str = DEFAULT_INDEX_NAME,
    create_index_if_missing: bool = False,
) -> None:
    """Convenience wrapper that always runs incremental embedding."""
    embed_products_to_opensearch(
        catalog_file=catalog_file,
        incremental=True,
        index_name=index_name,
        create_index_if_missing=create_index_if_missing,
    )


def convert_filters_to_opensearch(filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert application-level filters to an OpenSearch-compatible filter query."""
    filter_clauses: List[Dict[str, Any]] = []

    for field, value in filters.items():
        if field == "in_stock":
            if isinstance(value, dict) and "eq" in value:
                filter_clauses.append({"term": {"in_stock": value["eq"]}})
            else:
                filter_clauses.append({"term": {"in_stock": value}})
        elif field == "category":
            if isinstance(value, list):
                filter_clauses.append({"terms": {"category": value}})
            else:
                filter_clauses.append({"term": {"category": value}})
        elif field == "price":
            if isinstance(value, dict):
                range_clause: Dict[str, Any] = {}
                if "gte" in value:
                    range_clause["gte"] = value["gte"]
                if "lte" in value:
                    range_clause["lte"] = value["lte"]
                if "$gte" in value:
                    range_clause["gte"] = value["$gte"]
                if "$lte" in value:
                    range_clause["lte"] = value["$lte"]
                if range_clause:
                    filter_clauses.append({"range": {"price": range_clause}})
        elif field == "tags":
            if isinstance(value, list):
                if len(value) > 1:
                    filter_clauses.append({"terms": {"tags": value}})
                elif value:
                    filter_clauses.append({"term": {"tags": value[0]}})
            else:
                filter_clauses.append({"term": {"tags": value}})

    if not filter_clauses:
        return None

    return {"bool": {"filter": filter_clauses}}


def search_products_opensearch(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    index_name: str = DEFAULT_INDEX_NAME,
) -> Tuple[List[Dict[str, Any]], float]:
    """Execute a vector similarity search against OpenSearch."""
    start_time = time.time()

    query_embedding = create_embedding(query)
    if query_embedding is None:
        print(f"❌ Failed to generate embedding for query: '{query}'")
        return [], 0.0

    filter_query = convert_filters_to_opensearch(filters or {})

    query_body: Dict[str, Any] = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": k,
                }
            }
        },
    }

    if filter_query:
        query_body["query"]["knn"]["embedding"]["filter"] = filter_query

    try:
        response = client.search(index=index_name, body=query_body)
        hits = response.get("hits", {}).get("hits", [])
        results: List[Dict[str, Any]] = []

        for hit in hits:
            source = hit.get("_source", {})
            results.append(
                {
                    "product_id": source.get("product_id"),
                    "title": source.get("title"),
                    "price": source.get("price"),
                    "url": source.get("url"),
                    "image": source.get("image"),
                    "in_stock": source.get("in_stock"),
                    "category": source.get("category"),
                    "tags": source.get("tags", []),
                    "score": hit.get("_score"),
                }
            )
    except Exception as exc:
        print(f"❌ OpenSearch query error: {exc}")
        return [], 0.0

    duration_ms = (time.time() - start_time) * 1000
    return results, duration_ms


# --------------------------------------------------------------------------- #
# Diagnostic helpers
# --------------------------------------------------------------------------- #

def get_index_stats(index_name: str = DEFAULT_INDEX_NAME) -> Dict[str, Any]:
    """Return basic stats for the OpenSearch index."""
    try:
        count_response = client.count(index=index_name)
        doc_count = count_response.get("count", 0)
        return {
            "status": "healthy",
            "index": index_name,
            "documents": doc_count,
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "error": str(exc),
        }


def clear_index(index_name: str = DEFAULT_INDEX_NAME) -> bool:
    """Delete and recreate the OpenSearch index."""
    try:
        client.indices.delete(index=index_name, ignore=[404])
        ensure_index(index_name)
        print(f"✅ Index '{index_name}' cleared and recreated.")
        return True
    except Exception as exc:
        print(f"❌ Error clearing index '{index_name}': {exc}")
        return False


def debug_sample(index_name: str = DEFAULT_INDEX_NAME, sample_size: int = 3) -> None:
    """Print a small sample of indexed documents for sanity checking."""
    try:
        response = client.search(
            index=index_name,
            body={"size": sample_size, "query": {"match_all": {}}},
        )
        hits = response.get("hits", {}).get("hits", [])
        print(f"📊 Sample ({len(hits)} docs):")
        for hit in hits:
            src = hit.get("_source", {})
            print(f" - {src.get('product_id')}: {src.get('title')} (${src.get('price')})")
    except Exception as exc:
        print(f"❌ Error fetching sample docs: {exc}")


def list_indices(pattern: str = "*") -> List[str]:
    """Return a sorted list of index names matching the given pattern."""
    discovered: set[str] = set()

    # Try data-plane APIs first (may be unsupported on Serverless)
    try:
        indices = client.indices.get(index=pattern)
        discovered.update(indices.keys())
    except Exception:
        pass

    if not discovered:
        try:
            aliases = client.indices.get_alias(pattern)
            discovered.update(aliases.keys())
        except Exception:
            pass

    if not discovered:
        try:
            stats = client.indices.stats(pattern)
            discovered.update(stats.get("indices", {}).keys())
        except Exception:
            pass

    # Fallback to control-plane API for OpenSearch Serverless collections
    if not discovered:
        try:
            oss_client = boto3.client(
                "opensearchserverless",
                region_name=os.getenv("AWS_REGION", "us-east-1"),
            )
            paginator = oss_client.get_paginator("list_collections")
            for page in paginator.paginate(maxResults=100):
                for summary in page.get("collectionSummaries", []):
                    name = summary.get("name")
                    collection_type = summary.get("collectionType")
                    if collection_type and collection_type.upper() != "VECTORSEARCH":
                        continue
                    if name and fnmatch.fnmatch(name, pattern):
                        discovered.add(name)
        except Exception:
            pass

    return sorted(discovered)


# --------------------------------------------------------------------------- #
# Command-line entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("🧠 OpenSearch Serverless Product Embedder\n")
    data_dir = "./data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found. Please create it and add JSONL files.")
        raise SystemExit(1)

    jsonl_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".jsonl")
    ]

    if not jsonl_files:
        print("❌ No JSONL files found in ./data. Exiting.")
        raise SystemExit(1)

    print("📄 Available JSONL files:")
    for idx, file_path in enumerate(jsonl_files, start=1):
        print(f"  {idx}. {os.path.basename(file_path)}")

    choice = input("\n👉 Enter the number of the file to embed: ").strip()
    try:
        file_index = int(choice) - 1
        catalog_path = jsonl_files[file_index]
    except (ValueError, IndexError):
        print("❌ Invalid selection. Exiting.")
        raise SystemExit(1)

    existing_indices = list_indices()
    if existing_indices:
        print("\n📦 Existing indices:")
        for idx, name in enumerate(existing_indices, start=1):
            print(f"  {idx}. {name}")
    else:
        print("\n⚠️ No existing indices discovered via API (this may be due to permissions).")

    index_mode = input("\n➕ Create new index or use existing? (n/e): ").strip().lower()
    if index_mode == "n":
        index_name = input("🆕 Enter new index name: ").strip()
        if not index_name:
            print("❌ Index name cannot be empty. Exiting.")
            raise SystemExit(1)

        if client.indices.exists(index=index_name):
            print(f"⚠️ Index '{index_name}' already exists. Switching to existing index mode.")
            create_index = False
        else:
            create_index = True
    elif index_mode == "e":
        if existing_indices:
            index_choice = input("📦 Enter the number or name of the existing index: ").strip()
            if index_choice.isdigit():
                idx = int(index_choice) - 1
                if idx < 0 or idx >= len(existing_indices):
                    print("❌ Invalid index selection. Exiting.")
                    raise SystemExit(1)
                index_name = existing_indices[idx]
            else:
                index_name = index_choice
        else:
            index_name = input("📦 Enter the name of the existing index: ").strip()

        if not index_name:
            print("❌ Index name cannot be empty. Exiting.")
            raise SystemExit(1)

        if not client.indices.exists(index=index_name):
            print(f"❌ Index '{index_name}' not found. Exiting.")
            raise SystemExit(1)

        create_index = False
    else:
        print("❌ Invalid choice. Exiting.")
        raise SystemExit(1)

    mode = input("🔁 Run incremental update? (y/n): ").strip().lower()
    incremental_mode = mode != "n"

    print(f"\n🚀 Embedding '{catalog_path}' into '{index_name}' (incremental={incremental_mode})...\n")
    embed_products_to_opensearch(
        catalog_path,
        incremental=incremental_mode,
        index_name=index_name,
        create_index_if_missing=create_index,
    )
    print("\n✅ Done.")

