#!/usr/bin/env python3
"""
OpenSearch service for vector search
Uses bulk API for efficient indexing

Use this file to embed products into OpenSearch and cretae new indices
"""

import os
import json
import time
import random
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection

# Load environment variables
load_dotenv()

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))

# Initialize OpenSearch client
def get_opensearch_client():
    """Get OpenSearch client with configuration from environment"""
    host = os.getenv("OPENSEARCH_HOST", "").replace("https://", "").replace("http://", "")
    username = os.getenv("OPENSEARCH_USER", "")
    password = os.getenv("OPENSEARCH_PASS", "")
    
    if not host or not username or not password:
        raise ValueError("OpenSearch credentials not found in environment variables")
    
    # Create a custom connection with increased timeout
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    class CustomRequestsHttpConnection(RequestsHttpConnection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Configure session with longer timeout
            self.session.mount('https://', HTTPAdapter(
                max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            ))
    
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=(username, password),
        use_ssl=True,
        verify_certs=True,
        connection_class=CustomRequestsHttpConnection,
    )

# Global client and index name (will be set in main)
client = None
INDEX_NAME = "products"

def get_index_body():
    """Get index mapping configuration for OpenSearch"""
    return {
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

def create_embedding(text: str, model_id: str = 'amazon.titan-embed-text-v1', max_retries: int = 3) -> Optional[List[float]]:
    """Generate embedding for text using Amazon Bedrock with retry logic"""
    for attempt in range(max_retries):
        try:
            # Add small random delay to avoid rate limiting
            time.sleep(0.05 + random.uniform(0, 0.05))
            
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps({"inputText": text})
            )
            result = json.loads(response['body'].read())
            return result['embedding']
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️  Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Error creating embedding after {max_retries} attempts: {e}")
                return None
    return None

def build_text_for_embedding(product: Dict[str, Any]) -> str:
    """Build text for embedding from product data"""
    text_parts = [product['title']]
    
    if 'text' in product and product['text']:
        text_parts.append(product['text'])
    
    if 'category' in product and product['category']:
        text_parts.append(f"Category: {product['category']}")
    
    if 'tags' in product and product['tags']:
        tags_text = ", ".join(product['tags'][:5])  # Limit to top 5 tags
        text_parts.append(f"Tags: {tags_text}")
    
    return " ".join(text_parts)

def process_single_product_for_embedding(product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single product: generate embedding and return data for OpenSearch"""
    try:
        # Build text for embedding
        text_for_embedding = build_text_for_embedding(product)
        
        # Generate embedding
        embedding = create_embedding(text_for_embedding)
        if embedding is None:
            print(f"❌ Failed to generate embedding for {product['product_id']}")
            return None
        
        # Return data ready for OpenSearch
        return {
            'product_id': str(product['product_id']),
            'title': product.get('title', ''),
            'text': product.get('text', ''),
            'price': product.get('price', 0.0),
            'url': product.get('url', ''),
            'image': product.get('image', ''),
            'in_stock': product.get('in_stock', False),
            'category': product.get('category', ''),
            'tags': product.get('tags', []),
            'embedding': embedding
        }
    except Exception as e:
        print(f"❌ Error processing {product.get('product_id', 'unknown')}: {e}")
        return None

def ensure_index_exists(index_name: str):
    """Ensure OpenSearch index exists, create if it doesn't"""
    global client
    if client is None:
        client = get_opensearch_client()
    
    try:
        exists = client.indices.exists(index=index_name)
        if not exists:
            print(f"📦 Creating index '{index_name}'...")
            client.indices.create(index=index_name, body=get_index_body())
            print(f"✅ Index '{index_name}' created successfully")
        else:
            print(f"✅ Index '{index_name}' already exists")
    except Exception as e:
        print(f"❌ Error ensuring index exists: {e}")
        raise

def embed_products_to_opensearch(catalog_file: str = 'data/catalog.jsonl', index_name: str = 'products', incremental: bool = True):
    """Embed products from catalog.jsonl to OpenSearch (incremental by default)"""
    global client, INDEX_NAME
    INDEX_NAME = index_name
    
    if client is None:
        client = get_opensearch_client()
    
    print("🔄 Starting OpenSearch embedding process...")
    
    # Ensure index exists
    ensure_index_exists(index_name)
    
    # Check if index already has data
    try:
        stats = client.indices.stats(index=index_name)
        current_count = stats['indices'][index_name]['total']['docs']['count']
        if current_count > 0 and incremental:
            print(f"📊 Index already has {current_count} products")
            print("🔄 Running incremental update (only new products)...")
            return embed_products_incremental(catalog_file, index_name)
        elif current_count > 0 and not incremental:
            print(f"📊 Index already has {current_count} products")
            user_input = input("Do you want to clear and re-embed? (y/N): ").strip().lower()
            if user_input == 'y':
                print("🗑️  Clearing existing index...")
                clear_index(index_name)
            else:
                print("✅ Keeping existing data")
                return
        else:
            print("📊 Index is empty, proceeding with full embedding...")
    except Exception as e:
        print(f"⚠️  Could not check index stats: {e}")
        print("📊 Proceeding with full embedding...")
    
    # Load products
    products = []
    with open(catalog_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                product = json.loads(line.strip())
                products.append(product)
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing line {line_num}: {e}")
                continue
    
    print(f"📊 Loaded {len(products)} products from catalog")
    
    # Batch size for bulk operations - reduced for large embedding vectors
    BATCH_SIZE = 100  # Smaller batches to avoid timeout with large embedding vectors
    BULK_CHUNK_SIZE = 50  # Split bulk operations into smaller chunks
    total_added = 0
    MAX_WORKERS = 7  # Parallel workers (reduced to avoid throttling)
    
    print(f"⚡ Using {MAX_WORKERS} parallel workers for embeddings")
    
    for batch_start in range(0, len(products), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(products))
        batch = products[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_start//BATCH_SIZE + 1} (items {batch_start+1}-{batch_end})")
        
        # Process products in parallel using ThreadPoolExecutor
        results = {}
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_single_product_for_embedding, product): i 
                             for i, product in enumerate(batch)}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                completed += 1
                index = future_to_index[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        results[index] = result
                    
                    # Progress update every 100 items
                    if completed % 100 == 0 or completed == len(batch):
                        print(f"   ⚡ Processed {completed}/{len(batch)} embeddings...")
                except Exception as e:
                    print(f"❌ Error processing product: {e}")
        
        # Sort by index to maintain order
        sorted_results = [results[i] for i in sorted(results.keys())]
        
        # Prepare and execute bulk API request in smaller chunks to avoid timeout
        if sorted_results:
            try:
                # Split into smaller chunks for bulk operations
                for chunk_start in range(0, len(sorted_results), BULK_CHUNK_SIZE):
                    chunk_end = min(chunk_start + BULK_CHUNK_SIZE, len(sorted_results))
                    chunk = sorted_results[chunk_start:chunk_end]
                    
                    bulk_body = []
                    for doc in chunk:
                        # Action and metadata
                        bulk_body.append({
                            "index": {
                                "_index": index_name,
                                "_id": doc['product_id']
                            }
                        })
                        # Document source
                        bulk_body.append(doc)
                    
                    # Execute bulk operation with retry logic
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Use request_timeout parameter for bulk operations
                            response = client.bulk(body=bulk_body, refresh=False, request_timeout=300)
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt) + random.uniform(0, 1)
                                print(f"⚠️  Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s...")
                                time.sleep(wait_time)
                            else:
                                raise
                    
                    # Check for errors
                    if response.get('errors'):
                        error_count = sum(1 for item in response['items'] if 'error' in item.get('index', {}))
                        if error_count > 0:
                            print(f"⚠️  {error_count} errors in bulk chunk")
                            for item in response['items']:
                                if 'error' in item.get('index', {}):
                                    print(f"   ❌ Error indexing {item['index']['_id']}: {item['index']['error']}")
                    
                    success_count = len([item for item in response['items'] if 'error' not in item.get('index', {})])
                    total_added += success_count
                
                print(f"✅ Successfully added batch to OpenSearch (Total: {total_added}/{len(products)})")
                
            except Exception as e:
                print(f"❌ Error adding batch: {e}")
                print(f"⚠️  Partial progress: {total_added} products added so far")
                # Don't raise - continue with next batch
                print(f"⚠️  Continuing with next batch...")
    
    # Refresh index to make documents searchable
    print("🔄 Refreshing index...")
    client.indices.refresh(index=index_name)
    
    print(f"\n🎉 Successfully added {total_added} products to OpenSearch")

def embed_products_incremental(catalog_file: str = 'data/catalog.jsonl', index_name: str = 'products'):
    """Incrementally embed only new products to OpenSearch"""
    global client
    if client is None:
        client = get_opensearch_client()
    
    print("🔄 Starting incremental embedding process...")
    
    # Get existing product IDs
    try:
        # Use scroll API for large result sets
        existing_ids = set()
        response = client.search(
            index=index_name,
            body={
                "size": 10000,
                "_source": ["product_id"],
                "query": {"match_all": {}}
            },
            scroll='2m'
        )
        
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']
        
        while hits:
            for hit in hits:
                existing_ids.add(hit['_source']['product_id'])
            
            if scroll_id:
                response = client.scroll(scroll_id=scroll_id, scroll='2m')
                hits = response['hits']['hits']
            else:
                break
        
        print(f"📊 Found {len(existing_ids)} existing products")
    except Exception as e:
        print(f"⚠️  Could not get existing products: {e}")
        existing_ids = set()
    
    # Load all products from catalog
    all_products = []
    with open(catalog_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                product = json.loads(line.strip())
                all_products.append(product)
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing line {line_num}: {e}")
                continue
    
    print(f"📊 Loaded {len(all_products)} products from catalog")
    
    # Find new products (not in existing index)
    new_products = []
    for product in all_products:
        product_id = str(product['product_id'])
        if product_id not in existing_ids:
            new_products.append(product)
    
    print(f"🆕 Found {len(new_products)} new products to embed")
    
    if not new_products:
        print("✅ No new products to embed")
        return
    
    # Batch size for bulk operations - reduced for large embedding vectors
    BATCH_SIZE = 100  # Smaller batches to avoid timeout with large embedding vectors
    BULK_CHUNK_SIZE = 50  # Split bulk operations into smaller chunks
    total_added = 0
    MAX_WORKERS = 7
    
    print(f"⚡ Using {MAX_WORKERS} parallel workers for embeddings")
    
    for batch_start in range(0, len(new_products), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(new_products))
        batch = new_products[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_start//BATCH_SIZE + 1} (items {batch_start+1}-{batch_end})")
        
        # Process products in parallel
        results = {}
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {executor.submit(process_single_product_for_embedding, product): i 
                             for i, product in enumerate(batch)}
            
            completed = 0
            for future in as_completed(future_to_index):
                completed += 1
                index = future_to_index[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        results[index] = result
                    
                    if completed % 100 == 0 or completed == len(batch):
                        print(f"   ⚡ Processed {completed}/{len(batch)} embeddings...")
                except Exception as e:
                    print(f"❌ Error processing product: {e}")
        
        # Sort by index to maintain order
        sorted_results = [results[i] for i in sorted(results.keys())]
        
        # Prepare and execute bulk API request in smaller chunks to avoid timeout
        if sorted_results:
            try:
                # Split into smaller chunks for bulk operations
                for chunk_start in range(0, len(sorted_results), BULK_CHUNK_SIZE):
                    chunk_end = min(chunk_start + BULK_CHUNK_SIZE, len(sorted_results))
                    chunk = sorted_results[chunk_start:chunk_end]
                    
                    bulk_body = []
                    for doc in chunk:
                        bulk_body.append({
                            "index": {
                                "_index": index_name,
                                "_id": doc['product_id']
                            }
                        })
                        bulk_body.append(doc)
                    
                    # Execute bulk operation with retry logic
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Use request_timeout parameter for bulk operations
                            response = client.bulk(body=bulk_body, refresh=False, request_timeout=300)
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt) + random.uniform(0, 1)
                                print(f"⚠️  Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s...")
                                time.sleep(wait_time)
                            else:
                                raise
                    
                    if response.get('errors'):
                        error_count = sum(1 for item in response['items'] if 'error' in item.get('index', {}))
                        if error_count > 0:
                            print(f"⚠️  {error_count} errors in bulk chunk")
                    
                    success_count = len([item for item in response['items'] if 'error' not in item.get('index', {})])
                    total_added += success_count
                
                print(f"✅ Successfully added batch to OpenSearch (Total: {total_added}/{len(new_products)})")
                
            except Exception as e:
                print(f"❌ Error adding batch: {e}")
                print(f"⚠️  Partial progress: {total_added} products added so far")
                # Don't raise - continue with next batch
                print(f"⚠️  Continuing with next batch...")
    
    # Refresh index
    print("🔄 Refreshing index...")
    client.indices.refresh(index=index_name)
    
    # Get final count
    try:
        stats = client.indices.stats(index=index_name)
        final_count = stats['indices'][index_name]['total']['docs']['count']
        print(f"\n🎉 Successfully added {total_added} new products to OpenSearch")
        print(f"📊 Total products in index: {final_count}")
    except Exception as e:
        print(f"\n🎉 Successfully added {total_added} new products to OpenSearch")
        print(f"⚠️  Could not get final index count: {e}")

def search_products_opensearch(query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None, index_name: str = "products") -> tuple[List[Dict[str, Any]], float]:
    """Search products using OpenSearch
    
    Args:
        query: Search query string
        k: Number of results to return
        filters: Optional filters to apply
        index_name: Name of the index to search (defaults to "products")
    
    Returns:
        Tuple of (products list, search time in ms)
    """
    global client
    if client is None:
        client = get_opensearch_client()
    
    start_time = time.time()
    
    # Generate embedding
    query_embedding = create_embedding(query)
    if query_embedding is None:
        print(f"❌ Failed to generate embedding for query: '{query}'")
        return [], 0.0
    
    print(f"🔍 Searching OpenSearch for: '{query}'")
    
    # Build search query
    search_body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": k
                }
            }
        }
    }
    
    # Add filters if provided
    if filters:
        from utils.util_funcs import convert_filters_to_opensearch
        opensearch_filters = convert_filters_to_opensearch(filters)
        
        if opensearch_filters:
            # Combine KNN with filter query
            search_body["query"] = {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": k
                                }
                            }
                        }
                    ],
                    "filter": []
                }
            }
            
            # Add filters
            if "price" in opensearch_filters:
                price_filter = opensearch_filters["price"]
                if "$gte" in price_filter and "$lte" in price_filter:
                    search_body["query"]["bool"]["filter"].append({
                        "range": {
                            "price": {
                                "gte": price_filter["$gte"],
                                "lte": price_filter["$lte"]
                            }
                        }
                    })
                elif "$gte" in price_filter:
                    search_body["query"]["bool"]["filter"].append({
                        "range": {"price": {"gte": price_filter["$gte"]}}
                    })
                elif "$lte" in price_filter:
                    search_body["query"]["bool"]["filter"].append({
                        "range": {"price": {"lte": price_filter["$lte"]}}
                    })
            
            if "in_stock" in opensearch_filters:
                search_body["query"]["bool"]["filter"].append({
                    "term": {"in_stock": opensearch_filters["in_stock"]}
                })
            
            if "category" in opensearch_filters:
                search_body["query"]["bool"]["filter"].append({
                    "term": {"category": opensearch_filters["category"]}
                })
            
            if "tags" in opensearch_filters:
                if isinstance(opensearch_filters["tags"], list):
                    search_body["query"]["bool"]["filter"].append({
                        "terms": {"tags": opensearch_filters["tags"]}
                    })
                else:
                    search_body["query"]["bool"]["filter"].append({
                        "term": {"tags": opensearch_filters["tags"]}
                    })
    
    # Search OpenSearch
    try:
        response = client.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        
        print(f"🔍 OpenSearch query results: {len(hits)} items found")
    except Exception as e:
        print(f"❌ OpenSearch search error: {e}")
        return [], 0.0
    
    # Format results
    products = []
    for hit in hits:
        source = hit['_source']
        product = {
            'product_id': source['product_id'],
            'title': source['title'],
            'price': source['price'],
            'url': source['url'],
            'image': source['image'],
            'in_stock': source['in_stock'],
            'category': source['category'],
            'tags': source['tags'] if isinstance(source['tags'], list) else [source['tags']],
            'score': hit['_score']
        }
        products.append(product)
    
    search_time = (time.time() - start_time) * 1000
    return products, search_time

def get_index_stats(index_name: str) -> Dict[str, Any]:
    """Get OpenSearch index statistics"""
    global client
    if client is None:
        client = get_opensearch_client()
    
    try:
        stats = client.indices.stats(index=index_name)
        doc_count = stats['indices'][index_name]['total']['docs']['count']
        return {
            "status": "healthy",
            "index_name": index_name,
            "total_products": doc_count
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def list_all_indices() -> List[Dict[str, Any]]:
    """List all indices in OpenSearch with their statistics"""
    global client
    if client is None:
        client = get_opensearch_client()
    
    try:
        # Use cat.indices() API which is more reliable across versions
        indices_info = client.cat.indices(format='json')
        index_list = []
        
        for index_info in indices_info:
            # Skip system indices (starting with .)
            index_name = index_info.get('index', '')
            if index_name.startswith('.'):
                continue
                
            try:
                doc_count = int(index_info.get('docs.count', 0))
                index_list.append({
                    "name": index_name,
                    "count": doc_count
                })
            except Exception as e:
                index_list.append({
                    "name": index_name,
                    "count": "error",
                    "error": str(e)
                })
        
        return index_list
    except Exception as e:
        print(f"❌ Error listing indices: {e}")
        return []

def debug_index(index_name: str):
    """Debug function to check what's in the index"""
    global client
    if client is None:
        client = get_opensearch_client()
    
    try:
        stats = client.indices.stats(index=index_name)
        doc_count = stats['indices'][index_name]['total']['docs']['count']
        print(f"📊 Index has {doc_count} products")
        
        if doc_count > 0:
            # Get a sample of products
            response = client.search(
                index=index_name,
                body={
                    "size": 3,
                    "query": {"match_all": {}}
                }
            )
            print("📋 Sample products:")
            for i, hit in enumerate(response['hits']['hits'], 1):
                source = hit['_source']
                print(f"  {i}. {source['product_id']}: {source['title']} (${source['price']})")
        else:
            print("❌ Index is empty!")
            
    except Exception as e:
        print(f"❌ Error checking index: {e}")

def clear_index(index_name: str):
    """Clear all products from OpenSearch index"""
    global client
    if client is None:
        client = get_opensearch_client()
    
    try:
        # Delete and recreate index
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
        client.indices.create(index=index_name, body=get_index_body())
        print(f"✅ OpenSearch index '{index_name}' cleared")
        return True
    except Exception as e:
        print(f"❌ Error clearing index: {e}")
        return False

if __name__ == "__main__":
    print("🧠 Welcome to the OpenSearch Product Embedder\n")
    
    # Initialize OpenSearch client
    try:
        client = get_opensearch_client()
        info = client.info()
        print(f"✅ Connected to OpenSearch: {info['version']['number']}\n")
    except Exception as e:
        print(f"❌ Failed to connect to OpenSearch: {e}")
        exit(1)
    
    # --- STEP 1: Show available JSONL files ---
    data_dir = "./data"
    print(f"📂 Looking for product JSONL files in: {data_dir}\n")
    
    available_files = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".jsonl"):
                available_files.append(os.path.join(data_dir, f))
    
    if not available_files:
        print("❌ No JSONL files found. Please place your product files in ./data/")
        exit(1)
    
    print("📄 Available JSONL files:")
    for idx, file_path in enumerate(available_files, start=1):
        print(f"  {idx}. {os.path.basename(file_path)}")
    
    file_choice = input("\n👉 Enter the number of the file to use: ").strip()
    try:
        file_idx = int(file_choice) - 1
        catalog_file = available_files[file_idx]
    except (ValueError, IndexError):
        print("❌ Invalid choice. Exiting.")
        exit(1)
    
    print(f"\n✅ Selected file: {catalog_file}\n")
    
    # --- STEP 2: Show existing indices ---
    try:
        indices = list_all_indices()
        if indices:
            print("📦 Existing indices:")
            for idx, index_info in enumerate(indices, start=1):
                count = index_info.get('count', 'unknown')
                print(f"  {idx}. {index_info['name']} ({count} documents)")
        else:
            print("⚠️  No existing indices found.")
    except Exception as e:
        print(f"⚠️  Could not list indices: {e}")
        indices = []
    
    # Ask user whether to use existing or new index
    new_or_existing = input("\n➕ Create new index or use existing? (new/existing): ").strip().lower()
    
    if new_or_existing == "existing" and indices:
        col_choice = input("👉 Enter the number of the index to use: ").strip()
        try:
            col_idx = int(col_choice) - 1
            INDEX_NAME = indices[col_idx]['name']
        except (ValueError, IndexError):
            print("❌ Invalid choice. Exiting.")
            exit(1)
    else:
        INDEX_NAME = input("🆕 Enter new index name: ").strip()
        if not INDEX_NAME:
            print("❌ Index name cannot be empty. Exiting.")
            exit(1)
    
    # Ensure index exists
    print(f"\n🚀 Using index: {INDEX_NAME}")
    ensure_index_exists(INDEX_NAME)
    
    # --- STEP 3: Confirm and run embedding ---
    print(f"\n📥 You are about to embed data from '{catalog_file}' into '{INDEX_NAME}' index.")
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ Aborted by user.")
        exit(0)
    
    embed_products_to_opensearch(catalog_file=catalog_file, index_name=INDEX_NAME)

