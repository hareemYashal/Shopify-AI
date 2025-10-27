#!/usr/bin/env python3
"""
ChromaDB service for ultra-fast vector search
Provides 90%+ faster search compared to OpenSearch
"""

import os
import json
import time
import random
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import boto3

# Use shared ChromaDB client to avoid settings conflicts
from chroma_utils import get_chroma_client

# Load environment variables
load_dotenv()

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))

# Initialize ChromaDB using shared client
client = get_chroma_client()

# Create or get collection
COLLECTION_NAME = "products"
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

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
    """Process a single product: generate embedding and return data for ChromaDB"""
    try:
        # Build text for embedding
        text_for_embedding = build_text_for_embedding(product)
        
        # Generate embedding
        embedding = create_embedding(text_for_embedding)
        if embedding is None:
            print(f"❌ Failed to generate embedding for {product['product_id']}")
            return None
        
        # Return data ready for ChromaDB
        return {
            'id': str(product['product_id']),
            'embedding': embedding,
            'metadata': {
                'title': product['title'],
                'price': product['price'],
                'url': product['url'],
                'image': product['image'],
                'in_stock': product['in_stock'],
                'category': product['category'],
                'tags': ', '.join(product['tags']) if isinstance(product['tags'], list) else product['tags']
            }
        }
    except Exception as e:
        print(f"❌ Error processing {product.get('product_id', 'unknown')}: {e}")
        return None

def embed_products_to_chroma(catalog_file: str = 'data/catalog.jsonl', incremental: bool = True):
    """Embed products from catalog.jsonl to ChromaDB (incremental by default)"""
    print("🔄 Starting ChromaDB embedding process...")
    
    # Check if collection already has data
    current_count = collection.count()
    if current_count > 0 and incremental:
        print(f"📊 Collection already has {current_count} products")
        print("🔄 Running incremental update (only new products)...")
        return embed_products_incremental(catalog_file)
    elif current_count > 0 and not incremental:
        print(f"📊 Collection already has {current_count} products")
        user_input = input("Do you want to clear and re-embed? (y/N): ").strip().lower()
        if user_input == 'y':
            print("🗑️  Clearing existing collection...")
            clear_collection()
        else:
            print("✅ Keeping existing data")
            return
    else:
        print("📊 Collection is empty, proceeding with full embedding...")
    
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
    
    # Batch size to avoid ChromaDB limits (max batch size is ~5000)
    BATCH_SIZE = 1000
    total_added = 0
    MAX_WORKERS = 7  # Parallel workers (reduced to avoid throttling)
    
    print(f"⚡ Using {MAX_WORKERS} parallel workers for embeddings")
    
    for batch_start in range(0, len(products), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(products))
        batch = products[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_start//BATCH_SIZE + 1} (items {batch_start+1}-{batch_end})")
        
        # Process products in parallel using ThreadPoolExecutor
        ids = []
        embeddings = []
        metadatas = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_single_product_for_embedding, product): i 
                             for i, product in enumerate(batch)}
            
            # Collect results as they complete
            results = {}
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
        
        for result in sorted_results:
            ids.append(result['id'])
            embeddings.append(result['embedding'])
            metadatas.append(result['metadata'])
        
        # Add batch to ChromaDB
        if ids:
            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                total_added += len(ids)
                print(f"✅ Successfully added {len(ids)} products to ChromaDB (Total: {total_added}/{len(products)})")
            except Exception as e:
                print(f"❌ Error adding batch: {e}")
                print(f"⚠️  Partial progress: {total_added} products added so far")
                raise
    
    print(f"\n🎉 Successfully added {total_added} products to ChromaDB")

def embed_products_incremental(catalog_file: str = 'data/catalog.jsonl'):
    """Incrementally embed only new products to ChromaDB"""
    print("🔄 Starting incremental embedding process...")
    
    # Get existing product IDs
    try:
        existing_data = collection.get()
        existing_ids = set(existing_data['ids']) if existing_data['ids'] else set()
        print(f"📊 Found {len(existing_ids)} existing products")
    except Exception as e:
        print(f"❌ Error getting existing products: {e}")
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
    
    # Find new products (not in existing collection)
    new_products = []
    for product in all_products:
        product_id = str(product['product_id'])
        if product_id not in existing_ids:
            new_products.append(product)
    
    print(f"🆕 Found {len(new_products)} new products to embed")
    
    if not new_products:
        print("✅ No new products to embed")
        return
    
    # Batch size to avoid ChromaDB limits
    BATCH_SIZE = 1000
    total_added = 0
    MAX_WORKERS = 7  # Parallel workers (reduced to avoid throttling)
    
    print(f"⚡ Using {MAX_WORKERS} parallel workers for embeddings")
    
    for batch_start in range(0, len(new_products), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(new_products))
        batch = new_products[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_start//BATCH_SIZE + 1} (items {batch_start+1}-{batch_end})")
        
        # Process products in parallel using ThreadPoolExecutor
        ids = []
        embeddings = []
        metadatas = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_single_product_for_embedding, product): i 
                             for i, product in enumerate(batch)}
            
            # Collect results as they complete
            results = {}
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
        
        for result in sorted_results:
            ids.append(result['id'])
            embeddings.append(result['embedding'])
            metadatas.append(result['metadata'])
        
        # Add batch to ChromaDB
        if ids:
            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                total_added += len(ids)
                print(f"✅ Successfully added {len(ids)} products to ChromaDB (Total: {total_added}/{len(new_products)})")
            except Exception as e:
                print(f"❌ Error adding batch: {e}")
                print(f"⚠️  Partial progress: {total_added} products added so far")
                raise
    
    print(f"\n🎉 Successfully added {total_added} new products to ChromaDB")
    print(f"📊 Total products in collection: {collection.count()}")

def search_products_chroma(query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None, collection_name: str = "products") -> tuple[List[Dict[str, Any]], float]:
    """Search products using ChromaDB for maximum speed
    
    Args:
        query: Search query string
        k: Number of results to return
        filters: Optional filters to apply
        collection_name: Name of the collection to search (defaults to "products")
    
    Returns:
        Tuple of (products list, search time in ms)
    """
    start_time = time.time()
    
    # Generate embedding
    query_embedding = create_embedding(query)
    if query_embedding is None:
        print(f"❌ Failed to generate embedding for query: '{query}'")
        return [], 0.0
    
    # Get the specified collection
    try:
        search_collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"❌ Error getting collection '{collection_name}': {e}")
        return [], 0.0
    
    print(f"🔍 Searching ChromaDB for: '{query}'")
    print(f"📊 Collection: '{collection_name}' (count: {search_collection.count()})")
    
    # Convert filters to ChromaDB format
    where_clause = None
    if filters:
        where_clause = convert_filters_to_chroma(filters)
        print(f"🔍 ChromaDB filters: {where_clause}")
    
    # Search ChromaDB
    try:
        results = search_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause
        )
        print(f"🔍 ChromaDB query results: {len(results.get('ids', [[]])[0])} items found")
    except Exception as e:
        print(f"❌ ChromaDB search error: {e}")
        return [], 0.0
    
    # Format results
    products = []
    if results['ids'] and results['ids'][0]:
        for i, product_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            product = {
                'product_id': product_id,
                'title': metadata['title'],
                'price': metadata['price'],
                'url': metadata['url'],
                'image': metadata['image'],
                'in_stock': metadata['in_stock'],
                'category': metadata['category'],
                'tags': metadata['tags'].split(', ') if metadata['tags'] else [],  # Convert string back to list
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity
            }
            products.append(product)
    
    search_time = (time.time() - start_time) * 1000
    return products, search_time

def convert_filters_to_chroma(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Convert parsed filters to ChromaDB-compatible format with advanced filtering"""
    filter_conditions = []
    
    for field, value in filters.items():
        if field == "in_stock":
            # Handle in_stock filter
            if isinstance(value, dict) and "eq" in value:
                filter_conditions.append({"in_stock": {"$eq": value["eq"]}})
            else:
                filter_conditions.append({"in_stock": {"$eq": value}})
        elif field == "category":
            filter_conditions.append({"category": {"$eq": value}})
        elif field == "price":
            if isinstance(value, dict):
                if "gte" in value and "lte" in value:
                    # Both gte and lte - need $and
                    filter_conditions.append({
                        "$and": [
                            {"price": {"$gte": value["gte"]}},
                            {"price": {"$lte": value["lte"]}}
                        ]
                    })
                elif "gte" in value:
                    filter_conditions.append({"price": {"$gte": value["gte"]}})
                elif "lte" in value:
                    filter_conditions.append({"price": {"$lte": value["lte"]}})
        elif field == "tags":
            # For tag filtering, we'll use string contains
            if isinstance(value, list):
                if len(value) > 1:
                    filter_conditions.append({
                        "$or": [{"tags": {"$contains": tag}} for tag in value]
                    })
                else:
                    filter_conditions.append({"tags": {"$contains": value[0]}})
            else:
                filter_conditions.append({"tags": {"$contains": value}})
    
    # If we have multiple conditions, wrap in $and
    if len(filter_conditions) == 0:
        return None
    elif len(filter_conditions) == 1:
        return filter_conditions[0]
    else:
        return {"$and": filter_conditions}

def check_persistence():
    """Check if ChromaDB persistence is working correctly"""
    import os
    
    print("🔍 Checking ChromaDB persistence...")
    
    # Check if directory exists
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        print(f"✅ Database directory exists: {db_path}")
        
        # List files in directory
        files = os.listdir(db_path)
        print(f"📁 Files in database: {files}")
        
        if files:
            print("✅ Database has files - persistence is working")
        else:
            print("⚠️  Database directory is empty")
    else:
        print(f"❌ Database directory does not exist: {db_path}")
        print("🔄 Creating database directory...")
        os.makedirs(db_path, exist_ok=True)
    
    # Check collection count
    try:
        count = collection.count()
        print(f"📊 Collection has {count} products")
        
        if count > 0:
            print("✅ Data is persisted and accessible")
        else:
            print("⚠️  Collection is empty - may need to re-embed")
            
    except Exception as e:
        print(f"❌ Error accessing collection: {e}")

def get_collection_stats() -> Dict[str, Any]:
    """Get ChromaDB collection statistics"""
    try:
        count = collection.count()
        return {
            "status": "healthy",
            "collection_name": COLLECTION_NAME,
            "total_products": count,
            "persist_directory": "./chroma_db"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def list_all_collections() -> List[Dict[str, Any]]:
    """List all collections in ChromaDB with their statistics"""
    try:
        collections = client.list_collections()
        collection_list = []
        
        for col in collections:
            try:
                count = col.count()
                metadata = col.metadata or {}
                collection_list.append({
                    "name": col.name,
                    "count": count,
                    "metadata": metadata
                })
            except Exception as e:
                collection_list.append({
                    "name": col.name,
                    "count": "error",
                    "metadata": {},
                    "error": str(e)
                })
        
        return collection_list
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []

def debug_collection():
    """Debug function to check what's in the collection"""
    try:
        count = collection.count()
        print(f"📊 Collection has {count} products")
        
        if count > 0:
            # Get a sample of products
            sample = collection.get(limit=3)
            print("📋 Sample products:")
            for i, product_id in enumerate(sample['ids']):
                metadata = sample['metadatas'][i]
                print(f"  {i+1}. {product_id}: {metadata['title']} (${metadata['price']})")
        else:
            print("❌ Collection is empty!")
            
    except Exception as e:
        print(f"❌ Error checking collection: {e}")

def clear_collection():
    """Clear all products from ChromaDB collection"""
    try:
        # Delete and recreate collection
        client.delete_collection(COLLECTION_NAME)
        global collection
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ ChromaDB collection cleared")
        return True
    except Exception as e:
        print(f"❌ Error clearing collection: {e}")
        return False

def test_chroma_filters():
    """Test ChromaDB filtering capabilities"""
    print("🧪 Testing ChromaDB filtering capabilities...")
    
    # Test cases
    test_cases = [
        {
            "name": "Price Range Filter",
            "query": "shoes",
            "filters": {"price": {"gte": 50, "lte": 200}},
            "description": "Find shoes between $50-$200"
        },
        {
            "name": "Category Filter",
            "query": "clothing",
            "filters": {"category": "clothing"},
            "description": "Find clothing items"
        },
        {
            "name": "Stock Filter",
            "query": "electronics",
            "filters": {"in_stock": True},
            "description": "Find in-stock electronics"
        },
        {
            "name": "Tag Filter",
            "query": "workout",
            "filters": {"tags": ["athletic"]},
            "description": "Find athletic workout items"
        },
        {
            "name": "Combined Filters",
            "query": "kitchen",
            "filters": {
                "category": "kitchen-dining",
                "price": {"lte": 300},
                "in_stock": True
            },
            "description": "Find in-stock kitchen items under $300"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🔍 {test_case['name']}: {test_case['description']}")
        start_time = time.time()
        results, search_time = search_products_chroma(
            query=test_case['query'],
            k=5,
            filters=test_case['filters']
        )
        total_time = (time.time() - start_time) * 1000
        
        print(f"  Query: '{test_case['query']}'")
        print(f"  Filters: {test_case['filters']}")
        print(f"  Results: {len(results)}")
        print(f"  Search time: {search_time:.2f}ms")
        print(f"  Total time: {total_time:.2f}ms")
        
        if results:
            print(f"  Top result: {results[0]['title']} (${results[0]['price']})")

def test_chroma_performance():
    """Test ChromaDB performance with sample queries"""
    test_queries = [
        "running shoes",
        "black shorts",
        "electronics under $100",
        "workout clothes in stock",
        "kitchen cookware"
    ]
    
    print("🚀 Testing ChromaDB performance...")
    
    for query in test_queries:
        start_time = time.time()
        results, search_time = search_products_chroma(query, k=5)
        total_time = (time.time() - start_time) * 1000
        
        print(f"Query: '{query}'")
        print(f"  Search time: {search_time:.2f}ms")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Results: {len(results)}")
        print()

if __name__ == "__main__":
    print("🧠 Welcome to the ChromaDB Product Embedder\n")

    persist_path = "./chroma_db"
    client = get_chroma_client(persist_path)

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

    # --- STEP 2: Show existing collections ---
    collections = client.list_collections()
    if collections:
        print("📦 Existing collections:")
        for idx, col in enumerate(collections, start=1):
            print(f"  {idx}. {col.name}")
    else:
        print("⚠️ No existing collections found.")

    # Ask user whether to use existing or new collection
    new_or_existing = input("\n➕ Create new collection or use existing? (new/existing): ").strip().lower()

    if new_or_existing == "existing" and collections:
        col_choice = input("👉 Enter the number of the collection to use: ").strip()
        try:
            col_idx = int(col_choice) - 1
            COLLECTION_NAME = collections[col_idx].name
        except (ValueError, IndexError):
            print("❌ Invalid choice. Exiting.")
            exit(1)
    else:
        COLLECTION_NAME = input("🆕 Enter new collection name: ").strip()
        if not COLLECTION_NAME:
            print("❌ Collection name cannot be empty. Exiting.")
            exit(1)

    # Create or load collection
    print(f"\n🚀 Using collection: {COLLECTION_NAME}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # --- STEP 3: Confirm and run embedding ---
    print(f"\n📥 You are about to embed data from '{catalog_file}' into '{COLLECTION_NAME}' collection.")
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ Aborted by user.")
        exit(0)

    embed_products_to_chroma(catalog_file=catalog_file)

