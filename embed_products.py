#!/usr/bin/env python3
"""
Embed products from catalog.jsonl and upsert to OpenSearch
Uses incremental processing to avoid re-embedding existing products
"""

import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import boto3
from config.opensearch import client as opensearch_client


# Load environment variables
load_dotenv()

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))

def create_embedding(text, model_id='amazon.titan-embed-text-v1'):
    """Generate embedding for text using Amazon Bedrock"""
    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "inputText": text
            })
        )
        
        result = json.loads(response['body'].read())
        return result['embedding']
    
    except Exception as e:
        print(f"❌ Error creating embedding: {e}")
        return None

def get_processed_products():
    """Get list of already processed product IDs from OpenSearch"""
    try:
        # Get all existing product IDs
        response = opensearch_client.search(
            index='products',
            body={
                "size": 10000,  # Adjust based on your catalog size
                "_source": ["product_id"],
                "query": {"match_all": {}}
            }
        )
        
        processed_ids = set()
        for hit in response['hits']['hits']:
            processed_ids.add(hit['_source']['product_id'])
        
        return processed_ids
    
    except Exception as e:
        print(f"⚠️  Could not get processed products: {e}")
        return set()

def build_embedding_text(product):
    """Build text for embedding: title + text + category + tags"""
    # Primary content
    text_parts = [product['title'], product['text']]
    
    # Add category for better context
    if 'category' in product:
        text_parts.append(f"Category: {product['category']}")
    
    # Add top tags for better matching
    if 'tags' and len(product['tags']) > 0:
        tags_text = ", ".join(product['tags'][:5])  # Limit to top 5 tags
        text_parts.append(f"Tags: {tags_text}")
    
    return " ".join(text_parts)

def process_single_product(product):
    """Process a single product: generate embedding and return document"""
    try:
        product_id = product['product_id']
        
        # Build embedding text
        embedding_text = build_embedding_text(product)
        
        # Generate embedding
        embedding = create_embedding(embedding_text)
        
        if embedding is None:
            print(f"❌ Failed to generate embedding for {product_id}")
            return None
        
        # Prepare document for OpenSearch
        doc = {
            "product_id": product["product_id"],
            "title": product["title"],
            "text": product["text"],
            "price": product["price"],
            "url": product["url"],
            "image": product["image"],
            "in_stock": product["in_stock"],
            "category": product["category"],
            "tags": product["tags"],
            "embedding": embedding
        }
        
        return doc
        
    except Exception as e:
        print(f"❌ Error processing {product['product_id']}: {e}")
        return None

def process_catalog(max_workers=10):
    """Process catalog.jsonl and embed products with parallel processing"""
    
    print("🔄 Starting product embedding process with parallel processing...")
    print(f"⚡ Using {max_workers} parallel workers")
    
    start_time = time.time()
    
    # Get already processed products
    processed_ids = get_processed_products()
    print(f"📊 Found {len(processed_ids)} already processed products")
    
    # Load catalog
    products = []
    with open('data/catalog.jsonl', 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    product = json.loads(line.strip())
                    products.append(product)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping invalid JSON on line {line_num}: {e}")
                    continue
    
    print(f"📦 Loaded {len(products)} products from catalog")
    
    # Filter out already processed products
    products_to_process = [p for p in products if p['product_id'] not in processed_ids]
    print(f"🆕 Found {len(products_to_process)} new products to process")
    
    if not products_to_process:
        print("✅ No new products to process!")
        return
    
    # Process products in parallel
    new_count = 0
    error_count = 0
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_product = {executor.submit(process_single_product, product): product 
                            for product in products_to_process}
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_product), 1):
            product = future_to_product[future]
            
            try:
                doc = future.result()
                
                if doc is None:
                    error_count += 1
                    continue
                
                # Upsert to OpenSearch
                opensearch_client.index(
                    index='products',
                    id=doc['product_id'],
                    body=doc
                )
                
                print(f"✅ [{i}/{len(products_to_process)}] Indexed: {doc['title'][:50]}...")
                new_count += 1
                
            except Exception as e:
                print(f"❌ Error indexing {product['product_id']}: {e}")
                error_count += 1
    
    # Refresh index to make documents searchable
    print("🔄 Refreshing index...")
    opensearch_client.indices.refresh(index='products')
    
    # Calculate time taken
    elapsed_time = time.time() - start_time
    
    # Print summary
    skipped_count = len(products) - len(products_to_process)
    print(f"\n🎉 Processing complete!")
    print(f"⚡ Total time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} minutes)")
    print(f"⚡ Average time per product: {elapsed_time/len(products_to_process):.2f}s")
    print(f"📊 Summary:")
    print(f"   ✅ New products: {new_count}")
    print(f"   ⏭️  Skipped (already processed): {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📦 Total processed: {len(products_to_process)}")
    
    # Get final index stats
    try:
        stats = opensearch_client.indices.stats(index='products')
        doc_count = stats['indices']['products']['total']['docs']['count']
        print(f"   📈 Total documents in index: {doc_count}")
    except Exception as e:
        print(f"⚠️  Could not get index stats: {e}")

if __name__ == "__main__":
    process_catalog()
