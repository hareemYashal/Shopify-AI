#!/usr/bin/env python3
"""
Embed products from catalog.jsonl and upsert to OpenSearch
Uses incremental processing to avoid re-embedding existing products
"""

import json
import time
import os
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

def process_catalog():
    """Process catalog.jsonl and embed products"""
    
    print("🔄 Starting product embedding process...")
    
    # Get already processed products
    processed_ids = get_processed_products()
    print(f"📊 Found {len(processed_ids)} already processed products")
    
    # Load catalog
    products = []
    with open('data/catalog.jsonl', 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    product = json.loads(line.strip())
                    products.append(product)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping invalid JSON on line {line_num}: {e}")
                    continue
    
    print(f"📦 Loaded {len(products)} products from catalog")
    
    # Process products
    new_count = 0
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, product in enumerate(products):
        product_id = product['product_id']
        
        try:
            # Check if already processed
            if product_id in processed_ids:
                print(f"⏭️  Skipping {i+1}/{len(products)}: {product['title']} (already processed)")
                skipped_count += 1
                continue
            
            print(f"🔄 Processing {i+1}/{len(products)}: {product['title']}")
            
            # Build embedding text
            embedding_text = build_embedding_text(product)
            print(f"   📝 Embedding text: {embedding_text[:100]}...")
            
            # Generate embedding
            embedding = create_embedding(embedding_text)
            
            if embedding is None:
                print(f"❌ Failed to generate embedding for {product_id}")
                error_count += 1
                continue
            
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
            
            # Upsert to OpenSearch
            opensearch_client.index(
                index='products',
                id=product_id,
                body=doc
            )
            
            print(f"✅ Successfully indexed: {product['title']}")
            new_count += 1
            
            # Rate limiting to avoid throttling
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error processing {product_id}: {e}")
            error_count += 1
            continue
    
    # Refresh index to make documents searchable
    opensearch_client.indices.refresh(index='products')
    
    # Print summary
    print(f"\n🎉 Processing complete!")
    print(f"📊 Summary:")
    print(f"   ✅ New products: {new_count}")
    print(f"   ⏭️  Skipped (already processed): {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📦 Total processed: {len(products)}")
    
    # Get final index stats
    try:
        stats = opensearch_client.indices.stats(index='products')
        doc_count = stats['indices']['products']['total']['docs']['count']
        print(f"   📈 Total documents in index: {doc_count}")
    except Exception as e:
        print(f"⚠️  Could not get index stats: {e}")

if __name__ == "__main__":
    process_catalog()
