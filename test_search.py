#!/usr/bin/env python3
"""
Test search functionality with sample queries
"""
import os
import json
import time
from dotenv import load_dotenv
import boto3
from opensearch import client as opensearch_client

# Load environment variables
load_dotenv()

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-2'))

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

def search_products(query, k=5, filters=None):
    """Search for products using vector similarity"""
    
    print(f"🔍 Searching for: '{query}'")
    
    # Generate query embedding
    query_embedding = create_embedding(query)
    if query_embedding is None:
        return []
    
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
        
        for field, value in filters.items():
            if field == "in_stock":
                search_body["query"]["bool"]["filter"].append({
                    "term": {field: value}
                })
            elif field == "category":
                search_body["query"]["bool"]["filter"].append({
                    "term": {field: value}
                })
            elif field == "price":
                if "$lte" in value:
                    search_body["query"]["bool"]["filter"].append({
                        "range": {"price": {"lte": value["$lte"]}}
                    })
    
    # Execute search
    start_time = time.time()
    response = opensearch_client.search(
        index='products',
        body=search_body
    )
    search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Format results
    results = []
    for hit in response['hits']['hits']:
        product = hit['_source']
        results.append({
            "product_id": product["product_id"],
            "title": product["title"],
            "price": product["price"],
            "url": product["url"],
            "image": product["image"],
            "in_stock": product["in_stock"],
            "category": product["category"],
            "tags": product["tags"],
            "score": hit["_score"]
        })
    
    return results, search_time

def run_tests():
    """Run test queries and display results"""
    
    print("🧪 Testing AI Search System...")
    
    # Test queries
    test_queries = [
        {
            "query": "black running shorts under $60",
            "filters": {"in_stock": True},
            "k": 5
        },
        {
            "query": "women's athletic wear",
            "filters": {"category": "tops"},
            "k": 3
        },
        {
            "query": "waterproof shoes for hiking",
            "filters": {"in_stock": True},
            "k": 2
        },
        {
            "query": "yoga accessories",
            "filters": {"category": "accessories"},
            "k": 4
        }
    ]
    
    print("\n🔍 Testing Search Queries:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_queries):
        query = test_case['query']
        filters = test_case.get('filters', {})
        k = test_case.get('k', 5)
        
        print(f"\n{i+1}. Query: '{query}'")
        print(f"   Filters: {filters}")
        
        results, search_time = search_products(query, k, filters)
        
        print(f"   ⏱️  Search time: {search_time:.2f}ms")
        print(f"   📊 Results found: {len(results)}")
        
        for j, result in enumerate(results[:3]):  # Show top 3
            print(f"   {j+1}. {result['title']} - ${result['price']} ({result['category']})")
            print(f"      Score: {result['score']:.4f}")
            print(f"      In Stock: {result['in_stock']}")
    
    print("\n🎉 Search testing complete!")
    print(f"✅ All queries executed successfully")

if __name__ == "__main__":
    run_tests()
