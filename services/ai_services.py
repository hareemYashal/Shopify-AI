import os
import json
import time
import re
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
import boto3
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config.opensearch import client as opensearch_client
from config.bedrock import client as bedrock


# ================================
# Preferences Loading with Caching
# ================================
_system_prompt_cache = None
_cache_timestamp = None
_cache_file_mtime = None


def load_store_preferences(force_reload: bool = False) -> str:
    """Load store preferences from sys_prompt.txt file with caching"""
    global _system_prompt_cache, _cache_timestamp, _cache_file_mtime
    
    try:
        preferences_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'sys_prompt.txt')
        
        # Check if file has been modified
        if not force_reload and _system_prompt_cache:
            try:
                current_mtime = os.path.getmtime(preferences_path)
                if current_mtime == _cache_file_mtime:
                    # File hasn't changed, return cached version
                    return _system_prompt_cache
            except:
                pass
        
        # Load from file
        with open(preferences_path, 'r', encoding='utf-8') as f:
            _system_prompt_cache = f.read().strip()
            _cache_timestamp = time.time()
            _cache_file_mtime = os.path.getmtime(preferences_path)
        
        return _system_prompt_cache
    except Exception as e:
        print(f"❌ Error loading system prompt: {e}")
        return "Store preferences not available."


def reload_system_prompt_cache() -> bool:
    """Force reload of system prompt from file"""
    try:
        load_store_preferences(force_reload=True)
        print("✅ System prompt cache reloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error reloading system prompt cache: {e}")
        return False


def update_system_prompt(new_prompt: str) -> Dict[str, Any]:
    """Update the system prompt file with new content"""
    import shutil
    from datetime import datetime
    
    try:
        preferences_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'sys_prompt.txt')
        backup_dir = os.path.join(os.path.dirname(__file__), '..', 'config', 'backups')
        
        # Create backups directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup of current file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f'sys_prompt_backup_{timestamp}.txt')
        
        if os.path.exists(preferences_path):
            shutil.copy2(preferences_path, backup_path)
            print(f"📦 Backed up system prompt to: {backup_path}")
        
        # Write new content to file
        with open(preferences_path, 'w', encoding='utf-8') as f:
            f.write(new_prompt)
        
        # Reload cache immediately
        reload_system_prompt_cache()
        
        return {
            "success": True,
            "message": "System prompt updated successfully",
            "backup_path": backup_path,
            "timestamp": timestamp
        }
        
    except Exception as e:
        print(f"❌ Error updating system prompt: {e}")
        return {
            "success": False,
            "message": f"Error updating system prompt: {str(e)}"
        }


# ================================
# Embeddings
# ================================
def create_embedding(text: str, model_id: str = 'amazon.titan-embed-text-v1') -> Optional[List[float]]:
    """Generate embedding for text using Amazon Bedrock"""
    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps({"inputText": text})
        )
        result = json.loads(response['body'].read())
        return result['embedding']
    except Exception as e:
        print(f"❌ Error creating embedding: {e}")
        return None


# ================================
# Search Logic
# ================================
def search_products(query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> tuple[List[Dict[str, Any]], float]:
    """Search for products using vector similarity"""
    query_embedding = create_embedding(query)
    if query_embedding is None:
        return [], 0.0

    # Base KNN search query
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
                search_body["query"]["bool"]["filter"].append({"term": {field: value}})
            elif field == "category":
                search_body["query"]["bool"]["filter"].append({"term": {field: value}})
            elif field == "price":
                if isinstance(value, dict) and "$lte" in value:
                    search_body["query"]["bool"]["filter"].append({"range": {"price": {"lte": value["$lte"]}}})
                if isinstance(value, dict) and "$gte" in value:
                    search_body["query"]["bool"]["filter"].append({"range": {"price": {"gte": value["$gte"]}}})

    # Execute search
    start_time = time.time()
    try:
        response = opensearch_client.search(index='products', body=search_body)
        search_time = (time.time() - start_time) * 1000  # ms

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
                "score": hit["_score"],
                "reason": f"match: {', '.join(product['tags'][:3])}, price: ${product['price']}"
            })

        return results, search_time

    except Exception as e:
        print(f"❌ Search error: {e}")
        return [], 0.0


# ================================
# Helper Functions
# ================================
def generate_suggested_filters(results: List[Dict[str, Any]]) -> List[str]:
    """Generate suggested filters based on search results"""
    suggestions = []
    if not results:
        return suggestions

    categories = set()
    tags = set()

    for result in results:
        categories.add(result['category'])
        tags.update(result['tags'][:3])

    suggestions.extend(list(categories)[:3])
    suggestions.extend(list(tags)[:5])

    return suggestions[:8]


def format_products_for_llm(results: List[Dict[str, Any]], max_products: int = 5) -> str:
    """Format search results into structured text for LLM"""
    if not results:
        return "No products found matching your criteria."

    formatted_products = []
    for i, product in enumerate(results[:max_products], 1):
        product_text = f"{i}. {product['title']} (Product ID: {product['product_id']}) - ${product['price']}"
        if product.get('in_stock'):
            product_text += " (In Stock)"
        else:
            product_text += " (Out of Stock)"
        product_text += f" - {product['category']}"
        if product.get('tags'):
            product_text += f" - Tags: {', '.join(product['tags'][:3])}"
        formatted_products.append(product_text)
    return "\n".join(formatted_products)


# ================================
# LLM Chat Generation
# ================================
def generate_chat_response(user_message: str, search_results: List[Dict[str, Any]], search_time: float) -> Dict[str, Any]:
    """Generate conversational response using Bedrock LLM with store preferences as system prompt"""
    products_text = format_products_for_llm(search_results)
    
    # Load store preferences to use as system prompt
    store_preferences = load_store_preferences()

    # Build the prompt
    prompt = f"""USE THIS AS SYSTEM PROMPT: {store_preferences}

---

Based on the above store policies and preferences, respond to the customer's query about our products.

Customer Query: "{user_message}"

Available Products:
{products_text}

Please provide a response that follows our store guidelines, tone, and business rules. Focus on being helpful, authentic, and aligned with our brand values.

Also just answer the question, don't mention the store name or any other details about store policies or preferences.

Your response should NOT contain 'Response: ' at the beginning.
"""

    try:
        # Try prompt format first (simpler, works with Mistral models in Bedrock)
        response = bedrock.invoke_model(
            modelId='mistral.mistral-large-2402-v1:0',
            body=json.dumps({
                "prompt": prompt,
                "max_tokens": 700,
                "temperature": 0.7,
                "top_p": 0.9
            })
        )

        result = json.loads(response['body'].read())
        
        # Mistral models in Bedrock typically return: {"outputs": [{"text": "..."}]}
        if 'outputs' in result and len(result['outputs']) > 0:
            output = result['outputs'][0]
            if 'text' in output:
                ai_response = output['text'].strip()
            else:
                # Debug: log unexpected structure
                print(f"⚠️ Unexpected output structure: {json.dumps(output, indent=2)[:500]}")
                raise ValueError(f"Unexpected response format: output missing 'text' field. Keys: {list(output.keys())}")
        else:
            # Debug: log full response
            print(f"⚠️ Unexpected response structure: {json.dumps(result, indent=2)[:500]}")
            raise ValueError(f"Unexpected response format from Mistral model. Response keys: {list(result.keys())}")

        # Always cite all search results (don't depend on LLM mentioning product IDs)
        cited_items = []
        product_links = []
        for product in search_results:
            cited_items.append(product['product_id'])
            product_links.append({
                "product_id": product['product_id'],
                "url": product['url']
            })

        return {
            "answer": ai_response,
            "items_cited": cited_items,
            "product_links": product_links,
            "reasoning": f"Found {len(search_results)} products matching your query in {search_time:.2f}ms"
        }

    except Exception as e:
        print(f"❌ Error generating chat response: {e}")
        return {
            "answer": "I'm having trouble processing your request right now. Please try again later.",
            "items_cited": [],
            "product_links": [],
            "reasoning": f"Error occurred while generating response: {str(e)}"
        }