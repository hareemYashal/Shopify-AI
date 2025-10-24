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
    """Generate conversational response using Bedrock LLM"""
    products_text = format_products_for_llm(search_results)

    prompt = f"""
You are a helpful shopping assistant. Based on the following products, provide a natural, conversational response to the user's query.

User Query: "{user_message}"

Available Products:
{products_text}

Instructions:
1. Provide a helpful, conversational response about the products.
2. Mention specific product names and prices when relevant.
3. Use Product IDs when referring to products.
4. If no products match, suggest alternatives or ask for clarification.
5. Keep the response concise but informative.
"""

    try:
        response = bedrock.invoke_model(
            modelId='mistral.mistral-small-2402-v1:0',
            body=json.dumps({
                "prompt": prompt,
                "max_tokens": 300,
                "temperature": 0.3,
                "top_p": 0.9
            })
        )

        result = json.loads(response['body'].read())
        ai_response = result['outputs'][0]['text'].strip()

        cited_items = []
        product_links = []
        for product in search_results:
            if str(product['product_id']) in ai_response:
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