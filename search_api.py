"""
FastAPI endpoint for /search-fast
Sub-200ms vector search endpoint for Shopify AI Search
"""

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
from opensearch import client as opensearch_client
from preferences_parser import get_preferences

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Shopify AI Search API",
    description="AI-powered product search with vector similarity",
    version="1.0.0"
)

# Add CORS middleware for Shopify integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))

# Initialize preferences on startup
print("[INIT] Loading preferences...")
preferences = get_preferences()
print(f"[OK] Preferences loaded: k={preferences.get_config()['k']}, max_results={preferences.get_config()['max_results']}")


# ================================
# Models
# ================================
class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    items: List[Dict[str, Any]]
    suggested_filters: List[str]
    search_time_ms: float
    total_results: int


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    items_cited: List[str]
    search_time_ms: float
    reasoning: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


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
# Search Logic with Preferences Integration
# ================================
def search_products(query: str) -> tuple[List[Dict[str, Any]], float]:
    """Search for products using vector similarity with preferences-based ranking"""
    query_embedding = create_embedding(query)
    if query_embedding is None:
        return [], 0.0

    # Load preferences
    prefs = get_preferences()
    config = prefs.get_config()
    
    # Get all settings from preferences
    k = config.get('max_results', 24)  # Use max_results from preferences
    default_filters = prefs.get_default_filters()
    merged_filters = default_filters.copy()
    
    # Extract budget constraints from query
    budget_match = re.search(r'under\s+\$?(\d+)', query.lower())
    if budget_match:
        max_price = float(budget_match.group(1))
        merged_filters['max_price'] = max_price

    # Base KNN search query - fetch more for re-ranking
    fetch_size = min(k * 3, 100)  # Fetch 3x to allow for re-ranking, max 100
    search_body = {
        "size": fetch_size,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": fetch_size
                }
            }
        }
    }

    # Add filters
    if merged_filters:
        search_body["query"] = {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": fetch_size
                            }
                        }
                    }
                ],
                "filter": []
            }
        }

        for field, value in merged_filters.items():
            if field == "in_stock" and value is not None:
                search_body["query"]["bool"]["filter"].append({"term": {"in_stock": value}})
            elif field == "category" and value:
                search_body["query"]["bool"]["filter"].append({"term": {"category": value}})
            elif field == "max_price" and value is not None:
                search_body["query"]["bool"]["filter"].append({"range": {"price": {"lte": value}}})
            elif field == "min_price" and value is not None:
                search_body["query"]["bool"]["filter"].append({"range": {"price": {"gte": value}}})
            elif field == "price" and isinstance(value, dict):
                if "$lte" in value:
                    search_body["query"]["bool"]["filter"].append({"range": {"price": {"lte": value["$lte"]}}})
                if "$gte" in value:
                    search_body["query"]["bool"]["filter"].append({"range": {"price": {"gte": value["$gte"]}}})

    # Execute search
    start_time = time.time()
    try:
        response = opensearch_client.search(index='products', body=search_body)
        search_time = (time.time() - start_time) * 1000  # ms

        results = []
        for hit in response['hits']['hits']:
            product = hit['_source']
            base_score = hit['_score']
            
            # Apply preference-based boosting
            boosted_score, boost_reasons = prefs.apply_query_boost(query, product, base_score)
            
            # Build reason string
            reason_parts = []
            if boost_reasons:
                reason_parts.extend(boost_reasons)
            
            # Add tag matches
            matching_tags = [tag for tag in product.get('tags', [])[:3]]
            if matching_tags:
                reason_parts.append(f"tags: {', '.join(matching_tags)}")
            
            results.append({
                "product_id": product["product_id"],
                "title": product["title"],
                "price": product["price"],
                "url": product["url"],
                "image": product["image"],
                "in_stock": product["in_stock"],
                "category": product["category"],
                "tags": product["tags"],
                "score": base_score,
                "boosted_score": boosted_score,
                "reason": f"match: {', '.join(reason_parts)}, ${product['price']}"
            })

        # Sort by boosted score
        results.sort(key=lambda x: x['boosted_score'], reverse=True)
        
        # Return top k results
        results = results[:k]
        
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
# LLM Chat Generation with Preferences
# ================================
def generate_chat_response(user_message: str, search_results: List[Dict[str, Any]], search_time: float) -> Dict[str, Any]:
    """Generate conversational response using Bedrock LLM with preferences-based tone"""
    # Load preferences
    prefs = get_preferences()
    chat_config = prefs.get_chat_config()
    response_config = prefs.get_response_config()
    business_context = prefs.format_business_rules_for_llm()
    
    products_text = format_products_for_llm(search_results)

    # Build prompt with preferences
    tone = response_config.get('tone', 'friendly')
    include_price = response_config.get('include_price', True)
    include_availability = response_config.get('include_availability', True)
    
    prompt = f"""
You are a helpful shopping assistant for CoralBricks.ai. Based on the following products, provide a natural, conversational response to the user's query.

User Query: "{user_message}"

Available Products:
{products_text}

{business_context}

Instructions:
1. Tone: Be {tone}, concise, and matter-of-fact.
2. Always mention product names{"and prices" if include_price else ""} when relevant.
3. {"Always include stock availability information." if include_availability else ""}
4. Use Product IDs when referring to products for citation.
5. Explain why each product matches the customer's request.
6. NEVER invent specifications or details not provided in the product information.
7. If unsure about any detail, say "I don't have that detail" and provide the product link.
8. If no products match perfectly, show closest alternatives and explain the difference.
9. Keep the response helpful and informative without being pushy.
"""

    try:
        # Get temperature from preferences
        temperature = chat_config.get('temperature', 0.3)
        max_tokens = chat_config.get('max_tokens', 300)
        
        response = bedrock.invoke_model(
            modelId='mistral.mistral-small-2402-v1:0',
            body=json.dumps({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            })
        )

        result = json.loads(response['body'].read())
        ai_response = result['outputs'][0]['text'].strip()

        # Extract cited product IDs
        cited_items = []
        for product in search_results:
            product_id_str = str(product['product_id'])
            if product_id_str in ai_response or product['title'] in ai_response:
                cited_items.append(product_id_str)

        return {
            "answer": ai_response,
            "items_cited": cited_items,
            "reasoning": f"Found {len(search_results)} products matching your query in {search_time:.2f}ms"
        }

    except Exception as e:
        print(f"❌ Error generating chat response: {e}")
        return {
            "answer": "I'm having trouble processing your request right now. Please try again later.",
            "items_cited": [],
            "reasoning": f"Error occurred while generating response: {str(e)}"
        }


# ================================
# Routes
# ================================
@app.get("/", response_model=Dict[str, str])
async def root():
    """Health check endpoint"""
    return {"message": "Shopify AI Search API is running", "status": "healthy"}


@app.post("/search-fast", response_model=SearchResponse)
async def search_fast(request: SearchRequest):
    """Fast vector search endpoint (target: <200ms)"""
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 500:
            raise HTTPException(status_code=400, detail="Query too long (max 500 characters)")

        results, search_time = search_products(query=request.query)

        suggested_filters = generate_suggested_filters(results)

        if search_time > 200:
            print(f"⚠️ Search time {search_time:.2f}ms exceeds 200ms target")

        return SearchResponse(
            items=results,
            suggested_filters=suggested_filters,
            search_time_ms=search_time,
            total_results=len(results)
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during search")


@app.get("/search-fast", response_model=SearchResponse)
async def search_fast_get(
    query: str = Query(..., description="Search query")
):
    """GET version of search-fast endpoint for easier testing"""
    request = SearchRequest(query=query)
    return await search_fast(request)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Conversational AI endpoint with grounded product responses"""
    try:
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        if len(request.message) > 1000:
            raise HTTPException(status_code=400, detail="Message too long (max 1000 characters)")

        # Perform search (uses preferences for all settings)
        search_results, search_time = search_products(query=request.message)

        chat_response = generate_chat_response(
            user_message=request.message,
            search_results=search_results,
            search_time=search_time
        )

        return ChatResponse(
            answer=chat_response["answer"],
            items_cited=chat_response["items_cited"],
            search_time_ms=search_time,
            reasoning=chat_response["reasoning"]
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during chat")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        info = opensearch_client.info()
        opensearch_status = "healthy"
    except Exception as e:
        opensearch_status = f"unhealthy: {str(e)}"

    try:
        bedrock.list_foundation_models()
        bedrock_status = "healthy"
    except Exception as e:
        bedrock_status = f"unhealthy: {str(e)}"

    # Check preferences
    prefs = get_preferences()
    prefs_config = prefs.get_config()

    return {
        "status": "healthy",
        "opensearch": opensearch_status,
        "bedrock": bedrock_status,
        "preferences": {
            "loaded": True,
            "k": prefs_config.get('k'),
            "max_results": prefs_config.get('max_results'),
            "tone": prefs_config.get('response', {}).get('tone')
        },
        "timestamp": time.time()
    }


@app.post("/admin/reload-preferences")
async def reload_preferences_endpoint():
    """Reload preferences from file (admin endpoint)"""
    try:
        from preferences_parser import reload_preferences
        prefs = reload_preferences()
        config = prefs.get_config()
        
        return {
            "status": "success",
            "message": "Preferences reloaded successfully",
            "config": {
                "k": config.get('k'),
                "max_results": config.get('max_results'),
                "default_filters": config.get('default_filters'),
                "tone": config.get('response', {}).get('tone')
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload preferences: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
