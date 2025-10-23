"""
FastAPI endpoint for /search-fast
Sub-200ms vector search endpoint for Shopify AI Search
"""

import os
import json
import time
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
import boto3
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearch import client as opensearch_client

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
        for product in search_results:
            if str(product['product_id']) in ai_response:
                cited_items.append(product['product_id'])

        return {
            "answer": ai_response,
            "items_cited": cited_items,
            "reasoning": f"Found {len(search_results)} products matching your query in {search_time:.2f}ms"
        }

    except Exception as e:
        print(f"❌ Error generating chat response: {e}")
        return {
            "answer": "I’m having trouble processing your request right now. Please try again later.",
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

        results, search_time = search_products(
            query=request.query,
            k=24,
            filters=None
        )

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

        # Perform search (defaults to top 5 in-stock products)
        search_results, search_time = search_products(
            query=request.message,
            k=5,
            filters={"in_stock": True}
        )

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

    return {
        "status": "healthy",
        "opensearch": opensearch_status,
        "bedrock": bedrock_status,
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
