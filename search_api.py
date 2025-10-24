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
from utils.util_funcs import parse_filters, convert_filters_to_opensearch, clean_query_for_embedding
from models.models import SearchRequest, SearchResponse, ChatRequest, ChatResponse, ErrorResponse

from services.ai_services import search_products, generate_suggested_filters, generate_chat_response, create_embedding


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
# Routes
# ================================
@app.get("/", response_model=Dict[str, str])
async def root():
    """Health check endpoint"""
    return {"message": "Shopify AI Search API is running", "status": "healthy"}


@app.post("/search-fast", response_model=SearchResponse)
async def search_fast(request: SearchRequest):
    """Fast vector search endpoint with automatic filter extraction (target: <200ms)"""
    start_time = time.time()
    
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 500:
            raise HTTPException(status_code=400, detail="Query too long (max 500 characters)")

        # Extract filters from natural language query
        parsed_filters = parse_filters(request.query)
        opensearch_filters = convert_filters_to_opensearch(parsed_filters)
        
        # Clean query for better embedding (remove filter terms)
        cleaned_query = clean_query_for_embedding(request.query)
        
        # Use original query if cleaned query is too short
        if len(cleaned_query.strip()) < 3:
            cleaned_query = request.query
        
        # Log filter extraction for debugging
        if opensearch_filters:
            print(f"🔍 Extracted filters: {opensearch_filters}")
            print(f"📝 Cleaned query: '{cleaned_query}' (original: '{request.query}')")

        # Perform search with extracted filters
        results, search_time = search_products(
            query=cleaned_query,
            k=5,
            filters=opensearch_filters
        )

        suggested_filters = generate_suggested_filters(results)

        total_time = (time.time() - start_time) * 1000

        if total_time > 200:
            print(f"⚠️ Total time {total_time:.2f}ms exceeds 200ms target")

        return SearchResponse(
            items=results,
            suggested_filters=suggested_filters,
            search_time_ms=total_time,
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
    """Conversational AI endpoint with grounded product responses and filter extraction"""
    try:
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        if len(request.message) > 1000:
            raise HTTPException(status_code=400, detail="Message too long (max 1000 characters)")

        # Extract filters from natural language query
        parsed_filters = parse_filters(request.message)
        opensearch_filters = convert_filters_to_opensearch(parsed_filters)
        
        # Clean query for better embedding
        cleaned_query = clean_query_for_embedding(request.message)
        if len(cleaned_query.strip()) < 3:
            cleaned_query = request.message
        
        # Log filter extraction for debugging
        if opensearch_filters:
            print(f"💬 Chat filters: {opensearch_filters}")
            print(f"💬 Chat query: '{cleaned_query}' (original: '{request.message}')")

        # Perform search with extracted filters
        search_results, search_time = search_products(
            query=cleaned_query,
            k=5,
            filters=opensearch_filters
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
