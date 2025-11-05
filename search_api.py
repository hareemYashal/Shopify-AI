import os
import json
import time
import re
import datetime
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
import boto3
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.util_funcs import parse_filters, clean_query_for_embedding
from models.models import (SearchRequest, SearchResponse, ChatRequest, ChatResponse, ErrorResponse, 
                          CollectionsResponse, CollectionInfo, SystemPromptResponse, SystemPromptUpdateRequest)

from services.ai_services import generate_suggested_filters, generate_chat_response, load_store_preferences, update_system_prompt, reload_system_prompt_cache
from scripts.opensearch_db import search_products_opensearch, get_index_stats, list_all_indices


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Shopify AI Search API (OpenSearch)",
    description="AI-powered product search with OpenSearch vector similarity",
    version="2.0.0"
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
        # Use parsed filters directly for OpenSearch
        
        # Clean query for better embedding (remove filter terms)
        cleaned_query = clean_query_for_embedding(request.query)
        
        # Use original query if cleaned query is too short
        if len(cleaned_query.strip()) < 3:
            cleaned_query = request.query
        
        # Log filter extraction for debugging
        if parsed_filters:
            print(f"🔍 Extracted filters: {parsed_filters}")
            print(f"📝 Cleaned query: '{cleaned_query}' (original: '{request.query}')")

        # Perform search with extracted filters using OpenSearch
        # Map collection_name to index_name for OpenSearch
        index_name = request.collection_name or "products"
        results, search_time = search_products_opensearch(
            query=cleaned_query,
            k=request.k,
            filters=parsed_filters,
            index_name=index_name
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
        # Use parsed filters directly for OpenSearch
        
        # Clean query for better embedding
        cleaned_query = clean_query_for_embedding(request.message)
        if len(cleaned_query.strip()) < 3:
            cleaned_query = request.message
        
        # Log filter extraction for debugging
        if parsed_filters:
            print(f"💬 Chat filters: {parsed_filters}")
            print(f"💬 Chat query: '{cleaned_query}' (original: '{request.message}')")

        # Perform search with extracted filters using OpenSearch
        # Map collection_name to index_name for OpenSearch
        index_name = request.collection_name or "products"
        search_results, search_time = search_products_opensearch(
            query=cleaned_query,
            k=request.k,
            filters=parsed_filters,
            index_name=index_name
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
            reasoning=chat_response["reasoning"],
            product_links=chat_response["product_links"]
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during chat")


@app.get("/get-collections", response_model=CollectionsResponse)
async def get_collections():
    """Get all indices in OpenSearch with their statistics"""
    try:
        collections_data = list_all_indices()
        
        # Convert to CollectionInfo objects
        collection_list = []
        for col in collections_data:
            # Handle count that might be "error" string
            count_value = col.get("count", 0)
            if isinstance(count_value, str) and count_value == "error":
                continue  # Skip collections with errors
            
            collection_list.append(CollectionInfo(
                name=col.get("name", "unknown"),
                count=count_value,
                metadata=col.get("metadata", {})
            ))
        
        return CollectionsResponse(
            collections=collection_list,
            total_collections=len(collection_list)
        )
    except Exception as e:
        print(f"❌ Error getting collections: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving collections: {str(e)}")


@app.get("/system-prompt", response_model=SystemPromptResponse)
async def get_system_prompt():
    """
    Get the current system prompt content
    This endpoint is used by the View tab in the frontend
    """
    try:
        preferences_path = os.path.join("config", "sys_prompt.txt")
        
        # Get file modification time
        if os.path.exists(preferences_path):
            file_mtime = os.path.getmtime(preferences_path)
            last_updated = datetime.datetime.fromtimestamp(file_mtime).isoformat()
        else:
            last_updated = None
        
        # Load current prompt
        prompt = load_store_preferences()
        
        return SystemPromptResponse(
            prompt=prompt,
            last_updated=last_updated,
            filename="config/sys_prompt.txt"
        )
    except Exception as e:
        print(f"❌ Error getting system prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving system prompt: {str(e)}")


@app.put("/system-prompt", response_model=Dict[str, Any])
async def update_system_prompt_endpoint(request: SystemPromptUpdateRequest):
    """
    Update the system prompt content
    This endpoint is used by the Edit tab in the frontend when user clicks 'Modify'
    """
    try:
        # Validate the prompt content
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if len(request.prompt) < 50:
            raise HTTPException(status_code=400, detail="Prompt is too short (minimum 50 characters)")
        
        if len(request.prompt) > 50000:
            raise HTTPException(status_code=400, detail="Prompt is too long (maximum 50000 characters)")
        
        # Update the prompt
        result = update_system_prompt(request.prompt)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("message", "Failed to update system prompt"))
        
        return {
            "success": True,
            "message": "System prompt updated successfully and cache reloaded",
            "backup_path": result.get("backup_path"),
            "timestamp": result.get("timestamp")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating system prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating system prompt: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check OpenSearch health using default "products" index
        opensearch_stats = get_index_stats("products")
        opensearch_status = "healthy" if opensearch_stats["status"] == "healthy" else f"unhealthy: {opensearch_stats.get('error', 'Unknown error')}"
    except Exception as e:
        opensearch_status = f"unhealthy: {str(e)}"
        opensearch_stats = {"total_products": 0}

    try:
        bedrock.list_foundation_models()
        bedrock_status = "healthy"
    except Exception as e:
        bedrock_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if opensearch_status == "healthy" and bedrock_status == "healthy" else "unhealthy",
        "opensearch": opensearch_status,
        "opensearch_stats": opensearch_stats,
        "bedrock": bedrock_status,
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
