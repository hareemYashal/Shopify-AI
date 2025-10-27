from typing import Optional, Dict, Any, List
from pydantic import BaseModel




# ================================
# Models
# ================================
class SearchRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "products"  # Optional collection name, defaults to "products"
    k: Optional[int] = 24  # Number of results to return, defaults to 24


class SearchResponse(BaseModel):
    items: List[Dict[str, Any]]
    suggested_filters: List[str]
    search_time_ms: float
    total_results: int


class ChatRequest(BaseModel):
    message: str
    collection_name: Optional[str] = "products"  # Optional collection name, defaults to "products"
    k: Optional[int] = 24  # Number of results to return, defaults to 24


class ChatResponse(BaseModel):
    answer: str
    items_cited: List[str]
    search_time_ms: float
    reasoning: str
    product_links: List[Dict[str, str]]  # List of {product_id, url} dictionaries


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class CollectionInfo(BaseModel):
    name: str
    count: int
    metadata: Dict[str, Any]


class CollectionsResponse(BaseModel):
    collections: List[CollectionInfo]
    total_collections: int
