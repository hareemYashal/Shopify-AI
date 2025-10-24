from typing import Optional, Dict, Any, List
from pydantic import BaseModel




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
    product_links: List[Dict[str, str]]  # List of {product_id, url} dictionaries


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
