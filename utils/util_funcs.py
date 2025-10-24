from typing import Dict, Any
import re



# ================================
# Filter Parsing
# ================================
def parse_filters(query: str) -> Dict[str, Any]:
    """Extract filters from natural language query"""
    query_lower = query.lower()
    filters = {}

    # ---- PRICE FILTERS ----
    # Matches patterns like:
    # - under $500
    # - below 1000
    # - less than 200
    # - over $100
    # - more than 300
    # - between $100 and $300
    # - from 100 to 400

    if match := re.search(r'(?:under|below|less\s+than)\s*\$?(\d+)', query_lower):
        filters['price'] = {'lte': float(match.group(1))}
    elif match := re.search(r'(?:over|above|more\s+than)\s*\$?(\d+)', query_lower):
        filters['price'] = {'gte': float(match.group(1))}
    elif match := re.search(r'between\s*\$?(\d+)\s*(?:and|to)\s*\$?(\d+)', query_lower):
        filters['price'] = {
            'gte': float(match.group(1)),
            'lte': float(match.group(2))
        }
    elif match := re.search(r'around\s*\$?(\d+)', query_lower):
        price = float(match.group(1))
        filters['price'] = {
            'gte': price * 0.8,
            'lte': price * 1.2
        }

    # ---- STOCK / AVAILABILITY FILTERS ----
    # Handles:
    # - in stock, available
    # - out of stock, sold out, unavailable
    # - show only available, hide sold out, etc.

    if re.search(r'(out of stock|sold out|unavailable|not available)', query_lower):
        filters['in_stock'] = {'eq': False}
    elif re.search(r'(in stock|available|show available|only available|still available)', query_lower):
        filters['in_stock'] = {'eq': True}
    # default behavior (if not mentioned)
    else:
        filters.setdefault('in_stock', {'eq': True})

    return filters


def clean_query_for_embedding(query: str) -> str:
    """Remove filter-specific terms to get better semantic embedding"""
    # Remove price constraints
    cleaned = re.sub(r'(?:under|below|less\s+than)\s*\$?\d+', '', query, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?:over|above|more\s+than)\s*\$?\d+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'between\s*\$?\d+\s*(?:and|to)\s*\$?\d+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'around\s*\$?\d+', '', cleaned, flags=re.IGNORECASE)
    
    # Remove stock status phrases
    cleaned = re.sub(r'(?:in stock|out of stock|sold out|unavailable|available|show available|only available|still available)', '', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


def convert_filters_to_opensearch(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Convert parsed filters to OpenSearch-compatible format"""
    opensearch_filters = {}
    
    for field, value in filters.items():
        if field == "price":
            if isinstance(value, dict):
                opensearch_filters["price"] = {}
                if "gte" in value:
                    opensearch_filters["price"]["$gte"] = value["gte"]
                if "lte" in value:
                    opensearch_filters["price"]["$lte"] = value["lte"]
        elif field == "in_stock":
            if isinstance(value, dict) and "eq" in value:
                opensearch_filters["in_stock"] = value["eq"]
            else:
                opensearch_filters["in_stock"] = value
    
    return opensearch_filters