#!/usr/bin/env python3
"""
ChromaDB Filter Examples
Demonstrates all the filtering capabilities available with ChromaDB
"""

from chroma_db import search_products_chroma, get_collection_stats

def demonstrate_filters():
    """Demonstrate all ChromaDB filtering capabilities"""
    print("🔍 ChromaDB Filter Demonstration")
    print("=" * 50)
    
    # Check if ChromaDB is ready
    stats = get_collection_stats()
    if stats["total_products"] == 0:
        print("❌ No products in ChromaDB. Run 'python init_chroma.py' first.")
        return
    
    print(f"📊 ChromaDB Status: {stats['total_products']} products loaded\n")
    
    # Example 1: Basic Price Filter
    print("1️⃣ PRICE FILTERS")
    print("-" * 20)
    
    examples = [
        {
            "name": "Under $100",
            "query": "shoes",
            "filters": {"price": {"lte": 100}},
            "description": "Find shoes under $100"
        },
        {
            "name": "Over $200",
            "query": "electronics",
            "filters": {"price": {"gte": 200}},
            "description": "Find electronics over $200"
        },
        {
            "name": "Price Range",
            "query": "clothing",
            "filters": {"price": {"gte": 50, "lte": 150}},
            "description": "Find clothing between $50-$150"
        }
    ]
    
    for example in examples:
        print(f"\n🔍 {example['name']}: {example['description']}")
        results, search_time = search_products_chroma(
            query=example['query'],
            k=3,
            filters=example['filters']
        )
        print(f"   Results: {len(results)} | Time: {search_time:.2f}ms")
        for result in results:
            print(f"   - {result['title']} (${result['price']})")
    
    # Example 2: Category Filter
    print("\n\n2️⃣ CATEGORY FILTERS")
    print("-" * 20)
    
    categories = ["clothing", "electronics", "kitchen-dining", "home-garden"]
    for category in categories:
        print(f"\n🔍 Category: {category}")
        results, search_time = search_products_chroma(
            query="products",
            k=3,
            filters={"category": category}
        )
        print(f"   Results: {len(results)} | Time: {search_time:.2f}ms")
        for result in results:
            print(f"   - {result['title']} (${result['price']})")
    
    # Example 3: Stock Filter
    print("\n\n3️⃣ STOCK FILTERS")
    print("-" * 20)
    
    stock_examples = [
        {"name": "In Stock Only", "filters": {"in_stock": True}},
        {"name": "Out of Stock", "filters": {"in_stock": False}}
    ]
    
    for example in stock_examples:
        print(f"\n🔍 {example['name']}")
        results, search_time = search_products_chroma(
            query="products",
            k=3,
            filters=example['filters']
        )
        print(f"   Results: {len(results)} | Time: {search_time:.2f}ms")
        for result in results:
            print(f"   - {result['title']} (${result['price']}) - Stock: {result['in_stock']}")
    
    # Example 4: Tag Filter
    print("\n\n4️⃣ TAG FILTERS")
    print("-" * 20)
    
    tag_examples = [
        {"name": "Athletic Items", "filters": {"tags": ["athletic"]}},
        {"name": "Black Items", "filters": {"tags": ["black"]}},
        {"name": "Women's Items", "filters": {"tags": ["women"]}}
    ]
    
    for example in tag_examples:
        print(f"\n🔍 {example['name']}")
        results, search_time = search_products_chroma(
            query="products",
            k=3,
            filters=example['filters']
        )
        print(f"   Results: {len(results)} | Time: {search_time:.2f}ms")
        for result in results:
            print(f"   - {result['title']} (${result['price']}) - Tags: {result['tags']}")
    
    # Example 5: Combined Filters
    print("\n\n5️⃣ COMBINED FILTERS")
    print("-" * 20)
    
    combined_examples = [
        {
            "name": "Affordable Athletic Wear",
            "query": "workout",
            "filters": {
                "price": {"lte": 100},
                "tags": ["athletic"],
                "in_stock": True
            },
            "description": "In-stock athletic items under $100"
        },
        {
            "name": "Premium Electronics",
            "query": "electronics",
            "filters": {
                "category": "electronics",
                "price": {"gte": 200},
                "in_stock": True
            },
            "description": "In-stock electronics over $200"
        },
        {
            "name": "Kitchen Essentials",
            "query": "kitchen",
            "filters": {
                "category": "kitchen-dining",
                "price": {"lte": 300},
                "in_stock": True
            },
            "description": "In-stock kitchen items under $300"
        }
    ]
    
    for example in combined_examples:
        print(f"\n🔍 {example['name']}: {example['description']}")
        results, search_time = search_products_chroma(
            query=example['query'],
            k=3,
            filters=example['filters']
        )
        print(f"   Results: {len(results)} | Time: {search_time:.2f}ms")
        for result in results:
            print(f"   - {result['title']} (${result['price']}) - {result['category']}")
    
    # Example 6: Natural Language Queries with Filters
    print("\n\n6️⃣ NATURAL LANGUAGE + FILTERS")
    print("-" * 20)
    
    nl_examples = [
        {
            "query": "running shoes under $150",
            "description": "Natural language price filter"
        },
        {
            "query": "black athletic shorts in stock",
            "description": "Natural language stock + color filter"
        },
        {
            "query": "kitchen cookware over $200",
            "description": "Natural language category + price filter"
        }
    ]
    
    for example in nl_examples:
        print(f"\n🔍 Query: '{example['query']}'")
        print(f"   Description: {example['description']}")
        results, search_time = search_products_chroma(
            query=example['query'],
            k=3
        )
        print(f"   Results: {len(results)} | Time: {search_time:.2f}ms")
        for result in results:
            print(f"   - {result['title']} (${result['price']})")
    
    print("\n\n✅ Filter demonstration complete!")
    print("🚀 All filters are working with ChromaDB!")

if __name__ == "__main__":
    demonstrate_filters()
