#!/usr/bin/env python3
"""
Initialize ChromaDB with product data
Run this once to set up ChromaDB with your catalog
"""

from chroma_db import embed_products_to_chroma, get_collection_stats, test_chroma_performance, test_chroma_filters, debug_collection, check_persistence

if __name__ == "__main__":
    print("🚀 Initializing ChromaDB...")
    
    # Check persistence first
    print("\n🔍 Checking persistence...")
    check_persistence()
    
    # Check current status
    stats = get_collection_stats()
    print(f"\n📊 Current status: {stats}")
    
    # Debug collection contents
    print("\n🔍 Debugging collection...")
    debug_collection()
    
    if stats["total_products"] == 0:
        print("\n📥 No products found, embedding products...")
        embed_products_to_chroma(incremental=False)  # Full embedding for empty collection
    else:
        print(f"\n✅ Found {stats['total_products']} products in ChromaDB")
        print("🔄 Running incremental update (only new products)...")
        embed_products_to_chroma(incremental=True)  # Incremental embedding
    
    # Test performance
    print("\n🧪 Testing performance...")
    test_chroma_performance()
    
    # Test filters
    print("\n🔍 Testing filters...")
    test_chroma_filters()
    
    # Test specific Apple search
    print("\n🍎 Testing Apple product search...")
    from chroma_db import search_products_chroma
    results, search_time = search_products_chroma("apple products", k=5)
    print(f"Apple search results: {len(results)} items in {search_time:.2f}ms")
    for result in results:
        print(f"  - {result['title']} (${result['price']})")
    
    print("\n✅ ChromaDB initialization complete!")
    print("🚀 You can now run: python search_api.py")
