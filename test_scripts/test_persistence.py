#!/usr/bin/env python3
"""
Test ChromaDB Persistence
This script tests if ChromaDB data persists after closing the terminal
"""

from chroma_db import check_persistence, get_collection_stats, search_products_chroma

if __name__ == "__main__":
    print("🧪 Testing ChromaDB Persistence")
    print("=" * 40)
    
    # Check persistence
    check_persistence()
    
    # Get stats
    stats = get_collection_stats()
    print(f"\n📊 Collection Stats: {stats}")
    
    # Test search
    if stats["total_products"] > 0:
        print("\n🔍 Testing search functionality...")
        results, search_time = search_products_chroma("apple products", k=3)
        print(f"Search results: {len(results)} items in {search_time:.2f}ms")
        
        if results:
            print("✅ Search is working!")
            for result in results:
                print(f"  - {result['title']} (${result['price']})")
        else:
            print("❌ Search returned no results")
    else:
        print("\n⚠️  No products in collection - run 'python init_chroma.py' first")
    
    print("\n💡 To test persistence:")
    print("1. Run this script")
    print("2. Close terminal")
    print("3. Open new terminal")
    print("4. Run this script again")
    print("5. Data should still be there!")
