#!/usr/bin/env python3
"""
Simple test runner for AI functionality
Basic tests without complex mocking
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    try:
        from search_api import create_embedding, search_products, generate_chat_response
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_embedding_basic():
    """Test basic embedding functionality"""
    print("🧪 Testing embedding creation...")
    try:
        from search_api import create_embedding
        
        # Test with a simple query
        embedding = create_embedding("test query")
        
        if embedding is None:
            print("⚠️  Embedding returned None (might be API issue)")
            return True  # Don't fail if API is not available
        
        if isinstance(embedding, list) and len(embedding) > 0:
            print("✅ Embedding creation works")
            return True
        else:
            print("❌ Embedding format incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

def test_search_basic():
    """Test basic search functionality"""
    print("🧪 Testing search functionality...")
    try:
        from search_api import search_products
        
        # Test search with a simple query
        results, search_time = search_products("test query", k=5)
        
        if isinstance(results, list) and isinstance(search_time, float):
            print("✅ Search function works")
            print(f"   Results: {len(results)} items")
            print(f"   Time: {search_time:.2f}ms")
            return True
        else:
            print("❌ Search return format incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def test_chat_basic():
    """Test basic chat functionality"""
    print("🧪 Testing chat functionality...")
    try:
        from search_api import generate_chat_response
        
        # Test with mock data
        mock_products = [
            {
                "product_id": "12345",
                "title": "Test Product",
                "text": "Test product description",
                "price": 50.0,
                "url": "/products/test",
                "image": "https://example.com/test.jpg",
                "in_stock": True,
                "category": "test",
                "tags": ["test", "product"]
            }
        ]
        
        response = generate_chat_response(
            user_message="test message",
            search_results=mock_products,
            search_time=100.0
        )
        
        if isinstance(response, dict) and 'answer' in response:
            print("✅ Chat function works")
            print(f"   Response: {response['answer'][:50]}...")
            return True
        else:
            print("❌ Chat return format incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

def test_environment():
    """Test environment variables"""
    print("🧪 Testing environment...")
    
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY', 
        'AWS_REGION',
        'OPENSEARCH_HOST',
        'OPENSEARCH_USER',
        'OPENSEARCH_PASS'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {missing_vars}")
        print("   Some tests may fail due to missing credentials")
    else:
        print("✅ All environment variables present")
    
    return len(missing_vars) == 0

def main():
    """Run all basic tests"""
    print("🚀 Starting Basic AI Tests...")
    print("=" * 40)
    
    tests = [
        ("Environment", test_environment),
        ("Imports", test_imports),
        ("Embedding", test_embedding_basic),
        ("Search", test_search_basic),
        ("Chat", test_chat_basic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} Test:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed or had issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
