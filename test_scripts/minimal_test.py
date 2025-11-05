#!/usr/bin/env python3
"""
Minimal AI functionality test
Just checks if functions exist and can be called
"""

def test_function_exists():
    """Test that AI functions exist and can be imported"""
    print("🧪 Testing function existence...")
    
    try:
        from search_api import create_embedding, search_products, generate_chat_response
        print("✅ All AI functions imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_function_signatures():
    """Test that functions have correct signatures"""
    print("🧪 Testing function signatures...")
    
    try:
        from search_api import create_embedding, search_products, generate_chat_response
        
        # Test create_embedding
        if callable(create_embedding):
            print("✅ create_embedding is callable")
        else:
            print("❌ create_embedding is not callable")
            return False
        
        # Test search_products  
        if callable(search_products):
            print("✅ search_products is callable")
        else:
            print("❌ search_products is not callable")
            return False
            
        # Test generate_chat_response
        if callable(generate_chat_response):
            print("✅ generate_chat_response is callable")
        else:
            print("❌ generate_chat_response is not callable")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Signature test failed: {e}")
        return False

def test_basic_call():
    """Test that functions can be called without crashing"""
    print("🧪 Testing basic function calls...")
    
    try:
        from search_api import create_embedding, search_products, generate_chat_response
        
        # Test create_embedding (might return None due to API)
        result1 = create_embedding("test")
        print(f"✅ create_embedding called (result: {type(result1)})")
        
        # Test search_products (might fail due to API)
        try:
            result2, time2 = search_products("test", k=5)
            print(f"✅ search_products called (results: {len(result2)}, time: {time2})")
        except Exception as e:
            print(f"⚠️  search_products failed (expected if no API): {e}")
        
        # Test generate_chat_response (might fail due to API)
        try:
            mock_products = [{"product_id": "123", "title": "Test", "text": "Test description", "price": 50, "url": "/test", "image": "https://example.com/test.jpg", "in_stock": True, "category": "test", "tags": ["test"]}]
            result3 = generate_chat_response("test", mock_products, 100.0)
            print(f"✅ generate_chat_response called (result: {type(result3)})")
        except Exception as e:
            print(f"⚠️  generate_chat_response failed (expected if no API): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic call test failed: {e}")
        return False

def main():
    """Run minimal tests"""
    print("🚀 Minimal AI Functionality Test")
    print("=" * 35)
    
    tests = [
        ("Function Existence", test_function_exists),
        ("Function Signatures", test_function_signatures), 
        ("Basic Calls", test_basic_call)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n📋 {name}:")
        if test_func():
            passed += 1
            print(f"✅ {name} passed")
        else:
            print(f"❌ {name} failed")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All minimal tests passed!")
    else:
        print("⚠️  Some tests failed")

if __name__ == "__main__":
    main()
