#!/usr/bin/env python3
"""
Test script for preferences integration
Tests search with preference-based ranking and filtering
"""

import json
import requests
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"


def print_separator():
    print("\n" + "=" * 80 + "\n")


def test_health_check():
    """Test health check endpoint with preferences"""
    print("🔍 Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Health check passed")
        print(f"   Preferences loaded: {data.get('preferences', {})}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False


def test_preferences_reload():
    """Test preferences reload endpoint"""
    print("🔍 Testing Preferences Reload...")
    response = requests.post(f"{BASE_URL}/admin/reload-preferences")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Preferences reload passed")
        print(f"   Config: {json.dumps(data.get('config', {}), indent=2)}")
        return True
    else:
        print(f"❌ Preferences reload failed: {response.status_code}")
        return False


def test_search_with_budget(query: str, expected_max_price: float):
    """Test search with budget constraint from query"""
    print(f"🔍 Testing Search with Budget: '{query}'")
    
    response = requests.post(
        f"{BASE_URL}/search-fast",
        json={"query": query}
    )
    
    if response.status_code == 200:
        data = response.json()
        items = data.get('items', [])
        
        print(f"✅ Search completed in {data.get('search_time_ms', 0):.2f}ms")
        print(f"   Found {len(items)} results")
        
        # Check if all prices are under the budget
        over_budget = [item for item in items if item.get('price', 0) > expected_max_price]
        
        if over_budget:
            print(f"   ⚠️  Found {len(over_budget)} items over budget ${expected_max_price}")
        else:
            print(f"   ✅ All items under budget ${expected_max_price}")
        
        # Show top 3 results
        for i, item in enumerate(items[:3], 1):
            print(f"   {i}. {item['title']} - ${item['price']}")
            print(f"      Reason: {item.get('reason', 'N/A')}")
            print(f"      Score: {item.get('score', 0):.4f} → Boosted: {item.get('boosted_score', 0):.4f}")
        
        return True
    else:
        print(f"❌ Search failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def test_search_with_tags():
    """Test search with tag boosting"""
    print(f"🔍 Testing Search with Tag Boosting: 'quick-dry running shorts'")
    
    response = requests.post(
        f"{BASE_URL}/search-fast",
        json={"query": "quick-dry running shorts"}
    )
    
    if response.status_code == 200:
        data = response.json()
        items = data.get('items', [])
        
        print(f"✅ Search completed in {data.get('search_time_ms', 0):.2f}ms")
        print(f"   Found {len(items)} results")
        
        # Show top 3 results with tag analysis
        for i, item in enumerate(items[:3], 1):
            tags = item.get('tags', [])
            boosted_tags = [tag for tag in tags if tag in ['quick-dry', 'running', 'women']]
            
            print(f"   {i}. {item['title']} - ${item['price']}")
            print(f"      Tags: {', '.join(tags[:5])}")
            print(f"      Boosted tags: {', '.join(boosted_tags) if boosted_tags else 'None'}")
            print(f"      Score: {item.get('score', 0):.4f} → Boosted: {item.get('boosted_score', 0):.4f}")
        
        return True
    else:
        print(f"❌ Search failed: {response.status_code}")
        return False


def test_in_stock_filter():
    """Test default in-stock filtering from preferences"""
    print(f"🔍 Testing In-Stock Filter (from preferences)")
    
    response = requests.post(
        f"{BASE_URL}/search-fast",
        json={"query": "running shoes"}
    )
    
    if response.status_code == 200:
        data = response.json()
        items = data.get('items', [])
        
        print(f"✅ Search completed")
        print(f"   Found {len(items)} results")
        
        # Check stock status
        in_stock = [item for item in items if item.get('in_stock', False)]
        out_of_stock = [item for item in items if not item.get('in_stock', True)]
        
        print(f"   In stock: {len(in_stock)}")
        print(f"   Out of stock: {len(out_of_stock)}")
        
        if out_of_stock:
            print(f"   ⚠️  Found out-of-stock items (preferences may not be applied)")
        else:
            print(f"   ✅ All items in stock (preference filter working)")
        
        return True
    else:
        print(f"❌ Search failed: {response.status_code}")
        return False


def test_chat_with_preferences():
    """Test chat endpoint with preferences-based tone"""
    print(f"🔍 Testing Chat with Preferences")
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": "I need black running shorts under $50"}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"✅ Chat completed in {data.get('search_time_ms', 0):.2f}ms")
        print(f"   Answer: {data.get('answer', 'N/A')[:200]}...")
        print(f"   Items cited: {data.get('items_cited', [])}")
        print(f"   Reasoning: {data.get('reasoning', 'N/A')}")
        
        # Check if tone is appropriate (should be friendly, concise)
        answer = data.get('answer', '').lower()
        
        # Look for friendly indicators
        friendly_words = ['here', 'perfect', 'great', 'check out', 'recommend']
        has_friendly_tone = any(word in answer for word in friendly_words)
        
        if has_friendly_tone:
            print(f"   ✅ Response has friendly tone")
        else:
            print(f"   ⚠️  Response may not match preferred tone")
        
        return True
    else:
        print(f"❌ Chat failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def test_color_matching():
    """Test color-based boosting"""
    print(f"🔍 Testing Color Matching: 'black shorts'")
    
    response = requests.post(
        f"{BASE_URL}/search-fast",
        json={"query": "black shorts"}
    )
    
    if response.status_code == 200:
        data = response.json()
        items = data.get('items', [])
        
        print(f"✅ Search completed")
        
        # Check for color matches
        for i, item in enumerate(items[:3], 1):
            title = item.get('title', '').lower()
            has_black = 'black' in title or 'black' in str(item.get('tags', []))
            
            print(f"   {i}. {item['title']} - ${item['price']}")
            print(f"      Contains 'black': {has_black}")
            print(f"      Boosted score: {item.get('boosted_score', 0):.4f}")
        
        return True
    else:
        print(f"❌ Search failed: {response.status_code}")
        return False


def run_all_tests():
    """Run all integration tests"""
    print("🚀 Starting Preferences Integration Tests")
    print_separator()
    
    results = {
        "passed": 0,
        "failed": 0,
        "total": 0
    }
    
    tests = [
        ("Health Check", test_health_check),
        ("Preferences Reload", test_preferences_reload),
        ("Search with Budget", lambda: test_search_with_budget("running shorts under $50", 50)),
        ("Search with Tags", test_search_with_tags),
        ("In-Stock Filter", test_in_stock_filter),
        ("Chat with Preferences", test_chat_with_preferences),
        ("Color Matching", test_color_matching),
    ]
    
    for test_name, test_func in tests:
        results["total"] += 1
        print_separator()
        
        try:
            if test_func():
                results["passed"] += 1
            else:
                results["failed"] += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results["failed"] += 1
        
        time.sleep(0.5)  # Brief pause between tests
    
    # Summary
    print_separator()
    print("📊 Test Summary")
    print(f"   Total: {results['total']}")
    print(f"   Passed: {results['passed']} ✅")
    print(f"   Failed: {results['failed']} ❌")
    
    success_rate = (results['passed'] / results['total']) * 100 if results['total'] > 0 else 0
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if results['failed'] == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {results['failed']} test(s) failed")
    
    print_separator()


if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("⚡ Preferences Integration Test Suite")
    print("Make sure the API server is running on http://localhost:8000")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Server is running")
            run_all_tests()
        else:
            print("❌ Server returned unexpected status")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start the API server first:")
        print("   python search_api.py")

