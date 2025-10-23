#!/usr/bin/env python3
"""
Basic unit tests for AI functionalities
Tests embedding generation, search, and chat responses
"""

import unittest
import json
import os
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your modules
from search_api import create_embedding, search_products, generate_chat_response

class TestAIFunctionality(unittest.TestCase):
    """Basic tests for AI search and chat functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_query = "black running shorts"
        self.test_products = [
            {
                "product_id": "12345",
                "title": "Pace Run Shorts — Women",
                "text": "Featherweight 3\" quick-dry running shorts in black",
                "price": 49.0,
                "url": "/products/pace-run-shorts",
                "image": "https://cdn.shopify.com/s/files/1/0123/4567/8901/products/pace-shorts-black.jpg",
                "in_stock": True,
                "category": "shorts",
                "tags": ["quick-dry", "women", "running", "black"]
            },
            {
                "product_id": "67890", 
                "title": "Trail Running Shoes",
                "text": "Waterproof trail running shoes with aggressive grip",
                "price": 120.0,
                "url": "/products/trail-shoes",
                "image": "https://cdn.shopify.com/s/files/1/0123/4567/8901/products/trail-shoes-brown.jpg",
                "in_stock": True,
                "category": "shoes",
                "tags": ["waterproof", "trail", "running"]
            }
        ]
    
    def test_embedding_creation(self):
        """Test that embeddings are created successfully"""
        print("🧪 Testing embedding creation...")
        
        # Mock the Bedrock response
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = json.dumps({
            'embedding': [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        }).encode()
        
        with patch('search_api.bedrock.invoke_model', return_value=mock_response):
            embedding = create_embedding(self.test_query)
            
            # Assertions
            self.assertIsNotNone(embedding, "Embedding should not be None")
            self.assertIsInstance(embedding, list, "Embedding should be a list")
            self.assertEqual(len(embedding), 1536, "Embedding should have 1536 dimensions")
            print("✅ Embedding creation test passed")
    
    def test_embedding_creation_failure(self):
        """Test embedding creation handles errors gracefully"""
        print("🧪 Testing embedding creation error handling...")
        
        # Mock Bedrock to raise an exception
        with patch('search_api.bedrock.invoke_model', side_effect=Exception("API Error")):
            embedding = create_embedding(self.test_query)
            
            # Should return None on error
            self.assertIsNone(embedding, "Should return None on error")
            print("✅ Embedding error handling test passed")
    
    def test_search_products_basic(self):
        """Test basic product search functionality"""
        print("🧪 Testing product search...")
        
        # Mock the embedding creation
        mock_embedding = [0.1] * 1536
        with patch('search_api.create_embedding', return_value=mock_embedding):
            # Mock OpenSearch response
            mock_opensearch_response = {
                'hits': {
                    'hits': [
                        {
                            '_source': self.test_products[0],
                            '_score': 0.95
                        },
                        {
                            '_source': self.test_products[1], 
                            '_score': 0.87
                        }
                    ]
                }
            }
            
            with patch('search_api.opensearch_client.search', return_value=mock_opensearch_response):
                results, search_time = search_products(self.test_query, k=5)
                
                # Assertions
                self.assertIsInstance(results, list, "Results should be a list")
                self.assertGreater(len(results), 0, "Should return some results")
                self.assertIsInstance(search_time, float, "Search time should be a float")
                self.assertGreater(search_time, 0, "Search time should be positive")
                
                # Check result structure
                if results:
                    result = results[0]
                    self.assertIn('product_id', result, "Result should have product_id")
                    self.assertIn('title', result, "Result should have title")
                    self.assertIn('score', result, "Result should have score")
                
                print("✅ Product search test passed")
    
    def test_search_products_no_embedding(self):
        """Test search when embedding creation fails"""
        print("🧪 Testing search with no embedding...")
        
        # Mock embedding creation to return None
        with patch('search_api.create_embedding', return_value=None):
            results, search_time = search_products(self.test_query, k=5)
            
            # Should return empty results
            self.assertEqual(len(results), 0, "Should return empty results when no embedding")
            self.assertEqual(search_time, 0.0, "Search time should be 0 when no embedding")
            print("✅ Search no embedding test passed")
    
    def test_chat_response_generation(self):
        """Test chat response generation"""
        print("🧪 Testing chat response generation...")
        
        # Mock Bedrock response for chat
        mock_chat_response = {
            'body': MagicMock()
        }
        mock_chat_response['body'].read.return_value = json.dumps({
            'outputs': [{'text': 'Here are some great running shorts for you!'}]
        }).encode()
        
        with patch('search_api.bedrock.invoke_model', return_value=mock_chat_response):
            response = generate_chat_response(
                user_message=self.test_query,
                search_results=self.test_products,
                search_time=150.0
            )
            
            # Assertions
            self.assertIsInstance(response, dict, "Response should be a dictionary")
            self.assertIn('answer', response, "Response should have answer")
            self.assertIn('items_cited', response, "Response should have items_cited")
            self.assertIn('reasoning', response, "Response should have reasoning")
            
            self.assertIsInstance(response['answer'], str, "Answer should be a string")
            self.assertIsInstance(response['items_cited'], list, "Items cited should be a list")
            self.assertIsInstance(response['reasoning'], str, "Reasoning should be a string")
            
            print("✅ Chat response generation test passed")
    
    def test_chat_response_error_handling(self):
        """Test chat response error handling"""
        print("🧪 Testing chat response error handling...")
        
        # Mock Bedrock to raise an exception
        with patch('search_api.bedrock.invoke_model', side_effect=Exception("LLM Error")):
            response = generate_chat_response(
                user_message=self.test_query,
                search_results=self.test_products,
                search_time=150.0
            )
            
            # Should return error response
            self.assertIn('answer', response, "Should have answer field")
            self.assertIn("trouble processing", response['answer'].lower(), "Should indicate error")
            self.assertEqual(len(response['items_cited']), 0, "Should have empty items_cited on error")
            print("✅ Chat error handling test passed")

def run_tests():
    """Run all tests with simple output"""
    print("🚀 Starting AI Functionality Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIFunctionality)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All tests passed!")
        print(f"✅ Tests run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"❌ Errors: {len(result.errors)}")
    else:
        print("❌ Some tests failed!")
        print(f"✅ Tests run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"❌ Errors: {len(result.errors)}")
        
        # Print failure details
        for test, traceback in result.failures + result.errors:
            print(f"\n❌ {test}: {traceback}")

if __name__ == "__main__":
    run_tests()
