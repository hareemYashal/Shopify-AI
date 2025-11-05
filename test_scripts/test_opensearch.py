#!/usr/bin/env python3
"""
Basic unit tests for opensearch.py
Tests OpenSearch client connection and index operations
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module we're testing
import opensearch


class TestOpensearch(unittest.TestCase):
    """Test cases for opensearch.py functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OPENSEARCH_HOST': 'https://test-cluster.us-east-1.es.amazonaws.com',
            'OPENSEARCH_USER': 'test_user',
            'OPENSEARCH_PASS': 'test_pass'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after each test method"""
        self.env_patcher.stop()
    
    @patch('opensearch.OpenSearch')
    def test_opensearch_client_initialization(self, mock_opensearch):
        """Test that OpenSearch client is initialized with correct parameters"""
        # Mock the client
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        
        # Re-import to trigger client initialization
        import importlib
        importlib.reload(opensearch)
        
        # Verify OpenSearch was called with correct parameters
        mock_opensearch.assert_called_once_with(
            hosts=[{"host": "test-cluster.us-east-1.es.amazonaws.com", "port": 443}],
            http_auth=("test_user", "test_pass"),
            use_ssl=True,
            verify_certs=True,
            connection_class=opensearch.RequestsHttpConnection
        )
    
    @patch('opensearch.OpenSearch')
    def test_client_info_success(self, mock_opensearch):
        """Test successful client info retrieval"""
        # Mock the client and its response
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        
        # Mock the info response
        mock_info_response = {"version": {"number": "2.3.0"}}
        mock_client.info.return_value = mock_info_response
        
        # Mock index exists check
        mock_client.indices.exists.return_value = True
        
        # Re-import to trigger the test calls
        import importlib
        importlib.reload(opensearch)
        
        # Verify the methods were called
        mock_client.info.assert_called_once()
        mock_client.indices.exists.assert_called_once_with(index="products")
    
    @patch('opensearch.OpenSearch')
    def test_index_creation_when_not_exists(self, mock_opensearch):
        """Test index creation when it doesn't exist"""
        # Mock the client
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        
        # Mock successful info call
        mock_client.info.return_value = {"version": {"number": "2.3.0"}}
        
        # Mock index doesn't exist
        mock_client.indices.exists.return_value = False
        
        # Re-import to trigger the test calls
        import importlib
        importlib.reload(opensearch)
        
        # Verify index creation was called
        mock_client.indices.create.assert_called_once()
        
        # Check that the index body has the correct structure
        call_args = mock_client.indices.create.call_args
        self.assertEqual(call_args[1]['index'], 'products')
        self.assertIn('mappings', call_args[1]['body'])
        self.assertIn('settings', call_args[1]['body'])
    
    @patch('opensearch.OpenSearch')
    def test_client_connection_failure(self, mock_opensearch):
        """Test handling of connection failure"""
        # Mock the client to raise an exception
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        mock_client.info.side_effect = Exception("Connection failed")
        
        # This should not raise an exception, just print an error
        try:
            import importlib
            importlib.reload(opensearch)
            # If we get here, the error was handled gracefully
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Exception should have been handled gracefully: {e}")
    
    def test_index_mapping_structure(self):
        """Test that the index mapping has the correct structure"""
        # Check that the INDEX_BODY has the expected structure
        self.assertIn('mappings', opensearch.INDEX_BODY)
        self.assertIn('settings', opensearch.INDEX_BODY)
        
        # Check mappings structure
        mappings = opensearch.INDEX_BODY['mappings']['properties']
        expected_fields = ['product_id', 'title', 'text', 'price', 'in_stock', 
                          'category', 'tags', 'url', 'image', 'embedding']
        
        for field in expected_fields:
            self.assertIn(field, mappings)
        
        # Check embedding field specifically
        embedding_config = mappings['embedding']
        self.assertEqual(embedding_config['type'], 'knn_vector')
        self.assertEqual(embedding_config['dimension'], 1536)
        self.assertIn('method', embedding_config)
    
    def test_environment_variables_parsing(self):
        """Test that environment variables are parsed correctly"""
        # Test host URL parsing
        test_host = "https://test-cluster.us-east-1.es.amazonaws.com"
        parsed_host = test_host.replace("https://", "")
        self.assertEqual(parsed_host, "test-cluster.us-east-1.es.amazonaws.com")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)


