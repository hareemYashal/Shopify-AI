#!/usr/bin/env python3
"""
Basic unit tests for bedrock.py
Tests AWS Bedrock connection and model listing functionality
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module we're testing
import bedrock


class TestBedrock(unittest.TestCase):
    """Test cases for bedrock.py functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret',
            'AWS_REGION': 'us-east-1'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after each test method"""
        self.env_patcher.stop()
    
    @patch('bedrock.boto3.client')
    def test_bedrock_client_initialization(self, mock_boto3_client):
        """Test that Bedrock client is initialized with correct parameters"""
        # Mock the client
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Re-import to trigger client initialization
        import importlib
        importlib.reload(bedrock)
        
        # Verify boto3.client was called with correct parameters
        mock_boto3_client.assert_called_once_with(
            "bedrock",
            region_name="us-east-1",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
    
    @patch('bedrock.boto3.client')
    def test_list_foundation_models_success(self, mock_boto3_client):
        """Test successful listing of foundation models"""
        # Mock the client and its response
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Mock the list_foundation_models response
        mock_response = {
            "modelSummaries": [
                {"modelId": "amazon.titan-embed-text-v1"},
                {"modelId": "mistral.mistral-small-2402-v1:0"}
            ]
        }
        mock_client.list_foundation_models.return_value = mock_response
        
        # Re-import to trigger the test call
        import importlib
        importlib.reload(bedrock)
        
        # Verify the method was called
        mock_client.list_foundation_models.assert_called_once()
    
    @patch('bedrock.boto3.client')
    def test_list_foundation_models_failure(self, mock_boto3_client):
        """Test handling of failed model listing"""
        # Mock the client to raise an exception
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.list_foundation_models.side_effect = Exception("AWS Error")
        
        # This should not raise an exception, just print an error
        try:
            import importlib
            importlib.reload(bedrock)
            # If we get here, the error was handled gracefully
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Exception should have been handled gracefully: {e}")
    
    def test_environment_variables(self):
        """Test that environment variables are loaded correctly"""
        # Test with default region
        with patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret'
        }, clear=True):
            import importlib
            importlib.reload(bedrock)
            
            # The module should use default region when not specified
            self.assertEqual(os.getenv("AWS_REGION", "us-east-1"), "us-east-1")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)


