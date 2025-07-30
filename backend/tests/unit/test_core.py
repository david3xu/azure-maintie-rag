"""
Unit tests for core infrastructure layer
Tests Azure client implementations and utilities
"""

import pytest
from unittest.mock import Mock, patch


class TestAzureClients:
    """Test Azure client implementations"""
    
    def test_openai_client_initialization(self):
        """Test OpenAI client can be initialized"""
        from core.azure_openai.openai_client import UnifiedAzureOpenAIClient
        # Mock config to avoid actual Azure connection
        with patch.dict('os.environ', {'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com'}):
            client = UnifiedAzureOpenAIClient()
            assert client is not None
    
    def test_storage_client(self):
        """Test storage client initialization"""
        from core.azure_storage.storage_client import UnifiedStorageClient
        # Test that the class can be imported (instantiation requires credentials)
        assert UnifiedStorageClient is not None
    
    def test_search_client_initialization(self):
        """Test search client initialization"""
        from core.azure_search.search_client import UnifiedSearchClient
        # Mock to avoid actual connection
        with patch('core.azure_search.search_client.SearchClient'):
            client = UnifiedSearchClient()
            assert client is not None


class TestCoreModels:
    """Test core data models"""
    
    def test_universal_rag_models(self):
        """Test Universal RAG model imports"""
        from core.models.universal_rag_models import (
            UniversalEntity,
            UniversalRelation,
            UniversalDocument
        )
        
        # Test entity creation
        entity = UniversalEntity(
            entity_id="test_1",
            text="Test Entity",
            entity_type="test"
        )
        assert entity.entity_id == "test_1"
        assert entity.text == "Test Entity"
        assert entity.entity_type == "test"
    
    def test_gnn_data_models(self):
        """Test GNN data model imports"""
        from core.models.gnn_data_models import (
            StandardizedEntity,
            StandardizedRelation,
            StandardizedGraphData
        )
        
        # Test entity creation
        entity = StandardizedEntity(
            entity_id="test_1",
            text="Test Entity",
            entity_type="test",
            confidence=0.95
        )
        assert entity.entity_id == "test_1"
        assert entity.text == "Test Entity"


class TestCoreUtilities:
    """Test core utility functions"""
    
    def test_config_loader(self):
        """Test configuration loader utility"""
        from core.utilities.config_loader import ConfigLoader
        # Test basic class instantiation
        loader = ConfigLoader("config")
        assert loader is not None
    
    def test_file_utils(self):
        """Test file utility functions"""
        from core.utilities.file_utils import FileUtils
        from pathlib import Path
        
        # Test directory creation
        test_path = FileUtils.ensure_directory("/tmp/test_dir")
        assert test_path.exists()
        
        # Cleanup
        test_path.rmdir()