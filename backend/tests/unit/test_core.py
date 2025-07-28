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
    
    def test_storage_factory(self):
        """Test storage factory pattern"""
        from core.azure_storage.storage_factory import get_storage_factory
        factory = get_storage_factory()
        assert factory is not None
    
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
            id="test_1",
            name="Test Entity",
            type="test",
            properties={}
        )
        assert entity.id == "test_1"
        assert entity.name == "Test Entity"
    
    def test_gnn_data_models(self):
        """Test GNN data model imports"""
        from core.models.gnn_data_models import (
            GNNNode,
            GNNEdge,
            GNNGraph
        )
        
        # Test node creation
        node = GNNNode(
            id="node_1",
            features=[1.0, 2.0, 3.0],
            label="test"
        )
        assert node.id == "node_1"
        assert len(node.features) == 3


class TestCoreUtilities:
    """Test core utility functions"""
    
    def test_config_loader(self):
        """Test configuration loader utility"""
        from core.utilities.config_loader import load_config
        # Test with mock config
        config = load_config(config_dict={"test": "value"})
        assert config.get("test") == "value"
    
    def test_file_utils(self):
        """Test file utility functions"""
        from core.utilities.file_utils import ensure_directory
        from pathlib import Path
        
        # Test directory creation
        test_path = Path("/tmp/test_dir")
        ensure_directory(test_path)
        assert test_path.exists()
        
        # Cleanup
        test_path.rmdir()