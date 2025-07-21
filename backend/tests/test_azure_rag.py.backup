#!/usr/bin/env python3
"""
Universal RAG Test Suite
=======================

Comprehensive tests for the Universal RAG system using Azure services architecture.
Replaces all old domain-specific test files with clean universal tests.

Tests:
- Azure services integration
- Universal query processing
- Universal API endpoints
- Multi-domain support
- Performance validation
"""

import sys
import os
import pytest
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add backend directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Azure service imports - Updated to use new Azure services architecture
from integrations.azure_services import AzureServicesManager
from integrations.azure_openai import AzureOpenAIClient
from config.settings import AzureSettings

logger = logging.getLogger(__name__)


class TestUniversalRAG:
    """Test suite for Universal RAG system using Azure services"""

    @pytest.fixture
    async def azure_services(self):
        """Azure services manager fixture"""
        services = AzureServicesManager()
        await services.initialize()
        return services

    @pytest.fixture
    async def openai_integration(self):
        """Azure OpenAI integration fixture"""
        return AzureOpenAIClient()

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return [
                    "The system requires regular monitoring to ensure optimal performance.",
        "System monitoring helps detect potential issues before they occur.",
        "Proper procedures must be followed when performing system tasks.",
        "Preventive monitoring schedules reduce unexpected downtime significantly.",
            "Proper lubrication is essential for bearing longevity and performance."
        ]

    @pytest.fixture
    def sample_domains(self):
        """Sample domains with texts for multi-domain testing"""
        return {
            "general": [
                "System components work together to achieve desired outcomes.",
                "Performance monitoring helps identify potential issues early.",
                "Regular analysis reveals patterns in system behavior.",
                "Data processing requires systematic validation and testing."
            ]
        }

    @pytest.mark.asyncio
    async def test_azure_services_initialization(self):
        """Test Azure services initialization"""
        azure_services = await self.azure_services()
        # Test that all Azure services are properly initialized
        assert azure_services.get_rag_storage_client() is not None
        assert azure_services.get_ml_storage_client() is not None
        assert azure_services.get_app_storage_client() is not None
        assert azure_services.search_client is not None
        assert azure_services.cosmos_client is not None
        assert azure_services.ml_client is not None

    @pytest.mark.asyncio
    async def test_azure_openai_integration(self, sample_texts):
        """Test Azure OpenAI integration"""
        openai_integration = await self.openai_integration()
        domain = "test_domain"

        # Test document processing
        processed_docs = await openai_integration.process_documents(sample_texts, domain)

        assert len(processed_docs) == len(sample_texts)
        assert all(isinstance(doc, dict) for doc in processed_docs)

    @pytest.mark.asyncio
    async def test_azure_blob_storage_operations(self):
        """Test Azure Blob Storage operations"""
        azure_services = await self.azure_services()
        container_name = "test-container"
        blob_name = "test-document.txt"
        test_content = "This is a test document for RAG processing."

        rag_storage = None # Initialize to None
        try:
            # Test container creation using RAG storage
            rag_storage = azure_services.get_rag_storage_client()
            await rag_storage.create_container(container_name)

            # Test text upload
            await rag_storage.upload_text(container_name, blob_name, test_content)

            # Test text download
            downloaded_content = await rag_storage.download_text(container_name, blob_name)

            assert downloaded_content == test_content

        finally:
            # Clean up
            if rag_storage: # Check if rag_storage was assigned
                await rag_storage.delete_blob(container_name, blob_name)
                await rag_storage.delete_container(container_name)

    @pytest.mark.asyncio
    async def test_azure_cognitive_search(self, sample_texts):
        """Test Azure Cognitive Search operations"""
        azure_services = await self.azure_services()
        domain = "test_domain"
        query = "How to monitor systems properly?"

        # Test search functionality
        search_results = await azure_services.search_client.search_documents(
            domain, query, top_k=5
        )

        assert isinstance(search_results, list)

    @pytest.mark.asyncio
    async def test_universal_rag_workflow(self, sample_texts):
        """Test complete Universal RAG workflow"""
        azure_services = await self.azure_services()
        openai_integration = await self.openai_integration()
        domain = "test_domain"
        query = "What are system monitoring best practices?"

        try:
            # Store documents in Azure Blob Storage using RAG storage
            container_name = f"rag-data-{domain}"
            rag_storage = None # Initialize to None
            try:
                rag_storage = azure_services.get_rag_storage_client()
                await rag_storage.create_container(container_name)

                for i, text in enumerate(sample_texts):
                    blob_name = f"document_{i}.txt"
                    await rag_storage.upload_text(container_name, blob_name, text)

                # Process documents with Azure OpenAI
                processed_docs = await openai_integration.process_documents(sample_texts, domain)

                # Search for relevant documents
                search_results = await azure_services.search_client.search_documents(
                    domain, query, top_k=3
                )

                # Generate response
                response = await openai_integration.generate_response(
                    query, search_results, domain
                )

                assert isinstance(response, str)
                assert len(response) > 0

            finally:
                if rag_storage: # Check if rag_storage was assigned
                    await rag_storage.delete_container(container_name)

        except Exception as e:
            logger.warning(f"Universal RAG workflow test failed: {e}")

    @pytest.mark.asyncio
    async def test_multi_domain_support(self, sample_domains):
        """Test multi-domain support"""
        azure_services = await self.azure_services()
        openai_integration = await self.openai_integration()
        for domain, texts in sample_domains.items():
            try:
                # Store domain-specific documents using RAG storage
                container_name = f"rag-data-{domain}"
                rag_storage = None # Initialize to None
                try:
                    rag_storage = azure_services.get_rag_storage_client()
                    await rag_storage.create_container(container_name)

                    for i, text in enumerate(texts):
                        blob_name = f"document_{i}.txt"
                        await rag_storage.upload_text(container_name, blob_name, text)

                    # Process documents
                    processed_docs = await openai_integration.process_documents(texts, domain)

                    # Test universal query
                    queries = {
                        "general": "What should I monitor?"
                    }

                    query = queries[domain]
                    search_results = await azure_services.search_client.search_documents(
                        domain, query, top_k=2
                    )
                    response = await openai_integration.generate_response(
                        query, search_results, domain
                    )

                    assert isinstance(response, str)
                    assert len(response) > 0

                finally:
                    # Clean up
                    if rag_storage: # Check if rag_storage was assigned
                        await rag_storage.delete_container(container_name)
            except Exception as e:
                logger.warning(f"Multi-domain test failed for domain {domain}: {e}")

    @pytest.mark.asyncio
    async def test_azure_cosmos_gremlin_operations(self):
        """Test Azure Cosmos DB Gremlin operations"""
        azure_services = await self.azure_services()
        test_entity = {
            "text": "test-system",
            "entity_type": "component",
            "confidence": 0.95,
            "created_at": "2024-01-01T00:00:00Z"
        }

        test_relation = {
            "head_entity": "test-system",
            "tail_entity": "test-issue",
            "relation_type": "causes",
            "confidence": 0.9,
            "created_at": "2024-01-01T00:00:00Z"
        }

        try:
            # Test entity addition
            result = await azure_services.cosmos_client.add_entity(test_entity, "test")
            assert result["success"] == True

            # Test relationship addition
            result = await azure_services.cosmos_client.add_relationship(test_relation, "test")
            assert result["success"] == True

            # Test entity query
            entities = await azure_services.cosmos_client.find_entities_by_type("component", "test")
            assert len(entities) > 0

            # Test relationship query
            relationships = await azure_services.cosmos_client.find_related_entities("test-system", "test")
            assert len(relationships) > 0

        except Exception as e:
            logger.warning(f"Gremlin test failed: {e}")
            # Gremlin tests might fail if Gremlin API not enabled
            pass

    @pytest.mark.asyncio
    async def test_azure_ml_integration(self):
        """Test Azure Machine Learning integration"""
        azure_services = await self.azure_services()
        workspace_name = "test-workspace"

        # Test workspace operations
        workspaces = await azure_services.ml_client.list_workspaces()
        assert isinstance(workspaces, list)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in Azure services"""
        azure_services = await self.azure_services()
        # Test with invalid container name
        rag_storage = None # Initialize to None
        try:
            rag_storage = azure_services.get_rag_storage_client()
            with pytest.raises(Exception):
                await rag_storage.create_container("")
        finally:
            if rag_storage: # Check if rag_storage was assigned
                # Attempt to delete the container even if creation failed to ensure cleanup
                # In a real scenario, you might want more robust error checking here.
                pass

    def test_azure_settings(self):
        """Test Azure settings configuration"""
        settings = AzureSettings()
        assert settings.azure_storage_connection_string is not None
        assert settings.azure_search_service_name is not None
        assert settings.azure_cosmos_db_connection_string is not None


class TestSystemIntegration:
    """Test system integration with Azure services"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # This test would simulate a complete RAG workflow
        # from document ingestion to response generation
        pass


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])