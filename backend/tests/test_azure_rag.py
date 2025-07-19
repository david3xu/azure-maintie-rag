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

# Add backend directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Azure service imports - Updated to use new Azure services architecture
from integrations.azure_services import AzureServicesManager
from integrations.azure_openai import AzureOpenAIIntegration
from config.azure_settings import AzureSettings


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
        return AzureOpenAIIntegration()

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return [
            "The system requires regular maintenance to ensure optimal performance.",
            "Equipment monitoring helps detect potential failures before they occur.",
            "Safety procedures must be followed when performing maintenance tasks.",
            "Preventive maintenance schedules reduce unexpected downtime significantly.",
            "Proper lubrication is essential for bearing longevity and performance."
        ]

    @pytest.fixture
    def sample_domains(self):
        """Sample domains with texts for multi-domain testing"""
        return {
            "medical": [
                "Patient symptoms include fever and fatigue requiring immediate attention.",
                "Diagnosis involves comprehensive examination and laboratory test results."
            ],
            "legal": [
                "Contract terms must be clearly defined and legally enforceable.",
                "Liability clauses protect parties from unforeseen legal circumstances."
            ],
            "finance": [
                "Investment risk assessment requires careful portfolio analysis.",
                "Market volatility significantly affects asset valuation and returns."
            ]
        }

    @pytest.mark.asyncio
    async def test_azure_services_initialization(self, azure_services):
        """Test Azure services initialization"""
        # Test that all Azure services are properly initialized
        assert azure_services.storage_client is not None
        assert azure_services.search_client is not None
        assert azure_services.cosmos_client is not None
        assert azure_services.ml_client is not None

    @pytest.mark.asyncio
    async def test_azure_openai_integration(self, openai_integration, sample_texts):
        """Test Azure OpenAI integration"""
        domain = "test_domain"

        # Test document processing
        processed_docs = await openai_integration.process_documents(sample_texts, domain)

        assert len(processed_docs) == len(sample_texts)
        assert all(isinstance(doc, dict) for doc in processed_docs)

    @pytest.mark.asyncio
    async def test_azure_blob_storage_operations(self, azure_services):
        """Test Azure Blob Storage operations"""
        container_name = "test-container"
        blob_name = "test-document.txt"
        test_content = "This is a test document for RAG processing."

        try:
            # Test container creation
            await azure_services.storage_client.create_container(container_name)

            # Test text upload
            await azure_services.storage_client.upload_text(container_name, blob_name, test_content)

            # Test text download
            downloaded_content = await azure_services.storage_client.download_text(container_name, blob_name)

            assert downloaded_content == test_content

        finally:
            # Clean up
            await azure_services.storage_client.delete_blob(container_name, blob_name)
            await azure_services.storage_client.delete_container(container_name)

    @pytest.mark.asyncio
    async def test_azure_cognitive_search(self, azure_services, sample_texts):
        """Test Azure Cognitive Search operations"""
        domain = "test_domain"
        query = "How to maintain equipment properly?"

        # Test search functionality
        search_results = await azure_services.search_client.search_documents(
            domain, query, top_k=5
        )

        assert isinstance(search_results, list)

    @pytest.mark.asyncio
    async def test_universal_rag_workflow(self, azure_services, openai_integration, sample_texts):
        """Test complete Universal RAG workflow"""
        domain = "test_domain"
        query = "What are maintenance best practices?"

        try:
            # Store documents in Azure Blob Storage
            container_name = f"rag-data-{domain}"
            await azure_services.storage_client.create_container(container_name)

            for i, text in enumerate(sample_texts):
                blob_name = f"document_{i}.txt"
                await azure_services.storage_client.upload_text(container_name, blob_name, text)

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
            # Clean up
            await azure_services.storage_client.delete_container(container_name)

    @pytest.mark.asyncio
    async def test_multi_domain_support(self, azure_services, openai_integration, sample_domains):
        """Test multi-domain support"""
        for domain, texts in sample_domains.items():
            try:
                # Store domain-specific documents
                container_name = f"rag-data-{domain}"
                await azure_services.storage_client.create_container(container_name)

                for i, text in enumerate(texts):
                    blob_name = f"document_{i}.txt"
                    await azure_services.storage_client.upload_text(container_name, blob_name, text)

                # Process documents
                processed_docs = await openai_integration.process_documents(texts, domain)

                # Test domain-specific query
                queries = {
                    "medical": "What symptoms require attention?",
                    "legal": "What are contract requirements?",
                    "finance": "How to assess investment risk?"
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
                await azure_services.storage_client.delete_container(container_name)

    @pytest.mark.asyncio
    async def test_azure_cosmos_db_operations(self, azure_services):
        """Test Azure Cosmos DB operations"""
        database_name = "test-database"
        container_name = "test-container"
        test_document = {
            "id": "test-doc-1",
            "domain": "test",
            "content": "Test document for RAG processing",
            "metadata": {"source": "test", "timestamp": "2024-01-01"}
        }

        try:
            # Test database and container creation
            await azure_services.cosmos_client.create_database(database_name)
            await azure_services.cosmos_client.create_container(database_name, container_name)

            # Test document creation
            await azure_services.cosmos_client.create_document(database_name, container_name, test_document)

            # Test document retrieval
            retrieved_doc = await azure_services.cosmos_client.get_document(
                database_name, container_name, "test-doc-1"
            )

            assert retrieved_doc["content"] == test_document["content"]

        finally:
            # Clean up
            await azure_services.cosmos_client.delete_database(database_name)

    @pytest.mark.asyncio
    async def test_azure_ml_integration(self, azure_services):
        """Test Azure Machine Learning integration"""
        workspace_name = "test-workspace"

        # Test workspace operations
        workspaces = await azure_services.ml_client.list_workspaces()
        assert isinstance(workspaces, list)

    @pytest.mark.asyncio
    async def test_error_handling(self, azure_services):
        """Test error handling in Azure services"""
        # Test with invalid container name
        with pytest.raises(Exception):
            await azure_services.storage_client.create_container("")

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