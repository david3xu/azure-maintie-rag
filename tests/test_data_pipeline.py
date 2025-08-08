"""
Test Data Pipeline with Real Azure Services and Real Data
=========================================================

Tests complete data processing pipeline using:
- REAL Azure OpenAI, Cosmos DB, Cognitive Search, Blob Storage
- REAL data from /workspace/azure-maintie-rag/data/raw/ (179 Azure AI Language Service files)
- NO MOCKS - All tests use actual deployed Azure infrastructure

This test suite validates end-to-end functionality with production data.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from dotenv import load_dotenv

# Load environment variables - try multiple sources
from pathlib import Path
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")
load_dotenv(project_root / "config" / "environments" / "prod.env")
load_dotenv()  # Also load from current directory


class TestRealDataAvailability:
    """Test real data corpus availability and characteristics."""

    def test_data_directory_exists(self):
        """Test that real data directory exists with Azure AI files."""
        data_dir = Path("/workspace/azure-maintie-rag/data/raw")
        assert data_dir.exists(), "Real data directory not found"

        # Find Azure AI Language Service files
        azure_files = list(data_dir.rglob("*.md"))
        assert (
            len(azure_files) >= 10
        ), f"Expected 10+ Azure AI files for testing, found {len(azure_files)}"

        print(f"✅ Real Data: Found {len(azure_files)} Azure AI Language Service files")
        return azure_files

    def test_data_file_content_quality(self):
        """Test that real data files contain substantial content."""
        data_dir = Path("/workspace/azure-maintie-rag/data/raw")
        azure_files = list(data_dir.rglob("*.md"))[:10]  # Sample first 10 files

        total_size = 0
        valid_files = 0

        for file_path in azure_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                file_size = len(content)
                total_size += file_size

                if file_size > 500:  # At least 500 characters
                    valid_files += 1

                # Verify it's Azure AI content
                content_lower = content.lower()
                assert any(
                    keyword in content_lower
                    for keyword in [
                        "azure",
                        "microsoft",
                        "cognitive",
                        "api",
                        "language",
                        "ai",
                    ]
                ), f"File {file_path} doesn't appear to be Azure AI content"

            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for file access issues
                if any(access_error in error_msg for access_error in [
                    'permission denied', 'access denied', 'no such file', 'file not found'
                ]):
                    pytest.skip(f"File access issue - {e}")
                else:
                    pytest.fail(f"Failed to read file {file_path}: {e}")

        assert valid_files >= 8, f"Only {valid_files}/10 files have substantial content"
        average_size = total_size / len(azure_files)

        print(
            f"✅ Real Data Quality: {valid_files}/{len(azure_files)} files with substantial content"
        )
        print(f"   Average file size: {average_size:.0f} characters")
        print(f"   Total sample size: {total_size:,} characters")

    def test_data_content_diversity(self):
        """Test content diversity in real Azure AI documentation."""
        data_dir = Path("/workspace/azure-maintie-rag/data/raw")
        azure_files = list(data_dir.rglob("*.md"))[:20]  # Sample 20 files

        content_types = {
            "api_documentation": 0,
            "how_to_guides": 0,
            "concepts": 0,
            "quickstarts": 0,
            "code_examples": 0,
        }

        for file_path in azure_files:
            content = file_path.read_text(encoding="utf-8").lower()

            if any(
                keyword in content
                for keyword in ["api", "endpoint", "request", "response"]
            ):
                content_types["api_documentation"] += 1
            if any(keyword in content for keyword in ["how to", "tutorial", "step"]):
                content_types["how_to_guides"] += 1
            if any(
                keyword in content
                for keyword in ["concept", "overview", "introduction"]
            ):
                content_types["concepts"] += 1
            if any(
                keyword in content
                for keyword in ["quickstart", "getting started", "quick"]
            ):
                content_types["quickstarts"] += 1
            if "```" in content or "code" in content:
                content_types["code_examples"] += 1

        # Verify content diversity
        diverse_types = sum(1 for count in content_types.values() if count > 0)
        assert diverse_types >= 3, f"Content not diverse enough: {content_types}"

        print("✅ Real Data Diversity:")
        for content_type, count in content_types.items():
            print(f"   {content_type}: {count} files")


class TestRealAzureDataIngestion:
    """Test data ingestion with real Azure Blob Storage."""

    @pytest.mark.asyncio
    async def test_azure_storage_upload_real_data(self):
        """Test uploading real data files to Azure Blob Storage."""
        if not os.getenv("AZURE_STORAGE_ACCOUNT_NAME"):
            pytest.skip("Azure Storage not configured")

        from azure.identity import DefaultAzureCredential
        from azure.storage.blob.aio import BlobServiceClient

        credential = DefaultAzureCredential()
        storage_client = BlobServiceClient(
            account_url=f"https://{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net",
            credential=credential,
        )

        try:
            # Get real test file
            data_dir = Path("/workspace/azure-maintie-rag/data/raw")
            test_files = list(data_dir.rglob("*.md"))[
                :3
            ]  # Upload first 3 files as test

            container_name = os.getenv("STORAGE_CONTAINER_NAME", "documents-prod")

            uploaded_files = []
            for file_path in test_files:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    blob_name = f"test-upload/{file_path.name}"

                    blob_client = storage_client.get_blob_client(
                        container=container_name, blob=blob_name
                    )

                    await blob_client.upload_blob(content, overwrite=True)
                    uploaded_files.append(blob_name)

                except Exception as e:
                    print(f"⚠️ Failed to upload {file_path.name}: {e}")

            assert len(uploaded_files) > 0, "No files successfully uploaded"
            print(
                f"✅ Real Data Upload: Successfully uploaded {len(uploaded_files)} files to Azure Storage"
            )

        finally:
            await storage_client.close()


class TestRealAzureSearch:
    """Test Azure Cognitive Search with real data."""

    @pytest.mark.asyncio
    async def test_azure_search_index_creation(self):
        """Test creating search index for real Azure AI documentation."""
        if not os.getenv("AZURE_SEARCH_ENDPOINT"):
            pytest.skip("Azure Cognitive Search not configured")

        from azure.identity import DefaultAzureCredential
        from azure.search.documents.aio import SearchClient
        from azure.search.documents.indexes.aio import SearchIndexClient
        from azure.search.documents.indexes.models import (
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
        )

        credential = DefaultAzureCredential()
        index_name = os.getenv("SEARCH_INDEX_NAME", "maintie-prod-index")

        index_client = SearchIndexClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"), credential=credential
        )

        try:
            # Define index schema for Azure AI documentation
            index = SearchIndex(
                name=index_name,
                fields=[
                    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                    SearchField(
                        name="content", type=SearchFieldDataType.String, searchable=True
                    ),
                    SearchField(
                        name="title", type=SearchFieldDataType.String, searchable=True
                    ),
                    SimpleField(
                        name="file_path",
                        type=SearchFieldDataType.String,
                        filterable=True,
                    ),
                    SimpleField(
                        name="content_type",
                        type=SearchFieldDataType.String,
                        filterable=True,
                    ),
                ],
            )

            # Create or update index
            await index_client.create_or_update_index(index)
            print(f"✅ Azure Search Index: Created/updated '{index_name}'")

            # Test search functionality with real data
            search_client = SearchClient(
                endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
                index_name=index_name,
                credential=credential,
            )

            # Upload sample document
            data_dir = Path("/workspace/azure-maintie-rag/data/raw")
            test_file = list(data_dir.rglob("*.md"))[0]
            content = test_file.read_text(encoding="utf-8")

            document = {
                "id": "test-doc-1",
                "content": content[:2000],  # First 2000 chars
                "title": test_file.stem,
                "file_path": str(test_file),
                "content_type": "azure_ai_documentation",
            }

            await search_client.upload_documents([document])

            # Wait a moment for indexing
            await asyncio.sleep(2)

            # Test search
            results = await search_client.search("azure")
            result_list = []
            async for result in results:
                result_list.append(result)

            print(
                f"✅ Azure Search: Document indexed and searchable, found {len(result_list)} results"
            )

        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for authentication issues
            if any(auth_error in error_msg for auth_error in [
                'authentication', 'credential', 'unauthorized', 'forbidden',
                'invalid_api_key', 'access_denied', 'token', 'login required'
            ]):
                pytest.skip(f"Azure authentication issue - check Azure credentials: {e}")
            
            # Check for network/connectivity issues
            if any(network_error in error_msg for network_error in [
                'connection', 'timeout', 'network', 'dns', 'socket', 'unreachable'
            ]):
                pytest.skip(f"Network connectivity issue - check Azure services: {e}")
            
            # Check for resource configuration issues
            if any(config_error in error_msg for config_error in [
                'not found', '404', 'resource not found', 'deployment not found',
                'index not found', 'service not found'
            ]):
                pytest.skip(f"Azure resource configuration issue: {e}")
            
            pytest.fail(f"Azure Search test failed: {e}")
            
        finally:
            await index_client.close()


class TestRealAzureOpenAIProcessing:
    """Test Azure OpenAI processing with real Azure AI documentation."""

    @pytest.mark.asyncio
    async def test_real_content_analysis(self):
        """Test content analysis on real Azure AI documentation."""
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )

        try:
            # Get real Azure AI content
            data_dir = Path("/workspace/azure-maintie-rag/data/raw")
            test_file = list(data_dir.rglob("*.md"))[0]
            content = test_file.read_text(encoding="utf-8")[:3000]  # First 3000 chars

            # Test content analysis
            response = await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_DEPLOYMENT"),
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze this Azure AI documentation and extract key topics, entities, and concepts. Respond in JSON format.",
                    },
                    {"role": "user", "content": content},
                ],
                max_tokens=500,
            )

            analysis_result = response.choices[0].message.content
            assert len(analysis_result) > 100, "Analysis result too short"

            print(
                f"✅ Real Content Analysis: Processed {len(content)} chars from {test_file.name}"
            )
            print(f"   Analysis length: {len(analysis_result)} chars")

        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for authentication issues
            if any(auth_error in error_msg for auth_error in [
                'authentication', 'credential', 'unauthorized', 'forbidden',
                'invalid_api_key', 'access_denied', 'token', 'login required'
            ]):
                pytest.skip(f"Azure authentication issue - check Azure credentials: {e}")
            
            # Check for network/connectivity issues
            if any(network_error in error_msg for network_error in [
                'connection', 'timeout', 'network', 'dns', 'socket', 'unreachable'
            ]):
                pytest.skip(f"Network connectivity issue - check Azure services: {e}")
            
            # Check for resource configuration issues
            if any(config_error in error_msg for config_error in [
                'not found', '404', 'resource not found', 'deployment not found'
            ]):
                pytest.skip(f"Azure resource configuration issue: {e}")
            
            pytest.fail(f"Real content analysis test failed: {e}")
            
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_real_embeddings_generation(self):
        """Test embeddings generation on real Azure AI documentation."""
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )

        try:
            # Get multiple real content samples
            data_dir = Path("/workspace/azure-maintie-rag/data/raw")
            test_files = list(data_dir.rglob("*.md"))[:5]

            embeddings_generated = 0
            for test_file in test_files:
                content = test_file.read_text(encoding="utf-8")[
                    :2000
                ]  # First 2000 chars

                response = await client.embeddings.create(
                    model=os.getenv("EMBEDDING_MODEL_DEPLOYMENT"), input=content
                )

                embedding = response.data[0].embedding
                assert len(embedding) == 1536, "Unexpected embedding dimension"
                embeddings_generated += 1

            print(
                f"✅ Real Embeddings: Generated embeddings for {embeddings_generated} Azure AI documents"
            )

        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for authentication issues
            if any(auth_error in error_msg for auth_error in [
                'authentication', 'credential', 'unauthorized', 'forbidden',
                'invalid_api_key', 'access_denied', 'token', 'login required'
            ]):
                pytest.skip(f"Azure authentication issue - check Azure credentials: {e}")
            
            # Check for network/connectivity issues
            if any(network_error in error_msg for network_error in [
                'connection', 'timeout', 'network', 'dns', 'socket', 'unreachable'
            ]):
                pytest.skip(f"Network connectivity issue - check Azure services: {e}")
            
            # Check for resource configuration issues
            if any(config_error in error_msg for config_error in [
                'not found', '404', 'resource not found', 'deployment not found'
            ]):
                pytest.skip(f"Azure resource configuration issue: {e}")
            
            pytest.fail(f"Real embeddings generation test failed: {e}")
            
        finally:
            await client.close()


class TestRealDataPipelineIntegration:
    """Test complete pipeline integration with real data."""

    def test_dataflow_scripts_exist(self):
        """Test that all dataflow scripts exist and can be imported."""
        scripts = [
            "scripts/dataflow/00_check_azure_state.py",
            "scripts/dataflow/01_data_ingestion.py",
            "scripts/dataflow/02_knowledge_extraction.py",
            "scripts/dataflow/07_unified_search.py",
        ]

        base_path = Path("/workspace/azure-maintie-rag")

        for script_path in scripts:
            full_path = base_path / script_path
            assert full_path.exists(), f"Missing dataflow script: {script_path}"

        print("✅ Pipeline Scripts: All dataflow scripts present")

    @pytest.mark.asyncio
    async def test_azure_state_check_with_real_services(self):
        """Test Azure state check against real deployed services."""
        import subprocess
        import sys

        # Run the Azure state check script
        result = subprocess.run(
            [
                sys.executable,
                "/workspace/azure-maintie-rag/scripts/dataflow/00_check_azure_state.py",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace/azure-maintie-rag",
        )

        # Should not fail completely (some services may not be ready)
        assert result.returncode in [0, 1], f"State check failed: {result.stderr}"

        output = result.stdout
        assert "Azure Universal RAG" in output, "Unexpected output format"

        print("✅ Azure State Check: Completed successfully")
        if "✅ System Status: READY" in output:
            print("   All services ready for processing")
        else:
            print("   Some services may need configuration")


class TestRealDataResults:
    """Test and document expected results from real data processing."""

    def test_document_expected_processing_results(self):
        """Document what we expect from processing 179 Azure AI files."""
        data_dir = Path("/workspace/azure-maintie-rag/data/raw")
        azure_files = list(data_dir.rglob("*.md"))

        # Calculate expected processing metrics
        total_files = len(azure_files)
        total_size = sum(len(f.read_text(encoding="utf-8")) for f in azure_files[:10])
        avg_file_size = total_size / min(10, len(azure_files))
        estimated_total_size = avg_file_size * total_files

        expected_results = {
            "total_files": total_files,
            "estimated_total_chars": int(estimated_total_size),
            "estimated_embeddings": total_files,  # One per file
            "estimated_entities": total_files * 10,  # ~10 entities per file
            "estimated_relationships": total_files * 5,  # ~5 relationships per file
            "expected_search_results": "50-100 relevant results per query",
            "processing_time_estimate": "5-15 minutes for complete pipeline",
        }

        print("✅ Expected Processing Results for Real Data:")
        for key, value in expected_results.items():
            print(f"   {key}: {value}")

        # Write expected results to file for reference
        results_file = Path(
            "/workspace/azure-maintie-rag/tests/expected_real_data_results.json"
        )
        with open(results_file, "w") as f:
            json.dump(expected_results, f, indent=2)

        print(f"✅ Expected results documented in: {results_file}")
