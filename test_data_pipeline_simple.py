#!/usr/bin/env python3
"""
Simple Data Pipeline Test
Tests the core data processing pipeline with real Azure services
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_document_processing():
    """Test document processing with Azure OpenAI"""
    logger.info("🔍 Testing document processing with Azure OpenAI...")

    try:
        from openai import AzureOpenAI

        # Load the test document
        doc_path = Path("data/raw/azure-ml/azure-machine-learning-azureml-api-2.md")
        if not doc_path.exists():
            raise FileNotFoundError(f"Test document not found: {doc_path}")

        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Truncate content to first 2000 characters for testing
        content = content[:2000]

        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
        )

        # Test 1: Generate embeddings
        logger.info("  📊 Testing embedding generation...")
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=content
        )

        embedding = embedding_response.data[0].embedding
        logger.info(f"  ✅ Generated embedding with {len(embedding)} dimensions")

        # Test 2: Extract key information with GPT
        logger.info("  🧠 Testing knowledge extraction...")
        completion_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting key information from technical documentation. Extract the main topics, key concepts, and important entities from the provided text."
                },
                {
                    "role": "user",
                    "content": f"Extract key information from this Azure ML documentation:\n\n{content}"
                }
            ],
            max_tokens=500,
            temperature=0.1
        )

        extracted_info = completion_response.choices[0].message.content
        logger.info(f"  ✅ Extracted {len(extracted_info)} characters of key information")
        logger.info(f"  📝 Sample extraction: {extracted_info[:200]}...")

        return True, {
            'embedding_dimensions': len(embedding),
            'extracted_info_length': len(extracted_info),
            'document_length': len(content)
        }

    except Exception as e:
        logger.error(f"❌ Document processing failed: {str(e)}")
        return False, {}


async def test_azure_search_integration():
    """Test Azure Search integration"""
    logger.info("🔍 Testing Azure Search integration...")

    try:
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents import SearchClient
        from azure.identity import DefaultAzureCredential
        from azure.search.documents.indexes.models import (
            SearchIndex,
            SearchField,
            SearchFieldDataType,
            SimpleField,
            SearchableField,
            VectorSearch,
            HnswAlgorithmConfiguration,
            VectorSearchProfile,
            SemanticConfiguration,
            SemanticPrioritizedFields,
            SemanticField,
            SemanticSearch
        )

        endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        credential = DefaultAzureCredential()

        # Test index creation/verification
        index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

        # Create a test index if it doesn't exist
        test_index_name = "azure-ml-docs-test"

        try:
            index = index_client.get_index(test_index_name)
            logger.info(f"  ✅ Found existing index: {test_index_name}")
        except Exception:
            logger.info(f"  📝 Creating test index: {test_index_name}")

            # Define search fields
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=1536,
                    vector_search_profile_name="my-vector-config"
                ),
            ]

            # Configure vector search
            vector_search = VectorSearch(
                profiles=[VectorSearchProfile(
                    name="my-vector-config",
                    algorithm_configuration_name="my-algorithms-config"
                )],
                algorithms=[HnswAlgorithmConfiguration(
                    name="my-algorithms-config"
                )]
            )

            # Create index
            index = SearchIndex(
                name=test_index_name,
                fields=fields,
                vector_search=vector_search
            )

            result = index_client.create_index(index)
            logger.info(f"  ✅ Created search index: {result.name}")

        # Test basic search functionality
        search_client = SearchClient(endpoint=endpoint, index_name=test_index_name, credential=credential)

        # Simple connectivity test
        logger.info("  🔍 Testing search client connectivity...")
        results = list(search_client.search("*", top=1))  # Get any document
        logger.info(f"  ✅ Search connectivity successful, found {len(results)} results")

        return True, {'index_name': test_index_name}

    except Exception as e:
        logger.error(f"❌ Azure Search integration failed: {str(e)}")
        return False, {}


async def test_cosmos_db_storage():
    """Test Cosmos DB storage functionality"""
    logger.info("🔍 Testing Cosmos DB storage...")

    try:
        from azure.cosmos import CosmosClient

        endpoint = os.getenv('AZURE_COSMOS_ENDPOINT')
        key = os.getenv('AZURE_COSMOS_KEY')

        # Convert Gremlin endpoint to regular Cosmos endpoint
        cosmos_endpoint = endpoint.replace('.gremlin.cosmosdb.', '.documents.')
        cosmos_endpoint = cosmos_endpoint.replace(':443/', '')

        client = CosmosClient(cosmos_endpoint, key)

        # Test database access
        database_name = "maintie-rag-development"
        try:
            database = client.get_database_client(database_name)
            logger.info(f"  ✅ Connected to database: {database_name}")
        except Exception:
            logger.info(f"  📝 Creating database: {database_name}")
            database = client.create_database(database_name)

        # Test container access
        container_name = "knowledge-entities"
        try:
            container = database.get_container_client(container_name)
            logger.info(f"  ✅ Connected to container: {container_name}")
        except Exception:
            logger.info(f"  📝 Creating container: {container_name}")
            container = database.create_container(
                id=container_name,
                partition_key={'paths': ['/domain'], 'kind': 'Hash'}
            )

        # Test document operations
        test_doc = {
            'id': f'test-entity-{datetime.utcnow().timestamp()}',
            'domain': 'azure-ml',
            'type': 'concept',
            'name': 'Azure Machine Learning',
            'description': 'Cloud-based platform for machine learning',
            'timestamp': datetime.utcnow().isoformat()
        }

        # Insert test document
        result = container.create_item(test_doc)
        logger.info(f"  ✅ Created test document with id: {result['id']}")

        # Query test
        query = "SELECT * FROM c WHERE c.domain = 'azure-ml' ORDER BY c.timestamp DESC"
        items = list(container.query_items(query, enable_cross_partition_query=True))
        logger.info(f"  ✅ Query successful, found {len(items)} documents")

        return True, {
            'database': database_name,
            'container': container_name,
            'document_count': len(items)
        }

    except Exception as e:
        logger.error(f"❌ Cosmos DB storage failed: {str(e)}")
        return False, {}


async def test_storage_account():
    """Test Azure Storage Account functionality"""
    logger.info("🔍 Testing Azure Storage Account...")

    try:
        from azure.storage.blob import BlobServiceClient
        from azure.identity import DefaultAzureCredential

        account_name = os.getenv('AZURE_STORAGE_ACCOUNT')
        account_url = f"https://{account_name}.blob.core.windows.net"

        credential = DefaultAzureCredential()
        client = BlobServiceClient(account_url=account_url, credential=credential)

        # Test container access
        container_name = "documents"
        try:
            container_client = client.get_container_client(container_name)
            # Test if container exists
            container_client.get_container_properties()
            logger.info(f"  ✅ Connected to container: {container_name}")
        except Exception:
            logger.info(f"  📝 Creating container: {container_name}")
            container_client = client.create_container(container_name)

        # Test file upload
        test_content = "This is a test document for Azure Storage validation"
        blob_name = f"test-doc-{datetime.utcnow().timestamp()}.txt"

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(test_content, overwrite=True)
        logger.info(f"  ✅ Uploaded test blob: {blob_name}")

        # Test file listing
        blobs = list(container_client.list_blobs())
        logger.info(f"  ✅ Listed {len(blobs)} blobs in container")

        return True, {
            'container': container_name,
            'blob_count': len(blobs),
            'test_blob': blob_name
        }

    except Exception as e:
        logger.error(f"❌ Azure Storage failed: {str(e)}")
        return False, {}


async def main():
    """Main testing function"""
    print("🧪 Azure Universal RAG - Data Pipeline Testing")
    print("Testing core data processing with real Azure services")
    print("-" * 60)

    start_time = datetime.utcnow()

    # Run all pipeline tests
    tests = [
        ("Document Processing", test_document_processing()),
        ("Azure Search Integration", test_azure_search_integration()),
        ("Cosmos DB Storage", test_cosmos_db_storage()),
        ("Azure Storage Account", test_storage_account()),
    ]

    results = {}
    test_details = {}

    for test_name, test_coro in tests:
        success, details = await test_coro
        results[test_name] = success
        test_details[test_name] = details

    # Print summary
    total_time = (datetime.utcnow() - start_time).total_seconds()
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    print("\n" + "="*60)
    print("🧪 DATA PIPELINE TEST SUMMARY")
    print("="*60)
    print(f"⏱️  Total Testing Time: {total_time:.2f} seconds")
    print(f"📊 Tests Run: {total_tests}")
    print(f"✅ Tests Passed: {passed_tests}")
    print(f"❌ Tests Failed: {total_tests - passed_tests}")
    print(f"📈 Success Rate: {passed_tests/total_tests*100:.1f}%")

    print("\n📋 TEST RESULTS:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")

        if passed and test_name in test_details:
            details = test_details[test_name]
            if details:
                for key, value in details.items():
                    print(f"     - {key}: {value}")

    if passed_tests == total_tests:
        print("\n🎉 ALL DATA PIPELINE TESTS PASSED!")
        print("✅ Document processing working with Azure OpenAI")
        print("✅ Azure Search integration functional")
        print("✅ Cosmos DB storage operational")
        print("✅ Azure Storage Account accessible")
        print("✅ Ready for search system testing")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} TESTS FAILED")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
