#!/usr/bin/env python3
"""
Direct Azure Services Connectivity Test
Tests Azure services directly using SDKs with API keys for local development
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_azure_openai():
    """Test Azure OpenAI connectivity directly"""
    logger.info("🔍 Testing Azure OpenAI connectivity...")

    try:
        from openai import AzureOpenAI

        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')

        if not endpoint or not api_key:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY")

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )

        # Test embedding generation
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="This is a test document for Azure OpenAI connectivity validation."
        )

        embedding = response.data[0].embedding
        if len(embedding) < 1500:  # text-embedding-ada-002 should return 1536 dimensions
            raise ValueError(f"Unexpected embedding dimensions: {len(embedding)}")

        logger.info(f"✅ Azure OpenAI: Connected, embedding dimensions: {len(embedding)}")
        return True

    except Exception as e:
        logger.error(f"❌ Azure OpenAI connection failed: {str(e)}")
        return False


async def test_azure_search():
    """Test Azure Cognitive Search connectivity directly"""
    logger.info("🔍 Testing Azure Cognitive Search connectivity...")

    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        from azure.identity import DefaultAzureCredential

        endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')

        if not endpoint:
            raise ValueError("Missing AZURE_SEARCH_ENDPOINT")

        # Use Azure CLI credentials for authentication
        credential = DefaultAzureCredential()

        # Create a dummy index name for testing connectivity
        index_name = "test-connectivity"

        search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential
        )

        # Test service connectivity by trying to get service statistics
        from azure.search.documents.indexes import SearchIndexClient

        index_client = SearchIndexClient(
            endpoint=endpoint,
            credential=credential
        )

        # List indexes to verify connectivity
        indexes = list(index_client.list_indexes())

        logger.info(f"✅ Azure Search: Connected, found {len(indexes)} indexes")
        return True

    except Exception as e:
        logger.error(f"❌ Azure Search connection failed: {str(e)}")
        return False


async def test_azure_cosmos():
    """Test Azure Cosmos DB connectivity directly"""
    logger.info("🔍 Testing Azure Cosmos DB Gremlin connectivity...")

    try:
        from azure.cosmos import CosmosClient
        from azure.identity import DefaultAzureCredential

        endpoint = os.getenv('AZURE_COSMOS_ENDPOINT')
        key = os.getenv('AZURE_COSMOS_KEY')

        if not endpoint:
            raise ValueError("Missing AZURE_COSMOS_ENDPOINT")

        # Convert Gremlin endpoint to regular Cosmos endpoint
        cosmos_endpoint = endpoint.replace('.gremlin.cosmosdb.', '.documents.')
        cosmos_endpoint = cosmos_endpoint.replace(':443/', '')

        if key:
            # Use key-based authentication
            client = CosmosClient(cosmos_endpoint, key)
        else:
            # Use managed identity
            credential = DefaultAzureCredential()
            client = CosmosClient(cosmos_endpoint, credential)

        # Test connectivity by listing databases
        databases = list(client.list_databases())

        logger.info(f"✅ Azure Cosmos DB: Connected, found {len(databases)} databases")
        return True

    except Exception as e:
        logger.error(f"❌ Azure Cosmos DB connection failed: {str(e)}")
        return False


async def test_azure_storage():
    """Test Azure Storage connectivity directly"""
    logger.info("🔍 Testing Azure Storage connectivity...")

    try:
        from azure.storage.blob import BlobServiceClient
        from azure.identity import DefaultAzureCredential

        account_name = os.getenv('AZURE_STORAGE_ACCOUNT')

        if not account_name:
            raise ValueError("Missing AZURE_STORAGE_ACCOUNT")

        account_url = f"https://{account_name}.blob.core.windows.net"

        # Use Azure CLI credentials
        credential = DefaultAzureCredential()

        client = BlobServiceClient(
            account_url=account_url,
            credential=credential
        )

        # Test connectivity by listing containers
        containers = list(client.list_containers())

        logger.info(f"✅ Azure Storage: Connected, found {len(containers)} containers")
        return True

    except Exception as e:
        logger.error(f"❌ Azure Storage connection failed: {str(e)}")
        return False


async def main():
    """Main testing function"""
    print("🧪 Azure Universal RAG - Direct Service Connectivity Testing")
    print("Testing real Azure services with SDK authentication")
    print("-" * 60)

    start_time = datetime.utcnow()

    # Test all services
    tests = [
        ("Azure OpenAI", test_azure_openai()),
        ("Azure Search", test_azure_search()),
        ("Azure Cosmos DB", test_azure_cosmos()),
        ("Azure Storage", test_azure_storage()),
    ]

    results = {}
    for service_name, test_coro in tests:
        result = await test_coro
        results[service_name] = result

    # Print summary
    total_time = (datetime.utcnow() - start_time).total_seconds()
    total_services = len(results)
    passed_services = sum(1 for result in results.values() if result)

    print("\n" + "="*60)
    print("🧪 AZURE SERVICES CONNECTIVITY TEST SUMMARY")
    print("="*60)
    print(f"⏱️  Total Testing Time: {total_time:.2f} seconds")
    print(f"📊 Services Tested: {total_services}")
    print(f"✅ Services Passed: {passed_services}")
    print(f"❌ Services Failed: {total_services - passed_services}")
    print(f"📈 Success Rate: {passed_services/total_services*100:.1f}%")

    print("\n📋 SERVICE STATUS:")
    for service_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {service_name}")

    if passed_services == total_services:
        print("\n🎉 ALL AZURE SERVICES CONNECTED SUCCESSFULLY!")
        print("✅ Ready to proceed with data pipeline testing")
        return 0
    else:
        print(f"\n⚠️  {total_services - passed_services} SERVICES FAILED")
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
