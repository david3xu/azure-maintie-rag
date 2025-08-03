#!/usr/bin/env python3
"""
Azure Service Connectivity Testing Script
Tests real Azure service connections following coding standards - no mocks, real services only
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from infrastructure.azure_openai.openai_client import UnifiedAzureOpenAIClient as AzureOpenAIClient
from infrastructure.azure_search.search_client import UnifiedSearchClient as AzureSearchClient
from infrastructure.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient as CosmosGremlinClient
from infrastructure.azure_storage.storage_client import UnifiedStorageClient as AzureStorageClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AzureConnectivityTester:
    """Test connectivity to all required Azure services"""

    def __init__(self):
        self.results: Dict[str, bool] = {}
        self.errors: Dict[str, str] = {}
        self.start_time = datetime.utcnow()

    async def test_azure_openai(self) -> bool:
        """Test Azure OpenAI connectivity and basic functionality"""
        logger.info("🔍 Testing Azure OpenAI connectivity...")

        try:
            if not settings.azure_openai_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT not configured")

            # Test client initialization
            client = AzureOpenAIClient()

            # Test basic embedding generation
            test_text = "This is a test document for Azure OpenAI connectivity validation."
            embeddings = await client.get_embeddings([test_text])

            if not embeddings or len(embeddings) == 0:
                raise ValueError("No embeddings returned from Azure OpenAI")

            if len(embeddings[0]) < 1500:  # text-embedding-ada-002 should return 1536 dimensions
                raise ValueError(f"Unexpected embedding dimensions: {len(embeddings[0])}")

            logger.info(f"✅ Azure OpenAI: Connected, embedding dimensions: {len(embeddings[0])}")
            return True

        except Exception as e:
            error_msg = f"Azure OpenAI connection failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.errors["azure_openai"] = error_msg
            return False

    async def test_azure_search(self) -> bool:
        """Test Azure Cognitive Search connectivity"""
        logger.info("🔍 Testing Azure Cognitive Search connectivity...")

        try:
            if not settings.azure_search_endpoint:
                raise ValueError("AZURE_SEARCH_ENDPOINT not configured")

            # Test client initialization
            client = AzureSearchClient()

            # Test service connectivity (list indexes)
            indexes = await client.list_indexes()

            logger.info(f"✅ Azure Search: Connected, found {len(indexes)} indexes")
            return True

        except Exception as e:
            error_msg = f"Azure Search connection failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.errors["azure_search"] = error_msg
            return False

    async def test_azure_cosmos(self) -> bool:
        """Test Azure Cosmos DB Gremlin API connectivity"""
        logger.info("🔍 Testing Azure Cosmos DB Gremlin connectivity...")

        try:
            if not settings.azure_cosmos_endpoint:
                raise ValueError("AZURE_COSMOS_ENDPOINT not configured")

            # Test client initialization
            client = CosmosGremlinClient()

            # Test basic connectivity with simple query
            result = await client.execute_query("g.V().limit(1)")

            logger.info("✅ Azure Cosmos DB: Connected, Gremlin query executed")
            return True

        except Exception as e:
            error_msg = f"Azure Cosmos DB connection failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.errors["azure_cosmos"] = error_msg
            return False

    async def test_azure_storage(self) -> bool:
        """Test Azure Storage Account connectivity"""
        logger.info("🔍 Testing Azure Storage connectivity...")

        try:
            if not settings.azure_storage_account:
                raise ValueError("AZURE_STORAGE_ACCOUNT not configured")

            # Test client initialization
            client = AzureStorageClient()

            # Test basic connectivity (list containers)
            containers = await client.list_containers()

            logger.info(f"✅ Azure Storage: Connected, found {len(containers)} containers")
            return True

        except Exception as e:
            error_msg = f"Azure Storage connection failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.errors["azure_storage"] = error_msg
            return False

    async def test_all_services(self) -> Dict[str, bool]:
        """Test all Azure services concurrently"""
        logger.info("🚀 Starting Azure services connectivity testing...")
        logger.info(f"Testing environment: {os.getenv('AZURE_ENV_NAME', 'local')}")

        # Test all services in parallel for efficiency
        test_tasks = [
            ("azure_openai", self.test_azure_openai()),
            ("azure_search", self.test_azure_search()),
            ("azure_cosmos", self.test_azure_cosmos()),
            ("azure_storage", self.test_azure_storage()),
        ]

        # Execute all tests concurrently
        results = await asyncio.gather(*[task for _, task in test_tasks], return_exceptions=True)

        # Process results
        for i, (service_name, _) in enumerate(test_tasks):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"❌ {service_name}: Test failed with exception: {result}")
                self.results[service_name] = False
                self.errors[service_name] = str(result)
            else:
                self.results[service_name] = result

        return self.results

    def print_summary(self):
        """Print detailed test results summary"""
        total_time = (datetime.utcnow() - self.start_time).total_seconds()

        print("\n" + "="*60)
        print("🧪 AZURE SERVICES CONNECTIVITY TEST SUMMARY")
        print("="*60)

        total_services = len(self.results)
        passed_services = sum(1 for result in self.results.values() if result)

        print(f"⏱️  Total Testing Time: {total_time:.2f} seconds")
        print(f"📊 Services Tested: {total_services}")
        print(f"✅ Services Passed: {passed_services}")
        print(f"❌ Services Failed: {total_services - passed_services}")
        print(f"📈 Success Rate: {passed_services/total_services*100:.1f}%")

        print("\n📋 SERVICE STATUS:")
        for service_name, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} {service_name}")

            if not passed and service_name in self.errors:
                print(f"     Error: {self.errors[service_name]}")

        if passed_services == total_services:
            print("\n🎉 ALL AZURE SERVICES CONNECTED SUCCESSFULLY!")
            print("✅ Ready to proceed with data pipeline testing")
            print("✅ Environment configured for real Azure services testing")
        else:
            print(f"\n⚠️  {total_services - passed_services} SERVICES FAILED")
            print("❌ Fix service connectivity before proceeding")
            print("\nTroubleshooting steps:")
            print("1. Verify Azure CLI login: az account show")
            print("2. Check environment variables are set correctly")
            print("3. Verify service principal has proper permissions")
            print("4. Check Azure service endpoints are accessible")

        return passed_services == total_services


async def main():
    """Main testing function"""
    print("🧪 Azure Universal RAG - Service Connectivity Testing")
    print("Following coding standards: Real Azure services only, no mocks")
    print("-" * 60)

    # Validate environment configuration
    required_settings = [
        "azure_openai_endpoint",
        "azure_search_endpoint",
        "azure_cosmos_endpoint",
        "azure_storage_account"
    ]

    missing_settings = []
    for setting in required_settings:
        if not getattr(settings, setting, None):
            missing_settings.append(setting.upper())

    if missing_settings:
        print(f"❌ Missing required environment variables: {', '.join(missing_settings)}")
        print("Set these in your environment or .env file before testing")
        return 1

    # Run connectivity tests
    tester = AzureConnectivityTester()
    await tester.test_all_services()

    # Print results and return appropriate exit code
    success = tester.print_summary()
    return 0 if success else 1


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
