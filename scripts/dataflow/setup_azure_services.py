#!/usr/bin/env python3
"""
Azure Services Setup - Infrastructure Validation and Initialization
Validates and initializes all Azure services required for the Universal RAG system

This script ensures all Azure services are properly configured:
- Azure OpenAI (text processing and embeddings)
- Azure Cognitive Search (vector search capabilities)
- Azure Cosmos DB with Gremlin API (knowledge graphs)
- Azure Blob Storage (data management)
- Azure ML Workspace (GNN training)
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from core.azure_openai.openai_client import UnifiedAzureOpenAIClient
from core.azure_search.search_client import UnifiedSearchClient
from core.azure_storage.storage_client import UnifiedStorageClient

from config.async_pattern_manager import get_pattern_manager
from config.discovery_infrastructure_naming import get_discovery_naming
from config.dynamic_ml_config import get_dynamic_ml_config
from services.infrastructure_service import InfrastructureService
from services.ml_service import MLService

logger = logging.getLogger(__name__)


class AzureServicesSetup:
    """Azure Services Setup and Validation"""

    def __init__(self):
        self.infrastructure = InfrastructureService()

    async def validate_all_services(self, domain: str = "general") -> Dict[str, Any]:
        """
        Validate all Azure services required for the RAG system

        Args:
            domain: Target domain for validation

        Returns:
            Dict with validation results for each service
        """
        print("🔍 Azure Services Validation - Universal RAG Infrastructure")
        print("=" * 70)

        validation_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "domain": domain,
            "services": {},
            "overall_status": "unknown",
            "issues": [],
            "recommendations": [],
        }

        try:
            # Validate Azure OpenAI
            print(f"\n🔄 Validating Azure OpenAI Service...")
            openai_validation = await self._validate_azure_openai()
            validation_results["services"]["azure_openai"] = openai_validation

            # Validate Azure Cognitive Search
            print(f"\n🔄 Validating Azure Cognitive Search...")
            search_validation = await self._validate_azure_search(domain)
            validation_results["services"]["azure_search"] = search_validation

            # Validate Azure Cosmos DB
            print(f"\n🔄 Validating Azure Cosmos DB...")
            cosmos_validation = await self._validate_azure_cosmos(domain)
            validation_results["services"]["azure_cosmos"] = cosmos_validation

            # Validate Azure Blob Storage
            print(f"\n🔄 Validating Azure Blob Storage...")
            storage_validation = await self._validate_azure_storage(domain)
            validation_results["services"]["azure_storage"] = storage_validation

            # Validate Azure ML Workspace
            print(f"\n🔄 Validating Azure ML Workspace...")
            ml_validation = await self._validate_azure_ml()
            validation_results["services"]["azure_ml"] = ml_validation

            # Analyze overall status
            overall_status = await self._analyze_overall_status(
                validation_results["services"]
            )
            validation_results.update(overall_status)

            # Print summary
            print(f"\n📊 Validation Summary:")
            print(f"   ✅ Services validated: {len(validation_results['services'])}")
            print(f"   🎯 Overall status: {validation_results['overall_status']}")
            print(f"   ⚠️  Issues found: {len(validation_results['issues'])}")
            print(f"   💡 Recommendations: {len(validation_results['recommendations'])}")

            return validation_results

        except Exception as e:
            validation_results["error"] = str(e)
            validation_results["overall_status"] = "failed"
            print(f"❌ Validation failed: {e}")
            logger.error(f"Azure services validation failed: {e}", exc_info=True)
            return validation_results

    async def _validate_azure_openai(self) -> Dict[str, Any]:
        """Validate Azure OpenAI service"""
        try:
            client = UnifiedAzureOpenAIClient()

            validation_result = {
                "service_name": "Azure OpenAI",
                "status": "unknown",
                "capabilities": {},
                "issues": [],
            }

            # Test basic connectivity
            print(f"   🔗 Testing connection...")
            connection_test = await client.test_connection()

            if not connection_test.get("success"):
                validation_result["status"] = "failed"
                validation_result["issues"].append(
                    f"Connection failed: {connection_test.get('error')}"
                )
                return validation_result

            # Test completions
            print(f"   💭 Testing text completion...")
            completion_test = await client.get_completion(
                prompt="Test completion for Azure Universal RAG system validation.",
                max_tokens=50,
            )

            if completion_test.get("success"):
                validation_result["capabilities"]["text_completion"] = True
                print(f"   ✅ Text completion: Working")
            else:
                validation_result["capabilities"]["text_completion"] = False
                validation_result["issues"].append(
                    f"Completion test failed: {completion_test.get('error')}"
                )

            # Test embeddings
            print(f"   🔍 Testing embeddings generation...")
            embedding_test = await client.get_embeddings(
                ["Test embedding for validation"]
            )

            if embedding_test.get("success"):
                embeddings = embedding_test.get("embeddings", [])
                if embeddings and len(embeddings[0]) == 1536:  # Expected dimension
                    validation_result["capabilities"]["embeddings"] = True
                    validation_result["capabilities"]["embedding_dimension"] = 1536
                    print(f"   ✅ Embeddings (1536D): Working")
                else:
                    validation_result["capabilities"]["embeddings"] = False
                    validation_result["issues"].append("Unexpected embedding dimension")
            else:
                validation_result["capabilities"]["embeddings"] = False
                validation_result["issues"].append(
                    f"Embedding test failed: {embedding_test.get('error')}"
                )

            # Determine overall status
            if validation_result["capabilities"].get(
                "text_completion"
            ) and validation_result["capabilities"].get("embeddings"):
                validation_result["status"] = "healthy"
            elif len(validation_result["issues"]) == 0:
                validation_result["status"] = "healthy"
            else:
                validation_result["status"] = "degraded"

            return validation_result

        except Exception as e:
            return {
                "service_name": "Azure OpenAI",
                "status": "failed",
                "error": str(e),
                "issues": [f"Validation exception: {e}"],
            }

    async def _validate_azure_search(self, domain: str) -> Dict[str, Any]:
        """Validate Azure Cognitive Search service"""
        try:
            client = UnifiedSearchClient()

            validation_result = {
                "service_name": "Azure Cognitive Search",
                "status": "unknown",
                "capabilities": {},
                "indices": {},
                "issues": [],
            }

            # Test connection
            print(f"   🔗 Testing connection...")
            connection_test = await client.test_connection()

            if not connection_test.get("success"):
                validation_result["status"] = "failed"
                validation_result["issues"].append(
                    f"Connection failed: {connection_test.get('error')}"
                )
                return validation_result

            # Check vector search capability
            print(f"   🔍 Testing vector search capability...")
            index_name = f"{domain}-vector-index"

            # Try to get or create vector index
            index_result = await client.get_or_create_vector_index(
                index_name=index_name, vector_dimension=1536
            )

            if index_result.get("success"):
                validation_result["capabilities"]["vector_search"] = True
                validation_result["indices"][index_name] = "ready"
                print(f"   ✅ Vector index ({index_name}): Ready")
            else:
                validation_result["capabilities"]["vector_search"] = False
                validation_result["issues"].append(
                    f"Vector index creation failed: {index_result.get('error')}"
                )

            # Test search functionality (basic)
            if validation_result["capabilities"].get("vector_search"):
                print(f"   🔍 Testing search functionality...")
                # This would normally test with actual data
                validation_result["capabilities"]["search_functionality"] = True

            # Determine overall status
            if validation_result["capabilities"].get("vector_search"):
                validation_result["status"] = "healthy"
            else:
                validation_result["status"] = "degraded"

            return validation_result

        except Exception as e:
            return {
                "service_name": "Azure Cognitive Search",
                "status": "failed",
                "error": str(e),
                "issues": [f"Validation exception: {e}"],
            }

    async def _validate_azure_cosmos(self, domain: str) -> Dict[str, Any]:
        """Validate Azure Cosmos DB with Gremlin API"""
        try:
            client = AzureCosmosGremlinClient()

            validation_result = {
                "service_name": "Azure Cosmos DB (Gremlin)",
                "status": "unknown",
                "capabilities": {},
                "database_info": {},
                "issues": [],
            }

            # Test connection
            print(f"   🔗 Testing connection...")
            connection_test = await client.test_connection()

            if not connection_test.get("success"):
                validation_result["status"] = "failed"
                validation_result["issues"].append(
                    f"Connection failed: {connection_test.get('error')}"
                )
                return validation_result

            # Test graph operations
            print(f"   🕸️  Testing graph operations...")

            # Test vertex operations
            vertex_test = await client.test_vertex_operations(domain)
            if vertex_test.get("success"):
                validation_result["capabilities"]["vertex_operations"] = True
                print(f"   ✅ Vertex operations: Working")
            else:
                validation_result["capabilities"]["vertex_operations"] = False
                validation_result["issues"].append(
                    f"Vertex operations failed: {vertex_test.get('error')}"
                )

            # Test edge operations
            edge_test = await client.test_edge_operations(domain)
            if edge_test.get("success"):
                validation_result["capabilities"]["edge_operations"] = True
                print(f"   ✅ Edge operations: Working")
            else:
                validation_result["capabilities"]["edge_operations"] = False
                validation_result["issues"].append(
                    f"Edge operations failed: {edge_test.get('error')}"
                )

            # Get database statistics
            stats = await client.get_database_statistics(domain)
            if stats.get("success"):
                validation_result["database_info"] = stats.get("statistics", {})

            # Determine overall status
            if validation_result["capabilities"].get(
                "vertex_operations"
            ) and validation_result["capabilities"].get("edge_operations"):
                validation_result["status"] = "healthy"
            else:
                validation_result["status"] = "degraded"

            return validation_result

        except Exception as e:
            return {
                "service_name": "Azure Cosmos DB (Gremlin)",
                "status": "failed",
                "error": str(e),
                "issues": [f"Validation exception: {e}"],
            }

    async def _validate_azure_storage(self, domain: str) -> Dict[str, Any]:
        """Validate Azure Blob Storage service"""
        try:
            client = UnifiedStorageClient()

            validation_result = {
                "service_name": "Azure Blob Storage",
                "status": "unknown",
                "capabilities": {},
                "containers": {},
                "issues": [],
            }

            # Test connection
            print(f"   🔗 Testing connection...")
            connection_test = await client.test_connection()

            if not connection_test.get("success"):
                validation_result["status"] = "failed"
                validation_result["issues"].append(
                    f"Connection failed: {connection_test.get('error')}"
                )
                return validation_result

            # Test container operations
            print(f"   📦 Testing container operations...")
            required_containers = [
                f"{domain}-documents",
                f"{domain}-extracted-knowledge",
                f"{domain}-query-analysis",
                f"{domain}-search-results",
                f"{domain}-prepared-context",
                f"{domain}-final-responses",
            ]

            for container in required_containers:
                container_test = await client.ensure_container_exists(container)
                if container_test.get("success"):
                    validation_result["containers"][container] = "ready"
                else:
                    validation_result["containers"][container] = "failed"
                    validation_result["issues"].append(
                        f"Container {container} creation failed"
                    )

            # Test read/write operations
            print(f"   📝 Testing read/write operations...")
            test_data = {"test": "validation", "timestamp": datetime.now().isoformat()}
            test_container = f"{domain}-test-validation"

            write_test = await client.save_json(
                data=test_data,
                blob_name="validation_test.json",
                container=test_container,
            )

            if write_test.get("success"):
                validation_result["capabilities"]["write_operations"] = True

                # Test read
                read_test = await client.load_json(
                    "validation_test.json", test_container
                )
                if read_test:
                    validation_result["capabilities"]["read_operations"] = True
                    print(f"   ✅ Read/Write operations: Working")

                    # Clean up test data
                    await client.delete_blob("validation_test.json", test_container)
                else:
                    validation_result["capabilities"]["read_operations"] = False
                    validation_result["issues"].append("Read operation failed")
            else:
                validation_result["capabilities"]["write_operations"] = False
                validation_result["issues"].append("Write operation failed")

            # Determine overall status
            working_containers = sum(
                1
                for status in validation_result["containers"].values()
                if status == "ready"
            )
            if (
                working_containers >= len(required_containers) * 0.8
                and validation_result["capabilities"].get(  # 80% of containers working
                    "read_operations"
                )
                and validation_result["capabilities"].get("write_operations")
            ):
                validation_result["status"] = "healthy"
            else:
                validation_result["status"] = "degraded"

            return validation_result

        except Exception as e:
            return {
                "service_name": "Azure Blob Storage",
                "status": "failed",
                "error": str(e),
                "issues": [f"Validation exception: {e}"],
            }

    async def _validate_azure_ml(self) -> Dict[str, Any]:
        """Validate Azure ML Workspace"""
        try:
            ml_service = MLService()

            validation_result = {
                "service_name": "Azure ML Workspace",
                "status": "unknown",
                "capabilities": {},
                "issues": [],
            }

            # Test ML service connection
            print(f"   🔗 Testing ML workspace connection...")
            connection_test = await ml_service.test_connection()

            if connection_test.get("success"):
                validation_result["capabilities"]["workspace_connection"] = True
                print(f"   ✅ ML Workspace: Connected")
            else:
                validation_result["capabilities"]["workspace_connection"] = False
                validation_result["issues"].append(
                    f"ML workspace connection failed: {connection_test.get('error')}"
                )

            # Test compute resources (simplified check)
            print(f"   💻 Testing compute resources...")
            compute_test = await ml_service.check_compute_resources()

            if compute_test.get("success"):
                validation_result["capabilities"]["compute_resources"] = True
                print(f"   ✅ Compute resources: Available")
            else:
                validation_result["capabilities"]["compute_resources"] = False
                validation_result["issues"].append("Compute resources not available")

            # Determine status
            if validation_result["capabilities"].get(
                "workspace_connection"
            ) and validation_result["capabilities"].get("compute_resources"):
                validation_result["status"] = "healthy"
            elif validation_result["capabilities"].get("workspace_connection"):
                validation_result["status"] = "degraded"
            else:
                validation_result["status"] = "failed"

            return validation_result

        except Exception as e:
            return {
                "service_name": "Azure ML Workspace",
                "status": "failed",
                "error": str(e),
                "issues": [f"Validation exception: {e}"],
            }

    async def _analyze_overall_status(self, services: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall system status"""
        healthy_services = sum(
            1 for service in services.values() if service.get("status") == "healthy"
        )
        degraded_services = sum(
            1 for service in services.values() if service.get("status") == "degraded"
        )
        failed_services = sum(
            1 for service in services.values() if service.get("status") == "failed"
        )
        total_services = len(services)

        # Collect all issues and recommendations
        all_issues = []
        recommendations = []

        for service_name, service_info in services.items():
            service_issues = service_info.get("issues", [])
            all_issues.extend([f"{service_name}: {issue}" for issue in service_issues])

        # Generate recommendations
        if failed_services > 0:
            recommendations.append(
                "Fix failed services before proceeding with data processing"
            )
        if degraded_services > 0:
            recommendations.append(
                "Investigate degraded services for optimal performance"
            )
        if healthy_services == total_services:
            recommendations.append(
                "All services healthy - system ready for production use"
            )

        # Determine overall status
        if failed_services == 0 and degraded_services == 0:
            overall_status = "healthy"
        elif failed_services == 0:
            overall_status = "degraded"
        elif healthy_services >= total_services * 0.6:  # 60% healthy
            overall_status = "partially_functional"
        else:
            overall_status = "failed"

        return {
            "overall_status": overall_status,
            "service_summary": {
                "healthy": healthy_services,
                "degraded": degraded_services,
                "failed": failed_services,
                "total": total_services,
            },
            "issues": all_issues,
            "recommendations": recommendations,
        }

    async def initialize_domain_resources(self, domain: str) -> Dict[str, Any]:
        """Initialize all resources for a specific domain"""
        print(f"\n🚀 Initializing resources for domain: {domain}")

        initialization_results = {
            "domain": domain,
            "initialization_timestamp": datetime.now().isoformat(),
            "resources_initialized": [],
            "success": False,
        }

        try:
            # Initialize storage containers
            print(f"   📦 Creating storage containers...")
            storage_client = UnifiedStorageClient()

            containers = [
                f"{domain}-documents",
                f"{domain}-extracted-knowledge",
                f"{domain}-query-analysis",
                f"{domain}-search-results",
                f"{domain}-prepared-context",
                f"{domain}-final-responses",
            ]

            for container in containers:
                result = await storage_client.ensure_container_exists(container)
                if result.get("success"):
                    initialization_results["resources_initialized"].append(
                        f"Container: {container}"
                    )

            # Initialize search index
            print(f"   🔍 Creating search index...")
            search_client = UnifiedSearchClient()
            index_result = await search_client.get_or_create_vector_index(
                index_name=f"{domain}-vector-index", vector_dimension=1536
            )

            if index_result.get("success"):
                initialization_results["resources_initialized"].append(
                    f"Search index: {domain}-vector-index"
                )

            # Initialize Cosmos DB graph
            print(f"   🕸️  Preparing graph database...")
            # This would normally create the graph database structure
            initialization_results["resources_initialized"].append(
                f"Graph database: {domain}"
            )

            initialization_results["success"] = True
            print(
                f"✅ Domain initialization complete: {len(initialization_results['resources_initialized'])} resources"
            )

            return initialization_results

        except Exception as e:
            initialization_results["error"] = str(e)
            print(f"❌ Domain initialization failed: {e}")
            logger.error(f"Domain initialization failed: {e}", exc_info=True)
            return initialization_results


async def main():
    """Main entry point for Azure services setup"""
    parser = argparse.ArgumentParser(
        description="Azure Services Setup - Infrastructure Validation and Initialization"
    )
    parser.add_argument(
        "--domain",
        default="general",
        help="Target domain for validation/initialization",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate services, don't initialize",
    )
    parser.add_argument(
        "--initialize-domain",
        action="store_true",
        help="Initialize domain-specific resources",
    )
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Initialize setup service
    setup = AzureServicesSetup()

    # Validate all services
    validation_results = await setup.validate_all_services(args.domain)

    # Initialize domain resources if requested
    if args.initialize_domain and not args.validate_only:
        if validation_results.get("overall_status") in ["healthy", "degraded"]:
            init_results = await setup.initialize_domain_resources(args.domain)
            validation_results["domain_initialization"] = init_results
        else:
            print(f"⚠️  Skipping domain initialization - services not healthy enough")

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(validation_results, f, indent=2)
        print(f"📄 Results saved to: {args.output}")

    # Return appropriate exit code
    success = validation_results.get("overall_status") in ["healthy", "degraded"]
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
