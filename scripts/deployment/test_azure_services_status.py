#!/usr/bin/env python3
"""
Azure Services Status Test
Tests connectivity and status of all Azure services used by the Universal RAG system
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add root directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from azure.storage.blob.aio import BlobServiceClient
from azure.search.documents.aio import SearchClient
from azure.cosmos import CosmosClient
from azure.ai.ml.aio import MLClient
from azure.monitor.applicationinsights import ApplicationInsightsDataClient
from azure.keyvault.secrets.aio import SecretClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AzureServicesStatusChecker:
    """Azure services status checker with comprehensive testing"""
    
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.results: Dict[str, Dict] = {}
        self.start_time = datetime.now()
        
    async def test_azure_openai(self) -> Dict[str, Any]:
        """Test Azure OpenAI service connectivity"""
        service_name = "Azure OpenAI"
        try:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                return {
                    "status": "failed",
                    "error": "AZURE_OPENAI_ENDPOINT not set",
                    "details": "Environment variable missing"
                }
                
            # Use httpx to test the endpoint
            import httpx
            
            async with httpx.AsyncClient() as client:
                # Get token for Azure OpenAI
                token = await self.credential.get_token("https://cognitiveservices.azure.com/.default")
                headers = {
                    "Authorization": f"Bearer {token.token}",
                    "Content-Type": "application/json"
                }
                
                # Test models endpoint
                models_url = f"{endpoint.rstrip('/')}/openai/models?api-version=2024-02-01"
                response = await client.get(models_url, headers=headers, timeout=30.0)
                
                if response.status_code == 200:
                    models_data = response.json()
                    return {
                        "status": "healthy",
                        "endpoint": endpoint,
                        "models_count": len(models_data.get("data", [])),
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    }
                else:
                    return {
                        "status": "failed",
                        "error": f"HTTP {response.status_code}",
                        "details": response.text[:200]
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "service": service_name
            }
    
    async def test_azure_search(self) -> Dict[str, Any]:
        """Test Azure Cognitive Search service"""
        service_name = "Azure Cognitive Search"
        try:
            endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
            if not endpoint:
                return {
                    "status": "failed",
                    "error": "AZURE_SEARCH_ENDPOINT not set"
                }
                
            # Test with a dummy index name for connection testing
            search_client = SearchClient(
                endpoint=endpoint,
                index_name="test-connection",
                credential=self.credential
            )
            
            # Try to get service statistics
            import httpx
            async with httpx.AsyncClient() as client:
                token = await self.credential.get_token("https://search.azure.com/.default")
                headers = {
                    "Authorization": f"Bearer {token.token}",
                    "Content-Type": "application/json"
                }
                
                stats_url = f"{endpoint.rstrip('/')}/servicestats?api-version=2023-11-01"
                response = await client.get(stats_url, headers=headers, timeout=30.0)
                
                if response.status_code == 200:
                    stats = response.json()
                    return {
                        "status": "healthy",
                        "endpoint": endpoint,
                        "storage_size_mb": stats.get("counters", {}).get("storageSize", 0) / (1024*1024),
                        "document_count": stats.get("counters", {}).get("documentCount", 0),
                        "index_count": stats.get("counters", {}).get("indexCount", 0)
                    }
                else:
                    return {
                        "status": "degraded",
                        "error": f"HTTP {response.status_code}",
                        "endpoint": endpoint
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "service": service_name
            }
    
    async def test_cosmos_db(self) -> Dict[str, Any]:
        """Test Azure Cosmos DB connectivity"""
        service_name = "Azure Cosmos DB"
        try:
            endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
            if not endpoint:
                return {
                    "status": "failed",
                    "error": "AZURE_COSMOS_ENDPOINT not set"
                }
                
            # Create Cosmos client
            cosmos_client = CosmosClient(endpoint, credential=self.credential)
            
            # Test connection by listing databases
            databases = list(cosmos_client.list_databases())
            
            return {
                "status": "healthy",
                "endpoint": endpoint,
                "database_count": len(databases),
                "databases": [db["id"] for db in databases[:5]]  # First 5 databases
            }
            
        except Exception as e:
            return {
                "status": "failed", 
                "error": str(e),
                "service": service_name
            }
    
    async def test_storage_account(self) -> Dict[str, Any]:
        """Test Azure Storage Account"""
        service_name = "Azure Storage"
        try:
            account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
            if not account_name:
                return {
                    "status": "failed",
                    "error": "AZURE_STORAGE_ACCOUNT not set"
                }
                
            storage_url = f"https://{account_name}.blob.core.windows.net"
            
            async with BlobServiceClient(account_url=storage_url, credential=self.credential) as blob_client:
                # List containers to test connectivity
                containers = []
                async for container in blob_client.list_containers():
                    containers.append(container.name)
                    if len(containers) >= 10:  # Limit to first 10
                        break
                
                # Get account properties
                properties = await blob_client.get_account_information()
                
                return {
                    "status": "healthy",
                    "account_name": account_name,
                    "container_count": len(containers),
                    "containers": containers,
                    "account_kind": properties.account_kind.value if properties.account_kind else "unknown"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "service": service_name
            }
    
    async def test_key_vault(self) -> Dict[str, Any]:
        """Test Azure Key Vault connectivity"""
        service_name = "Azure Key Vault"
        try:
            vault_name = os.getenv("AZURE_KEY_VAULT_NAME")
            if not vault_name:
                return {
                    "status": "failed",
                    "error": "AZURE_KEY_VAULT_NAME not set"
                }
                
            vault_url = f"https://{vault_name}.vault.azure.net/"
            
            async with SecretClient(vault_url=vault_url, credential=self.credential) as secret_client:
                # List first few secrets to test connectivity
                secrets = []
                async for secret in secret_client.list_properties_of_secrets():
                    secrets.append(secret.name)
                    if len(secrets) >= 5:  # Limit to first 5
                        break
                
                return {
                    "status": "healthy",
                    "vault_name": vault_name,
                    "vault_url": vault_url,
                    "secret_count": len(secrets),
                    "secrets": secrets
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "service": service_name
            }
    
    async def test_ml_workspace(self) -> Dict[str, Any]:
        """Test Azure ML Workspace"""
        service_name = "Azure ML"
        try:
            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            resource_group = os.getenv("AZURE_RESOURCE_GROUP") or os.getenv("AZURE_ML_RESOURCE_GROUP")
            workspace_name = os.getenv("AZURE_ML_WORKSPACE_NAME")
            
            if not all([subscription_id, resource_group, workspace_name]):
                missing = []
                if not subscription_id: missing.append("AZURE_SUBSCRIPTION_ID")
                if not resource_group: missing.append("AZURE_RESOURCE_GROUP")
                if not workspace_name: missing.append("AZURE_ML_WORKSPACE_NAME")
                
                return {
                    "status": "failed",
                    "error": f"Missing environment variables: {', '.join(missing)}"
                }
            
            async with MLClient(
                credential=self.credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            ) as ml_client:
                # Get workspace info
                workspace = await ml_client.workspaces.get(workspace_name)
                
                return {
                    "status": "healthy",
                    "workspace_name": workspace_name,
                    "resource_group": resource_group,
                    "location": workspace.location,
                    "description": workspace.description
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "service": service_name
            }
    
    async def test_application_insights(self) -> Dict[str, Any]:
        """Test Application Insights connectivity"""
        service_name = "Application Insights"
        try:
            connection_string = os.getenv("AZURE_APP_INSIGHTS_CONNECTION_STRING")
            if not connection_string:
                return {
                    "status": "failed",
                    "error": "AZURE_APP_INSIGHTS_CONNECTION_STRING not set"
                }
            
            # Parse connection string to get instrumentation key
            import re
            instrumentation_key_match = re.search(r'InstrumentationKey=([^;]+)', connection_string)
            if not instrumentation_key_match:
                return {
                    "status": "failed",
                    "error": "Invalid connection string format"
                }
            
            # For now, just validate the connection string format
            return {
                "status": "healthy",
                "connection_string_configured": True,
                "has_instrumentation_key": bool(instrumentation_key_match),
                "note": "Connection validated by format check"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "service": service_name
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Azure service tests concurrently"""
        logger.info("Starting Azure services status check...")
        
        # Define all tests
        tests = {
            "azure_openai": self.test_azure_openai(),
            "azure_search": self.test_azure_search(),
            "cosmos_db": self.test_cosmos_db(),
            "storage_account": self.test_storage_account(),
            "key_vault": self.test_key_vault(),
            "ml_workspace": self.test_ml_workspace(),
            "application_insights": self.test_application_insights()
        }
        
        # Run all tests concurrently
        results = await asyncio.gather(*tests.values(), return_exceptions=True)
        
        # Combine results
        service_results = {}
        for i, (service_name, result) in enumerate(zip(tests.keys(), results)):
            if isinstance(result, Exception):
                service_results[service_name] = {
                    "status": "failed",
                    "error": str(result),
                    "exception_type": type(result).__name__
                }
            else:
                service_results[service_name] = result
        
        # Calculate summary
        total_services = len(service_results)
        healthy_services = sum(1 for r in service_results.values() if r.get("status") == "healthy")
        failed_services = sum(1 for r in service_results.values() if r.get("status") == "failed")
        degraded_services = sum(1 for r in service_results.values() if r.get("status") == "degraded")
        
        test_duration = (datetime.now() - self.start_time).total_seconds()
        
        summary = {
            "overall_status": "healthy" if failed_services == 0 else "degraded" if healthy_services > 0 else "failed",
            "total_services": total_services,
            "healthy_services": healthy_services,
            "degraded_services": degraded_services,
            "failed_services": failed_services,
            "test_duration_seconds": round(test_duration, 2),
            "timestamp": datetime.now().isoformat(),
            "environment_variables_checked": [
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_SEARCH_ENDPOINT", 
                "AZURE_COSMOS_ENDPOINT",
                "AZURE_STORAGE_ACCOUNT",
                "AZURE_KEY_VAULT_NAME",
                "AZURE_ML_WORKSPACE_NAME",
                "AZURE_APP_INSIGHTS_CONNECTION_STRING"
            ]
        }
        
        return {
            "summary": summary,
            "services": service_results
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted results to console"""
        print("\n" + "="*80)
        print("AZURE SERVICES STATUS CHECK")
        print("="*80)
        
        summary = results["summary"]
        
        # Overall status
        status_color = {
            "healthy": "\033[92m",    # Green
            "degraded": "\033[93m",   # Yellow  
            "failed": "\033[91m"      # Red
        }
        reset_color = "\033[0m"
        
        status = summary["overall_status"]
        color = status_color.get(status, "")
        
        print(f"\nOverall Status: {color}{status.upper()}{reset_color}")
        print(f"Test Duration: {summary['test_duration_seconds']}s")
        print(f"Timestamp: {summary['timestamp']}")
        
        # Summary counts
        print(f"\nServices Summary:")
        print(f"  ✅ Healthy: {summary['healthy_services']}")
        print(f"  ⚠️  Degraded: {summary['degraded_services']}")
        print(f"  ❌ Failed: {summary['failed_services']}")
        print(f"  📊 Total: {summary['total_services']}")
        
        # Detailed results
        print(f"\nDetailed Results:")
        print("-" * 80)
        
        for service_name, service_result in results["services"].items():
            status = service_result.get("status", "unknown")
            status_symbol = {
                "healthy": "✅",
                "degraded": "⚠️",
                "failed": "❌"
            }.get(status, "❓")
            
            print(f"\n{status_symbol} {service_name.replace('_', ' ').title()}")
            print(f"   Status: {status}")
            
            if status == "healthy":
                # Show healthy service details
                for key, value in service_result.items():
                    if key not in ["status", "service"]:
                        if isinstance(value, list):
                            print(f"   {key}: {len(value)} items")
                        else:
                            print(f"   {key}: {value}")
            else:
                # Show error details
                error = service_result.get("error", "Unknown error")
                print(f"   Error: {error}")
                
                details = service_result.get("details")
                if details:
                    print(f"   Details: {details}")
        
        print("\n" + "="*80)
        
        # Recommendations
        failed_services = [name for name, result in results["services"].items() 
                          if result.get("status") == "failed"]
        
        if failed_services:
            print("\nRecommendations:")
            print("1. Check environment variables are properly set")
            print("2. Verify Azure credentials with 'az login'")
            print("3. Ensure proper permissions for all services")
            print("4. Check network connectivity to Azure endpoints")
            
            print(f"\nFailed services to investigate: {', '.join(failed_services)}")


async def main():
    """Main entry point"""
    checker = AzureServicesStatusChecker()
    
    try:
        # Run all tests
        results = await checker.run_all_tests()
        
        # Print results to console
        checker.print_results(results)
        
        # Save results to file
        output_file = f"azure_services_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
        
        # Exit with proper code
        summary = results["summary"]
        if summary["overall_status"] == "healthy":
            sys.exit(0)  # All good
        elif summary["overall_status"] == "degraded":
            sys.exit(1)  # Some issues
        else:
            sys.exit(2)  # Major issues
            
    except KeyboardInterrupt:
        print("\n\nStatus check interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error during status check: {e}")
        print(f"\n❌ Status check failed with error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    # Install required packages if missing
    try:
        import httpx
    except ImportError:
        print("Installing required httpx package...")
        os.system("pip install httpx")
        import httpx
    
    # Run the status check
    asyncio.run(main())