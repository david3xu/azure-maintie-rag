#!/usr/bin/env python3
"""
Azure Services Cleanup Script
Safely cleans up Azure resources created by the Universal RAG system
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Set

# Add root directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.search import SearchManagementClient
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cosmosdb import CosmosDBManagementClient
from azure.mgmt.keyvault import KeyVaultManagementClient
from azure.mgmt.applicationinsights import ApplicationInsightsManagementClient
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from azure.mgmt.web import WebSiteManagementClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AzureServicesCleanup:
    """Azure services cleanup with safety checks"""

    def __init__(self, subscription_id: str, dry_run: bool = True):
        self.subscription_id = subscription_id
        self.dry_run = dry_run
        self.credential = DefaultAzureCredential()
        self.deleted_resources: List[Dict] = []
        self.skipped_resources: List[Dict] = []
        self.failed_deletions: List[Dict] = []

        # Initialize management clients
        self._init_clients()

        # Resource patterns to identify RAG system resources
        self.rag_patterns = [
            "maintie-rag",
            "azure-maintie-rag",
            "universal-rag",
            "maintie",
            "rag-system"
        ]

        # Protected resource patterns (never delete these)
        self.protected_patterns = [
            "production",
            "prod-",
            "-prod",
            "shared",
            "common",
            "platform"
        ]

    def _init_clients(self):
        """Initialize Azure management clients"""
        try:
            self.resource_client = ResourceManagementClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )
            self.storage_client = StorageManagementClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )
            self.search_client = SearchManagementClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )
            self.cognitive_client = CognitiveServicesManagementClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )
            self.cosmos_client = CosmosDBManagementClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )
            self.keyvault_client = KeyVaultManagementClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )
            self.appinsights_client = ApplicationInsightsManagementClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )
            self.registry_client = ContainerRegistryManagementClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )
            self.webapp_client = WebSiteManagementClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )

            logger.info("Azure management clients initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure clients: {e}")
            raise

    def _is_rag_resource(self, resource_name: str) -> bool:
        """Check if resource belongs to RAG system"""
        resource_lower = resource_name.lower()
        return any(pattern in resource_lower for pattern in self.rag_patterns)

    def _is_protected_resource(self, resource_name: str) -> bool:
        """Check if resource is protected from deletion"""
        resource_lower = resource_name.lower()
        return any(pattern in resource_lower for pattern in self.protected_patterns)

    def discover_rag_resource_groups(self) -> List[Dict]:
        """Discover resource groups that contain RAG system resources"""
        logger.info("Discovering RAG system resource groups...")

        rag_resource_groups = []

        try:
            for rg in self.resource_client.resource_groups.list():
                if self._is_rag_resource(rg.name) and not self._is_protected_resource(rg.name):
                    # Get resource count in the group
                    resources = list(self.resource_client.resources.list_by_resource_group(rg.name))

                    rg_info = {
                        "name": rg.name,
                        "location": rg.location,
                        "resource_count": len(resources),
                        "tags": rg.tags or {},
                        "resources": [{"name": r.name, "type": r.type} for r in resources[:10]]  # First 10
                    }

                    rag_resource_groups.append(rg_info)
                    logger.info(f"Found RAG resource group: {rg.name} ({len(resources)} resources)")

        except Exception as e:
            logger.error(f"Error discovering resource groups: {e}")

        return rag_resource_groups

    def discover_rag_resources(self) -> Dict[str, List]:
        """Discover all RAG system resources across the subscription"""
        logger.info("Discovering RAG system resources...")

        resources = {
            "storage_accounts": [],
            "search_services": [],
            "cognitive_services": [],
            "cosmos_accounts": [],
            "key_vaults": [],
            "app_insights": [],
            "container_registries": [],
            "web_apps": [],
            "other_resources": []
        }

        try:
            # Get all resources in subscription
            all_resources = self.resource_client.resources.list()

            for resource in all_resources:
                if self._is_rag_resource(resource.name) and not self._is_protected_resource(resource.name):
                    resource_info = {
                        "name": resource.name,
                        "type": resource.type,
                        "resource_group": resource_group_from_id(resource.id),
                        "location": resource.location,
                        "tags": resource.tags or {},
                        "id": resource.id
                    }

                    # Categorize by resource type
                    if "Microsoft.Storage/storageAccounts" in resource.type:
                        resources["storage_accounts"].append(resource_info)
                    elif "Microsoft.Search/searchServices" in resource.type:
                        resources["search_services"].append(resource_info)
                    elif "Microsoft.CognitiveServices/accounts" in resource.type:
                        resources["cognitive_services"].append(resource_info)
                    elif "Microsoft.DocumentDB/databaseAccounts" in resource.type:
                        resources["cosmos_accounts"].append(resource_info)
                    elif "Microsoft.KeyVault/vaults" in resource.type:
                        resources["key_vaults"].append(resource_info)
                    elif "Microsoft.Insights/components" in resource.type:
                        resources["app_insights"].append(resource_info)
                    elif "Microsoft.ContainerRegistry/registries" in resource.type:
                        resources["container_registries"].append(resource_info)
                    elif "Microsoft.Web/sites" in resource.type:
                        resources["web_apps"].append(resource_info)
                    else:
                        resources["other_resources"].append(resource_info)

        except Exception as e:
            logger.error(f"Error discovering resources: {e}")

        # Log summary
        total_resources = sum(len(resources[key]) for key in resources)
        logger.info(f"Discovered {total_resources} RAG system resources")

        for resource_type, resource_list in resources.items():
            if resource_list:
                logger.info(f"  {resource_type}: {len(resource_list)}")

        return resources

    async def delete_resource_group(self, resource_group_name: str) -> bool:
        """Delete a resource group and all its contents"""
        if self._is_protected_resource(resource_group_name):
            logger.warning(f"Skipping protected resource group: {resource_group_name}")
            self.skipped_resources.append({
                "name": resource_group_name,
                "type": "ResourceGroup",
                "reason": "Protected resource"
            })
            return False

        if self.dry_run:
            logger.info(f"[DRY RUN] Would delete resource group: {resource_group_name}")
            return True

        try:
            logger.info(f"Deleting resource group: {resource_group_name}")

            # Start async deletion
            delete_operation = self.resource_client.resource_groups.begin_delete(resource_group_name)

            # Wait for completion (this can take several minutes)
            logger.info(f"Waiting for resource group deletion to complete: {resource_group_name}")
            result = delete_operation.result()  # This blocks until complete

            self.deleted_resources.append({
                "name": resource_group_name,
                "type": "ResourceGroup",
                "deletion_time": datetime.now().isoformat()
            })

            logger.info(f"Successfully deleted resource group: {resource_group_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete resource group {resource_group_name}: {e}")
            self.failed_deletions.append({
                "name": resource_group_name,
                "type": "ResourceGroup",
                "error": str(e)
            })
            return False

    async def delete_individual_resource(self, resource_info: Dict) -> bool:
        """Delete an individual resource"""
        resource_name = resource_info["name"]
        resource_type = resource_info["type"]
        resource_group = resource_info["resource_group"]

        if self._is_protected_resource(resource_name):
            logger.warning(f"Skipping protected resource: {resource_name}")
            self.skipped_resources.append({
                "name": resource_name,
                "type": resource_type,
                "reason": "Protected resource"
            })
            return False

        if self.dry_run:
            logger.info(f"[DRY RUN] Would delete {resource_type}: {resource_name}")
            return True

        try:
            logger.info(f"Deleting {resource_type}: {resource_name}")

            # Use generic resource deletion
            delete_operation = self.resource_client.resources.begin_delete(
                resource_group_name=resource_group,
                resource_provider_namespace=resource_type.split('/')[0],
                resource_type=resource_type.split('/')[1],
                resource_name=resource_name,
                api_version="2021-04-01"
            )

            # Wait for completion
            delete_operation.result()

            self.deleted_resources.append({
                "name": resource_name,
                "type": resource_type,
                "resource_group": resource_group,
                "deletion_time": datetime.now().isoformat()
            })

            logger.info(f"Successfully deleted {resource_type}: {resource_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete {resource_type} {resource_name}: {e}")
            self.failed_deletions.append({
                "name": resource_name,
                "type": resource_type,
                "resource_group": resource_group,
                "error": str(e)
            })
            return False

    async def cleanup_by_resource_groups(self, resource_groups: List[str]) -> Dict:
        """Clean up by deleting entire resource groups"""
        logger.info(f"Starting cleanup of {len(resource_groups)} resource groups...")

        results = {
            "deleted": [],
            "failed": [],
            "skipped": []
        }

        for rg_name in resource_groups:
            success = await self.delete_resource_group(rg_name)
            if success:
                results["deleted"].append(rg_name)
            else:
                if any(s["name"] == rg_name for s in self.skipped_resources):
                    results["skipped"].append(rg_name)
                else:
                    results["failed"].append(rg_name)

        return results

    async def cleanup_individual_resources(self, resources: Dict[str, List]) -> Dict:
        """Clean up individual resources without deleting resource groups"""
        logger.info("Starting cleanup of individual resources...")

        results = {
            "deleted": [],
            "failed": [],
            "skipped": []
        }

        # Process each resource type
        for resource_type, resource_list in resources.items():
            logger.info(f"Processing {len(resource_list)} {resource_type}...")

            for resource_info in resource_list:
                success = await self.delete_individual_resource(resource_info)
                resource_name = resource_info["name"]

                if success:
                    results["deleted"].append(resource_name)
                else:
                    if any(s["name"] == resource_name for s in self.skipped_resources):
                        results["skipped"].append(resource_name)
                    else:
                        results["failed"].append(resource_name)

        return results

    def generate_cleanup_report(self) -> Dict:
        """Generate comprehensive cleanup report"""
        return {
            "cleanup_summary": {
                "dry_run": self.dry_run,
                "timestamp": datetime.now().isoformat(),
                "subscription_id": self.subscription_id,
                "total_deleted": len(self.deleted_resources),
                "total_skipped": len(self.skipped_resources),
                "total_failed": len(self.failed_deletions)
            },
            "deleted_resources": self.deleted_resources,
            "skipped_resources": self.skipped_resources,
            "failed_deletions": self.failed_deletions,
            "patterns_used": {
                "rag_patterns": self.rag_patterns,
                "protected_patterns": self.protected_patterns
            }
        }

    def print_cleanup_summary(self):
        """Print cleanup summary to console"""
        print("\n" + "="*80)
        print("AZURE SERVICES CLEANUP SUMMARY")
        print("="*80)

        print(f"\nMode: {'DRY RUN' if self.dry_run else 'LIVE DELETION'}")
        print(f"Subscription: {self.subscription_id}")
        print(f"Timestamp: {datetime.now().isoformat()}")

        print(f"\nResults:")
        print(f"  ✅ Deleted: {len(self.deleted_resources)}")
        print(f"  ⚠️  Skipped: {len(self.skipped_resources)}")
        print(f"  ❌ Failed: {len(self.failed_deletions)}")

        if self.deleted_resources:
            print(f"\nDeleted Resources:")
            for resource in self.deleted_resources:
                print(f"  - {resource['type']}: {resource['name']}")

        if self.skipped_resources:
            print(f"\nSkipped Resources:")
            for resource in self.skipped_resources:
                print(f"  - {resource['type']}: {resource['name']} ({resource['reason']})")

        if self.failed_deletions:
            print(f"\nFailed Deletions:")
            for resource in self.failed_deletions:
                print(f"  - {resource['type']}: {resource['name']} - {resource['error']}")

        print("\n" + "="*80)


def resource_group_from_id(resource_id: str) -> str:
    """Extract resource group name from Azure resource ID"""
    parts = resource_id.split('/')
    try:
        rg_index = parts.index('resourceGroups')
        return parts[rg_index + 1]
    except (ValueError, IndexError):
        return "unknown"


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Clean up Azure Universal RAG system resources")
    parser.add_argument("--subscription-id", required=True, help="Azure subscription ID")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Run in dry-run mode (default: True)")
    parser.add_argument("--live", action="store_true",
                       help="Run in live mode (actually delete resources)")
    parser.add_argument("--resource-groups", nargs="+",
                       help="Specific resource groups to delete")
    parser.add_argument("--individual", action="store_true",
                       help="Delete individual resources instead of entire resource groups")
    parser.add_argument("--output", help="Output file for cleanup report")

    args = parser.parse_args()

    # Determine run mode
    dry_run = not args.live  # Default to dry run unless --live is specified

    if not dry_run:
        print("\n⚠️  WARNING: This will DELETE Azure resources permanently!")
        print("⚠️  This action cannot be undone!")
        confirmation = input("\nType 'DELETE_RESOURCES' to confirm: ")

        if confirmation != "DELETE_RESOURCES":
            print("Cleanup cancelled.")
            sys.exit(0)

    try:
        # Initialize cleanup manager
        cleanup = AzureServicesCleanup(
            subscription_id=args.subscription_id,
            dry_run=dry_run
        )

        if args.resource_groups:
            # Clean up specific resource groups
            logger.info(f"Cleaning up specified resource groups: {args.resource_groups}")
            results = await cleanup.cleanup_by_resource_groups(args.resource_groups)

        elif args.individual:
            # Clean up individual resources
            logger.info("Discovering and cleaning up individual resources...")
            resources = cleanup.discover_rag_resources()
            results = await cleanup.cleanup_individual_resources(resources)

        else:
            # Discover and clean up resource groups
            logger.info("Discovering and cleaning up RAG system resource groups...")
            resource_groups = cleanup.discover_rag_resource_groups()
            rg_names = [rg["name"] for rg in resource_groups]

            if not rg_names:
                print("No RAG system resource groups found.")
                sys.exit(0)

            print(f"\nFound {len(rg_names)} RAG system resource groups:")
            for rg in resource_groups:
                print(f"  - {rg['name']} ({rg['resource_count']} resources)")

            if not dry_run:
                confirm = input(f"\nProceed with deleting {len(rg_names)} resource groups? (y/N): ")
                if confirm.lower() != 'y':
                    print("Cleanup cancelled.")
                    sys.exit(0)

            results = await cleanup.cleanup_by_resource_groups(rg_names)

        # Generate and display report
        report = cleanup.generate_cleanup_report()
        cleanup.print_cleanup_summary()

        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nCleanup report saved to: {args.output}")

        # Exit with appropriate code
        if cleanup.failed_deletions:
            sys.exit(1)  # Some failures
        else:
            sys.exit(0)  # Success

    except KeyboardInterrupt:
        print("\n\nCleanup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}")
        print(f"\n❌ Cleanup failed with error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
