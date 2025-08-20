#!/usr/bin/env python3
"""
Centralized Azure ML endpoint management to prevent duplicate endpoints.

This module ensures we reuse existing endpoints instead of creating new ones.
"""

import logging
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    logger.warning("Azure ML SDK not available")

from config.azure_settings import azure_settings
from infrastructure.azure_auth_utils import get_azure_credential


class EndpointManager:
    """Centralized manager for Azure ML endpoints to prevent duplicates."""

    def __init__(self):
        self.credential = get_azure_credential()
        self.ml_client = None
        self._initialized = False

    async def initialize(self):
        """Initialize Azure ML client."""
        if self._initialized:
            return

        if not AZURE_ML_AVAILABLE:
            raise ImportError("Azure ML SDK required for endpoint management")

        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=azure_settings.azure_subscription_id,
            resource_group_name=azure_settings.azure_resource_group,
            workspace_name=azure_settings.azure_ml_workspace_name,
        )
        self._initialized = True
        logger.info("EndpointManager initialized")

    async def list_gnn_endpoints(self) -> List[Any]:
        """List all existing GNN endpoints."""
        await self.initialize()

        try:
            endpoints = list(self.ml_client.online_endpoints.list())
            gnn_endpoints = [
                ep for ep in endpoints
                if (ep.name.startswith('gnn-') or
                    (ep.tags and ep.tags.get('model_type') in ['gnn', 'universal_gnn']))
            ]
            return gnn_endpoints
        except Exception as e:
            logger.error(f"Failed to list GNN endpoints: {e}")
            return []

    async def get_primary_gnn_endpoint(self) -> Optional[str]:
        """Get the primary GNN endpoint name, or None if none exists."""
        endpoints = await self.list_gnn_endpoints()

        if not endpoints:
            return None

        # Prefer endpoints tagged as 'shared' or 'primary'
        for ep in endpoints:
            if ep.tags and (ep.tags.get('shared') == 'true' or ep.tags.get('primary') == 'true'):
                logger.info(f"✅ Found primary GNN endpoint: {ep.name}")
                return ep.name

        # Otherwise, use the most recent
        latest = max(endpoints, key=lambda x: x.creation_context.created_at if x.creation_context else '')
        logger.info(f"✅ Using latest GNN endpoint: {latest.name}")
        return latest.name

    async def create_shared_gnn_endpoint(self, force_new: bool = False) -> str:
        """Create a single shared GNN endpoint if none exists."""
        await self.initialize()

        # Check for existing endpoint first
        if not force_new:
            existing = await self.get_primary_gnn_endpoint()
            if existing:
                logger.info(f"♻️  Reusing existing GNN endpoint: {existing}")
                return existing

        # Create new shared endpoint
        timestamp = int(time.time())
        endpoint_name = f"gnn-shared-{timestamp}"

        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Shared GNN endpoint for Azure Universal RAG tri-modal search",
            auth_mode="key",
            tags={
                "model_type": "universal_gnn",
                "purpose": "tri_modal_search",
                "shared": "true",
                "created_by": "endpoint_manager",
            },
        )

        logger.info(f"🌐 Creating shared GNN endpoint: {endpoint_name}")
        try:
            endpoint_result = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            logger.info(f"✅ Shared GNN endpoint created: {endpoint_name}")
            return endpoint_name
        except Exception as e:
            logger.error(f"Failed to create shared GNN endpoint: {e}")
            raise

    async def cleanup_duplicate_endpoints(self, keep_count: int = 1) -> Dict[str, Any]:
        """Clean up duplicate GNN endpoints, keeping only the most recent ones."""
        await self.initialize()

        endpoints = await self.list_gnn_endpoints()

        if len(endpoints) <= keep_count:
            logger.info(f"Only {len(endpoints)} GNN endpoints found, no cleanup needed")
            return {"cleaned": 0, "kept": len(endpoints)}

        # Sort by creation time (newest first)
        sorted_endpoints = sorted(
            endpoints,
            key=lambda x: x.creation_context.created_at if x.creation_context else '',
            reverse=True
        )

        # Keep the newest ones
        to_keep = sorted_endpoints[:keep_count]
        to_delete = sorted_endpoints[keep_count:]

        logger.info(f"🧹 Cleaning up {len(to_delete)} duplicate GNN endpoints")

        deleted_count = 0
        for endpoint in to_delete:
            try:
                logger.info(f"   Deleting: {endpoint.name}")
                self.ml_client.online_endpoints.begin_delete(endpoint.name).wait()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"   Failed to delete {endpoint.name}: {e}")

        logger.info(f"✅ Cleanup complete: {deleted_count} deleted, {len(to_keep)} kept")
        return {"cleaned": deleted_count, "kept": len(to_keep)}


# Global instance
_endpoint_manager = None

async def get_endpoint_manager() -> EndpointManager:
    """Get the global endpoint manager instance."""
    global _endpoint_manager
    if _endpoint_manager is None:
        _endpoint_manager = EndpointManager()
    return _endpoint_manager
