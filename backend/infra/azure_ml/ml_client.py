"""Azure Machine Learning client for Universal RAG model training."""

import logging
import time
from typing import Dict, List, Any, Optional
from azure.ai.ml import MLClient
from azure.core.exceptions import AzureError

from config.settings import azure_settings
from infra.models.universal_rag_models import (
    UniversalTrainingConfig, UniversalTrainingResult
)

logger = logging.getLogger(__name__)


class AzureMLClient:
    """Azure ML client for workspace management and model deployment"""
    
    def __init__(self):
        """Initialize Azure ML client"""
        from azure.identity import DefaultAzureCredential
        
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=azure_settings.azure_subscription_id,
            resource_group_name=azure_settings.azure_resource_group,
            workspace_name=azure_settings.azure_ml_workspace_name,
        )
        logger.info("Azure ML client initialized")

    def get_workspace(self) -> Any:
        """Get Azure ML workspace"""
        return self.ml_client.workspaces.get(azure_settings.azure_ml_workspace_name)

    async def invoke_gnn_endpoint(self, endpoint_name: str, deployment_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke GNN model endpoint for embedding generation."""
        try:
            import aiohttp
            import json
            
            # Get endpoint details
            scoring_uri = endpoint_name if endpoint_name.startswith("http") else None
            if not scoring_uri:
                try:
                    endpoint = self.ml_client.online_endpoints.get(name=endpoint_name)
                    scoring_uri = endpoint.scoring_uri
                except Exception as e:
                    logger.error(f"Failed to get endpoint URI: {e}")
                    return {"success": False, "error": f"Endpoint not found: {endpoint_name}"}
            
            # Prepare headers
            headers = {"Content-Type": "application/json"}
            timeout = getattr(azure_settings, 'azure_ml_inference_timeout', 300)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    scoring_uri,
                    headers=headers,
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        return {"success": True, "data": result_data}
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
                        
        except Exception as e:
            logger.error(f"Failed to invoke endpoint {endpoint_name}: {e}")
            return {"success": False, "error": str(e)}

__all__ = ['AzureMLClient']