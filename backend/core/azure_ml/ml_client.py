"""Azure Machine Learning client for Universal RAG model training."""

import logging
import time
from typing import Dict, List, Any, Optional
from azure.ai.ml import MLClient
from azure.core.exceptions import AzureError

from config.settings import azure_settings
from core.models.universal_rag_models import (
    UniversalTrainingConfig, UniversalTrainingResult
)

logger = logging.getLogger(__name__)


class AzureMLClient:

    async def invoke_gnn_endpoint(self, endpoint_name: str, deployment_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke GNN model endpoint for embedding generation."""
        try:
            import aiohttp
            import json
            # Get endpoint details (assume endpoint_name is the full name)
            # In real Azure ML, you may need to fetch the scoring URI from the endpoint object
            # For this example, assume endpoint_name is the scoring URI
            scoring_uri = endpoint_name if endpoint_name.startswith("http") else None
            if not scoring_uri:
                # Try to get endpoint from MLClient if not a URI
                try:
                    endpoint = self.ml_client.online_endpoints.get(name=endpoint_name)
                    scoring_uri = endpoint.scoring_uri
                except Exception as e:
                    logger.error(f"Failed to get endpoint URI: {e}")
                    return {"success": False, "error": f"Endpoint not found: {endpoint_name}"}
            # Prepare authentication headers (assume admin key or token)
            headers = {"Content-Type": "application/json"}
            # If using Azure ML token auth, add Authorization header
            # (In production, use DefaultAzureCredential or ManagedIdentityCredential)
            # For now, skip auth for local dev
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
                        return {
                            "success": True,
                            "embeddings": result_data.get("embeddings", {}),
                            "model_version": result_data.get("model_version"),
                            "inference_time_ms": result_data.get("inference_time_ms")
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}",
                            "endpoint": endpoint_name
                        }
        except Exception as e:
            logger.error(f"GNN endpoint invocation failed: {e}")
            return {"success": False, "error": str(e)}
    """Universal Azure ML client for model training - follows azure_openai.py pattern"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Azure ML client"""
        self.config = config or {}

        # Load from environment (matches azure_openai.py pattern)
        self.subscription_id = self.config.get('subscription_id') or azure_settings.azure_subscription_id
        self.resource_group = self.config.get('resource_group') or azure_settings.azure_resource_group
        self.workspace_name = self.config.get('workspace_name') or azure_settings.azure_ml_workspace

        if not all([self.subscription_id, self.resource_group, self.workspace_name]):
            raise ValueError("Azure ML subscription_id, resource_group, and workspace_name are required")

        # Initialize client (follows azure_openai.py error handling pattern)
        try:
            self.credential = self._get_azure_credential()
            self.ml_client = MLClient(
                credential=self.credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure ML client: {e}")
            raise

        logger.info(f"AzureMLClient initialized for workspace: {self.workspace_name}")

    def _get_azure_credential(self):
        """Enterprise credential management - data-driven from config"""
        if azure_settings.azure_use_managed_identity and azure_settings.azure_managed_identity_client_id:
            from azure.identity import ManagedIdentityCredential
            return ManagedIdentityCredential(client_id=azure_settings.azure_managed_identity_client_id)

        # Final fallback to DefaultAzureCredential
        from azure.identity import DefaultAzureCredential
        return DefaultAzureCredential()

    def create_compute_instance(self, compute_name: str, vm_size: str = "Standard_DS3_v2") -> Dict[str, Any]:
        """Create compute instance for training - data-driven configuration"""
        try:
            from azure.ai.ml.entities import ComputeInstance

            compute_instance = ComputeInstance(
                name=compute_name,
                size=vm_size,
                idle_time_before_shutdown_minutes=30  # Cost optimization
            )

            result = self.ml_client.compute.begin_create_or_update(compute_instance)

            return {
                "success": True,
                "compute_name": compute_name,
                "vm_size": vm_size,
                "status": "creating"
            }

        except Exception as e:
            logger.error(f"Compute instance creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "compute_name": compute_name
            }

    def submit_training_job(self, training_config: UniversalTrainingConfig) -> UniversalTrainingResult:
        """Submit universal RAG training job using universal models"""
        try:
            from azure.ai.ml.entities import CommandJob
            from azure.ai.ml import command, Input, Output

            # Convert universal config to Azure ML job
            job = command(
                inputs={
                    "training_data": Input(type="uri_folder", path=training_config.training_data_path),
                    "validation_data": Input(type="uri_folder", path=training_config.validation_data_path) if training_config.validation_data_path else None,
                    "model_config": Input(type="uri_file", path=training_config.model_config.get("config_path", ""))
                },
                outputs={
                    "model": Output(type="uri_folder", mode="rw_mount")
                },
                code=training_config.model_config.get("code_path", "./"),
                command=training_config.model_config.get("command", "python train.py"),
                environment=training_config.model_config.get("environment", "AzureML-sklearn-1.0-ubuntu20.04-py38-cpu"),
                compute=training_config.model_config.get("compute_target", "cpu-cluster"),
                display_name=f"Universal RAG {training_config.model_type.upper()} Training - {training_config.domain}",
                description=f"Universal RAG {training_config.model_type} model training for {training_config.domain} domain"
            )

            submitted_job = self.ml_client.jobs.create_or_update(job)

            return UniversalTrainingResult(
                model_id=submitted_job.name,
                model_type=training_config.model_type,
                domain=training_config.domain,
                training_metrics={},
                validation_metrics={},
                model_path=None,
                training_time=None,
                metadata={
                    "job_status": submitted_job.status,
                    "display_name": submitted_job.display_name,
                    "hyperparameters": training_config.hyperparameters,
                    "training_metadata": training_config.training_metadata
                }
            )

        except Exception as e:
            logger.error(f"Universal training job submission failed: {e}")
            return UniversalTrainingResult(
                model_id="failed",
                model_type=training_config.model_type,
                domain=training_config.domain,
                training_metrics={},
                validation_metrics={},
                metadata={"error": str(e)}
            )

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status"""
        try:
            job = self.ml_client.jobs.get(job_id)

            return {
                "success": True,
                "job_id": job_id,
                "status": job.status,
                "display_name": job.display_name,
                "start_time": str(job.creation_context.created_at) if job.creation_context else None,
                "duration": job.duration if hasattr(job, 'duration') else None
            }

        except Exception as e:
            logger.error(f"Job status query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "job_id": job_id
            }

    def register_model(self, training_result: UniversalTrainingResult) -> UniversalTrainingResult:
        """Register trained universal RAG model"""
        try:
            from azure.ai.ml.entities import Model

            model = Model(
                path=training_result.model_path or "",
                name=f"universal-rag-{training_result.model_type}-{training_result.domain}",
                version="1",
                description=f"Universal RAG {training_result.model_type} model for {training_result.domain} domain",
                tags={
                    "domain": training_result.domain,
                    "type": "rag",
                    "model_type": training_result.model_type,
                    "universal": "true"
                }
            )

            registered_model = self.ml_client.models.create_or_update(model)

            # Update training result with registration info
            training_result.model_id = registered_model.id
            training_result.metadata.update({
                "registered_model_name": registered_model.name,
                "registered_model_version": registered_model.version,
                "registration_success": True
            })

            return training_result

        except Exception as e:
            logger.error(f"Universal model registration failed: {e}")
            training_result.metadata.update({
                "registration_success": False,
                "registration_error": str(e)
            })
            return training_result

    def get_workspace_status(self) -> Dict[str, Any]:
        """Get ML workspace status - follows azure_openai.py pattern"""
        try:
            workspace = self.ml_client.workspaces.get(self.workspace_name)

            return {
                "status": "healthy",
                "subscription_id": self.subscription_id,
                "resource_group": self.resource_group,
                "workspace_name": self.workspace_name,
                "workspace_id": workspace.id,
                "location": workspace.location
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "subscription_id": self.subscription_id,
                "resource_group": self.resource_group,
                "workspace_name": self.workspace_name
            }

    def list_compute_targets(self) -> List[Dict[str, Any]]:
        """List available compute targets"""
        try:
            compute_targets = self.ml_client.compute.list()

            return [
                {
                    "name": target.name,
                    "type": target.type,
                    "status": target.provisioning_state if hasattr(target, 'provisioning_state') else "unknown",
                    "vm_size": target.size if hasattr(target, 'size') else None
                }
                for target in compute_targets
            ]

        except Exception as e:
            logger.error(f"Compute targets listing failed: {e}")
            return []

    def list_environments(self) -> List[Dict[str, Any]]:
        """List available environments"""
        try:
            environments = self.ml_client.environments.list()

            return [
                {
                    "name": env.name,
                    "version": env.version,
                    "conda_file": env.conda_file if hasattr(env, 'conda_file') else None,
                    "image": env.image if hasattr(env, 'image') else None
                }
                for env in environments
            ]

        except Exception as e:
            logger.error(f"Environments listing failed: {e}")
            return []

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status for service validation"""
        try:
            # Simple test to check if required ML settings are configured
            if not self.subscription_id:
                return {
                    "status": "unhealthy",
                    "error": "ML subscription ID not configured",
                    "service": "ml"
                }

            if not self.resource_group:
                return {
                    "status": "unhealthy",
                    "error": "ML resource group not configured",
                    "service": "ml"
                }

            if not self.workspace_name:
                return {
                    "status": "unhealthy",
                    "error": "ML workspace name not configured",
                    "service": "ml"
                }

            return {
                "status": "healthy",
                "service": "ml",
                "subscription_id": self.subscription_id,
                "resource_group": self.resource_group,
                "workspace_name": self.workspace_name
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service": "ml"
            }