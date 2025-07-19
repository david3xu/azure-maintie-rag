"""Azure Machine Learning client for Universal RAG model training."""

import logging
from typing import Dict, List, Any, Optional
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError

from backend.config.azure_settings import azure_settings

logger = logging.getLogger(__name__)


class AzureMLClient:
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
            self.credential = DefaultAzureCredential()
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

    def submit_training_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Submit GNN training job - universal configuration"""
        try:
            from azure.ai.ml.entities import CommandJob
            from azure.ai.ml import command, Input, Output

            job = command(
                inputs={
                    "data": Input(type="uri_folder", path=job_config.get("data_path", "")),
                    "model_config": Input(type="uri_file", path=job_config.get("config_path", ""))
                },
                outputs={
                    "model": Output(type="uri_folder", mode="rw_mount")
                },
                code=job_config.get("code_path", "./"),
                command=job_config.get("command", "python train.py"),
                environment=job_config.get("environment", "AzureML-sklearn-1.0-ubuntu20.04-py38-cpu"),
                compute=job_config.get("compute_target", "cpu-cluster"),
                display_name=job_config.get("display_name", "Universal RAG Training"),
                description=job_config.get("description", "Universal RAG model training job")
            )

            submitted_job = self.ml_client.jobs.create_or_update(job)

            return {
                "success": True,
                "job_id": submitted_job.name,
                "job_status": submitted_job.status,
                "display_name": submitted_job.display_name
            }

        except Exception as e:
            logger.error(f"Training job submission failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "job_config": job_config.get("display_name", "unknown")
            }

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

    def register_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Register trained model - universal model registration"""
        try:
            from azure.ai.ml.entities import Model

            model = Model(
                path=model_config.get("model_path", ""),
                name=model_config.get("model_name", "universal-rag-model"),
                version=model_config.get("version", "1"),
                description=model_config.get("description", "Universal RAG trained model"),
                tags=model_config.get("tags", {"domain": "universal", "type": "rag"})
            )

            registered_model = self.ml_client.models.create_or_update(model)

            return {
                "success": True,
                "model_name": registered_model.name,
                "model_version": registered_model.version,
                "model_id": registered_model.id
            }

        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_config.get("model_name", "unknown")
            }

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
                "workspace_name": self.workspace_name
            }