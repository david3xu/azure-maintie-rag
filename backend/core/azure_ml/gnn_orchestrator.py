"""
Azure ML GNN Training Orchestrator for Enterprise Pipeline
Handles automated training, model deployment, and real-time inference
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceNotFoundError

from ..azure_cosmos.enhanced_gremlin_client import EnterpriseGremlinGraphManager
from .gnn.trainer import UniversalGNNTrainer
from .gnn.data_loader import load_graph_data_from_cosmos

logger = logging.getLogger(__name__)


class AzureGNNTrainingOrchestrator:
    """Enterprise GNN training orchestration with Azure ML"""

    def __init__(self, ml_client: MLClient, cosmos_client: EnterpriseGremlinGraphManager):
        self.ml_client = ml_client
        self.cosmos_client = cosmos_client
        self.training_schedule = self._load_training_schedule()

    def _load_training_schedule(self) -> Dict[str, Any]:
        """Load training schedule configuration"""
        return {
            "trigger_threshold": 100,  # New entities/relations
            "training_frequency": "daily",
            "model_retention_days": 30,
            "deployment_tier": "standard"
        }

    async def orchestrate_incremental_training(
        self,
        domain: str,
        trigger_threshold: Optional[int] = None
    ) -> Dict[str, Any]:
        """Orchestrate incremental GNN training based on graph changes"""

        try:
            # Check graph change metrics
            change_metrics = await self.cosmos_client.get_graph_change_metrics(domain)
            threshold = trigger_threshold or self.training_schedule["trigger_threshold"]

            total_changes = change_metrics["new_entities"] + change_metrics["new_relations"]

            if total_changes >= threshold:
                logger.info(f"Triggering GNN retraining for domain {domain} (changes: {total_changes})")

                # Export fresh graph data
                graph_export = await self.cosmos_client.export_graph_for_training(domain)

                # Submit Azure ML training job
                training_job = await self._submit_azure_ml_training(
                    domain=domain,
                    graph_data=graph_export,
                    training_type="incremental"
                )

                # Track training progress
                training_results = await self._monitor_training_progress(training_job.id)

                # Update graph embeddings upon completion
                if training_results["status"] == "completed":
                    await self._update_graph_embeddings(
                        model_uri=training_results["model_uri"],
                        domain=domain
                    )

                return training_results

            return {
                "status": "no_training_required",
                "change_metrics": change_metrics,
                "threshold": threshold
            }

        except Exception as e:
            logger.error(f"Failed to orchestrate incremental training: {e}")
            raise

    async def _submit_azure_ml_training(
        self,
        domain: str,
        graph_data: Dict[str, Any],
        training_type: str = "full"
    ) -> Job:
        """Submit GNN training job to Azure ML"""

        try:
            # Prepare job configuration
            job_name = f"gnn-training-{domain}-{int(time.time())}"

            job_config = {
                "display_name": job_name,
                "experiment_name": "universal-rag-gnn",
                "description": f"GNN training for domain {domain} ({training_type})",
                "compute": "gnn-cluster",  # Your compute cluster name
                "environment": "gnn-training-env:latest",
                "code": "backend/",
                "command": [
                    "python", "scripts/train_comprehensive_gnn.py",
                    "--domain", domain,
                    "--training_type", training_type,
                    "--graph_data_path", "${{inputs.graph_data}}",
                    "--output_model_path", "${{outputs.trained_model}}",
                    "--config_path", "${{inputs.config}}"
                ],
                "inputs": {
                    "graph_data": {
                        "type": AssetTypes.URI_FOLDER,
                        "path": "azureml://datastores/workspaceblobstore/paths/graph_data/"
                    },
                    "config": {
                        "type": AssetTypes.URI_FILE,
                        "path": "azureml://datastores/workspaceblobstore/paths/configs/gnn_config.json"
                    }
                },
                "outputs": {
                    "trained_model": {
                        "type": AssetTypes.MLFLOW_MODEL
                    }
                }
            }

            # Create and submit job
            job = Job(**job_config)
            submitted_job = self.ml_client.jobs.create_or_update(job)

            logger.info(f"Submitted GNN training job: {submitted_job.id}")
            return submitted_job

        except Exception as e:
            logger.error(f"Failed to submit Azure ML training job: {e}")
            raise

    async def _monitor_training_progress(self, job_id: str) -> Dict[str, Any]:
        """Monitor training job progress"""

        try:
            max_wait_time = 3600  # 1 hour
            check_interval = 60  # 1 minute
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                job = self.ml_client.jobs.get(job_id)

                if job.status in ["Completed", "Failed", "Canceled"]:
                    if job.status == "Completed":
                        # Get model URI from job outputs
                        model_uri = job.outputs.get("trained_model", {}).get("path")

                        return {
                            "status": "completed",
                            "job_id": job_id,
                            "model_uri": model_uri,
                            "training_metrics": job.properties.get("metrics", {}),
                            "duration_minutes": elapsed_time / 60
                        }
                    else:
                        return {
                            "status": "failed",
                            "job_id": job_id,
                            "error": job.properties.get("error", "Unknown error")
                        }

                await asyncio.sleep(check_interval)
                elapsed_time += check_interval

            return {
                "status": "timeout",
                "job_id": job_id,
                "error": "Training job timed out"
            }

        except Exception as e:
            logger.error(f"Failed to monitor training progress: {e}")
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(e)
            }

    async def _update_graph_embeddings(
        self,
        model_uri: str,
        domain: str
    ) -> Dict[str, Any]:
        """Update graph embeddings with trained model"""

        try:
            # Load trained model
            trained_model = await self._load_trained_model(model_uri)

            # Update embeddings in Cosmos DB
            update_stats = await self.cosmos_client.update_graph_embeddings(
                trained_model=trained_model,
                domain=domain
            )

            logger.info(f"Updated graph embeddings: {update_stats}")
            return update_stats

        except Exception as e:
            logger.error(f"Failed to update graph embeddings: {e}")
            raise

    async def _load_trained_model(self, model_uri: str) -> torch.nn.Module:
        """Load trained model from Azure ML model registry"""

        try:
            # Load model from Azure ML
            model = self.ml_client.models.get(model_uri)

            # Download and load the model
            # This is a simplified version - actual implementation depends on your model format
            model_path = model.path
            trained_model = torch.load(model_path)

            return trained_model

        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            raise


class AzureGNNModelService:
    """Enterprise GNN model deployment and serving with Azure ML"""

    def __init__(self, ml_client: MLClient):
        self.ml_client = ml_client
        self.model_registry = self._initialize_model_registry()

    def _initialize_model_registry(self):
        """Initialize model registry"""
        return self.ml_client.models

    async def deploy_trained_gnn_model(
        self,
        model_uri: str,
        domain: str,
        deployment_tier: str = "standard"
    ) -> Dict[str, Any]:
        """Deploy trained GNN model to Azure ML endpoint"""

        try:
            # Register model in Azure ML
            model_registration = await self._register_model(
                model_uri=model_uri,
                domain=domain
            )

            # Create managed endpoint
            endpoint_name = f"gnn-endpoint-{domain}"
            endpoint_config = {
                "name": endpoint_name,
                "auth_mode": "key",
                "description": f"GNN model endpoint for {domain} domain"
            }

            endpoint = self.ml_client.online_endpoints.begin_create_or_update(
                ManagedOnlineEndpoint(**endpoint_config)
            ).result()

            # Deploy model to endpoint
            deployment_config = {
                "name": "primary",
                "model": model_registration.id,
                "instance_type": self._get_instance_type(deployment_tier),
                "instance_count": 1,
                "environment": "gnn-inference-env:latest",
                "code_configuration": {
                    "code": "deployment/",
                    "scoring_script": "score.py"
                }
            }

            deployment = self.ml_client.online_deployments.begin_create_or_update(
                ManagedOnlineDeployment(**deployment_config)
            ).result()

            return {
                "endpoint_name": endpoint_name,
                "model_id": model_registration.id,
                "deployment_status": "deployed",
                "inference_uri": endpoint.scoring_uri,
                "deployment_id": deployment.id
            }

        except Exception as e:
            logger.error(f"Failed to deploy GNN model: {e}")
            raise

    async def get_gnn_embeddings(
        self,
        entities: List[str],
        domain: str,
        endpoint_name: str
    ) -> Dict[str, np.ndarray]:
        """Get GNN embeddings for entities via deployed model"""

        try:
            # Prepare inference request
            inference_request = {
                "entities": entities,
                "domain": domain,
                "operation": "generate_embeddings"
            }

            # Call Azure ML endpoint
            response = await self._call_ml_endpoint(
                endpoint_name=endpoint_name,
                request_data=inference_request
            )

            # Parse embeddings response
            embeddings = {
                entity: np.array(embedding)
                for entity, embedding in response.get("embeddings", {}).items()
            }

            return embeddings

        except Exception as e:
            logger.error(f"Failed to get GNN embeddings: {e}")
            raise

    async def _register_model(
        self,
        model_uri: str,
        domain: str
    ) -> Any:
        """Register model in Azure ML model registry"""

        try:
            model_name = f"gnn-{domain}-{int(time.time())}"

            # Register model
            model = self.ml_client.models.create_or_update(
                name=model_name,
                path=model_uri,
                type=AssetTypes.MLFLOW_MODEL,
                description=f"GNN model for {domain} domain"
            )

            return model

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def _get_instance_type(self, deployment_tier: str) -> str:
        """Get Azure ML instance type based on deployment tier"""

        instance_types = {
            "standard": "Standard_DS3_v2",
            "premium": "Standard_DS4_v2",
            "basic": "Standard_DS2_v2"
        }

        return instance_types.get(deployment_tier, "Standard_DS3_v2")

    async def _call_ml_endpoint(
        self,
        endpoint_name: str,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call Azure ML endpoint for inference"""

        try:
            # Get endpoint
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)

            # Make inference request
            response = endpoint.invoke(request_data)

            return response

        except Exception as e:
            logger.error(f"Failed to call ML endpoint: {e}")
            raise