#!/usr/bin/env python3
"""
Enterprise GNN Pipeline Orchestration Script
Orchestrates the complete Entity/Relation Graph → Cosmos DB → GNN Training → Model Deployment pipeline
"""

import asyncio
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from core.azure_cosmos.enhanced_gremlin_client import EnterpriseGremlinGraphManager
from core.azure_ml.gnn_orchestrator import AzureGNNTrainingOrchestrator, AzureGNNModelService
from config.settings import azure_settings
from config.azure_config_validator import AzureConfigValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnterpriseGNNPipelineOrchestrator:
    """Complete GNN pipeline orchestration"""

    def __init__(self):
        self.settings = azure_settings
        # Azure configuration validation
        validator = AzureConfigValidator()
        cosmos_validation = validator.validate_cosmos_db_configuration()
        if not cosmos_validation["valid"]:
            for error in cosmos_validation["errors"]:
                logger.error(f"Azure Cosmos DB configuration error: {error}")
            raise ValueError(f"Invalid Azure Cosmos DB configuration: {cosmos_validation['errors']}")
        for warning in cosmos_validation["warnings"]:
            logger.warning(f"Azure Cosmos DB configuration warning: {warning}")
        ml_validation = validator.validate_azure_ml_configuration()
        if not ml_validation["valid"]:
            for error in ml_validation["errors"]:
                logger.error(f"Azure ML configuration error: {error}")
            raise ValueError(f"Invalid Azure ML configuration: {ml_validation['errors']}")
        logger.info("✅ Azure configuration validation passed")
        self.ml_client = self._initialize_ml_client()
        self.cosmos_client = self._initialize_cosmos_client()
        self.training_orchestrator = AzureGNNTrainingOrchestrator(
            ml_client=self.ml_client,
            cosmos_client=self.cosmos_client
        )
        self.model_service = AzureGNNModelService(ml_client=self.ml_client)

    def _initialize_ml_client(self) -> MLClient:
        """Initialize Azure ML client"""
        try:
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=self.settings.azure_subscription_id,
                resource_group_name=self.settings.azure_resource_group,
                workspace_name=self.settings.azure_ml_workspace_name
            )
            logger.info(f"Initialized ML client for workspace: {self.settings.azure_ml_workspace_name}")
            return ml_client
        except Exception as e:
            logger.error(f"Failed to initialize ML client: {e}")
            raise

    def _initialize_cosmos_client(self) -> EnterpriseGremlinGraphManager:
        """Initialize Cosmos DB Gremlin client"""
        try:
            cosmos_client = EnterpriseGremlinGraphManager(
                cosmos_endpoint=self.settings.azure_cosmos_endpoint,
                cosmos_key=self.settings.azure_cosmos_key,
                database_name=self.settings.azure_cosmos_database,
                container_name=self.settings.azure_cosmos_container
            )
            logger.info(f"Initialized Cosmos DB client for database: {self.settings.azure_cosmos_database}")
            return cosmos_client
        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB client: {e}")
            raise

    def run_complete_pipeline(
        self,
        domain: str,
        trigger_training: bool = True,
        deploy_model: bool = True
    ) -> Dict[str, Any]:
        """Run complete GNN pipeline"""

        pipeline_results = {
            "domain": domain,
            "start_time": datetime.now().isoformat(),
            "steps": {}
        }

        try:
            logger.info(f"Starting GNN pipeline for domain: {domain}")

            # Step 1: Check graph change metrics
            pipeline_results["steps"]["change_metrics"] = self._check_graph_changes(domain)

            # Step 2: Orchestrate incremental training (if triggered)
            if trigger_training:
                pipeline_results["steps"]["training"] = self._orchestrate_training(domain)

                # Step 3: Deploy model (if training completed successfully)
                if deploy_model and pipeline_results["steps"]["training"]["status"] == "completed":
                    pipeline_results["steps"]["deployment"] = self._deploy_model(
                        model_uri=pipeline_results["steps"]["training"]["model_uri"],
                        domain=domain
                    )

            # Step 4: Generate embeddings for new entities
            pipeline_results["steps"]["embeddings"] = self._generate_embeddings(domain)

            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["status"] = "completed"

            logger.info(f"GNN pipeline completed successfully for domain: {domain}")
            return pipeline_results

        except Exception as e:
            logger.error(f"GNN pipeline failed for domain {domain}: {e}")
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)
            return pipeline_results

    def _check_graph_changes(self, domain: str) -> Dict[str, Any]:
        """Check graph change metrics"""
        try:
            change_metrics = self.cosmos_client.get_graph_change_metrics(domain)
            logger.info(f"Graph change metrics for {domain}: {change_metrics}")
            return change_metrics
        except Exception as e:
            logger.error(f"Failed to check graph changes: {e}")
            raise

    def _orchestrate_training(self, domain: str) -> Dict[str, Any]:
        """Orchestrate GNN training"""
        try:
            training_results = self.training_orchestrator.orchestrate_incremental_training(
                domain=domain,
                trigger_threshold=self.settings.gnn_training_trigger_threshold
            )
            logger.info(f"Training orchestration results: {training_results}")
            return training_results
        except Exception as e:
            logger.error(f"Failed to orchestrate training: {e}")
            raise

    def _deploy_model(self, model_uri: str, domain: str) -> Dict[str, Any]:
        """Deploy trained model"""
        try:
            deployment_results = self.model_service.deploy_trained_gnn_model(
                model_uri=model_uri,
                domain=domain,
                deployment_tier=self.settings.gnn_model_deployment_tier
            )
            logger.info(f"Model deployment results: {deployment_results}")
            return deployment_results
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise

    def _generate_embeddings(self, domain: str) -> Dict[str, Any]:
        """Generate embeddings for new entities"""
        try:
            # Get new entities without embeddings
            new_entities = self._get_entities_without_embeddings(domain)

            if new_entities:
                # Generate embeddings using deployed model (if available)
                embeddings = self._generate_embeddings_for_entities(new_entities, domain)

                # Update entities with embeddings
                update_stats = self._update_entity_embeddings(new_entities, embeddings, domain)

                return {
                    "entities_processed": len(new_entities),
                    "embeddings_generated": len(embeddings),
                    "update_stats": update_stats
                }
            else:
                return {
                    "entities_processed": 0,
                    "embeddings_generated": 0,
                    "update_stats": {"entities_updated": 0}
                }

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def _get_entities_without_embeddings(self, domain: str) -> List[str]:
        """Get entities without GNN embeddings from Cosmos DB"""
        try:
            entities_query = f"""
                g.V().has('domain', '{domain}')
                    .not(__.has('gnn_embeddings'))
                    .values('entity_id')
            """
            query_results = self.cosmos_client._execute_gremlin_query(entities_query)
            entity_ids = [result for result in query_results if result]
            logger.info(f"Found {len(entity_ids)} entities without embeddings in domain '{domain}'")
            return entity_ids
        except Exception as e:
            logger.error(f"Failed to get entities without embeddings: {e}")
            return []

    def _generate_embeddings_for_entities(
        self,
        entities: List[str],
        domain: str
    ) -> Dict[str, Any]:
        """Generate embeddings using deployed Azure ML GNN model endpoint"""
        try:
            ml_client = self.ml_client
            if not ml_client:
                raise RuntimeError("Azure ML client not initialized")
            endpoint_name = f"{azure_settings.azure_ml_endpoint_prefix}-{domain}"
            deployment_name = getattr(azure_settings, 'azure_ml_deployment_name', 'default')
            entity_features = []
            for entity_id in entities:
                entity_query = f"""
                    g.V().has('entity_id', '{entity_id}')
                        .project('text', 'entity_type', 'confidence')
                        .by('text')
                        .by('entity_type')
                        .by('confidence')
                """
                entity_result = self.cosmos_client._execute_gremlin_query(entity_query)
                if entity_result:
                    entity_data = entity_result[0]
                    entity_features.append({
                        "entity_id": entity_id,
                        "text": entity_data.get("text", ""),
                        "entity_type": entity_data.get("entity_type", "unknown"),
                        "confidence": entity_data.get("confidence", 1.0)
                    })
            if not entity_features:
                return {}
            inference_request = {
                "domain": domain,
                "entities": entity_features,
                "model_version": getattr(azure_settings, 'gnn_model_version', 'latest')
            }
            inference_result = ml_client.invoke_gnn_endpoint(
                endpoint_name=endpoint_name,
                deployment_name=deployment_name,
                request_data=inference_request
            )
            if inference_result.get("success", False):
                embeddings = inference_result.get("embeddings", {})
                logger.info(f"Generated embeddings for {len(embeddings)} entities")
                return embeddings
            else:
                logger.error(f"Inference failed: {inference_result.get('error', 'Unknown error')}")
                return {}
        except Exception as e:
            logger.error(f"Failed to generate embeddings for entities: {e}")
            return {}

    def _update_entity_embeddings(
        self,
        entities: List[str],
        embeddings: Dict[str, Any],
        domain: str
    ) -> Dict[str, int]:
        """Update entity embeddings in Cosmos DB"""
        updated_count = 0
        try:
            for entity_id in entities:
                embedding = embeddings.get(entity_id)
                if embedding is not None:
                    # Convert embedding to comma-separated string if needed
                    if hasattr(embedding, 'tolist'):
                        embedding_str = ','.join(map(str, embedding.tolist()))
                        embedding_dim = len(embedding)
                    elif isinstance(embedding, (list, tuple)):
                        embedding_str = ','.join(map(str, embedding))
                        embedding_dim = len(embedding)
                    else:
                        embedding_str = str(embedding)
                        embedding_dim = 0
                    update_query = f"""
                        g.V().has('entity_id', '{entity_id}')
                            .property('gnn_embeddings', '{embedding_str}')
                            .property('embedding_dimension', {embedding_dim})
                            .property('embedding_updated_at', '{datetime.now().isoformat()}')
                    """
                    self.cosmos_client._execute_gremlin_query(update_query)
                    updated_count += 1
            logger.info(f"Updated {updated_count} entity embeddings in Cosmos DB for domain '{domain}'")
            return {"entities_updated": updated_count}
        except Exception as e:
            logger.error(f"Failed to update entity embeddings: {e}")
            return {"entities_updated": updated_count}


async def main():
    """Main orchestration function"""
    parser = argparse.ArgumentParser(description="Enterprise GNN Pipeline Orchestration")
    parser.add_argument("--domain", default="general", help="Domain for GNN pipeline")
    parser.add_argument("--trigger-training", action="store_true", help="Trigger GNN training")
    parser.add_argument("--deploy-model", action="store_true", help="Deploy trained model")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Initialize orchestrator
    orchestrator = EnterpriseGNNPipelineOrchestrator()

    # Run pipeline
    results = orchestrator.run_complete_pipeline(
        domain=args.domain,
        trigger_training=args.trigger_training,
        deploy_model=args.deploy_model
    )

    # Print results
    print(f"\n🎯 GNN Pipeline Results for Domain: {args.domain}")
    print(f"Status: {results['status']}")
    print(f"Duration: {results.get('end_time', 'N/A')}")

    if results['status'] == 'completed':
        print("✅ Pipeline completed successfully!")

        # Print step results
        for step_name, step_result in results.get('steps', {}).items():
            print(f"\n📊 {step_name.title()}:")
            for key, value in step_result.items():
                print(f"  {key}: {value}")
    else:
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())