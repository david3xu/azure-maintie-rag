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
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnterpriseGNNPipelineOrchestrator:
    """Complete GNN pipeline orchestration"""

    def __init__(self):
        self.settings = get_settings()
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
                workspace_name=self.settings.ml_workspace_name
            )
            logger.info(f"Initialized ML client for workspace: {self.settings.ml_workspace_name}")
            return ml_client
        except Exception as e:
            logger.error(f"Failed to initialize ML client: {e}")
            raise

    def _initialize_cosmos_client(self) -> EnterpriseGremlinGraphManager:
        """Initialize Cosmos DB Gremlin client"""
        try:
            cosmos_client = EnterpriseGremlinGraphManager(
                cosmos_endpoint=self.settings.cosmos_db_endpoint,
                cosmos_key=self.settings.cosmos_db_key,
                database_name=self.settings.cosmos_db_database_name
            )
            logger.info(f"Initialized Cosmos DB client for database: {self.settings.cosmos_db_database_name}")
            return cosmos_client
        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB client: {e}")
            raise

    async def run_complete_pipeline(
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
            pipeline_results["steps"]["change_metrics"] = await self._check_graph_changes(domain)

            # Step 2: Orchestrate incremental training (if triggered)
            if trigger_training:
                pipeline_results["steps"]["training"] = await self._orchestrate_training(domain)

                # Step 3: Deploy model (if training completed successfully)
                if deploy_model and pipeline_results["steps"]["training"]["status"] == "completed":
                    pipeline_results["steps"]["deployment"] = await self._deploy_model(
                        model_uri=pipeline_results["steps"]["training"]["model_uri"],
                        domain=domain
                    )

            # Step 4: Generate embeddings for new entities
            pipeline_results["steps"]["embeddings"] = await self._generate_embeddings(domain)

            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["status"] = "completed"

            logger.info(f"GNN pipeline completed successfully for domain: {domain}")
            return pipeline_results

        except Exception as e:
            logger.error(f"GNN pipeline failed for domain {domain}: {e}")
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)
            return pipeline_results

    async def _check_graph_changes(self, domain: str) -> Dict[str, Any]:
        """Check graph change metrics"""
        try:
            change_metrics = await self.cosmos_client.get_graph_change_metrics(domain)
            logger.info(f"Graph change metrics for {domain}: {change_metrics}")
            return change_metrics
        except Exception as e:
            logger.error(f"Failed to check graph changes: {e}")
            raise

    async def _orchestrate_training(self, domain: str) -> Dict[str, Any]:
        """Orchestrate GNN training"""
        try:
            training_results = await self.training_orchestrator.orchestrate_incremental_training(
                domain=domain,
                trigger_threshold=self.settings.gnn_training_trigger_threshold
            )
            logger.info(f"Training orchestration results: {training_results}")
            return training_results
        except Exception as e:
            logger.error(f"Failed to orchestrate training: {e}")
            raise

    async def _deploy_model(self, model_uri: str, domain: str) -> Dict[str, Any]:
        """Deploy trained model"""
        try:
            deployment_results = await self.model_service.deploy_trained_gnn_model(
                model_uri=model_uri,
                domain=domain,
                deployment_tier=self.settings.gnn_model_deployment_tier
            )
            logger.info(f"Model deployment results: {deployment_results}")
            return deployment_results
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise

    async def _generate_embeddings(self, domain: str) -> Dict[str, Any]:
        """Generate embeddings for new entities"""
        try:
            # Get new entities without embeddings
            new_entities = await self._get_entities_without_embeddings(domain)

            if new_entities:
                # Generate embeddings using deployed model (if available)
                embeddings = await self._generate_embeddings_for_entities(new_entities, domain)

                # Update entities with embeddings
                update_stats = await self._update_entity_embeddings(new_entities, embeddings, domain)

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

    async def _get_entities_without_embeddings(self, domain: str) -> List[str]:
        """Get entities without GNN embeddings"""
        try:
            # This would query Cosmos DB for entities without embeddings
            # Simplified implementation
            return []
        except Exception as e:
            logger.error(f"Failed to get entities without embeddings: {e}")
            return []

    async def _generate_embeddings_for_entities(
        self,
        entities: List[str],
        domain: str
    ) -> Dict[str, Any]:
        """Generate embeddings for entities using deployed model"""
        try:
            # This would call the deployed model endpoint
            # Simplified implementation
            return {}
        except Exception as e:
            logger.error(f"Failed to generate embeddings for entities: {e}")
            return {}

    async def _update_entity_embeddings(
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
                    await self.cosmos_client._execute_gremlin_query(update_query)
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
    results = await orchestrator.run_complete_pipeline(
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