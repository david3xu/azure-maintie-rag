"""
Complete GNN Workflow Testing Service
Enterprise testing for Azure ML GNN training pipeline
"""
import pytest
import asyncio
from pathlib import Path
import tempfile
import json

class TestGNNWorkflowComplete:
    """Enterprise GNN workflow testing service"""

    @pytest.mark.asyncio
    async def test_complete_gnn_workflow_with_raw_data(self):
        """Test complete GNN workflow from raw data to trained model"""
        # Test data preparation
        raw_data_path = "data/raw"  # Use existing raw data
        domain = "test_domain"

        # 1. Test Azure services initialization
        from integrations.azure_services import AzureServicesManager
        azure_services = AzureServicesManager()
        assert azure_services.validate_configuration()["all_configured"]

        # 2. Test data migration to Azure services
        migration_result = await azure_services.migrate_data_to_azure(raw_data_path, domain)
        assert migration_result["success"]

        # 3. Test graph export service
        cosmos_client = azure_services.get_service('cosmos')
        graph_export = cosmos_client.export_graph_for_training(domain)
        assert graph_export["success"]
        assert graph_export["quality_metrics"]["sufficient_for_training"]

        # 4. Test GNN training orchestration
        from core.azure_ml.gnn_orchestrator import AzureGNNTrainingOrchestrator
        gnn_orchestrator = AzureGNNTrainingOrchestrator(
            azure_services.get_service('ml'),
            cosmos_client
        )
        training_result = await gnn_orchestrator.orchestrate_incremental_training(domain)
        assert training_result["status"] == "completed"

        # 5. Test model quality assessment
        assert "model_uri" in training_result
        # Optionally, load and assess model quality here

    @pytest.mark.asyncio
    async def test_gnn_model_quality_assessment(self):
        """Test GNN model quality assessment service"""
        from core.azure_ml.gnn.model_quality_assessor import GNNModelQualityAssessor
        from core.azure_ml.gnn.trainer import UniversalGNNTrainer, UniversalGNNConfig
        # Mock data loader for testing
        test_data_loader = self._create_test_data_loader()
        # Create and train test model
        config = UniversalGNNConfig(hidden_dim=32, num_layers=2)
        trainer = UniversalGNNTrainer(config)
        trainer.setup_model(num_node_features=5, num_classes=3)
        # Train for few epochs
        trainer.train(test_data_loader, num_epochs=5)
        # Test quality assessment
        quality_assessor = GNNModelQualityAssessor()
        quality_metrics = quality_assessor.assess_model_quality(
            trainer.model, test_data_loader, "test_domain"
        )
        # Validate quality metrics structure
        required_metrics = [
            "accuracy", "precision", "recall", "f1_score",
            "embedding_quality", "connectivity_understanding",
            "overall_quality_score", "quality_recommendations"
        ]
        for metric in required_metrics:
            assert metric in quality_metrics
        assert 0.0 <= quality_metrics["overall_quality_score"] <= 1.0
        assert isinstance(quality_metrics["quality_recommendations"], list)

    def test_graph_data_quality_validation(self):
        """Test graph data quality validation service"""
        from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
        # Test with insufficient data
        insufficient_entities = [{"id": "e1", "text": "entity1"}]
        insufficient_relations = []
        cosmos_client = AzureCosmosGremlinClient()
        quality_result = cosmos_client._validate_graph_quality(insufficient_entities, insufficient_relations)
        assert not quality_result["sufficient_for_training"]
        assert quality_result["entity_count"] == 1
        assert quality_result["relation_count"] == 0
        # Test with sufficient data
        sufficient_entities = [{"id": f"e{i}", "text": f"entity{i}"} for i in range(15)]
        sufficient_relations = [
            {"source_entity": f"entity{i}", "target_entity": f"entity{i+1}", "relation_type": "related"}
            for i in range(10)
        ]
        quality_result = cosmos_client._validate_graph_quality(sufficient_entities, sufficient_relations)
        assert quality_result["sufficient_for_training"]
        assert quality_result["quality_score"] > 0.0

    def _create_test_data_loader(self):
        import torch
        from torch_geometric.data import Data, DataLoader
        # Create synthetic graph data
        num_nodes = 20
        num_edges = 30
        num_features = 5
        num_classes = 3
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        y = torch.randint(0, num_classes, (num_nodes,))
        data = Data(x=x, edge_index=edge_index, y=y)
        return DataLoader([data], batch_size=1)