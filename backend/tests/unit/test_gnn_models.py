import pytest
import torch
from src.gnn.gnn_models import MaintenanceGNNModel

def test_gnn_model_initialization():
    config = {
        'input_dim': 8,
        'hidden_dim': 16,
        'output_dim': 4,
        'num_layers': 2,
        'num_entity_types': 3,
        'gnn_type': 'GraphSAGE',
        'dropout': 0.1
    }
    model = MaintenanceGNNModel(config)
    assert model is not None
    assert model.input_dim == 8
    assert model.hidden_dim == 16
    assert model.num_layers == 2


def test_gnn_model_forward():
    config = {
        'input_dim': 8,
        'hidden_dim': 16,
        'output_dim': 4,
        'num_layers': 2,
        'num_entity_types': 3,
        'gnn_type': 'GraphSAGE',
        'dropout': 0.1
    }
    model = MaintenanceGNNModel(config)
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    outputs = model(x, edge_index)
    assert 'node_embeddings' in outputs
    assert outputs['entity_logits'].shape[0] == 5