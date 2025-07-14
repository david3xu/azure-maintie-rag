import pytest
import torch
from src.gnn.comprehensive_trainer import ComprehensiveGNNTrainer, create_comprehensive_training_config

class DummyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x, edge_index=None):
        return {'entity_logits': self.linear(x)}

def test_trainer_initialization():
    config = create_comprehensive_training_config()
    trainer = ComprehensiveGNNTrainer(DummyModel, {}, config)
    assert trainer.model_class == DummyModel
    assert trainer.training_config == config
    assert trainer.device.type in ['cpu', 'cuda']

def test_setup_model_and_training():
    config = create_comprehensive_training_config()
    trainer = ComprehensiveGNNTrainer(DummyModel, {}, config)
    trainer.setup_model_and_training({'input_dim': 4, 'hidden_dim': 4, 'output_dim': 2, 'num_layers': 1, 'num_entity_types': 2, 'gnn_type': 'GCN', 'dropout': 0.0, 'use_batch_norm': False, 'use_residual': False})
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.criterion is not None