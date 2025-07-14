import pytest
from unittest.mock import MagicMock
from src.gnn.data_preparation import MaintIEGNNDataProcessor

class DummyTransformer:
    entities = {}
    relations = []

def test_gnn_data_processor_initialization():
    processor = MaintIEGNNDataProcessor(DummyTransformer())
    assert processor is not None
    assert hasattr(processor, 'entity_to_idx')
    assert hasattr(processor, 'relation_to_idx')


def test_prepare_gnn_data_empty():
    processor = MaintIEGNNDataProcessor(DummyTransformer())
    # Patch _create_torch_geometric_data to avoid torch_geometric dependency
    processor._create_torch_geometric_data = MagicMock(return_value=None)
    data = processor.prepare_gnn_data(use_cache=False)
    assert 'entity_to_idx' in data
    assert 'relation_to_idx' in data