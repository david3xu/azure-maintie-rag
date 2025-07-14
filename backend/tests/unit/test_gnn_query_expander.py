import pytest
from unittest.mock import MagicMock
from src.gnn.gnn_query_expander import GNNQueryExpander

class DummyTransformer:
    entities = {'e1': MagicMock(text='pump')}
    relations = []
    knowledge_graph = None

def test_gnn_query_expander_init():
    expander = GNNQueryExpander(DummyTransformer(), model_path=None)
    assert expander is not None
    assert hasattr(expander, 'expand_query_entities')


def test_gnn_query_expander_fallback():
    expander = GNNQueryExpander(DummyTransformer(), model_path=None)
    # Force fallback
    expander.enabled = False
    result = expander.expand_query_entities(['pump'])
    assert 'pump' in result