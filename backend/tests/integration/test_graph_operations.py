#!/usr/bin/env python3
"""Test graph operations specifically"""

from src.pipeline.rag_structured import MaintIEStructuredRAG

def test_graph_initialization():
    """Test graph component initialization step by step"""
    print("Testing graph initialization...")

    rag = MaintIEStructuredRAG()

    # Check data transformer
    print(f"Data transformer: {rag.data_transformer is not None}")
    if rag.data_transformer:
        if hasattr(rag.data_transformer, 'check_knowledge_graph_status'):
            rag.data_transformer.check_knowledge_graph_status()
        else:
            print(f"Knowledge graph: {getattr(rag.data_transformer, 'knowledge_graph', None) is not None}")
            if getattr(rag.data_transformer, 'knowledge_graph', None):
                kg = rag.data_transformer.knowledge_graph
                print(f"Graph stats: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")

    # Check entity index
    print(f"Entity index: {rag.entity_index is not None}")
    if rag.entity_index:
        if hasattr(rag.entity_index, 'check_index_status'):
            rag.entity_index.check_index_status()
        else:
            print(f"Index built: {getattr(rag.entity_index, 'index_built', None)}")

    # Check graph operations
    print(f"Graph operations enabled: {getattr(rag, 'graph_operations_enabled', None)}")
    print(f"Graph ranker: {getattr(rag, 'graph_ranker', None) is not None}")

if __name__ == "__main__":
    test_graph_initialization()