"""
Script to prepare GNN data from existing MaintIE data
Run this after weeks 1-8 implementation is complete
"""

import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge.data_transformer import MaintIEDataTransformer
from src.gnn.data_preparation import MaintIEGNNDataProcessor
from config.settings import settings

def main():
    """Prepare GNN data from MaintIE annotations"""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting GNN data preparation...")

    # Initialize data transformer (uses your existing implementation)
    logger.info("Loading MaintIE data...")
    data_transformer = MaintIEDataTransformer()

    # Extract knowledge if not already done
    if not hasattr(data_transformer, 'entities') or not data_transformer.entities:
        logger.info("Extracting MaintIE knowledge...")
        data_transformer.extract_maintenance_knowledge()

    # Initialize GNN data processor
    logger.info("Initializing GNN data processor...")
    gnn_processor = MaintIEGNNDataProcessor(data_transformer)

    # Prepare GNN dataset
    logger.info("Preparing GNN dataset...")
    gnn_dataset = gnn_processor.prepare_gnn_data(use_cache=False)

    # Print statistics
    stats = gnn_dataset['stats']
    logger.info("GNN Dataset Statistics:")
    logger.info(f"  Entities: {stats['num_entities']}")
    logger.info(f"  Relation Types: {stats['num_relations']}")
    logger.info(f"  Edges: {stats['num_edges']}")
    logger.info(f"  Feature Dimension: {stats['feature_dim']}")
    logger.info(f"  Entity Types: {stats['entity_types']}")

    logger.info("GNN data preparation complete!")

    if not gnn_dataset['full_data']:
        logger.warning("PyTorch Geometric not available. Install with: pip install torch-geometric")

    return gnn_dataset

if __name__ == "__main__":
    main()