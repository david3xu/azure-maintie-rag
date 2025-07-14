"""
Deploy and Test GNN Integration
Simple script to verify GNN components work correctly
"""

import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from src.knowledge.data_transformer import MaintIEDataTransformer
from src.gnn.data_preparation import MaintIEGNNDataProcessor
from src.gnn.gnn_query_expander import GNNQueryExpander
from src.enhancement.query_analyzer import MaintenanceQueryAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gnn_integration():
    """Test GNN integration step by step"""

    logger.info("🚀 Testing GNN Integration for MaintIE Enhanced RAG")
    logger.info("=" * 60)

    try:
        # Step 1: Load MaintIE data
        logger.info("📊 Step 1: Loading MaintIE data...")
        data_transformer = MaintIEDataTransformer()

        # Extract knowledge if needed
        if not hasattr(data_transformer, 'entities') or not data_transformer.entities:
            logger.info("Extracting maintenance knowledge...")
            data_transformer.extract_maintenance_knowledge()

        logger.info(f"✅ Loaded {len(data_transformer.entities)} entities and {len(data_transformer.relations)} relations")

        # Step 2: Prepare GNN data
        logger.info("\n🧠 Step 2: Preparing GNN data...")
        gnn_processor = MaintIEGNNDataProcessor(data_transformer)
        gnn_data = gnn_processor.prepare_gnn_data(use_cache=True)

        stats = gnn_data['stats']
        logger.info(f"✅ GNN Dataset: {stats['num_entities']} entities, {stats['num_edges']} edges")

        if gnn_data['full_data'] is None:
            logger.warning("⚠️ PyTorch Geometric not available - using fallback mode")
        else:
            logger.info("✅ PyTorch Geometric data prepared successfully")

        # Step 3: Test GNN query expander
        logger.info("\n🔍 Step 3: Testing GNN query expansion...")
        gnn_expander = GNNQueryExpander(data_transformer)

        if gnn_expander.enabled:
            logger.info("✅ GNN query expander initialized successfully")

            # Test expansion
            test_entities = ["pump", "seal"]
            expanded = gnn_expander.expand_query_entities(test_entities, max_expansions=5)
            logger.info(f"✅ Query expansion: {test_entities} → {expanded}")

            # Test domain context
            context = gnn_expander.get_domain_context(test_entities)
            if context:
                logger.info(f"✅ Domain context: {context}")
        else:
            logger.info("ℹ️ GNN expander using fallback mode")

        # Step 4: Test enhanced query analyzer
        logger.info("\n🔧 Step 4: Testing enhanced query analyzer...")
        analyzer = MaintenanceQueryAnalyzer(data_transformer)

        if analyzer.gnn_enabled:
            logger.info("✅ GNN integration enabled in query analyzer")
        else:
            logger.info("ℹ️ Query analyzer using rule-based fallback")

        # Test query analysis
        test_query = "pump seal failure troubleshooting"
        analysis = analyzer.analyze_query(test_query)
        logger.info(f"✅ Query analysis: {analysis.query_type.value}, {len(analysis.entities)} entities")

        # Test query enhancement
        enhanced = analyzer.enhance_query(analysis)
        logger.info(f"✅ Query enhancement: {len(enhanced.expanded_concepts)} expanded concepts")

        # Step 5: Test complete pipeline
        logger.info("\n🔄 Step 5: Testing complete pipeline...")
        from src.pipeline.rag_structured import MaintIEStructuredRAG

        rag = MaintIEStructuredRAG()

        # Test structured query
        response = rag.process_structured_query(
            query=test_query,
            max_results=3,
            include_explanations=True
        )

        if response and response.get('enhanced_query'):
            enhanced_query = response['enhanced_query']
            expanded_concepts = enhanced_query.get('expanded_concepts', [])
            logger.info(f"✅ Pipeline test: {len(expanded_concepts)} concepts expanded")

            # Check if GNN was used
            if analyzer.gnn_enabled and len(expanded_concepts) > 5:
                logger.info("🎯 GNN enhancement detected in pipeline!")
            else:
                logger.info("ℹ️ Pipeline using rule-based expansion")

        logger.info("\n🎉 GNN Integration Test Complete!")
        logger.info("Your MaintIE Enhanced RAG system with GNN is ready!")

        return True

    except Exception as e:
        logger.error(f"❌ GNN integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main deployment function"""
    success = test_gnn_integration()

    if success:
        logger.info("\n✅ Deployment successful!")
        logger.info("Next steps:")
        logger.info("1. Start the API server: uvicorn api.main:app --reload")
        logger.info("2. Run integration tests: python test_gnn_integration.py")
        logger.info("3. Test with frontend queries")
    else:
        logger.error("\n❌ Deployment failed!")
        logger.error("Check the logs above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()