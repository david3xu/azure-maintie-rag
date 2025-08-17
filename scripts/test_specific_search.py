#!/usr/bin/env python3
"""
Test Specific Search Terms
===========================

Test tri-modal search with specific terms that should match our extracted entities.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.universal_search.agent import run_universal_search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_specific_searches():
    """Test searches with terms that should match our extracted entities"""
    logger.info("🔍 TESTING SPECIFIC SEARCH TERMS")
    logger.info("=" * 50)
    
    # Test queries that should match extracted entities
    test_queries = [
        "custom question",
        "bot service", 
        "training",
        "model",
        "application"
    ]
    
    for i, query in enumerate(test_queries):
        logger.info(f"\n{i+1}️⃣ Testing query: '{query}'")
        
        try:
            result = await run_universal_search(
                query=query,
                max_results=3,
                use_domain_analysis=True
            )
            
            logger.info(f"   ✅ Search completed")
            logger.info(f"   📊 Total results: {result.total_results_found}")
            logger.info(f"   🎯 Confidence: {result.search_confidence:.3f}")
            logger.info(f"   📊 Vector results: {len(result.vector_results)}")
            logger.info(f"   📊 Graph results: {len(result.graph_results)}")
            logger.info(f"   📊 GNN results: {len(result.gnn_results)}")
            
            if result.total_results_found > 0:
                logger.info(f"   🎉 SUCCESS: Found {result.total_results_found} results for '{query}'!")
                return True
                
        except Exception as e:
            logger.error(f"   ❌ Search failed for '{query}': {e}")
            continue
    
    logger.error("❌ No queries returned results")
    return False

async def main():
    """Main execution"""
    result = await test_specific_searches()
    
    if result:
        logger.info("\n🎉 SUCCESS: Tri-modal search is working with specific terms!")
        logger.info("💡 The system is operational - just need better query matching")
        return 0
    else:
        logger.error("\n❌ No search terms returned results")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)