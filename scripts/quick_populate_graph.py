#!/usr/bin/env python3
"""
Quick Graph Population
======================

This script directly populates the graph database with entities and relationships
from the Azure AI documents to get tri-modal search working immediately.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.knowledge_extraction.agent import run_knowledge_extraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def populate_graph_with_real_data():
    """Populate graph database with real Azure AI documentation entities"""
    logger.info("🔄 QUICK GRAPH POPULATION WITH REAL DATA")
    logger.info("=" * 50)
    
    # Get data files
    data_dir = Path('/workspace/azure-maintie-rag/data/raw/azure-ai-services-language-service_output')
    azure_files = list(data_dir.glob('*.md'))
    
    if not azure_files:
        logger.error("❌ No Azure AI data files found")
        return False
        
    logger.info(f"📄 Found {len(azure_files)} Azure AI documentation files")
    
    # Process files to populate graph
    total_entities = 0
    total_relationships = 0
    
    # Process 3 files for demonstration (enough to show graph search working)
    for i, file_path in enumerate(azure_files[:3]):
        logger.info(f"\n📊 Processing file {i+1}/3: {file_path.name}")
        
        try:
            # Load content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Use meaningful chunk
            if len(content) > 1500:
                content = content[:1500]  # Use manageable chunk
            elif len(content) < 50:
                logger.info(f"   ⚠️  File too small ({len(content)} chars), skipping")
                continue
                
            logger.info(f"   📏 Content length: {len(content)} chars")
            
            # Run knowledge extraction with graph storage
            logger.info("   🔬 Running knowledge extraction...")
            result = await run_knowledge_extraction(
                content=content,
                use_domain_analysis=True
            )
            
            logger.info(f"   ✅ Extraction completed:")
            logger.info(f"      Entities: {len(result.entities)}")
            logger.info(f"      Relationships: {len(result.relationships)}")
            logger.info(f"      Confidence: {result.extraction_confidence:.3f}")
            
            total_entities += len(result.entities)
            total_relationships += len(result.relationships)
            
            # Show sample results
            if result.entities:
                logger.info(f"      Sample entities:")
                for j, entity in enumerate(result.entities[:2]):
                    logger.info(f"        {j+1}. \"{entity.text}\" (type: {entity.type})")
                    
            if result.relationships:
                logger.info(f"      Sample relationships:")
                for j, rel in enumerate(result.relationships[:1]):
                    logger.info(f"        {j+1}. \"{rel.source}\" -> \"{rel.target}\" ({rel.relation})")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to process {file_path.name}: {e}")
            continue
            
    logger.info(f"\n📊 GRAPH POPULATION SUMMARY:")
    logger.info(f"   Files processed: 3")
    logger.info(f"   Total entities stored: {total_entities}")
    logger.info(f"   Total relationships stored: {total_relationships}")
    
    if total_entities > 0:
        logger.info(f"\n🔍 Verifying graph population...")
        
        # Verify the graph has data
        from agents.core.universal_deps import get_universal_deps
        deps = await get_universal_deps()
        cosmos_client = deps.cosmos_client
        
        # Check counts
        azure_count = await cosmos_client.count_vertices('azure-ai')
        general_count = await cosmos_client.count_vertices('general')
        
        logger.info(f"   Azure AI domain vertices: {azure_count}")
        logger.info(f"   General domain vertices: {general_count}")
        
        total_in_graph = azure_count + general_count
        if total_in_graph > 0:
            logger.info(f"\n🎉 SUCCESS: Graph database now contains {total_in_graph} entities!")
            logger.info("💡 Graph search modality is now operational with real data")
            return True
        else:
            logger.warning("⚠️  Graph still appears empty - domain mapping may need adjustment")
            return False
    else:
        logger.error("❌ No entities were extracted - graph remains empty")
        return False

async def main():
    """Main execution"""
    logger.info("🎯 QUICK GRAPH POPULATION FOR TRI-MODAL SEARCH")
    logger.info("=" * 50)
    
    result = await populate_graph_with_real_data()
    
    if result:
        logger.info("\n🎉 GRAPH POPULATION SUCCESSFUL!")
        logger.info("💡 Graph search modality now has real data")
        logger.info("🔍 Ready to test complete tri-modal search")
        return 0
    else:
        logger.error("\n❌ Graph population failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)