#!/usr/bin/env python3
"""
Stage 03: Cosmos DB Storage - Store extracted knowledge to Azure Cosmos DB Gremlin
This is the MISSING STEP that stores extracted entities and relationships to live Cosmos DB
"""
import sys
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient

logger = logging.getLogger(__name__)

class CosmosStorageStage:
    """Stage 03: Store extracted knowledge to Azure Cosmos DB"""
    
    def __init__(self):
        self.cosmos_client = None
        
    def _initialize_cosmos_client(self, domain: str) -> bool:
        """Initialize Cosmos DB client with direct configuration"""
        try:
            # Use direct configuration - same as our standalone scripts
            cosmos_config = {
                'endpoint': 'https://cosmos-maintie-rag-staging-oeeopj3ksgnlo.documents.azure.com:443/',
                'database': 'maintie-rag-staging',
                'container': 'knowledge-graph-staging'
            }
            
            self.cosmos_client = AzureCosmosGremlinClient(cosmos_config)
            
            # Test connection
            health = self.cosmos_client.health_check()
            if not health.get('healthy', False):
                logger.error(f"Cosmos DB health check failed: {health}")
                return False
                
            logger.info("✅ Cosmos DB client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cosmos DB: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def store_entities_to_cosmos(self, entities: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        """Store extracted entities to Cosmos DB"""
        print(f"📦 Storing {len(entities)} entities to Cosmos DB...")
        
        stored_count = 0
        failed_count = 0
        entity_map = {}  # Track entity ID mapping for relationships
        
        for i, entity in enumerate(entities):
            try:
                # Prepare entity data for Cosmos DB
                entity_data = {
                    'text': entity.get('text', ''),
                    'entity_type': entity.get('type', 'unknown'),
                    'context': entity.get('context', ''),
                    'confidence': 1.0,
                    'source': 'knowledge_extraction',
                    'created_at': datetime.now().isoformat()
                }
                
                # Store entity
                result = self.cosmos_client.add_entity(entity_data, domain)
                
                if result.get('success', False):
                    stored_count += 1
                    # Map original entity_id to Cosmos vertex ID
                    original_id = entity.get('entity_id', f'entity_{i}')
                    cosmos_id = result.get('vertex_id')
                    if cosmos_id:
                        entity_map[original_id] = cosmos_id
                else:
                    failed_count += 1
                    logger.warning(f"Failed to store entity: {entity.get('text', 'N/A')}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error storing entity {i}: {e}")
        
        print(f"   ✅ Stored: {stored_count} entities")
        print(f"   ❌ Failed: {failed_count} entities")
        
        return {
            'entities_stored': stored_count,
            'entities_failed': failed_count,
            'entity_id_mapping': entity_map
        }
    
    def store_relationships_to_cosmos(self, relationships: List[Dict[str, Any]], 
                                     entity_map: Dict[str, str], domain: str) -> Dict[str, Any]:
        """Store extracted relationships to Cosmos DB"""
        print(f"🔗 Storing {len(relationships)} relationships to Cosmos DB...")
        
        stored_count = 0
        failed_count = 0
        
        for relationship in relationships:
            try:
                source_text = relationship.get('source', '')
                target_text = relationship.get('target', '')
                relation_type = relationship.get('relation', 'related')
                
                # Try to find source and target entities in Cosmos DB
                # Since we might not have perfect ID mapping, search by text
                source_entities = self.cosmos_client.find_entities_by_type('*', domain, limit=1000)
                target_entities = source_entities  # Same pool
                
                source_id = None
                target_id = None
                
                # Find source entity
                for entity in source_entities:
                    if entity.get('text', '').strip().lower() == source_text.strip().lower():
                        source_id = entity.get('id')
                        break
                
                # Find target entity  
                for entity in target_entities:
                    if entity.get('text', '').strip().lower() == target_text.strip().lower():
                        target_id = entity.get('id')
                        break
                
                if source_id and target_id:
                    # Prepare relationship data
                    relation_data = {
                        'source_entity': source_id,
                        'target_entity': target_id,
                        'relation_type': relation_type,
                        'context': relationship.get('context', ''),
                        'confidence': 1.0,
                        'source': 'knowledge_extraction',
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # Store relationship
                    result = self.cosmos_client.add_relationship(relation_data, domain)
                    
                    if result.get('success', False):
                        stored_count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to store relationship: {source_text} -> {target_text}")
                else:
                    failed_count += 1
                    logger.warning(f"Could not find entities for relationship: {source_text} -> {target_text}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error storing relationship: {e}")
        
        print(f"   ✅ Stored: {stored_count} relationships")
        print(f"   ❌ Failed: {failed_count} relationships")
        
        return {
            'relationships_stored': stored_count,
            'relationships_failed': failed_count
        }
    
    async def execute_cosmos_storage(self, extraction_file: str, domain: str = "maintenance") -> Dict[str, Any]:
        """
        Execute Cosmos DB storage stage
        
        Args:
            extraction_file: Path to knowledge extraction JSON file
            domain: Target domain
            
        Returns:
            Dict with storage results
        """
        print("🚀 Stage 03: Cosmos DB Storage")
        print("=" * 50)
        print(f"📁 Input: {extraction_file}")
        print(f"🏷️  Domain: {domain}")
        
        start_time = datetime.now()
        
        # Initialize Cosmos client
        if not self._initialize_cosmos_client(domain):
            return {
                'success': False,
                'error': 'Failed to initialize Cosmos DB client'
            }
        
        try:
            # Load extraction results
            if not Path(extraction_file).exists():
                return {
                    'success': False,
                    'error': f'Extraction file not found: {extraction_file}'
                }
            
            with open(extraction_file, 'r') as f:
                extraction_data = json.load(f)
            
            knowledge_data = extraction_data.get('knowledge_data', {})
            entities = knowledge_data.get('entities', [])
            relationships = knowledge_data.get('relationships', [])
            
            print(f"📊 Found {len(entities)} entities and {len(relationships)} relationships to store")
            
            # Store entities first
            entity_results = self.store_entities_to_cosmos(entities, domain)
            
            # Store relationships
            relationship_results = self.store_relationships_to_cosmos(
                relationships, 
                entity_results.get('entity_id_mapping', {}), 
                domain
            )
            
            # Get final graph statistics
            stats = self.cosmos_client.get_graph_statistics(domain)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                'stage': '03_cosmos_storage',
                'success': True,
                'domain': domain,
                'input_file': extraction_file,
                'execution_time': duration,
                'entities_processed': len(entities),
                'relationships_processed': len(relationships),
                'entities_stored': entity_results.get('entities_stored', 0),
                'entities_failed': entity_results.get('entities_failed', 0),
                'relationships_stored': relationship_results.get('relationships_stored', 0),
                'relationships_failed': relationship_results.get('relationships_failed', 0),
                'final_graph_stats': stats,
                'timestamp': end_time.isoformat()
            }
            
            print(f"\n✅ Cosmos DB storage completed in {duration:.2f}s")
            print(f"📊 Final graph: {stats.get('vertex_count', 0)} vertices, {stats.get('edge_count', 0)} edges")
            
            return results
            
        except Exception as e:
            logger.error(f"Cosmos storage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': '03_cosmos_storage'
            }
        
        finally:
            # Clean up
            if self.cosmos_client:
                try:
                    self.cosmos_client.close()
                except Exception:
                    pass

async def main():
    """Main execution for standalone testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Store extracted knowledge to Cosmos DB')
    parser.add_argument('--extraction-file', required=True, help='Path to knowledge extraction JSON file')
    parser.add_argument('--domain', default='maintenance', help='Target domain')
    parser.add_argument('--output-dir', default='data/outputs/step03', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute storage
    storage_stage = CosmosStorageStage()
    results = await storage_stage.execute_cosmos_storage(args.extraction_file, args.domain)
    
    # Save results
    output_file = output_dir / f"cosmos_storage_{args.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    if results.get('success', False):
        print("🎉 Cosmos DB storage completed successfully!")
        return 0
    else:
        print(f"💥 Cosmos DB storage failed: {results.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)