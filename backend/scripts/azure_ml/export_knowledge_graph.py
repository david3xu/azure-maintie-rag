#!/usr/bin/env python3
"""
Export complete knowledge graph from Cosmos DB to local file for Azure ML
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class StandaloneCosmosExporter:
    """Export knowledge graph from Cosmos DB"""
    
    def __init__(self, config: Dict[str, str]):
        self.endpoint = config['endpoint']
        self.database = config['database'] 
        self.container = config['container']
        self.gremlin_client = None
        self._initialized = False
        
    def _initialize_client(self):
        """Initialize Gremlin client"""
        try:
            from gremlin_python.driver import client, serializer
            from azure.identity import DefaultAzureCredential
            
            print(f"🔧 Connecting to Cosmos DB for export...")
            
            # Extract account name and create endpoint
            account_name = self.endpoint.replace('https://', '').replace('.documents.azure.com:443/', '')
            gremlin_endpoint = f"wss://{account_name}.gremlin.cosmosdb.azure.com:443/"
            
            # Get credentials
            credential = DefaultAzureCredential()
            token = credential.get_token("https://cosmos.azure.com/.default")
            
            # Create client
            self.gremlin_client = client.Client(
                gremlin_endpoint,
                'g',
                username=f"/dbs/{self.database}/colls/{self.container}",
                password=token.token,
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            
            self._initialized = True
            print("✅ Cosmos DB client initialized for export")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Cosmos client: {e}")
            return False
    
    def _execute_query(self, query: str):
        """Execute Gremlin query"""
        try:
            if not self._initialized:
                if not self._initialize_client():
                    return []
            
            result = self.gremlin_client.submit(query)
            return result.all().result(timeout=60)  # Longer timeout for large exports
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []
    
    def export_all_entities(self) -> List[Dict[str, Any]]:
        """Export all entities from Cosmos DB"""
        print("📥 Exporting all entities from Cosmos DB...")
        
        # Get all vertices with comprehensive data
        query = """
        g.V().project('id', 'label', 'text', 'entity_type', 'domain', 'created_at')
             .by(id())
             .by(label())
             .by(values('text').fold())
             .by(values('entity_type').fold())
             .by(values('domain').fold())
             .by(values('created_at').fold())
        """
        
        vertices = self._execute_query(query)
        
        entities = []
        for vertex in vertices:
            # Handle property lists (Cosmos DB returns lists for properties)
            text = vertex.get('text', [''])
            text = text[0] if isinstance(text, list) and text else ''
            
            entity_type = vertex.get('entity_type', ['unknown'])
            entity_type = entity_type[0] if isinstance(entity_type, list) and entity_type else 'unknown'
            
            domain = vertex.get('domain', ['maintenance'])
            domain = domain[0] if isinstance(domain, list) and domain else 'maintenance'
            
            created_at = vertex.get('created_at', [''])
            created_at = created_at[0] if isinstance(created_at, list) and created_at else ''
            
            entities.append({
                'id': vertex.get('id', ''),
                'label': vertex.get('label', ''),
                'text': text,
                'entity_type': entity_type,
                'domain': domain,
                'created_at': created_at
            })
        
        print(f"✅ Exported {len(entities)} entities")
        return entities
    
    def export_all_relationships(self) -> List[Dict[str, Any]]:
        """Export all relationships from Cosmos DB"""
        print("🔗 Exporting all relationships from Cosmos DB...")
        
        # Get all edges with comprehensive data
        query = """
        g.E().project('id', 'label', 'source', 'target', 'relation_type')
             .by(id())
             .by(label())
             .by(outV().id())
             .by(inV().id())
             .by(values('relation_type').fold())
        """
        
        edges = self._execute_query(query)
        
        relationships = []
        for edge in edges:
            relation_type = edge.get('relation_type', [edge.get('label', 'related')])
            relation_type = relation_type[0] if isinstance(relation_type, list) and relation_type else edge.get('label', 'related')
            
            relationships.append({
                'id': edge.get('id', ''),
                'label': edge.get('label', ''),
                'source': edge.get('source', ''),
                'target': edge.get('target', ''),
                'relation_type': relation_type
            })
        
        print(f"✅ Exported {len(relationships)} relationships")
        return relationships

def export_knowledge_graph():
    """Export complete knowledge graph to local file"""
    print("🚀 EXPORTING KNOWLEDGE GRAPH FOR AZURE ML")
    print("=" * 60)
    
    # Initialize exporter
    cosmos_config = {
        'endpoint': 'https://cosmos-maintie-rag-staging-oeeopj3ksgnlo.documents.azure.com:443/',
        'database': 'maintie-rag-staging',
        'container': 'knowledge-graph-staging'
    }
    
    exporter = StandaloneCosmosExporter(cosmos_config)
    
    try:
        # Export entities
        entities = exporter.export_all_entities()
        
        # Export relationships  
        relationships = exporter.export_all_relationships()
        
        # Create comprehensive knowledge graph export
        knowledge_graph = {
            'export_info': {
                'exported_at': datetime.now().isoformat(),
                'source': 'Azure Cosmos DB',
                'database': cosmos_config['database'],
                'container': cosmos_config['container'],
                'total_entities': len(entities),
                'total_relationships': len(relationships)
            },
            'entities': entities,
            'relationships': relationships,
            'statistics': {
                'entity_types': {},
                'relationship_types': {},
                'domains': {}
            }
        }
        
        # Calculate statistics
        for entity in entities:
            entity_type = entity.get('entity_type', 'unknown')
            domain = entity.get('domain', 'unknown')
            
            knowledge_graph['statistics']['entity_types'][entity_type] = \
                knowledge_graph['statistics']['entity_types'].get(entity_type, 0) + 1
            knowledge_graph['statistics']['domains'][domain] = \
                knowledge_graph['statistics']['domains'].get(domain, 0) + 1
        
        for relationship in relationships:
            rel_type = relationship.get('relation_type', 'unknown')
            knowledge_graph['statistics']['relationship_types'][rel_type] = \
                knowledge_graph['statistics']['relationship_types'].get(rel_type, 0) + 1
        
        # Save to file
        output_file = "/workspace/azure-maintie-rag/backend/scripts/azure_ml/knowledge_graph_export.json"
        
        with open(output_file, 'w') as f:
            json.dump(knowledge_graph, f, indent=2)
        
        print(f"\n📁 Knowledge graph exported to: {output_file}")
        print(f"📊 Export summary:")
        print(f"   Entities: {len(entities)}")
        print(f"   Relationships: {len(relationships)}")
        print(f"   Entity types: {len(knowledge_graph['statistics']['entity_types'])}")
        print(f"   Relationship types: {len(knowledge_graph['statistics']['relationship_types'])}")
        
        # Show sample data
        print(f"\n📝 Sample entities:")
        for i, entity in enumerate(entities[:5]):
            print(f"   {i+1}. [{entity['entity_type']}] {entity['text']}")
        
        print(f"\n🔗 Sample relationships:")
        for i, rel in enumerate(relationships[:5]):
            print(f"   {i+1}. {rel['source']} --[{rel['relation_type']}]--> {rel['target']}")
        
        print(f"\n🎉 EXPORT COMPLETE!")
        print(f"💡 Azure ML can now use this local file instead of accessing Cosmos DB")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    output_file = export_knowledge_graph()
    
    if output_file:
        print(f"\n🏆 SUCCESS: Knowledge graph exported!")
        print(f"📂 File: {output_file}")
        print(f"🔧 Next: Update Azure ML training script to use this file")
    else:
        print(f"\n💥 FAILED: Could not export knowledge graph")