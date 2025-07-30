#!/usr/bin/env python3
"""
Test small-scale storage to verify the fix works
"""
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class SimpleCosmosClient:
    """Simple Cosmos client for testing"""
    
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
            
            print(f"🔧 Connecting to Cosmos DB...")
            
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
            print("✅ Cosmos DB client initialized")
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
            return result.all().result(timeout=30)
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []
    
    def add_vertex(self, vertex_id: str, label: str, properties: Dict[str, Any]):
        """Add vertex to graph with proper Cosmos DB syntax"""
        try:
            # Build proper Gremlin query for Cosmos DB
            query = f"g.addV('{label}').property(id, '{vertex_id}')"
            
            # Add each property correctly
            for key, value in properties.items():
                # Escape single quotes in values
                escaped_value = str(value).replace("'", "\\'").replace('"', '\\"')
                query += f".property('{key}', '{escaped_value}')"
            
            print(f"🔍 Executing: {query[:100]}...")
            result = self._execute_query(query)
            
            if result:
                print(f"✅ Vertex created: {vertex_id}")
                return {'success': True, 'vertex_id': vertex_id}
            else:
                return {'success': False, 'error': 'No result returned'}
            
        except Exception as e:
            print(f"❌ Failed to add vertex {vertex_id}: {e}")
            return {'success': False, 'error': str(e)}

def test_small_storage():
    """Test storing a few entities"""
    print("🧪 TESTING SMALL-SCALE STORAGE")
    print("=" * 40)
    
    # Initialize Cosmos client
    cosmos_config = {
        'endpoint': 'https://cosmos-maintie-rag-staging-oeeopj3ksgnlo.documents.azure.com:443/',
        'database': 'maintie-rag-staging',
        'container': 'knowledge-graph-staging'
    }
    
    cosmos_client = SimpleCosmosClient(cosmos_config)
    
    # Test data
    test_entities = [
        {
            'text': 'hydraulic pump',
            'type': 'equipment',
            'context': 'hydraulic pump failure'
        },
        {
            'text': 'bearing',
            'type': 'component', 
            'context': 'bearing worn out'
        }
    ]
    
    print(f"📦 Testing {len(test_entities)} entities...")
    
    for i, entity in enumerate(test_entities):
        vertex_id = f"test-entity-{i+1}"
        label = entity['type']
        
        properties = {
            'partitionKey': 'maintenance',  # Required for Cosmos DB
            'text': entity['text'],
            'entity_type': entity['type'],
            'domain': 'maintenance',
            'context': entity['context'],
            'created_at': datetime.now().isoformat(),
            'source': 'test'
        }
        
        result = cosmos_client.add_vertex(vertex_id, label, properties)
        
        if result.get('success', False):
            print(f"  ✅ Stored: {entity['text']}")
        else:
            print(f"  ❌ Failed: {entity['text']} - {result.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Test completed!")

if __name__ == "__main__":
    test_small_storage()