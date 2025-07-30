#!/usr/bin/env python3
"""
Check what data actually exists in Azure Cosmos DB
Using standalone client to avoid import issues
"""
import os
import sys
from typing import Dict, List, Any

# Add the standalone Cosmos client (inline)
class StandaloneCosmosClient:
    """Standalone Cosmos DB client with no external dependencies"""
    
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
            
            # Extract account name from endpoint
            account_name = self.endpoint.replace('https://', '').replace('.documents.azure.com:443/', '')
            gremlin_endpoint = f"wss://{account_name}.gremlin.cosmosdb.azure.com:443/"
            
            # Get Azure credential
            credential = DefaultAzureCredential()
            token = credential.get_token("https://cosmos.azure.com/.default")
            
            # Create Gremlin client
            self.gremlin_client = client.Client(
                gremlin_endpoint,
                'g',
                username=f"/dbs/{self.database}/colls/{self.container}",
                password=token.token,
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            
            self._initialized = True
            print("✅ Standalone Cosmos DB client initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Cosmos client: {e}")
            return False
    
    def _execute_query_safe(self, query: str, timeout_seconds: int = 30):
        """Execute Gremlin query safely"""
        try:
            if not self._initialized:
                if not self._initialize_client():
                    return []
            
            result = self.gremlin_client.submit(query)
            return result.all().result(timeout=timeout_seconds)
            
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return []

def check_cosmos_data():
    """Check what data exists in Cosmos DB"""
    print("🔍 CHECKING AZURE COSMOS DB DATA")
    print("=" * 60)
    
    # Configuration
    cosmos_config = {
        'endpoint': os.environ.get('AZURE_COSMOS_ENDPOINT', 'https://cosmos-maintie-rag-staging-oeeopj3ksgnlo.documents.azure.com:443/'),
        'database': os.environ.get('COSMOS_DATABASE_NAME', 'maintie-rag-staging'),
        'container': os.environ.get('COSMOS_GRAPH_NAME', 'knowledge-graph-staging')
    }
    
    print(f"📋 Configuration:")
    print(f"   Endpoint: {cosmos_config['endpoint']}")
    print(f"   Database: {cosmos_config['database']}")
    print(f"   Container: {cosmos_config['container']}")
    print()
    
    try:
        # Create client
        client = StandaloneCosmosClient(cosmos_config)
        
        # 1. Check basic vertex count
        print("1️⃣ CHECKING VERTEX COUNT:")
        vertex_count_query = "g.V().count()"
        vertex_count = client._execute_query_safe(vertex_count_query)
        print(f"   Total vertices: {vertex_count}")
        
        # 2. Check vertex properties
        print("\n2️⃣ CHECKING VERTEX PROPERTIES:")
        sample_vertex_query = "g.V().limit(3).valueMap()"
        sample_vertices = client._execute_query_safe(sample_vertex_query)
        print(f"   Sample vertices ({len(sample_vertices)}):")
        for i, vertex in enumerate(sample_vertices):
            print(f"     Vertex {i+1}: {vertex}")
        
        # 3. Check vertex labels/types
        print("\n3️⃣ CHECKING VERTEX LABELS:")
        vertex_labels_query = "g.V().groupCount().by(label)"
        vertex_labels = client._execute_query_safe(vertex_labels_query)
        print(f"   Vertex labels: {vertex_labels}")
        
        # 4. Check edge count
        print("\n4️⃣ CHECKING EDGE COUNT:")
        edge_count_query = "g.E().count()"
        edge_count = client._execute_query_safe(edge_count_query)
        print(f"   Total edges: {edge_count}")
        
        # 5. Check edge properties if any exist
        print("\n5️⃣ CHECKING EDGE PROPERTIES:")
        if edge_count and edge_count[0] > 0:
            sample_edge_query = "g.E().limit(3).valueMap()"
            sample_edges = client._execute_query_safe(sample_edge_query)
            print(f"   Sample edges ({len(sample_edges)}):")
            for i, edge in enumerate(sample_edges):
                print(f"     Edge {i+1}: {edge}")
                
            # Check edge labels
            edge_labels_query = "g.E().groupCount().by(label)"
            edge_labels = client._execute_query_safe(edge_labels_query)
            print(f"   Edge labels: {edge_labels}")
        else:
            print("   ⚠️ No edges found in the graph")
        
        # 6. Check data structure for GNN requirements
        print("\n6️⃣ GNN TRAINING REQUIREMENTS CHECK:")
        
        # Get actual vertices with their properties
        vertices_query = "g.V().limit(10).valueMap()"
        vertices = client._execute_query_safe(vertices_query)
        
        if vertices:
            print(f"   ✅ Found {len(vertices)} sample vertices")
            
            # Check required properties for GNN training
            required_props = ['id', 'text']
            optional_props = ['entity_type', 'type']
            
            for i, vertex in enumerate(vertices[:3]):  # Check first 3
                print(f"     Vertex {i+1} properties:")
                for prop in required_props:
                    if prop in vertex:
                        print(f"       ✅ {prop}: {vertex[prop]}")
                    else:
                        print(f"       ❌ {prop}: MISSING")
                
                for prop in optional_props:
                    if prop in vertex:
                        print(f"       📝 {prop}: {vertex[prop]}")
        else:
            print("   ❌ No vertices found - cannot train GNN")
        
        print("\n" + "=" * 60)
        print("📊 SUMMARY:")
        print("=" * 60)
        
        total_vertices = vertex_count[0] if vertex_count and len(vertex_count) > 0 else 0
        total_edges = edge_count[0] if edge_count and len(edge_count) > 0 else 0
        
        print(f"📈 Vertices: {total_vertices}")
        print(f"🔗 Edges: {total_edges}")
        
        if total_vertices >= 10:
            print("✅ SUFFICIENT DATA: Can proceed with GNN training")
        else:
            print("❌ INSUFFICIENT DATA: Need at least 10 vertices for GNN training")
            
        if total_edges == 0:
            print("⚠️ WARNING: No edges found - GNN will create synthetic connectivity")
        
        return {
            "vertices": total_vertices,
            "edges": total_edges,
            "sufficient_for_training": total_vertices >= 10,
            "has_connectivity": total_edges > 0
        }
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR checking Cosmos DB data: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = check_cosmos_data()
    
    if result:
        print(f"\n🎯 RESULT: {'✅ READY FOR GNN TRAINING' if result['sufficient_for_training'] else '❌ NOT READY FOR GNN TRAINING'}")
    else:
        print(f"\n💥 FAILED to check Cosmos DB data")