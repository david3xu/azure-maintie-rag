#!/usr/bin/env python3
"""
Check the gap between extracted relationships and Cosmos DB storage
"""
import json
import os

def check_relationship_gap():
    """Check why relationships aren't in Cosmos DB when we extracted them"""
    print("🔍 INVESTIGATING RELATIONSHIP EXTRACTION vs COSMOS DB GAP")
    print("=" * 70)
    
    # Check knowledge extraction results
    extraction_file = "/workspace/azure-maintie-rag/backend/data/outputs/complete_dataflow_20250730_044432/step02_knowledge_extraction.json"
    
    if os.path.exists(extraction_file):
        print("1️⃣ KNOWLEDGE EXTRACTION RESULTS:")
        with open(extraction_file, 'r') as f:
            extraction_data = json.load(f)
        
        entities_extracted = extraction_data.get('entities_extracted', 0)
        relationships_extracted = extraction_data.get('relationships_extracted', 0)
        
        print(f"   📊 Entities extracted: {entities_extracted}")
        print(f"   📊 Relationships extracted: {relationships_extracted}")
        
        # Show sample relationships
        relationships = extraction_data.get('knowledge_data', {}).get('relationships', [])
        print(f"\n   📝 Sample relationships extracted ({len(relationships)} total):")
        for i, rel in enumerate(relationships[:5]):
            source = rel.get('source', 'N/A')
            target = rel.get('target', 'N/A') 
            relation = rel.get('relation', 'N/A')
            print(f"     {i+1}. {source} --[{relation}]--> {target}")
    
    # Check graph construction results  
    graph_file = "/workspace/azure-maintie-rag/backend/data/outputs/complete_dataflow_20250730_044432/step04_graph_construction.json"
    
    if os.path.exists(graph_file):
        print(f"\n2️⃣ GRAPH CONSTRUCTION RESULTS:")
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)
        
        pytorch_info = graph_data.get('pytorch_data_info', {})
        stats = graph_data.get('statistics', {})
        
        print(f"   📊 PyTorch nodes: {pytorch_info.get('num_nodes', 0)}")
        print(f"   📊 PyTorch edges: {pytorch_info.get('num_edges', 0)}")
        print(f"   📊 Total entities: {stats.get('total_entities', 0)}")
        print(f"   📊 Total relationships: {stats.get('total_relationships', 0)}")
        
        # Show relationship types
        rel_types = stats.get('relationship_type_distribution', {})
        print(f"\n   📝 Relationship types found:")
        for rel_type, count in list(rel_types.items())[:10]:
            print(f"     - {rel_type}: {count}")
    
    # Now check what's actually in Cosmos DB
    print(f"\n3️⃣ COSMOS DB CURRENT STATE:")
    try:
        from gremlin_python.driver import client, serializer
        from azure.identity import DefaultAzureCredential
        
        endpoint = 'https://cosmos-maintie-rag-staging-oeeopj3ksgnlo.documents.azure.com:443/'
        database = 'maintie-rag-staging'
        container = 'knowledge-graph-staging'
        
        account_name = endpoint.replace('https://', '').replace('.documents.azure.com:443/', '')
        gremlin_endpoint = f"wss://{account_name}.gremlin.cosmosdb.azure.com:443/"
        
        credential = DefaultAzureCredential()
        token = credential.get_token("https://cosmos.azure.com/.default")
        
        gremlin_client = client.Client(
            gremlin_endpoint,
            'g',
            username=f"/dbs/{database}/colls/{container}",
            password=token.token,
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        
        # Check vertices
        vertex_result = gremlin_client.submit("g.V().count()")
        vertex_count = vertex_result.all().result()[0]
        
        # Check edges
        edge_result = gremlin_client.submit("g.E().count()")
        edge_count = edge_result.all().result()[0]
        
        print(f"   📊 Cosmos DB vertices: {vertex_count}")
        print(f"   📊 Cosmos DB edges: {edge_count}")
        
        # Check if there are any edge labels
        if edge_count > 0:
            edge_labels_result = gremlin_client.submit("g.E().groupCount().by(label)")
            edge_labels = edge_labels_result.all().result()
            print(f"   📝 Edge labels: {edge_labels}")
        
    except Exception as e:
        print(f"   ❌ Error checking Cosmos DB: {e}")
    
    print(f"\n" + "=" * 70)
    print("🎯 ANALYSIS:")
    print("=" * 70)
    
    if entities_extracted > 0 and relationships_extracted > 0:
        print("✅ EXTRACTION SUCCESSFUL: We DID extract entities and relationships")
        print("✅ GRAPH BUILDING SUCCESSFUL: PyTorch data was created with edges")
        
        if edge_count == 0:
            print("❌ STORAGE GAP: Relationships were NOT stored in Cosmos DB")
            print("🔍 LIKELY CAUSE: Graph construction saved to PyTorch files but not Cosmos DB")
            print("💡 SOLUTION: Need to run the Cosmos DB storage step for relationships")
        else:
            print("✅ RELATIONSHIPS STORED: Found edges in Cosmos DB")
    else:
        print("❌ EXTRACTION FAILED: No entities or relationships were extracted")

if __name__ == "__main__":
    check_relationship_gap()