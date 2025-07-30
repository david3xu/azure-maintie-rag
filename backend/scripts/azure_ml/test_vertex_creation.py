#!/usr/bin/env python3
"""
Test how to create vertices properly in Cosmos DB with partition key
"""
def test_vertex_creation():
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
    
    # Test simple vertex creation with correct partition key syntax
    test_id = "test-vertex-12345"
    
    # Proper Cosmos DB partition key syntax
    query = f"""
    g.addV('test_entity')
      .property(id, '{test_id}')
      .property('partitionKey', 'maintenance')
      .property('text', 'test air conditioner')
      .property('domain', 'maintenance')
      .property('entity_type', 'test')
    """
    
    print(f"🧪 Testing vertex creation with query:")
    print(query)
    
    try:
        result = gremlin_client.submit(query.strip())
        result_data = result.all().result(timeout=30)
        print(f"✅ SUCCESS: Vertex created: {result_data}")
        
        # Now try to query it back
        query_back = f"g.V('{test_id}').valueMap()"
        result2 = gremlin_client.submit(query_back)
        vertex_data = result2.all().result(timeout=30)
        print(f"✅ Retrieved vertex: {vertex_data}")
        
        # Clean up - delete the test vertex
        delete_query = f"g.V('{test_id}').drop()"
        gremlin_client.submit(delete_query)
        print(f"🧹 Test vertex deleted")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    test_vertex_creation()