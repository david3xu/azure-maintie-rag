#!/usr/bin/env python3
"""
Check what partition key is used by existing vertices
"""
def check_partition_key():
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
    
    # Check existing vertex properties
    query = "g.V().limit(3).valueMap()"
    result = gremlin_client.submit(query)
    vertices = result.all().result()
    
    print("🔍 Existing vertex properties:")
    for i, vertex in enumerate(vertices, 1):
        print(f"Vertex {i}: {vertex}")
        if 'partitionKey' in vertex:
            print(f"  ✅ Partition key: {vertex['partitionKey']}")
        else:
            print(f"  ❌ No partition key found")

if __name__ == "__main__":
    check_partition_key()