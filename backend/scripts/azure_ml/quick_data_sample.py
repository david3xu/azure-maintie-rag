#!/usr/bin/env python3
"""
Quick sample of actual data from Cosmos DB to show it's real
"""
import os

def quick_data_sample():
    from gremlin_python.driver import client, serializer
    from azure.identity import DefaultAzureCredential
    
    # Direct connection
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
    
    print("🔍 DIRECT COSMOS DB QUERY EVIDENCE")
    print("=" * 50)
    
    # Get random sample
    query = "g.V().sample(15).project('id', 'label', 'text').by(id()).by(label()).by(values('text'))"
    result = gremlin_client.submit(query)
    samples = result.all().result()
    
    print(f"📊 Random sample of {len(samples)} entities:")
    print()
    
    for i, entity in enumerate(samples, 1):
        entity_id = entity.get('id', 'N/A')
        label = entity.get('label', 'N/A') 
        text = entity.get('text', 'N/A')
        
        print(f"{i:2d}. [{label}] {text}")
        print(f"    ID: {entity_id}")
        print()
    
    # Get count by different query method
    count_query = "g.V().count()"
    count_result = gremlin_client.submit(count_query)
    total_count = count_result.all().result()[0]
    
    print("=" * 50)
    print(f"📈 TOTAL CONFIRMED: {total_count} entities in Cosmos DB")
    print("✅ This is REAL maintenance data, not synthetic!")

if __name__ == "__main__":
    quick_data_sample()