#!/usr/bin/env python3
"""Check Azure data availability for GNN training"""

import sys
import os
import asyncio

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

async def check_azure_storage_data():
    """Check what data we have in Azure Storage"""
    print("=== Checking Azure Storage Data ===")
    
    try:
        from core.azure_storage.storage_client import UnifiedStorageClient
        
        storage_client = UnifiedStorageClient()
        
        # Test connection
        connection_result = await storage_client.test_connection()
        if not connection_result.get("success", False):
            print(f"❌ Storage connection failed: {connection_result.get('error')}")
            return False
        
        print(f"✅ Storage connected: {connection_result.get('account')}")
        print(f"   Container count: {connection_result.get('container_count', 0)}")
        
        # List containers to see what we have
        containers = await storage_client.list_containers()
        print(f"📁 Available containers:")
        for container in containers[:5]:  # Show first 5
            print(f"   - {container}")
        
        # Check default container for raw text data
        blobs = await storage_client.list_blobs()
        if blobs.get("success", False):
            blob_list = blobs.get("blobs", [])
            print(f"📄 Sample files in default container ({len(blob_list)} shown):")
            for i, blob in enumerate(blob_list):
                if i >= 5:  # Show only first 5
                    break
                print(f"   - {blob.get('name', 'Unknown')} ({blob.get('size', 0)} bytes)")
            return len(blob_list) > 0
        else:
            print("❌ Could not list blobs")
            return False
            
    except Exception as e:
        print(f"❌ Storage check failed: {e}")
        return False

async def check_cosmos_graph_data():
    """Check knowledge graph data in Cosmos DB"""
    print("\n=== Checking Cosmos DB Graph Data ===")
    
    try:
        from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
        
        cosmos_client = AzureCosmosGremlinClient()
        
        # Test connection
        connection_result = await cosmos_client.test_connection()
        if not connection_result.get("success", False):
            print(f"❌ Cosmos connection failed: {connection_result.get('error')}")
            return False
        
        print(f"✅ Cosmos DB connected: {connection_result.get('database')}")
        
        # Check for different domains
        domains_to_check = ["maintenance", "general", "default"]
        
        for domain in domains_to_check:
            try:
                stats = cosmos_client.get_graph_statistics(domain)
                if stats.get("success", False):
                    vertex_count = stats.get("vertex_count", 0)
                    edge_count = stats.get("edge_count", 0)
                    
                    if vertex_count > 0 or edge_count > 0:
                        print(f"📊 Domain '{domain}': {vertex_count} entities, {edge_count} relationships")
                        
                        # Get sample entities (safely)
                        try:
                            entities = cosmos_client.get_all_entities(domain)
                            if entities:
                                print(f"   Sample entities: {len(entities)} found")
                        except Exception as entity_error:
                            print(f"   Sample entities: Could not retrieve ({entity_error})")
                        
                        return True, domain
                    else:
                        print(f"   Domain '{domain}': No data")
                else:
                    print(f"   Domain '{domain}': Stats query failed")
            except Exception as e:
                print(f"   Domain '{domain}': Error - {e}")
        
        print("❌ No graph data found in any domain")
        return False, None
        
    except Exception as e:
        print(f"❌ Cosmos check failed: {e}")
        return False, None

async def check_azure_ml_workspace():
    """Check Azure ML workspace connectivity"""
    print("\n=== Checking Azure ML Workspace ===")
    
    try:
        from core.azure_ml.client import AzureMLClient
        
        ml_client = AzureMLClient()
        
        # Try to get workspace info
        workspace = ml_client.get_workspace()
        print(f"✅ Azure ML workspace connected: {workspace.name if workspace else 'Unknown'}")
        
        # Check if we can list compute resources
        try:  
            computes = list(ml_client.ml_client.compute.list())
            print(f"💻 Available compute resources: {len(computes)}")
            for compute in computes[:3]:
                print(f"   - {compute.name} ({compute.type})")
        except Exception as e:
            print(f"⚠️  Could not list compute resources: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Azure ML check failed: {e}")
        return False

async def main():
    """Check all Azure resources for GNN training readiness"""
    print("Checking Azure Data Availability for GNN Training...")
    print("=" * 60)
    
    # Check all resources
    storage_ok = await check_azure_storage_data()
    graph_data_available, graph_domain = await check_cosmos_graph_data()
    ml_workspace_ok = await check_azure_ml_workspace()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"📦 Azure Storage: {'✅ Available' if storage_ok else '❌ Issues'}")
    print(f"🌐 Graph Data: {'✅ Available' if graph_data_available else '❌ Missing'}")
    if graph_data_available:
        print(f"   Best domain for training: {graph_domain}")
    print(f"🤖 Azure ML: {'✅ Ready' if ml_workspace_ok else '❌ Issues'}")
    
    # Overall readiness assessment
    if storage_ok and graph_data_available and ml_workspace_ok:
        print("\n🎉 READY FOR GNN TRAINING!")
        print(f"Recommended next step: Run training on domain '{graph_domain}'")
        return 0, graph_domain
    else:
        print("\n❌ NOT READY - Missing required resources")
        return 1, None

if __name__ == "__main__":
    result_code, domain = asyncio.run(main())
    sys.exit(result_code)