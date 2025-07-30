#!/usr/bin/env python3
"""
Test script to verify Azure Cosmos DB connection from training environment
"""
import sys
import os

def test_cosmos_connection():
    """Test Azure Cosmos DB connection and data access"""
    
    # Fix Python path for different environments
    backend_path = os.path.join(os.getcwd(), 'backend')
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    print("🔍 Testing Azure Cosmos DB connection...")
    print(f"📂 Current working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[:3]}...")
    
    try:
        # Test import
        print("📦 Testing import...")
        try:
            from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
            print("✅ Successfully imported AzureCosmosGremlinClient")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            # Try alternative paths
            print("🔄 Trying alternative import paths...")
            sys.path.insert(0, '.')
            from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
            print("✅ Successfully imported AzureCosmosGremlinClient (alternative path)")
        
        # Test client creation
        print("🔗 Creating Cosmos DB client...")
        cosmos_client = AzureCosmosGremlinClient()
        print("✅ Successfully created Cosmos DB client")
        
        # Test basic connectivity
        print("🏓 Testing basic connectivity...")
        stats = cosmos_client.get_graph_statistics("maintenance")
        
        if stats.get('success'):
            vertex_count = stats.get('vertex_count', 0)
            edge_count = stats.get('edge_count', 0)
            print(f"✅ Connection successful! Found {vertex_count} vertices and {edge_count} edges")
            
            # Test graph export for training
            print("📊 Testing graph export for training...")
            graph_data = cosmos_client.export_graph_for_training("maintenance")
            if graph_data.get('success'):
                entities = graph_data.get('entities', [])
                relations = graph_data.get('relations', [])
                print(f"✅ Successfully exported {len(entities)} entities and {len(relations)} relations")
                if entities:
                    print(f"📝 Sample entity: {entities[0]}")
                if relations:
                    print(f"🔗 Sample relation: {relations[0]}")
            else:
                print("❌ Graph export failed")
                
            return True
        else:
            print(f"❌ Connection failed: {stats}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cosmos_connection()
    if success:
        print("\n🎉 SUCCESS: Azure Cosmos DB connection is working!")
        print("✅ GNN training should be able to access real data")
    else:
        print("\n💥 FAILURE: Azure Cosmos DB connection failed")
        print("❌ GNN training will fall back to synthetic data")
    
    sys.exit(0 if success else 1)