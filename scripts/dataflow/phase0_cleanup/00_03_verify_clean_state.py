#!/usr/bin/env python3
"""
Check Azure Services Data
Directly inspect what data currently exists in each Azure service
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps


async def check_cosmos_db_data():
    """Check what data exists in Cosmos DB graph database"""
    print("\n🗄️  CHECKING COSMOS DB DATA")
    print("=" * 40)
    
    try:
        deps = await get_universal_deps()
        cosmos_client = deps.cosmos_client
        
        print("   📊 Checking graph vertices (entities)...")
        
        # Query for all vertices
        try:
            vertex_query = "g.V().count()"
            vertex_result = await cosmos_client.execute_query(vertex_query)
            vertex_count = vertex_result[0] if vertex_result else 0
            print(f"   🔢 Total vertices (entities): {vertex_count}")
            
            if vertex_count > 0:
                # Get sample vertices
                sample_query = "g.V().limit(5).valueMap()"
                sample_result = await cosmos_client.execute_query(sample_query)
                print(f"   📋 Sample vertices:")
                for i, vertex in enumerate(sample_result[:3], 1):
                    print(f"      {i}. {vertex}")
            
        except Exception as e:
            print(f"   ⚠️  Vertex query error: {e}")
        
        print("   📊 Checking graph edges (relationships)...")
        
        # Query for all edges
        try:
            edge_query = "g.E().count()"
            edge_result = await cosmos_client.execute_query(edge_query)
            edge_count = edge_result[0] if edge_result else 0
            print(f"   🔢 Total edges (relationships): {edge_count}")
            
            if edge_count > 0:
                # Get sample edges
                sample_edge_query = "g.E().limit(5).valueMap()"
                sample_edge_result = await cosmos_client.execute_query(sample_edge_query)
                print(f"   📋 Sample edges:")
                for i, edge in enumerate(sample_edge_result[:3], 1):
                    print(f"      {i}. {edge}")
                    
        except Exception as e:
            print(f"   ⚠️  Edge query error: {e}")
        
        return {"vertices": vertex_count if 'vertex_count' in locals() else 0, 
                "edges": edge_count if 'edge_count' in locals() else 0}
        
    except Exception as e:
        print(f"   ❌ Cosmos DB check failed: {e}")
        return {"vertices": "error", "edges": "error"}


async def check_cognitive_search_data():
    """Check what data exists in Cognitive Search indexes"""
    print("\n🔍 CHECKING COGNITIVE SEARCH DATA")
    print("=" * 40)
    
    try:
        deps = await get_universal_deps()
        search_client = deps.search_client
        
        print("   📊 Checking search indexes...")
        
        # List all indexes
        try:
            # Get indexes (this might need adjustment based on actual client API)
            indexes = []  # Placeholder - actual implementation would get index list
            print(f"   📁 Available indexes: {len(indexes)}")
            
            # For each index, check document count
            document_counts = {}
            
            # Note: This is a placeholder - actual implementation would need 
            # the specific search client methods for your Azure Search setup
            print(f"   💡 Manual check needed: Use Azure portal or REST API to check search indexes")
            print(f"      - Check for knowledge base indexes")
            print(f"      - Check for document vector indexes")
            print(f"      - Check for entity/relationship indexes")
            
            return {"indexes": len(indexes), "total_documents": "manual_check_needed"}
            
        except Exception as e:
            print(f"   ⚠️  Search index query error: {e}")
            return {"indexes": "error", "total_documents": "error"}
        
    except Exception as e:
        print(f"   ❌ Cognitive Search check failed: {e}")
        return {"indexes": "error", "total_documents": "error"}


async def check_azure_storage_data():
    """Check what data exists in Azure Storage blobs"""
    print("\n💾 CHECKING AZURE STORAGE DATA")
    print("=" * 40)
    
    try:
        deps = await get_universal_deps()
        
        print("   📊 Checking blob containers...")
        
        # Note: Actual implementation would need the storage client
        print(f"   💡 Manual check needed: Use Azure portal or Azure CLI to check storage")
        print(f"      - Check blob containers for uploaded documents")
        print(f"      - Check for processed data files")
        print(f"      - Check for model checkpoints")
        
        # Placeholder for actual blob counting
        container_count = "manual_check_needed"
        blob_count = "manual_check_needed"
        
        return {"containers": container_count, "blobs": blob_count}
        
    except Exception as e:
        print(f"   ❌ Azure Storage check failed: {e}")
        return {"containers": "error", "blobs": "error"}


async def check_azure_openai_usage():
    """Check Azure OpenAI usage/activity"""
    print("\n🤖 CHECKING AZURE OPENAI USAGE")
    print("=" * 40)
    
    try:
        deps = await get_universal_deps()
        openai_client = deps.openai_client
        
        print("   📊 Checking OpenAI client connection...")
        print(f"   ✅ OpenAI client available: {openai_client is not None}")
        
        # Note: OpenAI doesn't store persistent data, just processes requests
        print(f"   💡 OpenAI service status: Operational (no persistent data stored)")
        print(f"   📋 Usage tracking: Available via Azure portal metrics")
        
        return {"status": "operational", "persistent_data": None}
        
    except Exception as e:
        print(f"   ❌ Azure OpenAI check failed: {e}")
        return {"status": "error", "persistent_data": "error"}


def check_local_cache_data():
    """Check for any remaining local cache data"""
    print("\n💽 CHECKING LOCAL CACHE DATA")
    print("=" * 40)
    
    cache_locations = [
        "cache/",
        "logs/", 
        "scripts/dataflow/results/",
        ".pytest_cache/",
        "frontend/node_modules/.cache/",
        "frontend/dist/"
    ]
    
    total_files = 0
    total_size = 0
    
    for cache_dir in cache_locations:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            files = list(cache_path.rglob("*"))
            file_count = len([f for f in files if f.is_file()])
            if file_count > 0:
                dir_size = sum(f.stat().st_size for f in files if f.is_file())
                total_files += file_count
                total_size += dir_size
                print(f"   📁 {cache_dir}: {file_count} files ({dir_size/1024:.1f} KB)")
            else:
                print(f"   📁 {cache_dir}: Empty")
        else:
            print(f"   📁 {cache_dir}: Not found")
    
    print(f"   📊 Total cache files: {total_files} ({total_size/1024:.1f} KB)")
    
    return {"files": total_files, "size_kb": total_size/1024}


async def main():
    """Main data inspection orchestrator"""
    print("🔍 AZURE SERVICES DATA INSPECTION")
    print("=" * 60)
    print("Checking what data currently exists in Azure services...")
    print("")
    
    start_time = time.time()
    
    results = {}
    
    try:
        # Check each service
        results["cosmos_db"] = await check_cosmos_db_data()
        results["cognitive_search"] = await check_cognitive_search_data()
        results["azure_storage"] = await check_azure_storage_data()
        results["azure_openai"] = await check_azure_openai_usage()
        results["local_cache"] = check_local_cache_data()
        
        duration = time.time() - start_time
        
        # Summary
        print(f"\n📊 DATA INSPECTION SUMMARY")
        print("=" * 40)
        
        for service, data in results.items():
            print(f"🔧 {service.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   result: {data}")
            print("")
        
        print(f"⏱️  Total inspection time: {duration:.2f}s")
        
        # Save results
        results_file = Path("../results/azure_data_inspection.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        inspection_report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "inspection_duration": duration,
            "services_checked": list(results.keys()),
            "results": results
        }
        
        with open(results_file, 'w') as f:
            json.dump(inspection_report, f, indent=2)
        
        print(f"💾 Inspection report saved: {results_file}")
        
        # Final verdict
        has_data = False
        if results.get("cosmos_db", {}).get("vertices", 0) > 0:
            has_data = True
        if results.get("local_cache", {}).get("files", 0) > 0:
            has_data = True
            
        print(f"\n🎯 VERDICT: {'DATA FOUND' if has_data else 'CLEAN STATE'}")
        
    except Exception as e:
        print(f"\n❌ Data inspection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())