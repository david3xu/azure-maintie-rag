#!/usr/bin/env python3
"""
Clean All Azure Data Script
Comprehensive cleanup of all data from Azure services while preserving:
- data/raw/ (original source data)
- Azure services (infrastructure remains operational, data cleaned)
- Core configuration files and codebase
"""

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import path utilities for consistent directory handling
from scripts.dataflow.utilities.path_utils import get_results_dir
from agents.core.universal_deps import get_universal_deps
from infrastructure.azure_ml.gnn_model import UniversalGNNConfig


async def clean_local_data():
    """Clean all local running results and cached data"""
    print("🧹 CLEANING LOCAL DATA")
    print("=" * 40)

    # Directories to clean (but preserve structure)
    dirs_to_clean = [
        "logs/",
        "cache/",
        "scripts/dataflow/results/",
        "tests/__pycache__/",
        "agents/__pycache__/",
        "api/__pycache__/",
        "infrastructure/__pycache__/",
        "config/__pycache__/",
        ".pytest_cache/",
        "frontend/dist/",
        "frontend/node_modules/.cache/",
    ]

    # Files to clean
    files_to_clean = [
        "*.log",
        "*.tmp",
        "*.cache",
        "*_debug_*.json",
        "*_tmp*.json",
        "debug_*.py",  # Any remaining debug scripts
    ]

    for dir_path in dirs_to_clean:
        full_path = Path(dir_path)
        if full_path.exists():
            if full_path.is_dir():
                print(f"   🗑️  Cleaning directory: {dir_path}")
                # Clean contents but preserve directory structure
                for item in full_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir() and item.name != "README.md":
                        shutil.rmtree(item)
            else:
                print(f"   🗑️  Removing file: {dir_path}")
                full_path.unlink()

    # Clean specific result files (keep only essential ones)
    results_dir = get_results_dir()  # Use path utilities for reliable directory access
    if results_dir.exists():
        print(f"   📁 Cleaning results directory...")
        # Keep only essential files
        essential_files = ["README.md", "SYSTEM_CAPABILITIES_DEMO.md"]

        for file_path in results_dir.iterdir():
            if file_path.is_file() and file_path.name not in essential_files:
                print(f"      🗑️  Removing: {file_path.name}")
                file_path.unlink()

    # Clean Python cache files recursively
    print(f"   🐍 Cleaning Python cache files...")
    for pycache_dir in Path(".").rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)

    for pyc_file in Path(".").rglob("*.pyc"):
        pyc_file.unlink()

    print("   ✅ Local data cleanup complete")


async def clean_azure_services():
    """Clean data from Azure services while keeping services running"""
    print("\n☁️  CLEANING AZURE SERVICES DATA")
    print("=" * 40)

    try:
        # Initialize dependencies to access Azure services
        deps = await get_universal_deps()
        available_services = deps.get_available_services()
        print(f"   🔧 Connected to services: {', '.join(available_services)}")

        cleaned_services = []

        # Clean Cosmos DB (Knowledge Graph data)
        if "cosmos" in available_services:
            try:
                print("   🗄️  Cleaning Cosmos DB knowledge graph data...")
                cosmos_client = deps.cosmos_client

                # Check current data
                vertex_result = await cosmos_client.execute_query("g.V().count()")
                edge_result = await cosmos_client.execute_query("g.E().count()")
                vertex_count = vertex_result[0] if vertex_result else 0
                edge_count = edge_result[0] if edge_result else 0

                print(
                    f"      📊 Current data: {vertex_count} vertices, {edge_count} edges"
                )

                if vertex_count > 0 or edge_count > 0:
                    print("      🧹 Clearing all graph data...")

                    # Clear all edges first
                    if edge_count > 0:
                        await cosmos_client.execute_query("g.E().drop()")
                        print(f"      ✅ Cleared {edge_count} edges")

                    # Clear all vertices
                    if vertex_count > 0:
                        await cosmos_client.execute_query("g.V().drop()")
                        print(f"      ✅ Cleared {vertex_count} vertices")

                    # Verify cleanup
                    final_vertex_count = (
                        await cosmos_client.execute_query("g.V().count()")
                    )[0]
                    final_edge_count = (
                        await cosmos_client.execute_query("g.E().count()")
                    )[0]
                    print(
                        f"      🔍 After cleanup: {final_vertex_count} vertices, {final_edge_count} edges"
                    )
                else:
                    print("      ✅ Cosmos DB already clean")

                cleaned_services.append("cosmos_db")

            except Exception as e:
                print(f"      ❌ Cosmos DB cleanup failed: {e}")
                print(f"      💡 May need manual cleanup via Azure portal")

        # Clean Cognitive Search (Vector indexes)
        if "search" in available_services:
            try:
                print("   🔍 Cleaning Cognitive Search indexes...")
                search_client = deps.search_client

                # DON'T trust get_index_stats() - it can report wrong counts
                # Instead, directly search for all documents
                print(
                    "      🔍 Searching for all documents (ignoring unreliable stats)..."
                )
                search_result = await search_client.search_documents("*", top=1000)

                if search_result.get("success"):
                    documents = search_result["data"]["documents"]
                    doc_count = len(documents)
                    print(f"      📊 Actually found {doc_count} documents in index")

                    if doc_count > 0:
                        print("      🧹 Clearing all search documents...")

                        # Get document IDs for deletion
                        doc_ids = [doc["id"] for doc in documents]
                        print(
                            f"      🗑️  Deleting {len(doc_ids)} documents: {doc_ids[:5]}{'...' if len(doc_ids) > 5 else ''}"
                        )

                        # Delete documents by ID
                        delete_result = await search_client.delete_documents(doc_ids)
                        if delete_result.get("success"):
                            deleted_count = delete_result["data"]["deleted_count"]
                            failed_count = delete_result["data"].get("failed_count", 0)
                            print(
                                f"      📊 Deletion result: {deleted_count} successful, {failed_count} failed"
                            )

                            # Show detailed results if any failed
                            if failed_count > 0:
                                results = delete_result["data"].get("results", [])
                                for i, r in enumerate(results):
                                    status = (
                                        "✅"
                                        if (hasattr(r, "succeeded") and r.succeeded)
                                        else "❌"
                                    )
                                    print(
                                        f"        {status} Doc {i+1}: {getattr(r, 'key', 'unknown')} - {getattr(r, 'status_code', 'unknown')}"
                                    )

                            # Verify deletion worked
                            verify_result = await search_client.search_documents(
                                "*", top=10
                            )
                            if verify_result.get("success"):
                                remaining = len(verify_result["data"]["documents"])
                                if remaining == 0:
                                    print(
                                        f"      ✅ Verified: Search index is now clean (0 documents)"
                                    )
                                else:
                                    print(
                                        f"      ⚠️  WARNING: {remaining} documents still remain after deletion"
                                    )
                                    # Show what's still there
                                    for i, doc in enumerate(
                                        verify_result["data"]["documents"][:3]
                                    ):
                                        print(
                                            f"        Still there: {doc.get('id', 'unknown-id')}"
                                        )
                        else:
                            print(
                                f"      ❌ Document deletion failed: {delete_result.get('error', 'Unknown error')}"
                            )
                    else:
                        print(
                            "      ✅ Search index already clean (no documents found)"
                        )
                else:
                    print(
                        f"      ❌ Could not search for documents: {search_result.get('error', 'Unknown error')}"
                    )

                cleaned_services.append("cognitive_search")

            except Exception as e:
                print(f"      ❌ Cognitive Search cleanup failed: {e}")
                print(f"      💡 Index may require manual cleanup via Azure portal")

        # Clean Azure Storage (Blob data - but keep containers)
        if "storage" in available_services:
            try:
                print("   💾 Cleaning Azure Storage blob data...")
                storage_client = deps.storage_client

                # Get storage service client for container operations
                storage_service = storage_client._blob_service
                print("      📊 Listing all containers...")

                container_count = 0
                total_blobs_deleted = 0

                # List all containers
                containers = storage_service.list_containers()
                container_names = []

                for container in containers:
                    container_names.append(container.name)
                    container_count += 1

                print(f"      📁 Found {container_count} containers: {container_names}")

                # Clean each container
                for container_name in container_names:
                    print(f"      🧹 Cleaning container: {container_name}")

                    try:
                        container_client = storage_service.get_container_client(
                            container_name
                        )
                        blobs = container_client.list_blobs()
                        blob_names = []

                        # Collect blob names
                        for blob in blobs:
                            blob_names.append(blob.name)

                        if blob_names:
                            print(
                                f"        🗑️  Deleting {len(blob_names)} blobs from {container_name}"
                            )

                            # Delete each blob
                            for blob_name in blob_names:
                                blob_client = storage_service.get_blob_client(
                                    container=container_name, blob=blob_name
                                )
                                blob_client.delete_blob()
                                total_blobs_deleted += 1

                            print(
                                f"        ✅ Deleted {len(blob_names)} blobs from {container_name}"
                            )
                        else:
                            print(
                                f"        ✅ Container {container_name} already clean"
                            )

                    except Exception as e:
                        print(
                            f"        ❌ Failed to clean container {container_name}: {e}"
                        )

                print(
                    f"      ✅ Storage cleanup complete: {total_blobs_deleted} blobs deleted from {container_count} containers"
                )
                cleaned_services.append("azure_storage")

            except Exception as e:
                print(f"      ❌ Azure Storage cleanup failed: {e}")
                print(f"      💡 Storage may require manual cleanup via Azure portal")

        # Clean GNN model data
        try:
            print("   🧠 Cleaning GNN model data...")

            # Clear any local GNN model cache/checkpoints
            gnn_cache_dirs = ["cache/gnn_models/", "logs/gnn_training/", "checkpoints/"]

            for cache_dir in gnn_cache_dirs:
                cache_path = Path(cache_dir)
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                    print(f"      🗑️  Removed: {cache_dir}")

            cleaned_services.append("gnn_models")

        except Exception as e:
            print(f"      ⚠️  GNN cleanup warning: {e}")

        if cleaned_services:
            print(
                f"   ✅ Azure services data cleanup prepared: {', '.join(cleaned_services)}"
            )
        else:
            print(f"   ⚠️  No Azure services data cleaned (may require manual steps)")

    except Exception as e:
        print(f"   ❌ Azure cleanup failed: {e}")
        print(f"   💡 Services remain operational, data cleaning skipped")


def clean_session_data():
    """Clean session-specific data and logs"""
    print("\n📋 CLEANING SESSION DATA")
    print("=" * 40)

    session_dirs = [
        "logs/pipeline_*/",
        "logs/session_*/",
        "cache/sessions/",
    ]

    for session_pattern in session_dirs:
        for session_dir in Path(".").glob(session_pattern):
            if session_dir.is_dir():
                print(f"   🗑️  Removing session: {session_dir}")
                shutil.rmtree(session_dir)

    # Clean session files
    session_files = [
        "logs/session_report.md",
        "logs/performance.log",
        "logs/current_session.log",
    ]

    for session_file in session_files:
        file_path = Path(session_file)
        if file_path.exists():
            print(f"   🗑️  Removing: {session_file}")
            file_path.unlink()

    print("   ✅ Session data cleanup complete")


def preserve_essential_data():
    """Show what data is being preserved"""
    print("\n🛡️  PRESERVED DATA")
    print("=" * 40)

    preserved_items = [
        "data/raw/ - Original source data (Azure AI documentation)",
        "config/ - Core configuration files",
        "agents/ - PydanticAI agent code",
        "infrastructure/ - Azure service client code",
        "api/ - FastAPI backend code",
        "frontend/src/ - React frontend code",
        "scripts/dataflow/ - Pipeline scripts (code only)",
        "tests/ - Test code (results cleaned)",
        "Azure Services - All services remain operational",
    ]

    for item in preserved_items:
        print(f"   ✅ {item}")


async def main():
    """Main cleanup orchestrator"""
    print("🧹 COMPREHENSIVE DATA CLEANUP")
    print("=" * 60)
    print(
        "This script will clean all running results and cached data while preserving:"
    )
    print("- Original source data (data/raw/)")
    print("- Core codebase and configuration")
    print("- Azure services (will remain operational)")
    print("")

    start_time = time.time()

    try:
        # Step 1: Clean local data
        await clean_local_data()

        # Step 2: Clean Azure services data
        await clean_azure_services()

        # Step 3: Clean session data
        clean_session_data()

        # Step 4: Show preserved data
        preserve_essential_data()

        duration = time.time() - start_time

        print(f"\n🎉 CLEANUP COMPLETE!")
        print(f"⏱️  Total time: {duration:.2f}s")
        print(f"💾 Data/raw preserved: {len(list(Path('data/raw').rglob('*')))} files")
        print(f"🏗️  Codebase preserved: Core functionality intact")
        print(f"☁️  Azure services: Operational and ready for fresh data")
        print("")
        print("🚀 System is now clean and ready for a fresh start!")

    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
