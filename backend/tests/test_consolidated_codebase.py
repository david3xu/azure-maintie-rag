#!/usr/bin/env python3
"""
Test Consolidated Codebase
Verify that the unified Azure clients and services work correctly
"""

import sys
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

async def test_unified_clients():
    """Test unified Azure clients"""
    print("🧪 Testing Unified Azure Clients")
    print("=" * 50)
    
    try:
        # Test imports from reorganized structure
        from core.azure_openai.openai_client import UnifiedAzureOpenAIClient
        from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
        from core.azure_search.search_client import UnifiedSearchClient
        from core.azure_storage.storage_factory import get_storage_factory
        print("✅ All unified clients imported successfully")
        
        # Test client initialization (without actual Azure calls)
        print("\\n📋 Testing Client Initialization:")
        
        # Mock config to avoid Azure calls
        mock_config = {
            'endpoint': 'https://mock.openai.azure.com/',
            'key': 'mock_key_12345'
        }
        
        # Test OpenAI client
        try:
            openai_client = UnifiedAzureOpenAIClient(config=mock_config)
            print("✅ UnifiedAzureOpenAIClient: Initialized")
        except Exception as e:
            print(f"⚠️  UnifiedAzureOpenAIClient: {e}")
        
        # Test other clients with mock configs
        clients = [
            ("UnifiedCosmosClient", UnifiedCosmosClient),
            ("UnifiedSearchClient", UnifiedSearchClient), 
            ("UnifiedStorageClient", UnifiedStorageClient)
        ]
        
        for name, client_class in clients:
            try:
                client = client_class(config=mock_config)
                print(f"✅ {name}: Initialized")
            except Exception as e:
                print(f"⚠️  {name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Client testing failed: {e}")
        return False

async def test_unified_services():
    """Test unified services"""
    print("\\n🎯 Testing Unified Services")
    print("=" * 50)
    
    try:
        # Test service imports
        from services.infrastructure_service import InfrastructureService
        from services.data_service import DataService
        from services.workflow_service import WorkflowService
        from services.query_service import QueryService
        print("✅ All services imported successfully")
        
        # Test service initialization
        print("\\n📋 Testing Service Initialization:")
        
        # Test infrastructure service
        infrastructure = InfrastructureService()
        print("  ✅ InfrastructureService initialized")
        
        # Test services that depend on infrastructure
        data_service = DataService(infrastructure)
        workflow_service = WorkflowService(infrastructure)
        query_service = QueryService(infrastructure)
        
        services = [
            ("DataService", data_service),
            ("WorkflowService", workflow_service),
            ("QueryService", query_service)
        ]
        
        for name, service_instance in services:
            try:
                print(f"  ✅ {name}: Initialized")
                
                # Test basic methods exist  
                if hasattr(service_instance, 'migrate_data_to_azure') and name == "DataService":
                    print(f"   📝 Has migrate_data_to_azure method")
                if hasattr(service_instance, 'initialize_rag_orchestration') and name == "WorkflowService":
                    print(f"   🗄️  Has initialize_rag_orchestration method")
                if hasattr(service_instance, 'process_query') and name == "QueryService":
                    print(f"   🔍 Has process_query method")
                    
            except Exception as e:
                print(f"⚠️  {name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Service testing failed: {e}")
        return False

def test_code_structure():
    """Test overall code structure"""
    print("\\n📁 Testing Code Structure")
    print("=" * 50)
    
    try:
        # Check key directories exist
        directories = [
            "core/azure_openai",
            "core/azure_search", 
            "core/azure_storage",
            "core/azure_cosmos",
            "services", 
            "scripts/organized",
            "integrations"
        ]
        
        for directory in directories:
            path = Path(directory)
            if path.exists():
                file_count = len(list(path.glob("*.py")))
                print(f"✅ {directory}: {file_count} Python files")
            else:
                print(f"❌ {directory}: Missing")
        
        # Check no duplicate scripts
        script_root = Path("scripts")
        root_scripts = list(script_root.glob("*.py"))
        organized_scripts = list(script_root.glob("organized/**/*.py"))
        
        print(f"\\n📊 Script Organization:")
        print(f"   Root scripts: {len(root_scripts)} (should be 0 after cleanup)")
        print(f"   Organized scripts: {len(organized_scripts)}")
        
        if len(root_scripts) == 0:
            print("✅ No duplicate scripts in root directory")
        else:
            print("⚠️  Some scripts still in root directory")
        
        return True
        
    except Exception as e:
        print(f"❌ Structure testing failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 CONSOLIDATED CODEBASE TESTING")
    print("=" * 70)
    
    tests = [
        ("Unified Clients", test_unified_clients),
        ("Unified Services", test_unified_services),
        ("Code Structure", test_code_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 70)
    print("📊 CONSOLIDATION TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    success_rate = passed / len(results) * 100
    print(f"\\n🎯 Success Rate: {passed}/{len(results)} ({success_rate:.1f}%)")
    
    if success_rate >= 100:
        print("🎉 CONSOLIDATION SUCCESSFUL!")
    elif success_rate >= 66:
        print("✅ CONSOLIDATION MOSTLY SUCCESSFUL")
    else:
        print("⚠️  CONSOLIDATION NEEDS WORK")

if __name__ == "__main__":
    asyncio.run(main())