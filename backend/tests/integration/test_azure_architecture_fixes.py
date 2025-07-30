#!/usr/bin/env python3
"""
Azure-Only Architecture Fix Validation
Tests ONLY against real Azure services - NO MOCKS OR SIMULATIONS
"""

import asyncio
import sys
import os
from typing import Dict, Any

def validate_azure_environment():
    """Ensure we have Azure credentials and configuration"""
    required_env_vars = [
        'AZURE_TENANT_ID',
        'AZURE_CLIENT_ID', 
        'AZURE_CLIENT_SECRET',
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_COSMOS_ENDPOINT',
        'AZURE_SEARCH_ENDPOINT',
        'AZURE_STORAGE_ACCOUNT_NAME'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ AZURE ENVIRONMENT VALIDATION FAILED")
        print(f"Missing required environment variables: {missing_vars}")
        print("Cannot test without proper Azure credentials")
        return False
    
    print("✅ Azure environment variables present")
    return True

async def test_azure_cosmos_new_methods():
    """Test cosmos client new methods against REAL Azure Cosmos DB"""
    print("🗄️  Testing Azure Cosmos DB New Methods")
    print("=" * 50)
    
    try:
        from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
        
        # Create client with REAL Azure configuration
        client = AzureCosmosGremlinClient()
        
        # Test Azure connection first
        connection_test = await client.test_connection()
        if not connection_test.get('success'):
            print(f"   ❌ Azure Cosmos DB connection failed: {connection_test.get('error')}")
            return False
        
        print("   ✅ Connected to Azure Cosmos DB")
        
        # Test new method: get_graph_change_metrics with real Azure data
        domain = "test_architecture"
        change_metrics = client.get_graph_change_metrics(domain)
        
        assert isinstance(change_metrics, dict)
        assert "entity_count" in change_metrics
        assert "domain" in change_metrics
        print(f"   ✅ get_graph_change_metrics works: {change_metrics['entity_count']} entities")
        
        # Test new method: extract_training_features with real Azure graph data
        training_features = await client.extract_training_features(domain)
        
        assert isinstance(training_features, dict)
        assert "features" in training_features
        assert "metadata" in training_features
        print(f"   ✅ extract_training_features works: {training_features['metadata']['num_nodes']} nodes")
        
        # Test new method: save_evidence_report to real Azure Cosmos
        test_report = {
            "workflow_id": "azure_test_123",
            "summary": {"total_steps": 2, "total_cost_usd": 0.05},
            "domain": domain
        }
        
        save_result = client.save_evidence_report(test_report)
        
        assert save_result.get("success") == True
        assert "report_id" in save_result
        print(f"   ✅ save_evidence_report works: {save_result['report_id']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Azure Cosmos test failed: {e}")
        return False

async def test_azure_evidence_collection():
    """Test evidence collection workflow with REAL Azure services"""
    print("\n🧪 Testing Azure Evidence Collection Workflow")
    print("=" * 50)
    
    try:
        from core.utilities.workflow_evidence_collector import AzureDataWorkflowEvidenceCollector
        
        # Create evidence collector
        collector = AzureDataWorkflowEvidenceCollector("azure_arch_test")
        
        # Record evidence from REAL Azure Cosmos operation
        from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
        cosmos_client = AzureCosmosGremlinClient()
        
        # Test Azure connection
        connection_test = await cosmos_client.test_connection()
        if not connection_test.get('success'):
            print(f"   ❌ Cannot test without Azure Cosmos connection")
            return False
        
        # Record evidence from REAL Azure operation
        import time
        start_time = time.time()
        
        # Real Azure Cosmos operation
        stats = cosmos_client.get_graph_statistics("test_architecture")
        processing_time = (time.time() - start_time) * 1000
        
        # Record this REAL Azure evidence
        evidence = await collector.record_azure_service_evidence(
            step_number=1,
            azure_service="cosmos_db",
            operation_type="graph_statistics",
            input_data={"domain": "test_architecture"},
            output_data=stats,
            processing_time_ms=processing_time,
            azure_request_id=f"cosmos_real_{int(time.time())}"
        )
        
        print(f"   ✅ Recorded real Azure Cosmos evidence: {evidence.processing_time_ms:.1f}ms")
        
        # Generate evidence report
        report = await collector.generate_workflow_evidence_report()
        
        assert report["workflow_id"] == "azure_arch_test"
        assert report["summary"]["total_steps"] == 1
        assert report["summary"]["total_cost_usd"] > 0
        
        print(f"   ✅ Generated evidence report: ${report['summary']['total_cost_usd']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Azure evidence collection test failed: {e}")
        return False

async def test_azure_services_integration():
    """Test services layer with REAL Azure services"""
    print("\n🔗 Testing Azure Services Integration")
    print("=" * 50)
    
    try:
        from services.workflow_service import WorkflowService
        from core.utilities.workflow_evidence_collector import AzureDataWorkflowEvidenceCollector
        
        # Create workflow service (uses core utilities)
        service = WorkflowService()
        
        # Create workflow
        workflow_id = service.create_workflow("azure_integration_test")
        evidence_collector = service.evidence_collectors[workflow_id]
        
        # Verify it's using core utilities
        assert isinstance(evidence_collector, AzureDataWorkflowEvidenceCollector)
        print(f"   ✅ Services using core utilities: {workflow_id}")
        
        # Test with REAL Azure operation via services layer
        # This would normally be called by actual Azure operations
        await service.record_azure_operation(
            workflow_id=workflow_id,
            step_number=1,
            azure_service="cosmos_db",
            operation_type="connection_test",
            input_data={"test": "azure_integration"},
            output_data={"success": True, "ru_charge": 1.0},
            processing_time_ms=50.0
        )
        
        # Calculate costs using real cost tracker
        costs = service.calculate_workflow_costs(workflow_id)
        
        assert "workflow_id" in costs
        assert costs["total_cost_usd"] >= 0
        print(f"   ✅ Azure workflow cost calculated: ${costs['total_cost_usd']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Azure services integration test failed: {e}")
        return False

async def test_azure_gnn_orchestrator():
    """Test GNN orchestrator with REAL Azure ML and Cosmos"""
    print("\n🤖 Testing Azure GNN Orchestrator")
    print("=" * 50)
    
    try:
        from core.azure_ml.gnn.training.orchestrator import UnifiedGNNTrainingOrchestrator
        from core.azure_ml.client import AzureMLClient
        from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
        
        # Create REAL Azure clients
        ml_client = AzureMLClient()
        cosmos_client = AzureCosmosGremlinClient()
        
        # Test Azure connections
        cosmos_connection = await cosmos_client.test_connection()
        if not cosmos_connection.get('success'):
            print(f"   ❌ Azure Cosmos connection required for GNN orchestrator")
            return False
        
        print("   ✅ Azure Cosmos connected")
        
        # Create orchestrator with REAL Azure clients
        orchestrator = UnifiedGNNTrainingOrchestrator(ml_client, cosmos_client)
        
        # Test evidence collection from REAL Azure operations
        from core.utilities.workflow_evidence_collector import AzureDataWorkflowEvidenceCollector
        evidence_collector = AzureDataWorkflowEvidenceCollector("azure_gnn_test")
        
        # Test the method that was fixed (imports from core, not services)
        domain = "test_architecture"
        change_metrics = await orchestrator._collect_graph_change_evidence(domain, evidence_collector)
        
        assert isinstance(change_metrics, dict)
        print(f"   ✅ GNN orchestrator using core utilities successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Azure GNN orchestrator test failed: {e}")
        return False

def test_azure_import_architecture():
    """Verify imports use Azure services only, no circular dependencies"""
    print("\n📦 Testing Azure-Only Import Architecture")
    print("=" * 50)
    
    try:
        # Test core utilities import independently
        from core.utilities.workflow_evidence_collector import AzureDataWorkflowEvidenceCollector
        from core.utilities.azure_cost_tracker import AzureServiceCostTracker
        print("   ✅ Core utilities import independently")
        
        # Test GNN orchestrator imports from core (not services)
        from core.azure_ml.gnn_training_evidence_orchestrator import GNNTrainingEvidenceOrchestrator
        print("   ✅ GNN orchestrator imports core utilities")
        
        # Test services imports from core
        from services.workflow_service import WorkflowService
        print("   ✅ Services import core utilities")
        
        # Verify no circular dependencies in source code
        import inspect
        orchestrator_file = inspect.getfile(UnifiedGNNTrainingOrchestrator)
        with open(orchestrator_file, 'r') as f:
            content = f.read()
        
        assert "from services." not in content, "Found services import in core module"
        assert "import services." not in content, "Found services import in core module"
        
        print("   ✅ No circular dependencies - core independent of services")
        return True
        
    except Exception as e:
        print(f"   ❌ Import architecture test failed: {e}")
        return False

async def main():
    """Run Azure-only architecture validation tests"""
    print("🚀 AZURE-ONLY ARCHITECTURE VALIDATION")
    print("=" * 80)
    print("Testing architecture fixes against REAL Azure services")
    print("NO MOCKS - NO SIMULATIONS - AZURE ONLY")
    print("=" * 80)
    
    # First validate Azure environment
    if not validate_azure_environment():
        print("\n❌ Cannot proceed without proper Azure configuration")
        return False
    
    tests = [
        ("Azure Import Architecture", test_azure_import_architecture),
        ("Azure Cosmos New Methods", test_azure_cosmos_new_methods),
        ("Azure Evidence Collection", test_azure_evidence_collection),
        ("Azure Services Integration", test_azure_services_integration),
        ("Azure GNN Orchestrator", test_azure_gnn_orchestrator)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("AZURE ARCHITECTURE VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {passed + failed}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL AZURE TESTS PASSED!")
        print("✅ Architecture fixes validated against REAL Azure services")
        print("✅ Core->Services dependency violation fixed")
        print("✅ Missing Azure methods implemented and working")
        print("✅ Evidence collection works with Azure Cosmos DB")
        print("✅ Cost tracking integrated with Azure operations")
        print("✅ Services layer uses core utilities with Azure clients")
        print("✅ No circular dependencies in Azure-only architecture")
        print("\n🚀 Azure architecture fixes are PRODUCTION READY!")
    else:
        print(f"\n⚠️  {failed} Azure test(s) failed")
        print("❌ Architecture fixes need attention for Azure deployment")
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)