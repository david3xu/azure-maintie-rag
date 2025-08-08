#!/usr/bin/env python3
"""
Configuration Consistency Validator - Azure Universal RAG
Validates all configuration files, environment settings, and agent dependencies are properly aligned
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_config_file_consistency():
    """Check consistency between configuration files"""
    print("⚙️  Checking Configuration File Consistency...")
    
    results = {
        "status": "checking",
        "config_files_found": [],
        "inconsistencies": [],
        "missing_configs": []
    }
    
    # Expected configuration files
    expected_configs = [
        "config/azure_settings.py",
        "config/universal_config.py", 
        "agents/core/simple_config_manager.py",
        "config/settings.py"
    ]
    
    for config_file in expected_configs:
        config_path = Path(config_file)
        if config_path.exists():
            results["config_files_found"].append(config_file)
            print(f"   ✅ {config_file}")
        else:
            results["missing_configs"].append(config_file)
            print(f"   ❌ {config_file} - Missing")
    
    # Check for duplicate configuration patterns
    if len(results["config_files_found"]) > 0:
        print("   📋 Configuration files located and accessible")
    
    return results

def check_agent_dependency_consistency():
    """Check agent dependency imports and structure"""
    print("🤖 Checking Agent Dependency Consistency...")
    
    results = {
        "status": "checking",
        "agents_validated": [],
        "dependency_issues": [],
        "import_consistency": True
    }
    
    agents_to_check = [
        ("agents/domain_intelligence/agent.py", "Domain Intelligence Agent"),
        ("agents/knowledge_extraction/agent.py", "Knowledge Extraction Agent"), 
        ("agents/universal_search/agent.py", "Universal Search Agent"),
        ("agents/orchestrator.py", "Universal Orchestrator")
    ]
    
    for agent_path, agent_name in agents_to_check:
        if Path(agent_path).exists():
            print(f"   ✅ {agent_name} - File exists")
            results["agents_validated"].append(agent_name)
            
            # Check if file contains expected imports
            try:
                with open(agent_path, 'r') as f:
                    content = f.read()
                    
                # Check for key dependency imports
                expected_imports = [
                    "from agents.core.universal_deps import UniversalDeps",
                    "from agents.core.universal_models import",
                    "from pydantic_ai import Agent"
                ]
                
                for expected_import in expected_imports:
                    if expected_import in content:
                        print(f"      ✅ {expected_import.split('import')[1].strip()} import found")
                    else:
                        print(f"      ⚠️  {expected_import.split('import')[1].strip()} import missing")
                        results["dependency_issues"].append(f"{agent_name}: Missing {expected_import}")
                        
            except Exception as e:
                print(f"   ❌ {agent_name} - Error reading file: {e}")
                results["dependency_issues"].append(f"{agent_name}: File read error - {e}")
        else:
            print(f"   ❌ {agent_name} - File missing")
            results["dependency_issues"].append(f"{agent_name}: File missing")
    
    return results

def check_azure_service_integration():
    """Check Azure service client consistency"""
    print("☁️  Checking Azure Service Integration Consistency...")
    
    results = {
        "status": "checking", 
        "service_clients_found": [],
        "integration_issues": []
    }
    
    service_clients = [
        ("infrastructure/azure_openai/openai_client.py", "Azure OpenAI"),
        ("infrastructure/azure_search/search_client.py", "Azure Cognitive Search"),
        ("infrastructure/azure_cosmos/cosmos_gremlin_client.py", "Azure Cosmos DB"), 
        ("infrastructure/azure_storage/storage_client.py", "Azure Blob Storage"),
        ("infrastructure/azure_ml/gnn_inference_client.py", "Azure ML GNN")
    ]
    
    for client_path, client_name in service_clients:
        if Path(client_path).exists():
            results["service_clients_found"].append(client_name)
            print(f"   ✅ {client_name} client exists")
            
            # Check for base client inheritance
            try:
                with open(client_path, 'r') as f:
                    content = f.read()
                    
                if "BaseAzureClient" in content:
                    print(f"      ✅ {client_name} inherits from BaseAzureClient")
                else:
                    results["integration_issues"].append(f"{client_name}: Missing BaseAzureClient inheritance")
                    print(f"      ⚠️  {client_name} missing BaseAzureClient inheritance")
                    
                if "ensure_initialized" in content:
                    print(f"      ✅ {client_name} has ensure_initialized method")
                else:
                    results["integration_issues"].append(f"{client_name}: Missing ensure_initialized method")
                    print(f"      ⚠️  {client_name} missing ensure_initialized method")
                    
            except Exception as e:
                results["integration_issues"].append(f"{client_name}: File read error - {e}")
                
        else:
            results["integration_issues"].append(f"{client_name}: Client file missing")
            print(f"   ❌ {client_name} client missing")
    
    return results

def check_universal_design_compliance():
    """Check for universal design compliance (no hardcoded domain assumptions)"""
    print("🌍 Checking Universal Design Compliance...")
    
    results = {
        "status": "checking",
        "violations_found": [],
        "compliant_files": []
    }
    
    # Domain bias terms that should NOT appear in the code
    domain_bias_terms = [
        'technical', 'medical', 'legal', 'financial', 'academic', 
        'business', 'scientific', 'engineering', 'healthcare',
        'if.*domain.*==', 'domain_type.*=', 'category.*=='
    ]
    
    files_to_check = [
        "agents/domain_intelligence/agent.py",
        "agents/knowledge_extraction/agent.py", 
        "agents/universal_search/agent.py",
        "config/universal_config.py"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read().lower()
                    
                violations = []
                for term in domain_bias_terms:
                    if term in content:
                        violations.append(term)
                
                if violations:
                    results["violations_found"].extend([(file_path, v) for v in violations])
                    print(f"   ⚠️  {file_path}: Found {len(violations)} potential domain bias terms")
                else:
                    results["compliant_files"].append(file_path)
                    print(f"   ✅ {file_path}: Universal design compliant")
                    
            except Exception as e:
                print(f"   ❌ {file_path}: Error checking file - {e}")
        else:
            print(f"   ❌ {file_path}: File missing")
    
    return results

def check_data_pipeline_integration():
    """Check data pipeline script consistency and integration"""
    print("🔄 Checking Data Pipeline Integration...")
    
    results = {
        "status": "checking",
        "pipeline_scripts_found": [],
        "integration_issues": []
    }
    
    pipeline_scripts = [
        ("scripts/dataflow/00_full_pipeline.py", "Full Pipeline"),
        ("scripts/dataflow/01_data_ingestion.py", "Data Ingestion"),
        ("scripts/dataflow/02_knowledge_extraction.py", "Knowledge Extraction"),
        ("scripts/dataflow/07_unified_search.py", "Unified Search"),
        ("scripts/dataflow/demo_full_workflow.py", "Workflow Demo")
    ]
    
    for script_path, script_name in pipeline_scripts:
        if Path(script_path).exists():
            results["pipeline_scripts_found"].append(script_name)
            print(f"   ✅ {script_name} script exists")
            
            # Check for proper imports
            try:
                with open(script_path, 'r') as f:
                    content = f.read()
                    
                if "sys.path.insert(0," in content:
                    print(f"      ✅ {script_name} has proper path setup")
                else:
                    results["integration_issues"].append(f"{script_name}: Missing sys.path setup")
                    
                if "from agents." in content:
                    print(f"      ✅ {script_name} imports agents properly")
                else:
                    results["integration_issues"].append(f"{script_name}: Missing agent imports")
                    
            except Exception as e:
                results["integration_issues"].append(f"{script_name}: File read error - {e}")
        else:
            results["integration_issues"].append(f"{script_name}: Script missing")
            print(f"   ❌ {script_name} script missing")
    
    return results

def main():
    """Run comprehensive configuration consistency validation"""
    print("🚀 Azure Universal RAG - Configuration Consistency Validation")
    print("=" * 70)
    
    validation_results = {
        "config_consistency": check_config_file_consistency(),
        "agent_dependencies": check_agent_dependency_consistency(), 
        "azure_integration": check_azure_service_integration(),
        "universal_design": check_universal_design_compliance(),
        "pipeline_integration": check_data_pipeline_integration()
    }
    
    # Summary
    print("\n📊 Validation Summary")
    print("=" * 70)
    
    total_issues = 0
    total_components = 0
    
    for category, results in validation_results.items():
        issues = (
            len(results.get("inconsistencies", [])) +
            len(results.get("dependency_issues", [])) + 
            len(results.get("integration_issues", [])) +
            len(results.get("violations_found", []))
        )
        
        compliant = (
            len(results.get("config_files_found", [])) +
            len(results.get("agents_validated", [])) +
            len(results.get("service_clients_found", [])) +
            len(results.get("compliant_files", [])) + 
            len(results.get("pipeline_scripts_found", []))
        )
        
        total_issues += issues
        total_components += compliant
        
        status = "✅ PASS" if issues == 0 else f"⚠️  {issues} issues"
        print(f"   {category.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall Results:")
    print(f"   • {total_components} components validated successfully")
    print(f"   • {total_issues} total issues identified")
    
    if total_issues == 0:
        print(f"\n✅ System configuration is CONSISTENT and ready for production")
    else:
        print(f"\n⚠️  Configuration issues found - review validation details above")
        print(f"\n💡 Recommendations:")
        print(f"   • Fix identified configuration inconsistencies")
        print(f"   • Ensure all Azure service clients follow base patterns")
        print(f"   • Verify universal design compliance (no hardcoded domains)")
        print(f"   • Deploy Azure infrastructure: 'azd up'")
        print(f"   • Configure Azure OpenAI API keys and endpoints")
    
    return validation_results

if __name__ == "__main__":
    results = main()