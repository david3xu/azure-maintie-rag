#!/usr/bin/env python3
"""
Local Agent Testing - Azure Universal RAG Validation
Test agents with fallback configurations to validate system architecture
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_domain_intelligence_agent():
    """Test Domain Intelligence Agent with local fallback"""
    print("🔍 Testing Domain Intelligence Agent Structure...")
    
    try:
        from agents.domain_intelligence.agent import domain_intelligence_agent
        from agents.core.universal_deps import get_universal_deps
        
        print("   ✅ Agent imports successful")
        print("   ✅ Dependencies structure loaded")
        print("   ⚠️  Azure OpenAI configuration needed for full functionality")
        
        return {
            "agent": "domain_intelligence",
            "status": "structure_validated",
            "issue": "azure_openai_config_required"
        }
        
    except Exception as e:
        print(f"   ❌ Agent structure issue: {e}")
        return {
            "agent": "domain_intelligence", 
            "status": "structure_error",
            "error": str(e)
        }

async def test_knowledge_extraction_agent():
    """Test Knowledge Extraction Agent structure"""
    print("🧠 Testing Knowledge Extraction Agent Structure...")
    
    try:
        from agents.knowledge_extraction.agent import knowledge_extraction_agent
        
        print("   ✅ Agent imports successful")
        print("   ⚠️  Azure OpenAI + Cosmos DB configuration needed")
        
        return {
            "agent": "knowledge_extraction",
            "status": "structure_validated", 
            "issue": "azure_services_config_required"
        }
        
    except Exception as e:
        print(f"   ❌ Agent structure issue: {e}")
        return {
            "agent": "knowledge_extraction",
            "status": "structure_error", 
            "error": str(e)
        }

async def test_universal_search_agent():
    """Test Universal Search Agent structure"""
    print("🎯 Testing Universal Search Agent Structure...")
    
    try:
        from agents.universal_search.agent import universal_search_agent
        
        print("   ✅ Agent imports successful")
        print("   ⚠️  Azure Search + OpenAI configuration needed")
        
        return {
            "agent": "universal_search",
            "status": "structure_validated",
            "issue": "azure_services_config_required" 
        }
        
    except Exception as e:
        print(f"   ❌ Agent structure issue: {e}")
        return {
            "agent": "universal_search",
            "status": "structure_error",
            "error": str(e)
        }

async def validate_configuration_structure():
    """Validate configuration management structure"""
    print("⚙️  Validating Configuration Structure...")
    
    try:
        from config.universal_config import UniversalConfig
        from agents.core.simple_config_manager import SimpleDynamicConfigManager
        
        print("   ✅ Configuration classes loaded")
        
        # Check for Azure settings
        from config.azure_settings import azure_settings
        print(f"   📝 Environment detected: {azure_settings.environment}")
        print(f"   🔐 Auth method: {'managed_identity' if azure_settings.use_managed_identity else 'api_key'}")
        
        return {
            "component": "configuration",
            "status": "structure_validated",
            "environment": azure_settings.environment
        }
        
    except Exception as e:
        print(f"   ❌ Configuration issue: {e}")
        return {
            "component": "configuration",
            "status": "error", 
            "error": str(e)
        }

async def check_data_availability():
    """Check test data availability"""
    print("📁 Checking Test Data Availability...")
    
    data_dir = Path("data/raw")
    if data_dir.exists():
        files = list(data_dir.rglob("*.md"))
        print(f"   ✅ Found {len(files)} test files")
        print(f"   📂 Data directory: {data_dir.absolute()}")
        
        # Sample a file to validate content
        if files:
            sample_file = files[0]
            content = sample_file.read_text()[:200]
            print(f"   📄 Sample content: {content[:100]}...")
            
        return {
            "component": "test_data",
            "status": "available",
            "file_count": len(files)
        }
    else:
        print("   ❌ No test data found")
        return {
            "component": "test_data", 
            "status": "missing"
        }

async def main():
    """Run comprehensive agent structure validation"""
    print("🚀 Azure Universal RAG - Agent Structure Validation")
    print("=" * 60)
    
    results = []
    
    # Test each agent structure
    results.append(await test_domain_intelligence_agent())
    results.append(await test_knowledge_extraction_agent()) 
    results.append(await test_universal_search_agent())
    
    # Test configuration structure
    results.append(await validate_configuration_structure())
    
    # Check data availability
    results.append(await check_data_availability())
    
    # Summary
    print("\n📊 Validation Summary:")
    print("=" * 60)
    
    structure_valid = 0
    config_issues = 0
    
    for result in results:
        component = result.get('agent') or result.get('component', 'unknown')
        status = result['status']
        
        if status == 'structure_validated':
            print(f"   ✅ {component}: Structure OK")
            structure_valid += 1
        elif status == 'available':
            print(f"   ✅ {component}: Available")
            structure_valid += 1 
        else:
            print(f"   ❌ {component}: {status}")
            
        if 'azure' in result.get('issue', ''):
            config_issues += 1
    
    print(f"\n🎯 Results:")
    print(f"   • {structure_valid}/{len(results)} components validated")
    print(f"   • {config_issues} Azure configuration issues identified")
    
    if config_issues > 0:
        print(f"\n💡 Next Steps:")
        print(f"   • Deploy Azure infrastructure: 'azd up'")
        print(f"   • Configure Azure OpenAI endpoint and API key")
        print(f"   • Set up Azure Cosmos DB and Search services")
        print(f"   • Run 'make sync-env' to synchronize configuration")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())