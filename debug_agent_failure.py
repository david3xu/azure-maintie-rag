#!/usr/bin/env python3
"""
Debug script to identify specific agent test failures
"""
import os
import sys
import asyncio
sys.path.insert(0, '/workspace/azure-maintie-rag')

from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from multiple sources
project_root = Path(__file__).parent
load_dotenv(project_root / ".env")
load_dotenv(project_root / "config" / "environments" / "prod.env")
load_dotenv()

print("🔍 Debug Agent Test Failures")
print("=" * 50)

print("\n1. Environment Variables:")
env_vars = [
    'AZURE_OPENAI_ENDPOINT',
    'AZURE_OPENAI_DEPLOYMENT_NAME', 
    'OPENAI_MODEL_DEPLOYMENT',
    'OPENAI_BASE_URL',
    'USE_MANAGED_IDENTITY',
    'OPENAI_API_VERSION'
]

for var in env_vars:
    value = os.getenv(var)
    if value:
        display = f"{value[:30]}..." if len(value) > 30 else value
        print(f"✅ {var}: {display}")
    else:
        print(f"❌ {var}: NOT SET")

print("\n2. Testing Agent Imports:")
try:
    from agents.domain_intelligence.agent import domain_intelligence_agent
    print("✅ Domain Intelligence Agent: Import successful")
    print(f"   Agent Type: {type(domain_intelligence_agent)}")
    
    # Check agent structure
    has_run = hasattr(domain_intelligence_agent, 'run')
    has_system_prompt = hasattr(domain_intelligence_agent, 'system_prompt')
    print(f"   Has run method: {has_run}")
    print(f"   Has system_prompt: {has_system_prompt}")
    
except Exception as e:
    print(f"❌ Domain Intelligence Agent: Import failed - {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing Universal Dependencies:")
try:
    from agents.core.universal_deps import get_universal_deps
    deps = await get_universal_deps()
    print("✅ Universal Dependencies: Async initialization successful")
    print(f"   Deps type: {type(deps)}")
    print(f"   Initialized: {getattr(deps, '_initialized', 'Unknown')}")
    
    # Test service status
    service_status = await deps._get_service_status()
    print(f"   Service Status: {service_status}")
    
except Exception as e:
    print(f"❌ Universal Dependencies: Failed - {e}")
    import traceback
    traceback.print_exc()

print("\n4. Testing Azure PydanticAI Provider:")
try:
    from agents.core.azure_pydantic_provider import get_azure_openai_model
    model = get_azure_openai_model()
    print("✅ Azure PydanticAI Provider: Model creation successful")
    print(f"   Model Type: {type(model)}")
    print(f"   Model Name: {model.model_name}")
    
except Exception as e:
    print(f"❌ Azure PydanticAI Provider: Failed - {e}")
    import traceback
    traceback.print_exc()

print("\n5. Testing Agent Execution (Simple):")
try:
    from agents.core.universal_deps import get_universal_deps
    from agents.domain_intelligence.agent import domain_intelligence_agent
    
    deps = await get_universal_deps()
    
    # Very simple prompt to test basic functionality
    simple_prompt = "Test basic agent functionality."
    
    print(f"   Running agent with prompt: '{simple_prompt}'")
    result = await domain_intelligence_agent.run(simple_prompt, deps=deps)
    
    print("✅ Agent Execution: Successful")
    print(f"   Result Type: {type(result)}")
    
    if hasattr(result, 'output'):
        print(f"   Output Type: {type(result.output)}")
    else:
        print(f"   Result Value: {result}")
    
except Exception as e:
    print(f"❌ Agent Execution: Failed - {e}")
    error_msg = str(e).lower()
    
    if '404' in error_msg or 'not found' in error_msg:
        print("💡 This suggests model deployment name issue")
    elif 'authentication' in error_msg or 'credential' in error_msg:
        print("💡 This suggests Azure authentication issue")
    elif 'connection' in error_msg or 'timeout' in error_msg:
        print("💡 This suggests network connectivity issue")
    
    import traceback
    traceback.print_exc()

async def main():
    """Main debug function to run all tests."""
    # The debug logic is above - this is just for proper async structure
    pass

if __name__ == "__main__":
    # Run the async parts
    asyncio.run(main())