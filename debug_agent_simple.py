#!/usr/bin/env python3
"""
Simple debug script to identify agent test failures
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

async def debug_agents():
    print("🔍 Debug Agent Test Failures")
    print("=" * 50)

    print("\n1. Environment Variables:")
    env_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_DEPLOYMENT_NAME', 
        'OPENAI_MODEL_DEPLOYMENT',
        'USE_MANAGED_IDENTITY',
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            display = f"{value[:30]}..." if len(value) > 30 else value
            print(f"✅ {var}: {display}")
        else:
            print(f"❌ {var}: NOT SET")

    print("\n2. Testing Basic Imports:")
    try:
        from agents.domain_intelligence.agent import domain_intelligence_agent
        print("✅ Domain Intelligence Agent: Import successful")
    except Exception as e:
        print(f"❌ Domain Intelligence Agent: Import failed - {e}")
        return

    print("\n3. Testing Universal Dependencies:")
    try:
        from agents.core.universal_deps import get_universal_deps
        deps = await get_universal_deps()
        print("✅ Universal Dependencies: Initialization successful")
        
        # Check service status
        service_status = await deps._get_service_status()
        print(f"   Service Status: {service_status}")
        
    except Exception as e:
        print(f"❌ Universal Dependencies: Failed - {e}")
        print(f"   Error type: {type(e)}")
        return

    print("\n4. Testing Simple Agent Execution:")
    try:
        result = await domain_intelligence_agent.run(
            "Analyze this simple text: Python programming.", 
            deps=deps
        )
        print("✅ Agent Execution: Successful")
        print(f"   Result Type: {type(result)}")
        
    except Exception as e:
        print(f"❌ Agent Execution: Failed - {e}")
        error_msg = str(e).lower()
        
        if '404' in error_msg or 'not found' in error_msg:
            print("💡 Likely issue: Model deployment name not found")
        elif 'authentication' in error_msg or 'credential' in error_msg:
            print("💡 Likely issue: Azure authentication problem")
        elif 'connection' in error_msg or 'timeout' in error_msg:
            print("💡 Likely issue: Network connectivity problem")
        
        return

    print("\n✅ All basic tests passed!")

if __name__ == "__main__":
    asyncio.run(debug_agents())