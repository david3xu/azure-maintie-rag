#!/usr/bin/env python3
"""
Quick environment test for Azure services
"""
import os
import sys
sys.path.insert(0, '/workspace/azure-maintie-rag')

from dotenv import load_dotenv

# Load from prod.env
load_dotenv('/workspace/azure-maintie-rag/config/environments/prod.env')

print("🔍 Environment Check:")
print("=" * 40)

vars_to_check = [
    'AZURE_OPENAI_ENDPOINT',
    'AZURE_OPENAI_DEPLOYMENT_NAME', 
    'OPENAI_MODEL_DEPLOYMENT',
    'USE_MANAGED_IDENTITY',
    'AZURE_COSMOS_ENDPOINT',
    'AZURE_SEARCH_ENDPOINT',
    'AZURE_STORAGE_ACCOUNT'
]

for var in vars_to_check:
    value = os.getenv(var)
    if value:
        display = f"{value[:30]}..." if len(value) > 30 else value
        print(f"✅ {var}: {display}")
    else:
        print(f"❌ {var}: NOT SET")

print("\n🧪 Testing Azure authentication:")
try:
    from azure.identity import DefaultAzureCredential
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    print(f"✅ Azure credential: Working (token length: {len(token.token)})")
except Exception as e:
    print(f"❌ Azure credential: {e}")

print("\n🧪 Testing agent imports:")
try:
    from agents.domain_intelligence.agent import domain_intelligence_agent
    print("✅ Domain Intelligence Agent: Import successful")
except Exception as e:
    print(f"❌ Domain Intelligence Agent: {e}")

try:
    from agents.core.universal_deps import get_universal_deps_sync
    deps = get_universal_deps_sync()
    print("✅ Universal Dependencies: Initialized successfully")
except Exception as e:
    print(f"❌ Universal Dependencies: {e}")