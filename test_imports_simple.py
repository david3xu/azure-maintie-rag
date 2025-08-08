#!/usr/bin/env python3
"""
Test imports to identify the exact issue
"""
import sys
sys.path.insert(0, '/workspace/azure-maintie-rag')

print("Testing imports step by step...")

print("1. Testing universal_models imports:")
try:
    from agents.core.universal_models import UniversalDomainCharacteristics
    print("✅ UniversalDomainCharacteristics: OK")
except Exception as e:
    print(f"❌ UniversalDomainCharacteristics: {e}")

try:
    from agents.core.universal_models import UniversalProcessingConfiguration
    print("✅ UniversalProcessingConfiguration: OK")
except Exception as e:
    print(f"❌ UniversalProcessingConfiguration: {e}")

try:
    from agents.core.universal_models import UniversalDomainAnalysis
    print("✅ UniversalDomainAnalysis: OK")
except Exception as e:
    print(f"❌ UniversalDomainAnalysis: {e}")

print("\n2. Testing domain intelligence agent import:")
try:
    from agents.domain_intelligence.agent import domain_intelligence_agent
    print("✅ domain_intelligence_agent: OK")
    print(f"   Type: {type(domain_intelligence_agent)}")
except Exception as e:
    print(f"❌ domain_intelligence_agent: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing agent attributes:")
try:
    from agents.domain_intelligence.agent import domain_intelligence_agent
    print(f"   Has run: {hasattr(domain_intelligence_agent, 'run')}")
    print(f"   Has system_prompt: {hasattr(domain_intelligence_agent, 'system_prompt')}")
except Exception as e:
    print(f"❌ Agent attributes test: {e}")

print("Done.")