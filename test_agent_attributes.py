#!/usr/bin/env python3
"""
Test Agent Attributes - Understand PydanticAI internal structure
"""

import os
import sys
sys.path.insert(0, '/workspace/azure-maintie-rag')

os.environ['OPENAI_MODEL_DEPLOYMENT'] = 'gpt-4o'
os.environ['OPENAI_API_KEY'] = 'dummy-key-for-testing'

from agents.domain_intelligence.pydantic_ai_agent import agent as domain_agent

print("Domain Agent attributes:")
for attr in sorted(dir(domain_agent)):
    if not attr.startswith('__'):
        value = getattr(domain_agent, attr)
        print(f"  {attr}: {type(value)} = {value}")
        
print("\nFunction tools:")
if hasattr(domain_agent, '_function_toolset'):
    print(f"  _function_toolset: {domain_agent._function_toolset}")
    if hasattr(domain_agent._function_toolset, 'functions'):
        print(f"    functions: {domain_agent._function_toolset.functions}")

print("\nModel info:")  
if hasattr(domain_agent, 'model'):
    print(f"  model: {domain_agent.model}")