#!/usr/bin/env python3
"""
Debug script to isolate agent test failures
"""

import asyncio
import os
import sys

# Set environment
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONPATH'] = '/workspace/azure-maintie-rag'

sys.path.insert(0, '/workspace/azure-maintie-rag')

async def debug_agent():
    try:
        print("🔍 Step 1: Testing imports...")
        from agents.core.universal_deps import get_universal_deps
        from agents.domain_intelligence.agent import domain_intelligence_agent
        print("✅ Imports successful")
        
        print("🔍 Step 2: Testing universal deps...")
        deps = await get_universal_deps()
        print(f"✅ Universal deps created: {type(deps)}")
        
        print("🔍 Step 3: Testing agent properties...")
        assert hasattr(domain_intelligence_agent, "run")
        print("✅ Agent has run method")
        
        print("🔍 Step 4: Testing simple agent call...")
        sample_prompt = "Analyze: Python programming tutorial with code examples"
        
        print(f"   Calling agent with: {sample_prompt}")
        result = await domain_intelligence_agent.run(sample_prompt, deps=deps)
        
        print(f"✅ Agent call completed!")
        print(f"   Result type: {type(result)}")
        print(f"   Has output: {hasattr(result, 'output')}")
        
        if hasattr(result, 'output'):
            output = result.output
            print(f"   Output type: {type(output)}")
            print(f"   Has characteristics: {hasattr(output, 'characteristics')}")
            print(f"   Has domain_signature: {hasattr(output, 'domain_signature')}")
            
            if hasattr(output, 'domain_signature'):
                print(f"   Domain signature: {output.domain_signature}")
            
            if hasattr(output, 'characteristics'):
                chars = output.characteristics
                print(f"   Characteristics type: {type(chars)}")
                print(f"   Has vocab_complexity_ratio: {hasattr(chars, 'vocabulary_complexity_ratio')}")
                
                if hasattr(chars, 'vocabulary_complexity_ratio'):
                    print(f"   Vocab complexity: {chars.vocabulary_complexity_ratio}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed at step: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(debug_agent())
    print(f"\n🎯 Final result: {'SUCCESS' if result else 'FAILED'}")