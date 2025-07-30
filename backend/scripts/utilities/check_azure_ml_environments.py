#!/usr/bin/env python3
"""
Check available Azure ML environments
"""
import sys
import os

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def check_environments():
    """Check available Azure ML environments"""
    try:
        from core.azure_ml.client import AzureMLClient
        
        ml_wrapper = AzureMLClient()
        ml_client = ml_wrapper.ml_client
        
        print("🔍 Checking available Azure ML environments...")
        
        # List environments
        environments = list(ml_client.environments.list())
        
        print(f"📋 Found {len(environments)} environments:")
        
        pytorch_envs = []
        for env in environments:
            print(f"   - {env.name} (v{env.version})")
            if 'pytorch' in env.name.lower():
                pytorch_envs.append(f"{env.name}:{env.version}")
        
        print(f"\n🔥 PyTorch environments found: {len(pytorch_envs)}")
        for env in pytorch_envs[:5]:  # Show first 5
            print(f"   - {env}")
        
        # Get a working environment
        if environments:
            first_env = environments[0]
            working_env = f"{first_env.name}:{first_env.version}"
            print(f"\n✅ Recommended environment: {working_env}")
            return working_env
        else:
            print("❌ No environments found!")
            return None
            
    except Exception as e:
        print(f"❌ Error checking environments: {e}")
        return None

if __name__ == "__main__":
    check_environments()