#!/usr/bin/env python3
"""
Load .env file and setup Azure ML with interactive authentication
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    """Load .env file"""
    env_file = Path(__file__).parent.parent / ".env"
    
    if not env_file.exists():
        print(f"❌ .env file not found at: {env_file}")
        return False
    
    print(f"📋 Loading environment from: {env_file}")
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
    
    print(f"✅ Environment variables loaded")
    return True

def main():
    """Load env and setup Azure ML"""
    
    print("🔧 LOADING ENVIRONMENT AND SETTING UP AZURE ML")
    print("=" * 60)
    
    # Load .env file
    if not load_env_file():
        return
    
    # Check key variables
    required_vars = ['AZURE_SUBSCRIPTION_ID', 'AZURE_RESOURCE_GROUP', 'AZURE_ML_WORKSPACE_NAME']
    for var in required_vars:
        value = os.environ.get(var, 'NOT_SET')
        print(f"   {var}: {value}")
    
    print("\n" + "=" * 60)
    
    # Now run the Azure ML setup
    print("🚀 Starting Azure ML setup...")
    
    try:
        # Import and run the interactive setup
        sys.path.append(str(Path(__file__).parent))
        from setup_azure_ml_interactive import main as setup_main
        setup_main()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()