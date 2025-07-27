#!/usr/bin/env python3
"""
Validate Azure ML Connection
Test connection to Azure ML and prepare for real training
"""

import os
import sys
from pathlib import Path

def check_azure_credentials():
    """Check if Azure credentials are set"""
    
    required_vars = [
        'AZURE_SUBSCRIPTION_ID',
        'AZURE_RESOURCE_GROUP',
        'AZURE_TENANT_ID', 
        'AZURE_CLIENT_ID',
        'AZURE_CLIENT_SECRET'
    ]
    
    print("🔐 Checking Azure credentials...")
    
    missing = []
    for var in required_vars:
        if not os.environ.get(var):
            missing.append(var)
        else:
            print(f"   ✅ {var}: {os.environ[var][:8]}...")
    
    if missing:
        print("❌ Missing credentials:")
        for var in missing:
            print(f"   • {var}")
        return False
    
    print("✅ All Azure credentials present")
    return True

def test_azure_ml_import():
    """Test Azure ML SDK import"""
    
    print("📦 Testing Azure ML SDK...")
    
    try:
        import azureml.core
        from azureml.core import Workspace
        from azureml.core.authentication import ServicePrincipalAuthentication
        print(f"✅ Azure ML SDK version: {azureml.core.VERSION}")
        return True
    except ImportError as e:
        print(f"❌ Azure ML SDK import failed: {e}")
        print("💡 Install with: pip install azureml-sdk")
        return False

def test_azure_connection():
    """Test connection to Azure ML"""
    
    if not check_azure_credentials():
        return False
    
    if not test_azure_ml_import():
        return False
    
    try:
        from azureml.core import Workspace
        from azureml.core.authentication import ServicePrincipalAuthentication
        
        print("🔗 Testing Azure ML connection...")
        
        # Create authentication
        auth = ServicePrincipalAuthentication(
            tenant_id=os.environ['AZURE_TENANT_ID'],
            service_principal_id=os.environ['AZURE_CLIENT_ID'],
            service_principal_password=os.environ['AZURE_CLIENT_SECRET']
        )
        
        # Try to connect (this will create workspace if it doesn't exist)
        workspace_name = "gnn-training-workspace"
        
        try:
            workspace = Workspace.get(
                name=workspace_name,
                subscription_id=os.environ['AZURE_SUBSCRIPTION_ID'],
                resource_group=os.environ['AZURE_RESOURCE_GROUP'],
                auth=auth
            )
            print(f"✅ Connected to existing workspace: {workspace_name}")
        except:
            print(f"🔄 Creating workspace: {workspace_name}")
            workspace = Workspace.create(
                name=workspace_name,
                subscription_id=os.environ['AZURE_SUBSCRIPTION_ID'],
                resource_group=os.environ['AZURE_RESOURCE_GROUP'],
                location="East US",
                auth=auth
            )
            print(f"✅ Created workspace: {workspace_name}")
        
        print(f"📍 Workspace details:")
        print(f"   • Name: {workspace.name}")
        print(f"   • Location: {workspace.location}")
        print(f"   • Resource Group: {workspace.resource_group}")
        
        return workspace
        
    except Exception as e:
        print(f"❌ Azure connection failed: {e}")
        return None

def main():
    """Main validation"""
    
    print("☁️ AZURE ML CONNECTION VALIDATION")
    print("=" * 50)
    
    # Test connection
    workspace = test_azure_connection()
    
    if workspace:
        print(f"\n🎉 AZURE ML CONNECTION SUCCESSFUL!")
        print(f"🚀 Ready for real Azure ML training!")
        
        # Save workspace config for training scripts
        config_dir = Path(__file__).parent.parent / "azure_ml_config"
        config_dir.mkdir(exist_ok=True)
        
        workspace.write_config(path=config_dir)
        print(f"💾 Workspace config saved to: {config_dir}")
        
        print(f"\n🔥 NEXT STEPS - REAL AZURE ML TRAINING:")
        print(f"   python scripts/real_azure_ml_gnn_training.py --partial --wait")
        
        return True
    else:
        print(f"\n❌ AZURE ML CONNECTION FAILED")
        print(f"📋 Setup required:")
        print(f"   1. Set Azure credentials (see azure_credentials_setup.sh)")
        print(f"   2. Install Azure ML SDK: pip install azureml-sdk")
        print(f"   3. Run this script again")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)