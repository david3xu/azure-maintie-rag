#!/usr/bin/env python3
"""
Setup Azure ML with Managed Identity (Alternative to Service Principal)
Use this if you don't have Service Principal credentials
"""

import os
import sys
from pathlib import Path

def setup_managed_identity_auth():
    """Setup Azure ML using Managed Identity instead of Service Principal"""
    
    print("🔐 AZURE ML WITH MANAGED IDENTITY")
    print("=" * 50)
    
    try:
        from azure.identity import DefaultAzureCredential
        from azureml.core import Workspace
        
        print("✅ Using DefaultAzureCredential (Managed Identity)")
        
        # Use DefaultAzureCredential which tries multiple auth methods
        credential = DefaultAzureCredential()
        
        # Connect to workspace using Managed Identity
        workspace = Workspace(
            subscription_id=os.environ['AZURE_SUBSCRIPTION_ID'],
            resource_group=os.environ['AZURE_RESOURCE_GROUP'], 
            workspace_name=os.environ['AZURE_ML_WORKSPACE_NAME'],
            auth=credential
        )
        
        print(f"✅ Connected to workspace: {workspace.name}")
        return workspace
        
    except Exception as e:
        print(f"❌ Managed Identity setup failed: {e}")
        print("\n💡 Alternatives:")
        print("1. Run on Azure VM with Managed Identity enabled")
        print("2. Use Azure Cloud Shell")
        print("3. Get Service Principal credentials from Azure admin")
        return None

def main():
    """Test Managed Identity setup"""
    
    # Check required environment variables
    required_vars = [
        'AZURE_SUBSCRIPTION_ID',
        'AZURE_RESOURCE_GROUP',
        'AZURE_ML_WORKSPACE_NAME'
    ]
    
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        return
    
    workspace = setup_managed_identity_auth()
    
    if workspace:
        print(f"\n🎉 MANAGED IDENTITY SETUP SUCCESSFUL!")
        print(f"🚀 Ready for Azure ML training without Service Principal!")
    else:
        print(f"\n❌ MANAGED IDENTITY SETUP FAILED")

if __name__ == "__main__":
    main()