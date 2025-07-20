#!/usr/bin/env python3
"""Test script to check Azure ML workspace configuration"""

import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

def test_azure_ml_config():
    """Test Azure ML workspace configuration"""
    try:
        from config.settings import azure_settings

        print("🔍 Azure ML Configuration Test:")
        print("=" * 50)

        # Check basic settings
        print(f"✅ Subscription ID: {azure_settings.azure_subscription_id}")
        print(f"✅ Resource Group: {azure_settings.azure_resource_group}")
        print(f"✅ ML Workspace: {azure_settings.azure_ml_workspace}")
        print(f"✅ ML Workspace Name: {azure_settings.azure_ml_workspace_name}")
        print(f"✅ API Version: {azure_settings.azure_ml_api_version}")
        print(f"✅ Tenant ID: {azure_settings.azure_tenant_id}")

        # Check if all required fields are set
        required_fields = [
            azure_settings.azure_subscription_id,
            azure_settings.azure_resource_group,
            azure_settings.azure_ml_workspace,
            azure_settings.azure_ml_workspace_name
        ]

        if all(required_fields):
            print("\n✅ All required Azure ML fields are configured!")
        else:
            print("\n❌ Some required fields are missing!")

        return True

    except Exception as e:
        print(f"❌ Azure ML Configuration Test Failed: {e}")
        return False

def test_azure_ml_connection():
    """Test Azure ML workspace connection"""
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
        from config.settings import azure_settings

        print("\n🔗 Testing Azure ML Connection:")
        print("=" * 50)

        # Create ML client
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=azure_settings.azure_subscription_id,
            resource_group_name=azure_settings.azure_resource_group,
            workspace_name=azure_settings.azure_ml_workspace
        )

        # Test connection by getting workspace info
        workspace = ml_client.workspaces.get(azure_settings.azure_ml_workspace)

        print(f"✅ Workspace Name: {workspace.name}")
        print(f"✅ Workspace Location: {workspace.location}")
        print(f"✅ Workspace ID: {workspace.workspace_id}")
        print(f"✅ Discovery URL: {workspace.discovery_url}")
        print(f"✅ MLflow URI: {workspace.mlflow_tracking_uri}")

        print("\n✅ Azure ML connection successful!")
        return True

    except Exception as e:
        print(f"❌ Azure ML Connection Failed: {e}")
        print("💡 Make sure you have Azure CLI installed and are logged in:")
        print("   az login")
        print("   az account set --subscription <your-subscription-id>")
        return False

if __name__ == "__main__":
    print("🚀 Azure ML Configuration Checker")
    print("=" * 60)

    # Test configuration
    config_success = test_azure_ml_config()

    # Test connection
    connection_success = test_azure_ml_connection()

    if config_success and connection_success:
        print("\n🎉 All Azure ML checks passed!")
    else:
        print("\n💥 Some checks failed. Please review the configuration.")