#!/usr/bin/env python3
"""Simple test to check Azure ML configuration"""

import os
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

def test_environment_variables():
    """Test if environment variables are set correctly"""
    print("🔍 Checking Environment Variables:")
    print("=" * 50)

    # Check Azure ML variables
    ml_vars = {
        "AZURE_SUBSCRIPTION_ID": "ccc6af52-5928-4dbe-8ceb-fa794974a30f",
        "AZURE_RESOURCE_GROUP": "maintie-rag-rg",
        "AZURE_ML_WORKSPACE": "maintie-dev-ml-1cdd8e11",
        "AZURE_ML_WORKSPACE_NAME": "maintie-dev-ml-1cdd8e11",
        "AZURE_ML_API_VERSION": "2023-04-01",
        "AZURE_TENANT_ID": "05894af0-cb28-46d8-8716-74cdb46e2226"
    }

    all_good = True
    for var_name, expected_value in ml_vars.items():
        actual_value = os.getenv(var_name, "NOT_SET")
        if actual_value == expected_value:
            print(f"✅ {var_name}: {actual_value}")
        else:
            print(f"❌ {var_name}: {actual_value} (expected: {expected_value})")
            all_good = False

    return all_good

def test_settings_import():
    """Test if settings can be imported correctly"""
    print("\n🔧 Testing Settings Import:")
    print("=" * 50)

    try:
        from config.settings import azure_settings

        print(f"✅ Settings imported successfully")
        print(f"✅ Subscription ID: {azure_settings.azure_subscription_id}")
        print(f"✅ Resource Group: {azure_settings.azure_resource_group}")
        print(f"✅ ML Workspace: {azure_settings.azure_ml_workspace}")
        print(f"✅ API Version: {azure_settings.azure_ml_api_version}")

        return True

    except Exception as e:
        print(f"❌ Settings import failed: {e}")
        return False

def test_configuration_matches():
    """Test if configuration matches expected values"""
    print("\n🎯 Testing Configuration Match:")
    print("=" * 50)

    try:
        from config.settings import azure_settings

        expected_values = {
            "azure_subscription_id": "ccc6af52-5928-4dbe-8ceb-fa794974a30f",
            "azure_resource_group": "maintie-rag-rg",
            "azure_ml_workspace": "maintie-dev-ml-1cdd8e11",
            "azure_ml_workspace_name": "maintie-dev-ml-1cdd8e11",
            "azure_ml_api_version": "2023-04-01",
            "azure_tenant_id": "05894af0-cb28-46d8-8716-74cdb46e2226"
        }

        all_match = True
        for setting_name, expected_value in expected_values.items():
            actual_value = getattr(azure_settings, setting_name, "NOT_FOUND")
            if actual_value == expected_value:
                print(f"✅ {setting_name}: {actual_value}")
            else:
                print(f"❌ {setting_name}: {actual_value} (expected: {expected_value})")
                all_match = False

        return all_match

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Azure ML Configuration Checker (Simple)")
    print("=" * 60)

    # Test environment variables
    env_success = test_environment_variables()

    # Test settings import
    import_success = test_settings_import()

    # Test configuration match
    config_success = test_configuration_matches()

    print("\n" + "=" * 60)
    if env_success and import_success and config_success:
        print("🎉 All checks passed! Your Azure ML configuration is correct.")
    else:
        print("💥 Some checks failed. Please review your configuration.")

    print("\n📋 Next Steps:")
    print("1. Set up your .env file with the correct values")
    print("2. Install Azure ML SDK: pip install azure-ai-ml")
    print("3. Login to Azure: az login")
    print("4. Test connection with the full test script")