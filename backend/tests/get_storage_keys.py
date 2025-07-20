#!/usr/bin/env python3
"""Script to help get Azure Storage account keys"""

import subprocess
import json
import sys

def get_storage_keys_azure_cli():
    """Get storage account keys using Azure CLI"""
    print("🔑 Getting Azure Storage Account Keys...")
    print("=" * 60)

    storage_accounts = [
        {
            "name": "maintiedevmlstor1cdd8e11",
            "purpose": "RAG Data & ML Models",
            "env_var": "AZURE_STORAGE_KEY"
        },
        {
            "name": "maintiedevstor1cdd8e11",
            "purpose": "Application Data",
            "env_var": "AZURE_APP_STORAGE_KEY"
        }
    ]

    for account in storage_accounts:
        print(f"\n📦 Storage Account: {account['name']}")
        print(f"🎯 Purpose: {account['purpose']}")
        print(f"🔧 Environment Variable: {account['env_var']}")
        print("-" * 40)

        try:
            # Get keys
            result = subprocess.run([
                "az", "storage", "account", "keys", "list",
                "--account-name", account["name"],
                "--resource-group", "maintie-rag-rg",
                "--output", "json"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                keys_data = json.loads(result.stdout)
                if keys_data:
                    key1 = keys_data[0].get("value", "NOT_FOUND")
                    print(f"✅ Key 1: {key1}")
                    print(f"💡 Add to your .env file: {account['env_var']}={key1}")
                else:
                    print("❌ No keys found")
            else:
                print(f"❌ Error getting keys: {result.stderr}")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Make sure you're logged in to Azure CLI:")
            print("   az login")
            print("   az account set --subscription ccc6af52-5928-4dbe-8ceb-fa794974a30f")

def get_connection_strings_azure_cli():
    """Get connection strings using Azure CLI"""
    print("\n🔗 Getting Connection Strings...")
    print("=" * 60)

    storage_accounts = [
        {
            "name": "maintiedevmlstor1cdd8e11",
            "purpose": "RAG Data & ML Models",
            "env_var": "AZURE_STORAGE_CONNECTION_STRING"
        },
        {
            "name": "maintiedevstor1cdd8e11",
            "purpose": "Application Data",
            "env_var": "AZURE_APP_STORAGE_CONNECTION_STRING"
        }
    ]

    for account in storage_accounts:
        print(f"\n📦 Storage Account: {account['name']}")
        print(f"🎯 Purpose: {account['purpose']}")
        print(f"🔧 Environment Variable: {account['env_var']}")
        print("-" * 40)

        try:
            # Get connection string
            result = subprocess.run([
                "az", "storage", "account", "show-connection-string",
                "--name", account["name"],
                "--resource-group", "maintie-rag-rg",
                "--output", "json"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                conn_data = json.loads(result.stdout)
                connection_string = conn_data.get("connectionString", "NOT_FOUND")
                print(f"✅ Connection String: {connection_string}")
                print(f"💡 Add to your .env file: {account['env_var']}={connection_string}")
            else:
                print(f"❌ Error getting connection string: {result.stderr}")

        except Exception as e:
            print(f"❌ Error: {e}")

def manual_instructions():
    """Show manual instructions for getting keys"""
    print("\n📋 Manual Instructions (if Azure CLI doesn't work):")
    print("=" * 60)

    print("\n1️⃣ Go to Azure Portal: https://portal.azure.com")
    print("2️⃣ Navigate to Storage accounts")
    print("3️⃣ Find your storage accounts:")
    print("   - maintiedevmlstor1cdd8e11 (RAG & ML)")
    print("   - maintiedevstor1cdd8e11 (App Data)")
    print("4️⃣ Click on each storage account")
    print("5️⃣ Go to Access keys in the left menu")
    print("6️⃣ Click 'Show' next to key1")
    print("7️⃣ Click 'Copy' to copy the key")
    print("8️⃣ Paste into your .env file")

    print("\n📝 Example .env file entries:")
    print("AZURE_STORAGE_KEY=your-actual-key-here")
    print("AZURE_APP_STORAGE_KEY=your-actual-key-here")
    print("AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=maintiedevmlstor1cdd8e11;AccountKey=your-actual-key;EndpointSuffix=core.windows.net")
    print("AZURE_APP_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=maintiedevstor1cdd8e11;AccountKey=your-actual-key;EndpointSuffix=core.windows.net")

if __name__ == "__main__":
    print("🚀 Azure Storage Key Helper")
    print("=" * 60)

    # Try to get keys via Azure CLI
    try:
        get_storage_keys_azure_cli()
        get_connection_strings_azure_cli()
    except Exception as e:
        print(f"❌ Azure CLI not available: {e}")

    # Show manual instructions
    manual_instructions()

    print("\n🎯 Summary:")
    print("- Storage Key = AccountKey (same thing)")
    print("- Get keys from Azure Portal or Azure CLI")
    print("- Update your .env file with the actual keys")
    print("- Test with: python backend/test_storage_integration.py")