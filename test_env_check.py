#!/usr/bin/env python3
"""
Quick environment variable check to debug test failures
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 Azure Environment Variable Check")
print("=" * 50)

azure_vars = {
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
    "AZURE_SEARCH_INDEX": os.getenv("AZURE_SEARCH_INDEX"),
    "AZURE_COSMOS_ENDPOINT": os.getenv("AZURE_COSMOS_ENDPOINT"),
    "AZURE_STORAGE_ACCOUNT": os.getenv("AZURE_STORAGE_ACCOUNT"),
    "AZURE_ENV_NAME": os.getenv("AZURE_ENV_NAME"),
    
    # Also check alternate variable names used in tests
    "OPENAI_MODEL_DEPLOYMENT": os.getenv("OPENAI_MODEL_DEPLOYMENT"),
    "EMBEDDING_MODEL_DEPLOYMENT": os.getenv("EMBEDDING_MODEL_DEPLOYMENT"),
    "SEARCH_INDEX_NAME": os.getenv("SEARCH_INDEX_NAME"),
    "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "Not Set"),
}

for var_name, var_value in azure_vars.items():
    if var_value:
        # Show partial value for security
        if "KEY" in var_name or "SECRET" in var_name:
            display_value = f"[SET - {len(var_value)} chars]"
        elif len(var_value) > 50:
            display_value = f"{var_value[:30]}...{var_value[-10:]}"
        else:
            display_value = var_value
        print(f"✅ {var_name}: {display_value}")
    else:
        print(f"❌ {var_name}: NOT SET")

print("\n🔍 Authentication Method Check")
print("-" * 30)

# Check which authentication method would be used
test_managed_identity = os.getenv("TEST_USE_MANAGED_IDENTITY", "false").lower() == "true"
print(f"TEST_USE_MANAGED_IDENTITY: {test_managed_identity}")

if test_managed_identity:
    print("Will use: DefaultAzureCredential (Managed Identity)")
else:
    print("Will try: Azure CLI Credential -> API Key fallback")
    
# Quick Azure CLI check
try:
    from azure.identity import AzureCliCredential
    cli_cred = AzureCliCredential()
    token = cli_cred.get_token("https://cognitiveservices.azure.com/.default")
    print("✅ Azure CLI Credential: Available")
except Exception as e:
    print(f"❌ Azure CLI Credential: {e}")