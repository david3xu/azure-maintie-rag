import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

REQUIRED_VARS = [
    "AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_KEY", "AZURE_BLOB_CONTAINER", "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_ML_STORAGE_ACCOUNT", "AZURE_ML_STORAGE_KEY", "AZURE_ML_BLOB_CONTAINER", "AZURE_ML_STORAGE_CONNECTION_STRING",
    "AZURE_APP_STORAGE_ACCOUNT", "AZURE_APP_STORAGE_KEY", "AZURE_APP_BLOB_CONTAINER", "AZURE_APP_STORAGE_CONNECTION_STRING",
    "AZURE_RESOURCE_PREFIX", "AZURE_ENVIRONMENT", "AZURE_REGION",
    "OPENAI_API_TYPE", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_API_BASE", "OPENAI_API_VERSION", "OPENAI_DEPLOYMENT_NAME",
    "AZURE_TEXT_ANALYTICS_ENDPOINT", "AZURE_TEXT_ANALYTICS_KEY",
    "AZURE_SEARCH_SERVICE", "AZURE_SEARCH_ADMIN_KEY", "AZURE_SEARCH_QUERY_KEY", "AZURE_SEARCH_SERVICE_NAME", "AZURE_SEARCH_API_VERSION",
    "AZURE_COSMOS_ENDPOINT", "AZURE_COSMOS_KEY", "AZURE_COSMOS_DATABASE", "AZURE_COSMOS_CONTAINER", "AZURE_COSMOS_API_VERSION", "AZURE_COSMOS_DB_CONNECTION_STRING",
    "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ML_WORKSPACE", "AZURE_ML_WORKSPACE_NAME", "AZURE_ML_API_VERSION", "AZURE_TENANT_ID"
]

PLACEHOLDER_STRINGS = [
    "[REPLACE_WITH_ACTUAL", "[YOUR_ACTUAL", "1234567890", "https://clu-project-foundry-instance.openai.azure.com/", "universal-rag-data", "ml-models", "app-data", "maintie", "dev", "eastus"
]

def is_placeholder(value):
    if not value:
        return True
    for ph in PLACEHOLDER_STRINGS:
        if ph in value:
            return True
    return False

missing = []
placeholder = []

for var in REQUIRED_VARS:
    val = os.getenv(var)
    if not val:
        missing.append(var)
    elif is_placeholder(val):
        placeholder.append((var, val))

if missing:
    print("❌ Missing required environment variables:")
    for var in missing:
        print(f"  - {var}")
else:
    print("✅ No missing required environment variables.")

if placeholder:
    print("❌ Variables with placeholder or default values:")
    for var, val in placeholder:
        print(f"  - {var}: {val}")
else:
    print("✅ No variables with placeholder/default values.")

if not missing and not placeholder:
    print("🎉 All required environment variables are set and have real values!")