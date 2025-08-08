#!/bin/bash

# Populate .env with real Azure service values
# This script fetches actual Azure resource information and updates the root .env file

echo "🔧 Populating .env with real Azure service values..."
echo "======================================================"

# Get Azure OpenAI details
echo "📋 Fetching Azure OpenAI configuration..."
OPENAI_RESOURCE_GROUP="rg-maintie-rag-prod"
OPENAI_RESOURCE_NAME="oai-maintie-rag-prod-fymhwfec3ra2w"

# Get OpenAI API key
OPENAI_API_KEY=$(az cognitiveservices account keys list \
    --name "$OPENAI_RESOURCE_NAME" \
    --resource-group "$OPENAI_RESOURCE_GROUP" \
    --query "key1" --output tsv 2>/dev/null)

if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ Azure OpenAI API key retrieved"
else
    echo "❌ Failed to retrieve Azure OpenAI API key"
fi

# Create the root .env file with real values
cat > /workspace/azure-maintie-rag/.env << EOF
# Azure Universal RAG Production Environment
# Auto-generated from real Azure services $(date)

# Azure Environment
AZURE_ENV_NAME="prod"
AZURE_LOCATION="westus2"
AZURE_SUBSCRIPTION_ID="ccc6af52-5928-4dbe-8ceb-fa794974a30f"
AZURE_RESOURCE_GROUP="rg-maintie-rag-prod"

# Azure OpenAI Service (Real Production Service)
AZURE_OPENAI_ENDPOINT="https://maintie-rag-prod-fymhwfec3ra2w.openai.azure.com/"
AZURE_OPENAI_RESOURCE_NAME="oai-maintie-rag-prod-fymhwfec3ra2w"
OPENAI_API_KEY="$OPENAI_API_KEY"
OPENAI_MODEL_DEPLOYMENT="gpt-4o"
EMBEDDING_MODEL_DEPLOYMENT="text-embedding-ada-002"
OPENAI_API_VERSION="2024-02-01"

# Azure Cognitive Search (To be deployed)
AZURE_SEARCH_ENDPOINT=""
AZURE_SEARCH_RESOURCE_NAME=""
SEARCH_INDEX_NAME="maintie-prod-index"

# Azure Cosmos DB (To be deployed)
AZURE_COSMOS_ENDPOINT=""
AZURE_COSMOS_RESOURCE_NAME=""
COSMOS_DATABASE_NAME="maintie-rag-prod"
COSMOS_GRAPH_NAME="knowledge-graph-prod"

# Azure Storage (To be deployed)
AZURE_STORAGE_ENDPOINT=""
AZURE_STORAGE_ACCOUNT_NAME=""
STORAGE_CONTAINER_NAME="documents-prod"

# Azure ML (To be deployed)
AZURE_ML_ENDPOINT=""
AZURE_ML_WORKSPACE_NAME=""

# Application Configuration
USE_MANAGED_IDENTITY="false"
ENVIRONMENT_TYPE="prod"
LOG_LEVEL="INFO"
MAX_WORKERS=4
BACKEND_PORT=8000
CACHE_TTL=600
EOF

echo ""
echo "✅ Root .env file populated with real Azure service values"
echo "🔑 OpenAI API Key: $(echo $OPENAI_API_KEY | cut -c1-10)..."
echo "🌐 OpenAI Endpoint: https://maintie-rag-prod-fymhwfec3ra2w.openai.azure.com/"
echo ""
echo "🚀 Ready to run Azure Universal RAG with real services!"