#!/bin/bash

# Auto-Discover and Populate .env with ALL Azure service values
# This script automatically discovers Azure resources and creates a complete .env file

set -e

echo "🔧 AUTO-DISCOVERING Azure Universal RAG Services..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get current azd environment or use prod as default
AZURE_ENV_NAME=$(azd env get-values 2>/dev/null | grep AZURE_ENV_NAME | cut -d'=' -f2 | tr -d '"' || echo "prod")
AZURE_LOCATION=$(azd env get-values 2>/dev/null | grep AZURE_LOCATION | cut -d'=' -f2 | tr -d '"' || echo "westus2")
AZURE_SUBSCRIPTION_ID=$(azd env get-values 2>/dev/null | grep AZURE_SUBSCRIPTION_ID | cut -d'=' -f2 | tr -d '"' || az account show --query id -o tsv)

# Resource group name pattern
RESOURCE_GROUP="rg-maintie-rag-${AZURE_ENV_NAME}"

echo -e "${YELLOW}📋 Environment: $AZURE_ENV_NAME${NC}"
echo -e "${YELLOW}📍 Location: $AZURE_LOCATION${NC}"
echo -e "${YELLOW}🔑 Subscription: $AZURE_SUBSCRIPTION_ID${NC}"
echo -e "${YELLOW}👥 Resource Group: $RESOURCE_GROUP${NC}"
echo ""

# Discover Azure OpenAI
echo "🤖 Discovering Azure OpenAI..."
OPENAI_RESOURCE=$(az cognitiveservices account list --resource-group "$RESOURCE_GROUP" --query "[?kind=='OpenAI'].name" -o tsv 2>/dev/null | head -1)
if [ -n "$OPENAI_RESOURCE" ]; then
    OPENAI_ENDPOINT=$(az cognitiveservices account show --name "$OPENAI_RESOURCE" --resource-group "$RESOURCE_GROUP" --query "properties.endpoint" -o tsv)
    OPENAI_API_KEY=$(az cognitiveservices account keys list --name "$OPENAI_RESOURCE" --resource-group "$RESOURCE_GROUP" --query "key1" --output tsv 2>/dev/null)
    echo -e "   ${GREEN}✅ Found: $OPENAI_RESOURCE${NC}"
    echo "   🔗 Endpoint: $OPENAI_ENDPOINT"
    echo "   🔑 API Key: $(echo $OPENAI_API_KEY | cut -c1-10)..."
else
    echo -e "   ${RED}❌ Azure OpenAI not found${NC}"
    OPENAI_ENDPOINT=""
    OPENAI_API_KEY=""
fi

# Discover Azure Cognitive Search
echo "🔍 Discovering Azure Cognitive Search..."
SEARCH_RESOURCE=$(az search service list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv 2>/dev/null)
if [ -n "$SEARCH_RESOURCE" ]; then
    SEARCH_ENDPOINT="https://${SEARCH_RESOURCE}.search.windows.net"
    echo -e "   ${GREEN}✅ Found: $SEARCH_RESOURCE${NC}"
    echo "   🔗 Endpoint: $SEARCH_ENDPOINT"
else
    echo -e "   ${RED}❌ Azure Cognitive Search not found${NC}"
    SEARCH_ENDPOINT=""
fi

# Discover Azure Cosmos DB
echo "🌐 Discovering Azure Cosmos DB..."
COSMOS_RESOURCE=$(az cosmosdb list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv 2>/dev/null)
if [ -n "$COSMOS_RESOURCE" ]; then
    COSMOS_ENDPOINT=$(az cosmosdb show --name "$COSMOS_RESOURCE" --resource-group "$RESOURCE_GROUP" --query "documentEndpoint" -o tsv)
    echo -e "   ${GREEN}✅ Found: $COSMOS_RESOURCE${NC}"
    echo "   🔗 Endpoint: $COSMOS_ENDPOINT"
else
    echo -e "   ${RED}❌ Azure Cosmos DB not found${NC}"
    COSMOS_ENDPOINT=""
fi

# Discover Azure Storage
echo "💾 Discovering Azure Storage..."
STORAGE_RESOURCE=$(az storage account list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv 2>/dev/null)
if [ -n "$STORAGE_RESOURCE" ]; then
    STORAGE_ENDPOINT="https://${STORAGE_RESOURCE}.blob.core.windows.net/"
    echo -e "   ${GREEN}✅ Found: $STORAGE_RESOURCE${NC}"
    echo "   🔗 Endpoint: $STORAGE_ENDPOINT"
else
    echo -e "   ${RED}❌ Azure Storage not found${NC}"
    STORAGE_ENDPOINT=""
fi

# Discover Azure Machine Learning
echo "🧠 Discovering Azure ML..."
ML_RESOURCE=$(az ml workspace list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv 2>/dev/null)
if [ -n "$ML_RESOURCE" ]; then
    echo -e "   ${GREEN}✅ Found: $ML_RESOURCE${NC}"
    
    # Discover GNN endpoints
    echo "🔬 Discovering GNN endpoints..."
    GNN_ENDPOINT=$(az ml online-endpoint list --workspace-name "$ML_RESOURCE" --resource-group "$RESOURCE_GROUP" --query "[?contains(tags.model_type, 'gnn')].name" -o tsv 2>/dev/null | head -1)
    if [ -n "$GNN_ENDPOINT" ]; then
        GNN_URI=$(az ml online-endpoint show --name "$GNN_ENDPOINT" --workspace-name "$ML_RESOURCE" --resource-group "$RESOURCE_GROUP" --query "scoring_uri" -o tsv 2>/dev/null)
        echo -e "   ${GREEN}✅ Found GNN endpoint: $GNN_ENDPOINT${NC}"
        echo "   🔗 Scoring URI: $GNN_URI"
    else
        echo -e "   ${YELLOW}⚠️  No GNN endpoints found${NC}"
        GNN_URI=""
    fi
else
    echo -e "   ${RED}❌ Azure ML not found${NC}"
    GNN_ENDPOINT=""
    GNN_URI=""
fi

# Discover Application Insights
echo "📊 Discovering Application Insights..."
APP_INSIGHTS_RESOURCE=$(az monitor app-insights component list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv 2>/dev/null)
if [ -n "$APP_INSIGHTS_RESOURCE" ]; then
    APP_INSIGHTS_CONNECTION_STRING=$(az monitor app-insights component show --app "$APP_INSIGHTS_RESOURCE" --resource-group "$RESOURCE_GROUP" --query "connectionString" -o tsv 2>/dev/null)
    echo -e "   ${GREEN}✅ Found: $APP_INSIGHTS_RESOURCE${NC}"
    echo "   🔗 Connection String: $(echo $APP_INSIGHTS_CONNECTION_STRING | cut -c1-50)..."
else
    echo -e "   ${YELLOW}⚠️  Application Insights not found${NC}"
    APP_INSIGHTS_CONNECTION_STRING=""
fi

# Discover Key Vault
echo "🔐 Discovering Key Vault..."
KEY_VAULT_RESOURCE=$(az keyvault list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv 2>/dev/null)
if [ -n "$KEY_VAULT_RESOURCE" ]; then
    KEY_VAULT_URL=$(az keyvault show --name "$KEY_VAULT_RESOURCE" --resource-group "$RESOURCE_GROUP" --query "properties.vaultUri" -o tsv 2>/dev/null)
    echo -e "   ${GREEN}✅ Found: $KEY_VAULT_RESOURCE${NC}"
    echo "   🔗 URL: $KEY_VAULT_URL"
else
    echo -e "   ${YELLOW}⚠️  Key Vault not found${NC}"
    KEY_VAULT_URL=""
fi

# Discover Container Apps
echo "📦 Discovering Container Apps..."
CONTAINER_APP_RESOURCE=$(az containerapp list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv 2>/dev/null)
if [ -n "$CONTAINER_APP_RESOURCE" ]; then
    echo -e "   ${GREEN}✅ Found: $CONTAINER_APP_RESOURCE${NC}"
else
    echo -e "   ${YELLOW}⚠️  Container Apps not found${NC}"
fi

# Discover Managed Identity
echo "🆔 Discovering Managed Identity..."
MANAGED_IDENTITY_RESOURCE=$(az identity list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv 2>/dev/null)
if [ -n "$MANAGED_IDENTITY_RESOURCE" ]; then
    MANAGED_IDENTITY_CLIENT_ID=$(az identity show --name "$MANAGED_IDENTITY_RESOURCE" --resource-group "$RESOURCE_GROUP" --query "clientId" -o tsv 2>/dev/null)
    echo -e "   ${GREEN}✅ Found: $MANAGED_IDENTITY_RESOURCE${NC}"
    echo "   🔑 Client ID: $MANAGED_IDENTITY_CLIENT_ID"
else
    echo -e "   ${YELLOW}⚠️  Managed Identity not found${NC}"
    MANAGED_IDENTITY_CLIENT_ID=""
fi

echo ""
echo -e "${GREEN}🔧 Generating complete .env configuration...${NC}"

# Create the root .env file with discovered Azure service values
cat > /workspace/azure-maintie-rag/.env << EOF
# Azure Universal RAG - Auto-Discovered Environment Configuration
# Generated on: $(date)
# Environment: $AZURE_ENV_NAME

# Azure Environment Settings
AZURE_ENV_NAME="$AZURE_ENV_NAME"
AZURE_LOCATION="$AZURE_LOCATION"
AZURE_SUBSCRIPTION_ID="$AZURE_SUBSCRIPTION_ID"
AZURE_RESOURCE_GROUP="$RESOURCE_GROUP"

# Authentication Settings
USE_MANAGED_IDENTITY="false"

# Azure OpenAI Service (Auto-Discovered)
AZURE_OPENAI_ENDPOINT="$OPENAI_ENDPOINT"
AZURE_OPENAI_RESOURCE_NAME="$OPENAI_RESOURCE"
OPENAI_API_KEY="$OPENAI_API_KEY"
OPENAI_MODEL_DEPLOYMENT="gpt-4o"
EMBEDDING_MODEL_DEPLOYMENT="text-embedding-ada-002"
OPENAI_API_VERSION="2024-02-01"

# Azure Cognitive Search (Auto-Discovered)
AZURE_SEARCH_ENDPOINT="$SEARCH_ENDPOINT"
AZURE_SEARCH_RESOURCE_NAME="$SEARCH_RESOURCE"
SEARCH_INDEX_NAME="maintie-index"

# Azure Cosmos DB (Auto-Discovered)
AZURE_COSMOS_ENDPOINT="$COSMOS_ENDPOINT"
AZURE_COSMOS_RESOURCE_NAME="$COSMOS_RESOURCE"
COSMOS_DATABASE_NAME="maintie-rag-db"
COSMOS_GRAPH_NAME="knowledge-graph"

# Azure Storage (Auto-Discovered)
AZURE_STORAGE_ENDPOINT="$STORAGE_ENDPOINT"
AZURE_STORAGE_ACCOUNT="$STORAGE_RESOURCE"
AZURE_STORAGE_CONTAINER="documents"

# Azure Machine Learning (Auto-Discovered)
AZURE_ML_WORKSPACE="$ML_RESOURCE"
GNN_MODEL_NAME="universal-gnn"
GNN_ENDPOINT_NAME="$GNN_ENDPOINT"
GNN_SCORING_URI="$GNN_URI"

# Azure Application Insights (Auto-Discovered)
AZURE_APPLICATION_INSIGHTS_RESOURCE="$APP_INSIGHTS_RESOURCE"
APPLICATIONINSIGHTS_CONNECTION_STRING="$APP_INSIGHTS_CONNECTION_STRING"

# Azure Key Vault (Auto-Discovered)
AZURE_KEY_VAULT_NAME="$KEY_VAULT_RESOURCE"
AZURE_KEY_VAULT_URL="$KEY_VAULT_URL"

# Azure Container Apps (Auto-Discovered)
AZURE_CONTAINER_APP_NAME="$CONTAINER_APP_RESOURCE"

# Azure Managed Identity (Auto-Discovered)
AZURE_MANAGED_IDENTITY_NAME="$MANAGED_IDENTITY_RESOURCE"
AZURE_MANAGED_IDENTITY_CLIENT_ID="$MANAGED_IDENTITY_CLIENT_ID"

# Development Settings
PYTHONPATH="/workspace/azure-maintie-rag"
ENVIRONMENT_TYPE="$AZURE_ENV_NAME"
LOG_LEVEL="INFO"
MAX_WORKERS=4
BACKEND_PORT=8000
CACHE_TTL=600
EOF

echo ""
echo -e "${GREEN}✅ AUTO-CONFIGURATION COMPLETE!${NC}"
echo "================================================"
echo -e "${YELLOW}📁 Configuration file: .env${NC}"
echo -e "${YELLOW}📊 Services discovered and configured:${NC}"
if [ -n "$OPENAI_RESOURCE" ]; then
    echo "   🤖 Azure OpenAI: $OPENAI_RESOURCE"
fi
if [ -n "$SEARCH_RESOURCE" ]; then
    echo "   🔍 Cognitive Search: $SEARCH_RESOURCE" 
fi
if [ -n "$COSMOS_RESOURCE" ]; then
    echo "   🌐 Cosmos DB: $COSMOS_RESOURCE"
fi
if [ -n "$STORAGE_RESOURCE" ]; then
    echo "   💾 Storage: $STORAGE_RESOURCE"
fi
if [ -n "$ML_RESOURCE" ]; then
    echo "   🧠 Azure ML: $ML_RESOURCE"
    if [ -n "$GNN_ENDPOINT" ]; then
        echo "   🔬 GNN Endpoint: $GNN_ENDPOINT"
    fi
fi
if [ -n "$APP_INSIGHTS_RESOURCE" ]; then
    echo "   📊 Application Insights: $APP_INSIGHTS_RESOURCE"
fi
if [ -n "$KEY_VAULT_RESOURCE" ]; then
    echo "   🔐 Key Vault: $KEY_VAULT_RESOURCE"
fi
if [ -n "$CONTAINER_APP_RESOURCE" ]; then
    echo "   📦 Container App: $CONTAINER_APP_RESOURCE"
fi
if [ -n "$MANAGED_IDENTITY_RESOURCE" ]; then
    echo "   🆔 Managed Identity: $MANAGED_IDENTITY_RESOURCE"
fi
echo ""
echo "🚀 Next steps:"
echo "   1. Review the generated .env file"
echo "   2. Run: source .env"
echo "   3. Start the API: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo -e "${GREEN}🎉 Your Azure Universal RAG system is ready to use!${NC}"