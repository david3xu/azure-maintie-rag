#!/bin/bash
# deploy-ml.sh - Deploy ML resources for Azure Universal RAG
# Deploys ML Workspace, Application Insights, Container Apps for GNN training

set -e

# Configuration
RESOURCE_GROUP="maintie-rag-rg"
ENVIRONMENT="dev"
LOCATION="eastus"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo "🚀 Azure ML Resources Deployment"
echo "Resource Group: $RESOURCE_GROUP"
echo "Environment: $ENVIRONMENT"
echo ""

# Check Azure CLI
if ! command -v az &> /dev/null; then
    print_error "Azure CLI not found. Please install Azure CLI first."
    exit 1
fi

# Check if logged in
if ! az account show &> /dev/null; then
    print_error "Not logged into Azure. Please run 'az login' first."
    exit 1
fi

print_status "Azure CLI and authentication verified"
echo ""

# Check if resource group exists
if ! az group show --name $RESOURCE_GROUP &> /dev/null; then
    print_error "Resource group '$RESOURCE_GROUP' does not exist. Please run deploy-core.sh first."
    exit 1
fi

print_status "Resource group '$RESOURCE_GROUP' exists"
echo ""

# Check if core resources exist
print_info "Checking core resources..."
STORAGE_ACCOUNT=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources-core \
  --query 'properties.outputs.storageAccountName.value' \
  --output tsv 2>/dev/null || echo "")

if [ -z "$STORAGE_ACCOUNT" ]; then
    print_error "Core resources not found. Please run deploy-core.sh first."
    exit 1
fi

print_status "Core resources found: $STORAGE_ACCOUNT"
echo ""

# Deploy ML resources
print_info "Deploying ML resources..."

# Check if ML deployment already exists
if az deployment group show --resource-group $RESOURCE_GROUP --name azure-resources-ml &> /dev/null; then
    print_warning "ML deployment already exists. Skipping deployment."
else
    # Deploy ML resources
    print_info "Creating ML deployment..."
    az deployment group create \
        --resource-group $RESOURCE_GROUP \
        --template-file infrastructure/azure-resources-ml.bicep \
        --name azure-resources-ml \
        --parameters environment=$ENVIRONMENT location=$LOCATION \
        --verbose

    print_status "ML resources deployment completed"
fi

# Get deployment outputs
print_info "Getting deployment outputs..."
ML_WORKSPACE=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources-ml \
  --query 'properties.outputs.mlWorkspaceName.value' \
  --output tsv 2>/dev/null || echo "")

APP_INSIGHTS=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources-ml \
  --query 'properties.outputs.appInsightsName.value' \
  --output tsv 2>/dev/null || echo "")

CONTAINER_APP=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources-ml \
  --query 'properties.outputs.containerAppName.value' \
  --output tsv 2>/dev/null || echo "")

CONTAINER_URL=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources-ml \
  --query 'properties.outputs.containerAppUrl.value' \
  --output tsv 2>/dev/null || echo "")

# Store secrets in Key Vault
print_info "Storing secrets in Key Vault..."
KEY_VAULT_NAME=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources-core \
  --query 'properties.outputs.keyVaultName.value' \
  --output tsv 2>/dev/null || echo "")

if [ ! -z "$KEY_VAULT_NAME" ]; then
    # Get storage connection string
    STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
      --name $STORAGE_ACCOUNT \
      --resource-group $RESOURCE_GROUP \
      --query connectionString \
      --output tsv 2>/dev/null || echo "")

    if [ ! -z "$STORAGE_CONNECTION_STRING" ]; then
        az keyvault secret set \
          --vault-name $KEY_VAULT_NAME \
          --name "AzureStorageConnectionString" \
          --value "$STORAGE_CONNECTION_STRING" \
          --output none 2>/dev/null || print_warning "Could not store storage connection string"
    fi

    # Get search admin key
    SEARCH_SERVICE=$(az deployment group show \
      --resource-group $RESOURCE_GROUP \
      --name azure-resources-core \
      --query 'properties.outputs.searchServiceName.value' \
      --output tsv 2>/dev/null || echo "")

    if [ ! -z "$SEARCH_SERVICE" ]; then
        SEARCH_ADMIN_KEY=$(az search admin-key show \
          --service-name $SEARCH_SERVICE \
          --resource-group $RESOURCE_GROUP \
          --query primaryKey \
          --output tsv 2>/dev/null || echo "")

        if [ ! -z "$SEARCH_ADMIN_KEY" ]; then
            az keyvault secret set \
              --vault-name $KEY_VAULT_NAME \
              --name "AzureSearchAdminKey" \
              --value "$SEARCH_ADMIN_KEY" \
              --output none 2>/dev/null || print_warning "Could not store search admin key"
        fi
    fi

    # Store ML workspace info
    if [ ! -z "$ML_WORKSPACE" ]; then
        az keyvault secret set \
          --vault-name $KEY_VAULT_NAME \
          --name "AzureMLWorkspaceName" \
          --value "$ML_WORKSPACE" \
          --output none 2>/dev/null || print_warning "Could not store ML workspace name"
    fi

    print_status "Secrets stored in Key Vault"
else
    print_warning "Key Vault not found, skipping secret storage"
fi

# Update environment file
print_info "Updating environment configuration..."
if [ -f "backend/.env" ]; then
    cp backend/.env backend/.env.backup.$(date +%Y%m%d_%H%M%S)
fi

cat > backend/.env << EOF
# Azure ML Resources Configuration
# Generated by deploy-ml.sh on $(date)

# ML Workspace
AZURE_ML_WORKSPACE_NAME=$ML_WORKSPACE
AZURE_ML_SUBSCRIPTION_ID=$(az account show --query id --output tsv)
AZURE_ML_RESOURCE_GROUP=$RESOURCE_GROUP

# Application Insights
AZURE_APP_INSIGHTS_NAME=$APP_INSIGHTS
AZURE_ENABLE_TELEMETRY=true

# Container App
AZURE_CONTAINER_APP_NAME=$CONTAINER_APP
AZURE_CONTAINER_APP_URL=$CONTAINER_URL

# Core Resources (from previous deployment)
AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT
AZURE_SEARCH_SERVICE=$SEARCH_SERVICE
AZURE_KEY_VAULT_NAME=$KEY_VAULT_NAME

# Environment
AZURE_ENVIRONMENT=$ENVIRONMENT
AZURE_LOCATION=$LOCATION
AZURE_USE_MANAGED_IDENTITY=true

# OpenAI Configuration (update with your values)
OPENAI_API_TYPE=azure
OPENAI_API_BASE=https://your-openai-instance.openai.azure.com/
OPENAI_API_VERSION=2025-03-01-preview
OPENAI_DEPLOYMENT_NAME=gpt-4.1
OPENAI_MODEL=gpt-4.1

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

# Application Settings
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
EOF

print_status "Environment configuration updated"

# Summary
echo ""
echo "📊 Deployment Summary:"
echo "======================"
print_status "ML Workspace: $ML_WORKSPACE"
print_status "Application Insights: $APP_INSIGHTS"
print_status "Container App: $CONTAINER_APP"
if [ ! -z "$CONTAINER_URL" ]; then
    print_status "Container App URL: https://$CONTAINER_URL"
fi
print_status "Key Vault: $KEY_VAULT_NAME"
print_status "Storage Account: $STORAGE_ACCOUNT"
print_status "Search Service: $SEARCH_SERVICE"

echo ""
echo "🎯 Next Steps:"
echo "1. Update OpenAI configuration in backend/.env"
echo "2. Build and deploy your application:"
echo "   docker build -t azure-maintie-rag:latest backend/"
echo "   az containerapp update --name $CONTAINER_APP --resource-group $RESOURCE_GROUP --image azure-maintie-rag:latest"
echo "3. Test GNN training:"
echo "   python backend/scripts/train_comprehensive_gnn.py --workspace $ML_WORKSPACE"
echo ""
print_status "ML resources deployment completed successfully!"