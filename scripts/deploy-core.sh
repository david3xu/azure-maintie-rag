#!/bin/bash
# deploy-core.sh - Core Azure Universal RAG deployment script
# Handles core infrastructure deployment without enterprise features
# Includes resource existence checks to prevent duplicate creation

set -e

# Configuration
RESOURCE_GROUP="maintie-rag-rg"
LOCATION="${LOCATION:-eastus}"  # Allow override from environment
ENVIRONMENT="dev"
RESOURCE_PREFIX="maintie"
PROJECT_NAME="Universal-RAG"

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

echo "🚀 Starting Azure Universal RAG core deployment..."
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "Environment: $ENVIRONMENT"

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

# Function to check if resource group exists
check_resource_group() {
    if az group show --name $RESOURCE_GROUP &> /dev/null; then
        print_warning "Resource group '$RESOURCE_GROUP' already exists"
        return 0
    else
        print_info "Resource group '$RESOURCE_GROUP' does not exist"
        return 1
    fi
}

# Function to check if deployment exists
check_deployment() {
    local deployment_name=$1
    if az deployment group show --resource-group $RESOURCE_GROUP --name $deployment_name &> /dev/null; then
        print_warning "Deployment '$deployment_name' already exists"
        return 0
    else
        print_info "Deployment '$deployment_name' does not exist"
        return 1
    fi
}

# Function to check if specific resources exist
check_storage_account() {
    local storage_name=$1
    if [ -z "$storage_name" ]; then
        print_error "Storage account name is empty"
        return 1
    fi
    if az resource show --name $storage_name --resource-group $RESOURCE_GROUP --resource-type Microsoft.Storage/storageAccounts &> /dev/null; then
        print_status "Storage account '$storage_name' exists"
        return 0
    else
        print_error "Storage account '$storage_name' does not exist"
        return 1
    fi
}

check_search_service() {
    local search_name=$1
    if az search service show --name $search_name --resource-group $RESOURCE_GROUP &> /dev/null; then
        print_warning "Search service '$search_name' already exists"
        return 0
    else
        print_info "Search service '$search_name' does not exist"
        return 1
    fi
}

check_key_vault() {
    local kv_name=$1
    if az keyvault show --name $kv_name --resource-group $RESOURCE_GROUP &> /dev/null; then
        print_warning "Key Vault '$kv_name' already exists"
        return 0
    else
        print_info "Key Vault '$kv_name' does not exist"
        return 1
    fi
}

# Phase 1: Infrastructure Deployment
echo ""
echo "📦 Phase 1: Checking and Deploying Core Infrastructure..."

# Check if resource group exists
if check_resource_group; then
    print_info "Using existing resource group"
else
    print_info "Creating resource group..."
    az group create \
      --name $RESOURCE_GROUP \
      --location $LOCATION \
      --tags Environment=$ENVIRONMENT Project=$PROJECT_NAME
    print_status "Resource group created"
fi

# Check if core deployment exists
if check_deployment "azure-resources-core"; then
    print_warning "Core deployment already exists. Skipping infrastructure deployment."
    print_info "If you want to redeploy, delete the existing deployment first:"
    print_info "  az deployment group delete --resource-group $RESOURCE_GROUP --name azure-resources-core"
else
    print_info "Deploying core infrastructure..."
    az deployment group create \
      --resource-group $RESOURCE_GROUP \
      --template-file infrastructure/azure-resources-core.bicep \
      --parameters environment=$ENVIRONMENT location=$LOCATION resourcePrefix=$RESOURCE_PREFIX \
      --verbose
    print_status "Core infrastructure deployment completed"
fi

# Phase 2: Configuration & Secrets Management
echo ""
echo "⚙️  Phase 2: Configuring Services..."

# Get deployment outputs
print_info "Getting deployment outputs..."
STORAGE_ACCOUNT=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources-core \
  --query 'properties.outputs.storageAccountName.value' \
  --output tsv)

SEARCH_SERVICE=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources-core \
  --query 'properties.outputs.searchServiceName.value' \
  --output tsv)

KEY_VAULT_NAME=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources-core \
  --query 'properties.outputs.keyVaultName.value' \
  --output tsv)

# COSMOS_DB_NAME=$(az deployment group show \
#   --resource-group $RESOURCE_GROUP \
#   --name azure-resources-core \
#   --query 'properties.outputs.cosmosDBName.value' \
#   --output tsv)

# COSMOS_DB_ENDPOINT=$(az deployment group show \
#   --resource-group $RESOURCE_GROUP \
#   --name azure-resources-core \
#   --query 'properties.outputs.cosmosDBEndpoint.value' \
#   --output tsv)

COSMOS_DB_NAME=""
COSMOS_DB_ENDPOINT=""

print_info "Storage Account: $STORAGE_ACCOUNT"
print_info "Search Service: $SEARCH_SERVICE"
print_info "Key Vault: $KEY_VAULT_NAME"
print_info "Cosmos DB: $COSMOS_DB_NAME"
print_info "Cosmos DB Endpoint: $COSMOS_DB_ENDPOINT"

# Verify resources exist
if ! check_storage_account $STORAGE_ACCOUNT; then
    print_error "Storage account not found. Deployment may have failed."
    exit 1
fi

if ! check_search_service $SEARCH_SERVICE; then
    print_error "Search service not found. Deployment may have failed."
    exit 1
fi

if ! check_key_vault $KEY_VAULT_NAME; then
    print_error "Key Vault not found. Deployment may have failed."
    exit 1
fi

# Check Cosmos DB (temporarily disabled)
if [ ! -z "$COSMOS_DB_NAME" ]; then
    if az cosmosdb show --name $COSMOS_DB_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
        print_status "Cosmos DB '$COSMOS_DB_NAME' exists"
    else
        print_warning "Cosmos DB not found (temporarily disabled)."
    fi
else
    print_warning "Cosmos DB temporarily disabled due to region availability."
fi

# Configure Key Vault (only if not already configured)
print_info "Configuring Key Vault..."
USER_OBJECT_ID=$(az ad signed-in-user show --query id --output tsv)

# Check if Key Vault uses RBAC (which doesn't support access policies)
if az keyvault show --name $KEY_VAULT_NAME --resource-group $RESOURCE_GROUP --query "properties.enableRbacAuthorization" --output tsv 2>/dev/null | grep -q "true"; then
    print_warning "Key Vault uses RBAC authorization. Access policies not needed."
    print_info "To access secrets, assign 'Key Vault Secrets User' role to your account:"
    print_info "  az role assignment create --role 'Key Vault Secrets User' --assignee $USER_OBJECT_ID --scope /subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KEY_VAULT_NAME"
else
    # Check if policy already exists
    if az keyvault show --name $KEY_VAULT_NAME --resource-group $RESOURCE_GROUP --query "properties.accessPolicies[?objectId=='$USER_OBJECT_ID']" --output tsv 2>/dev/null | grep -q "$USER_OBJECT_ID"; then
        print_warning "Key Vault policy already exists for current user"
    else
        print_info "Setting Key Vault policy..."
        az keyvault set-policy \
          --name $KEY_VAULT_NAME \
          --resource-group $RESOURCE_GROUP \
          --object-id $USER_OBJECT_ID \
          --secret-permissions get set list delete
        print_status "Key Vault policy configured"
    fi
fi

# Store secrets in Key Vault (only if not already stored)
print_info "Checking and storing secrets in Key Vault..."

# Check if storage connection string secret exists
if az keyvault secret show --vault-name $KEY_VAULT_NAME --name "AzureStorageConnectionString" &> /dev/null; then
    print_warning "Storage connection string secret already exists"
else
    print_info "Storing storage connection string..."
    STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=$STORAGE_ACCOUNT;AccountKey=$(az storage account keys list --account-name $STORAGE_ACCOUNT --query '[0].value' --output tsv);EndpointSuffix=core.windows.net"
    az keyvault secret set \
      --vault-name $KEY_VAULT_NAME \
      --name "AzureStorageConnectionString" \
      --value "$STORAGE_CONNECTION_STRING"
    print_status "Storage connection string stored"
fi

# Check if search admin key secret exists
if az keyvault secret show --vault-name $KEY_VAULT_NAME --name "AzureSearchAdminKey" &> /dev/null; then
    print_warning "Search admin key secret already exists"
else
    print_info "Storing search admin key..."
    SEARCH_ADMIN_KEY=$(az search admin-key show --service-name $SEARCH_SERVICE --resource-group $RESOURCE_GROUP --query primaryKey --output tsv)
    az keyvault secret set \
      --vault-name $KEY_VAULT_NAME \
      --name "AzureSearchAdminKey" \
      --value "$SEARCH_ADMIN_KEY"
    print_status "Search admin key stored"
fi

# Create Search Index (if not exists)
print_info "Creating search index..."
if ! az search index show --service-name $SEARCH_SERVICE --resource-group $RESOURCE_GROUP --name universal-rag-index &> /dev/null; then
    print_info "Creating universal-rag-index..."

    # Create index definition
    cat > /tmp/universal-rag-index.json << 'EOF'
{
  "name": "universal-rag-index",
  "fields": [
    {
      "name": "id",
      "type": "Edm.String",
      "key": true,
      "searchable": false,
      "filterable": false,
      "sortable": false,
      "facetable": false
    },
    {
      "name": "content",
      "type": "Edm.String",
      "searchable": true,
      "filterable": false,
      "sortable": false,
      "facetable": false
    },
    {
      "name": "metadata",
      "type": "Edm.String",
      "searchable": false,
      "filterable": true,
      "sortable": false,
      "facetable": false
    },
    {
      "name": "embedding",
      "type": "Collection(Edm.Single)",
      "searchable": false,
      "filterable": false,
      "sortable": false,
      "facetable": false,
      "dimensions": 1536
    },
    {
      "name": "domain",
      "type": "Edm.String",
      "searchable": true,
      "filterable": true,
      "sortable": true,
      "facetable": true
    },
    {
      "name": "timestamp",
      "type": "Edm.DateTimeOffset",
      "searchable": false,
      "filterable": true,
      "sortable": true,
      "facetable": false
    }
  ],
  "suggesters": [
    {
      "name": "sg",
      "searchMode": "analyzingInfixMatching",
      "sourceFields": ["content", "domain"]
    }
  ],
  "scoringProfiles": [
    {
      "name": "semantic",
      "textWeights": {
        "weights": {
          "content": 1.0,
          "domain": 0.5
        }
      }
    }
  ],
  "semantic": {
    "configurations": [
      {
        "name": "semantic-config",
        "prioritizedFields": {
          "titleField": {
            "fieldName": "content"
          },
          "prioritizedKeywordsFields": [
            {
              "fieldName": "domain"
            }
          ],
          "prioritizedContentFields": [
            {
              "fieldName": "content"
            }
          ]
        }
      }
    ]
  }
}
EOF

    az search index create \
      --service-name $SEARCH_SERVICE \
      --resource-group $RESOURCE_GROUP \
      --name universal-rag-index \
      --body @/tmp/universal-rag-index.json

    print_status "Search index 'universal-rag-index' created"
else
    print_warning "Search index 'universal-rag-index' already exists"
fi

# Generate environment configuration (only if .env doesn't exist or is different)
print_info "Checking environment configuration..."
if [ -f "backend/.env" ]; then
    print_warning "Environment file already exists. Backing up..."
    cp backend/.env backend/.env.backup
fi

print_info "Generating environment configuration..."
cat > backend/.env << EOF
# Azure Environment
AZURE_ENVIRONMENT=$ENVIRONMENT
AZURE_RESOURCE_GROUP=$RESOURCE_GROUP
AZURE_REGION=$LOCATION

# Azure Storage
AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT
AZURE_BLOB_CONTAINER=universal-rag-data

# Azure Search
AZURE_SEARCH_SERVICE=$SEARCH_SERVICE
AZURE_SEARCH_INDEX=universal-rag-index
AZURE_SEARCH_API_VERSION=2023-11-01

# Azure Cosmos DB
AZURE_COSMOS_DB_NAME=$COSMOS_DB_NAME
AZURE_COSMOS_DB_ENDPOINT=$COSMOS_DB_ENDPOINT
AZURE_COSMOS_DB_DATABASE=universal-rag-db
AZURE_COSMOS_DB_CONTAINER=knowledge-graph

# Azure Key Vault
AZURE_KEY_VAULT_URL=https://$KEY_VAULT_NAME.vault.azure.net/
AZURE_USE_MANAGED_IDENTITY=true

# Application Insights (disabled for core deployment)
AZURE_ENABLE_TELEMETRY=false

# OpenAI Configuration
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
ENVIRONMENT=development
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
EOF

print_status "Configuration completed"

# Phase 3: Validation & Testing
echo ""
echo "🧪 Phase 3: Validating Deployment..."

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    print_info "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if not already installed
if [ ! -f ".venv/installed" ]; then
    print_info "Installing dependencies..."
    pip install -r backend/requirements.txt
    touch .venv/installed
else
    print_warning "Dependencies already installed"
fi

# Run enterprise integration tests
print_info "Running enterprise integration tests..."
cd backend
python scripts/test_enterprise_simple.py
cd ..

print_status "Validation completed"

echo ""
print_status "Core deployment completed successfully!"
echo ""
echo "📋 Deployment Summary:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Storage Account: $STORAGE_ACCOUNT"
echo "  Search Service: $SEARCH_SERVICE"
echo "  Key Vault: $KEY_VAULT_NAME"
echo "  Cosmos DB: $COSMOS_DB_NAME"
echo ""
echo "🔧 Next Steps:"
echo "  1. Update OpenAI configuration in backend/.env"
echo "  2. Deploy enterprise features when ready: ./scripts/deploy-enterprise.sh"
echo "  3. Deploy application: ./scripts/deploy-app.sh"
echo ""
echo "🧹 To clean up: ./scripts/teardown.sh"