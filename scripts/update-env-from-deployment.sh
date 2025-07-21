#!/bin/bash
# Update .env file with actual Azure service endpoints from deployment
# This script extracts service endpoints from deployment output and updates backend/.env

set -euo pipefail

# Color coding for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Configuration
ENVIRONMENT="${AZURE_ENVIRONMENT:-dev}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-maintie-rag-rg}"
ENV_FILE="backend/.env"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    print_error "Environment file not found: $ENV_FILE"
    print_info "Please run: cp backend/config/environment_example.env backend/.env"
    exit 1
fi

# Function to get deployment token from resource names
get_deployment_token() {
    local resource_name="$1"
    # Extract token from resource name (last part after prefix)
    echo "$resource_name" | sed 's/.*-//' | sed 's/[0-9]*$//'
}

# Function to update .env file with actual values
update_env_file() {
    local search_service="$1"
    local storage_account="$2"
    local key_vault_name="$3"
    local cosmos_name="$4"

    print_info "Updating $ENV_FILE with actual Azure service endpoints..."

    # Create backup
    cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d-%H%M%S)"

    # Update Azure Storage settings
    if [ -n "$storage_account" ]; then
        print_info "Setting AZURE_STORAGE_ACCOUNT=$storage_account"
        sed -i "s|AZURE_STORAGE_ACCOUNT=.*|AZURE_STORAGE_ACCOUNT=${storage_account//|/\\|}|" "$ENV_FILE"
    fi

    # Update Azure Search settings
    if [ -n "$search_service" ]; then
        print_info "Setting AZURE_SEARCH_SERVICE=$search_service"
        sed -i "s|AZURE_SEARCH_SERVICE=.*|AZURE_SEARCH_SERVICE=${search_service//|/\\|}|" "$ENV_FILE"
        sed -i "s|AZURE_SEARCH_SERVICE_NAME=.*|AZURE_SEARCH_SERVICE_NAME=${search_service//|/\\|}|" "$ENV_FILE"
    fi

    # Update Azure Key Vault settings
    if [ -n "$key_vault_name" ]; then
        print_info "Setting AZURE_KEY_VAULT_URL=https://$key_vault_name.vault.azure.net/"
        sed -i "s|AZURE_KEY_VAULT_URL=.*|AZURE_KEY_VAULT_URL=https://$key_vault_name.vault.azure.net/|" "$ENV_FILE"
    fi

    # Update Azure Cosmos DB settings
    if [ -n "$cosmos_name" ]; then
        print_info "Setting AZURE_COSMOS_ENDPOINT=https://$cosmos_name.documents.azure.com:443/"
        sed -i "s|AZURE_COSMOS_ENDPOINT=.*|AZURE_COSMOS_ENDPOINT=https://$cosmos_name.documents.azure.com:443/|" "$ENV_FILE"
    fi

    # Update connection strings
    if [ -n "$storage_account" ]; then
        print_info "Setting AZURE_STORAGE_CONNECTION_STRING for $storage_account"
        sed -i "s|AZURE_STORAGE_CONNECTION_STRING=.*|AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=${storage_account//|/\\|};AccountKey=your-key;EndpointSuffix=core.windows.net|" "$ENV_FILE"
    fi
    if [ -n "$cosmos_name" ]; then
        print_info "Setting AZURE_COSMOS_DB_CONNECTION_STRING for $cosmos_name"
        sed -i "s|AZURE_COSMOS_DB_CONNECTION_STRING=.*|AZURE_COSMOS_DB_CONNECTION_STRING=AccountEndpoint=https://$cosmos_name.documents.azure.com:443/;AccountKey=your-key;|" "$ENV_FILE"
    fi

    print_status "Environment file updated successfully"
    print_warning "⚠️  IMPORTANT: You need to manually update the API keys in $ENV_FILE:"
    print_info "   - AZURE_STORAGE_KEY (get from Azure Portal)"
    print_info "   - AZURE_SEARCH_KEY (get from Azure Portal)"
    print_info "   - AZURE_COSMOS_KEY (get from Azure Portal)"
    print_info "   - OPENAI_API_KEY (your Azure OpenAI key)"
}

# Function to extract service names from deployment files
extract_service_names() {
    local search_service=""
    local storage_account=""
    local key_vault_name=""
    local cosmos_name=""

    # Try to get from deployment files
    if [ -f ".deployment_search_name" ]; then
        search_service=$(cat ".deployment_search_name" | tr -d '\n\r')
        print_status "Found Search Service: $search_service" >&2
    fi

    if [ -f ".deployment_storage_name" ]; then
        storage_account=$(cat ".deployment_storage_name" | tr -d '\n\r')
        print_status "Found Storage Account: $storage_account" >&2
    fi

    if [ -f ".deployment_keyvault_name" ]; then
        key_vault_name=$(cat ".deployment_keyvault_name" | tr -d '\n\r')
        print_status "Found Key Vault: $key_vault_name" >&2
    fi

    if [ -f ".deployment_cosmos_name" ]; then
        cosmos_name=$(cat ".deployment_cosmos_name" | tr -d '\n\r')
        print_status "Found Cosmos DB: $cosmos_name" >&2
    fi

    # If not found in files, try to get from Azure
    if [ -z "$search_service" ]; then
        search_service=$(az search service list --resource-group "$RESOURCE_GROUP" --query "[0].name" --output tsv 2>/dev/null || echo "")
        if [ -n "$search_service" ]; then
            print_status "Found Search Service from Azure: $search_service" >&2
        fi
    fi

    if [ -z "$storage_account" ]; then
        storage_account=$(az storage account list --resource-group "$RESOURCE_GROUP" --query "[0].name" --output tsv 2>/dev/null || echo "")
        if [ -n "$storage_account" ]; then
            print_status "Found Storage Account from Azure: $storage_account" >&2
        fi
    fi

    if [ -z "$key_vault_name" ]; then
        key_vault_name=$(az keyvault list --resource-group "$RESOURCE_GROUP" --query "[0].name" --output tsv 2>/dev/null || echo "")
        if [ -n "$key_vault_name" ]; then
            print_status "Found Key Vault from Azure: $key_vault_name" >&2
        fi
    fi

    if [ -z "$cosmos_name" ]; then
        cosmos_name=$(az cosmosdb list --resource-group "$RESOURCE_GROUP" --query "[0].name" --output tsv 2>/dev/null || echo "")
        if [ -n "$cosmos_name" ]; then
            print_status "Found Cosmos DB from Azure: $cosmos_name" >&2
        fi
    fi

    # Return values
    echo "$search_service|$storage_account|$key_vault_name|$cosmos_name"
}

main() {
    print_info "🔧 Updating .env file with Azure service endpoints..."
    print_info "Resource Group: $RESOURCE_GROUP"
    print_info "Environment: $ENVIRONMENT"

    # Check Azure CLI authentication
    if ! az account show --output none 2>/dev/null; then
        print_error "Azure CLI not authenticated. Please run 'az login'"
        exit 1
    fi

    # Extract service names
    local service_names=$(extract_service_names)
    local search_service=$(echo "$service_names" | cut -d'|' -f1)
    local storage_account=$(echo "$service_names" | cut -d'|' -f2)
    local key_vault_name=$(echo "$service_names" | cut -d'|' -f3)
    local cosmos_name=$(echo "$service_names" | cut -d'|' -f4)

    # Validate we have the required services
    if [ -z "$search_service" ] || [ -z "$storage_account" ] || [ -z "$key_vault_name" ]; then
        print_error "Could not find required Azure services"
        print_info "Please ensure you have run the deployment script first:"
        print_info "  ./scripts/enhanced-complete-redeploy.sh"
        exit 1
    fi

    # Update .env file
    update_env_file "$search_service" "$storage_account" "$key_vault_name" "$cosmos_name"

    print_status "✅ Environment file updated successfully!"
    print_info ""
    print_info "Next steps:"
    print_info "1. Update API keys in $ENV_FILE"
    print_info "2. Run: make dev"
    print_info "3. Test: make health"
}

main "$@"