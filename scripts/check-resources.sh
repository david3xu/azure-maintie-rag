#!/bin/bash
# check-resources.sh - Resource existence checker
# Checks what Azure resources exist to prevent duplicate creation and waste credits

set -e

# Configuration
RESOURCE_GROUP="maintie-rag-rg"

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

echo "🔍 Azure Resource Checker"
echo "Resource Group: $RESOURCE_GROUP"
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

# Function to check if resource exists
check_resource_exists() {
    local resource_type=$1
    local resource_name=$2
    local resource_group=$3

    # Map resource types to Azure resource types
    case $resource_type in
        "storage account")
            local azure_type="Microsoft.Storage/storageAccounts"
            ;;
        "search service")
            local azure_type="Microsoft.Search/searchServices"
            ;;
        "keyvault")
            local azure_type="Microsoft.KeyVault/vaults"
            ;;
        "cosmosdb")
            local azure_type="Microsoft.DocumentDB/databaseAccounts"
            ;;
        "ml workspace")
            local azure_type="Microsoft.MachineLearningServices/workspaces"
            ;;
        "monitor app-insights component")
            local azure_type="Microsoft.Insights/components"
            ;;
        "containerapp")
            local azure_type="Microsoft.App/containerApps"
            ;;
        *)
            local azure_type=$resource_type
            ;;
    esac

    # Use generic resource check with resource type
    if az resource show --name $resource_name --resource-group $resource_group --resource-type $azure_type &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check if deployment exists
check_deployment_exists() {
    local deployment_name=$1
    if az deployment group show --resource-group $RESOURCE_GROUP --name $deployment_name &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check Resource Group
echo "📋 Resource Group Status:"
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    print_status "Resource group '$RESOURCE_GROUP' exists"
    RG_LOCATION=$(az group show --name $RESOURCE_GROUP --query location --output tsv)
    print_info "  Location: $RG_LOCATION"
else
    print_warning "Resource group '$RESOURCE_GROUP' does not exist"
fi
echo ""

# Check Deployments
echo "📋 Deployment Status:"
DEPLOYMENTS=("azure-resources-core" "deploy_enterprise" "azure-resources-simple")

for deployment in "${DEPLOYMENTS[@]}"; do
    if check_deployment_exists "$deployment"; then
        STATE=$(az deployment group show --resource-group $RESOURCE_GROUP --name $deployment --query properties.provisioningState --output tsv)
        print_status "Deployment '$deployment' exists (State: $STATE)"
    else
        print_warning "Deployment '$deployment' does not exist"
    fi
done
echo ""

# Check Core Resources
echo "📋 Core Resources Status:"

# Get resource names from deployments
STORAGE_ACCOUNT=""
SEARCH_SERVICE=""
KEY_VAULT_NAME=""

if check_deployment_exists "azure-resources-core"; then
    STORAGE_ACCOUNT=$(az deployment group show \
      --resource-group $RESOURCE_GROUP \
      --name azure-resources-core \
      --query 'properties.outputs.storageAccountName.value' \
      --output tsv 2>/dev/null || echo "")

    SEARCH_SERVICE=$(az deployment group show \
      --resource-group $RESOURCE_GROUP \
      --name azure-resources-core \
      --query 'properties.outputs.searchServiceName.value' \
      --output tsv 2>/dev/null || echo "")

    KEY_VAULT_NAME=$(az deployment group show \
      --resource-group $RESOURCE_GROUP \
      --name azure-resources-core \
      --query 'properties.outputs.keyVaultName.value' \
      --output tsv 2>/dev/null || echo "")
fi

# Check Storage Account
if [ ! -z "$STORAGE_ACCOUNT" ]; then
    if check_resource_exists "storage account" "$STORAGE_ACCOUNT" "$RESOURCE_GROUP"; then
        print_status "Storage Account '$STORAGE_ACCOUNT' exists"
        # SKU info removed due to Azure CLI environment issues
    else
        print_warning "Storage Account '$STORAGE_ACCOUNT' not found"
    fi
else
    print_warning "Storage Account name not available from deployment"
fi

# Check Search Service
if [ ! -z "$SEARCH_SERVICE" ]; then
    if check_resource_exists "search service" "$SEARCH_SERVICE" "$RESOURCE_GROUP"; then
        print_status "Search Service '$SEARCH_SERVICE' exists"
        # SKU info removed due to Azure CLI environment issues
    else
        print_warning "Search Service '$SEARCH_SERVICE' not found"
    fi
else
    print_warning "Search Service name not available from deployment"
fi

# Check Key Vault
if [ ! -z "$KEY_VAULT_NAME" ]; then
    if check_resource_exists "keyvault" "$KEY_VAULT_NAME" "$RESOURCE_GROUP"; then
        print_status "Key Vault '$KEY_VAULT_NAME' exists"
        # Check if secrets exist
        if az keyvault secret show --vault-name $KEY_VAULT_NAME --name "AzureStorageConnectionString" &> /dev/null; then
            print_info "  Storage connection string secret exists"
        else
            print_warning "  Storage connection string secret not found"
        fi

        if az keyvault secret show --vault-name $KEY_VAULT_NAME --name "AzureSearchAdminKey" &> /dev/null; then
            print_info "  Search admin key secret exists"
        else
            print_warning "  Search admin key secret not found"
        fi
    else
        print_warning "Key Vault '$KEY_VAULT_NAME' not found"
    fi
else
    print_warning "Key Vault name not available from deployment"
fi
echo ""

# Check Additional Resources
echo "📋 Additional Resources Status:"

# Check Cosmos DB
COSMOS_ACCOUNT="maintie-dev-cosmos"
if check_resource_exists "cosmosdb" "$COSMOS_ACCOUNT" "$RESOURCE_GROUP"; then
    print_status "Cosmos DB '$COSMOS_ACCOUNT' exists"
else
    print_warning "Cosmos DB '$COSMOS_ACCOUNT' not found"
fi

# Check ML Workspace
ML_WORKSPACE="maintie-dev-ml"
if check_resource_exists "ml workspace" "$ML_WORKSPACE" "$RESOURCE_GROUP"; then
    print_status "ML Workspace '$ML_WORKSPACE' exists"
else
    print_warning "ML Workspace '$ML_WORKSPACE' not found"
fi

# Check Application Insights
APP_INSIGHTS="maintie-dev-app-insights"
if check_resource_exists "monitor app-insights component" "$APP_INSIGHTS" "$RESOURCE_GROUP"; then
    print_status "Application Insights '$APP_INSIGHTS' exists"
else
    print_warning "Application Insights '$APP_INSIGHTS' not found"
fi

# Check Container App
if check_resource_exists "containerapp" "maintie-rag-app" "$RESOURCE_GROUP"; then
    print_status "Container App 'maintie-rag-app' exists"
    # Get app URL
    APP_URL=$(az containerapp show --name maintie-rag-app --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn --output tsv 2>/dev/null || echo "")
    if [ ! -z "$APP_URL" ]; then
        print_info "  URL: https://$APP_URL"
    fi
else
    print_warning "Container App 'maintie-rag-app' not found"
fi

# Check Log Analytics Workspace
LOG_ANALYTICS="maintie-dev-laworkspace"
if check_resource_exists "Microsoft.OperationalInsights/workspaces" "$LOG_ANALYTICS" "$RESOURCE_GROUP"; then
    print_status "Log Analytics Workspace '$LOG_ANALYTICS' exists"
else
    print_warning "Log Analytics Workspace '$LOG_ANALYTICS' not found"
fi
echo ""

# Check Local Environment
echo "📋 Local Environment Status:"

if [ -f "backend/.env" ]; then
    print_status "Environment file 'backend/.env' exists"
else
    print_warning "Environment file 'backend/.env' not found"
fi

if [ -d ".venv" ]; then
    print_status "Virtual environment '.venv' exists"
else
    print_warning "Virtual environment '.venv' not found"
fi

if docker images | grep -q "azure-maintie-rag"; then
    print_status "Docker image 'azure-maintie-rag' exists"
else
    print_warning "Docker image 'azure-maintie-rag' not found"
fi
echo ""

# Summary
echo "📊 Summary:"
TOTAL_RESOURCES=0
EXISTING_RESOURCES=0

# Count resources
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    TOTAL_RESOURCES=$((TOTAL_RESOURCES + 1))
    EXISTING_RESOURCES=$((EXISTING_RESOURCES + 1))
fi

if [ ! -z "$STORAGE_ACCOUNT" ] && check_resource_exists "storage account" "$STORAGE_ACCOUNT" "$RESOURCE_GROUP"; then
    TOTAL_RESOURCES=$((TOTAL_RESOURCES + 1))
    EXISTING_RESOURCES=$((EXISTING_RESOURCES + 1))
fi

if [ ! -z "$SEARCH_SERVICE" ] && check_resource_exists "search service" "$SEARCH_SERVICE" "$RESOURCE_GROUP"; then
    TOTAL_RESOURCES=$((TOTAL_RESOURCES + 1))
    EXISTING_RESOURCES=$((EXISTING_RESOURCES + 1))
fi

if [ ! -z "$KEY_VAULT_NAME" ] && check_resource_exists "keyvault" "$KEY_VAULT_NAME" "$RESOURCE_GROUP"; then
    TOTAL_RESOURCES=$((TOTAL_RESOURCES + 1))
    EXISTING_RESOURCES=$((EXISTING_RESOURCES + 1))
fi

print_info "Total expected resources: $TOTAL_RESOURCES"
print_info "Existing resources: $EXISTING_RESOURCES"

if [ $EXISTING_RESOURCES -eq $TOTAL_RESOURCES ]; then
    print_status "All expected resources exist"
elif [ $EXISTING_RESOURCES -gt 0 ]; then
    print_warning "Some resources exist. Consider using --force flag for deployment."
else
    print_info "No resources exist. Safe to deploy."
fi

echo ""
echo "💡 Usage:"
echo "  ./scripts/check-resources.sh                    # Check current status"
echo "  ./scripts/deploy-core.sh                        # Deploy core resources"
echo "  ./scripts/teardown.sh                           # Clean up all resources"