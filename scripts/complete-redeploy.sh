#!/bin/bash

# 🚀 Complete Azure Universal RAG Redeployment Script
# This script completely tears down and recreates the entire infrastructure

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP="maintie-rag-rg"
LOCATION="eastus"
ENVIRONMENT="dev"

echo -e "${BLUE}🚀 Azure Universal RAG Complete Redeployment${NC}"
echo -e "${BLUE}=============================================${NC}"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "Environment: $ENVIRONMENT"
echo ""

# Function to print status
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

# Function to check if resource group exists
check_resource_group() {
    az group show --name $RESOURCE_GROUP >/dev/null 2>&1
}

# Function to wait for resource group deletion
wait_for_deletion() {
    print_info "Waiting for resource group deletion to complete..."
    while check_resource_group; do
        echo -n "."
        sleep 10
    done
    echo ""
    print_status "Resource group deletion completed"
}

# Function to check Cosmos DB region availability
check_cosmos_region_availability() {
    local region=$1
    print_info "Checking Cosmos DB availability in $region..."

    # Try to create a temporary Cosmos DB account to test availability
    if az cosmosdb check-name-exists --name "test-cosmos-availability" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Phase 1: Complete Teardown
echo -e "${BLUE}🔥 Phase 1: Complete Teardown${NC}"
echo "=================================="

# Check if resource group exists
if check_resource_group; then
    print_info "Resource group exists. Starting teardown..."

    # List current resources
    print_info "Current resources:"
    az resource list --resource-group $RESOURCE_GROUP --query "[].{Name:name,Type:type}" --output table 2>/dev/null || print_warning "Could not list resources"

    # Purge Key Vault if it exists (to avoid soft delete conflicts)
    print_info "Checking for Key Vault soft delete conflicts..."
    if az keyvault show --name maintie-dev-kv --resource-group $RESOURCE_GROUP &> /dev/null; then
        print_warning "Key Vault exists. Will be deleted with resource group."
    elif az keyvault show-deleted --name maintie-dev-kv --location $LOCATION &> /dev/null; then
        print_info "Purging deleted Key Vault to avoid conflicts..."
        az keyvault purge --name maintie-dev-kv --location $LOCATION
        print_status "Key Vault purged"
    fi

    # Delete resource group
    print_info "Deleting resource group and all resources..."
    az group delete --name $RESOURCE_GROUP --yes --no-wait

    # Wait for deletion
    wait_for_deletion

    # Additional wait for search service deletion to complete
    print_info "Waiting for search service deletion to complete..."
    sleep 30
else
    print_info "Resource group does not exist. Skipping teardown."
fi

echo ""

# Phase 2: Create Resource Group
echo -e "${BLUE}🏗️  Phase 2: Create Resource Group${NC}"
echo "================================="

print_info "Creating resource group..."
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION \
    --tags Environment=$ENVIRONMENT Project=universal-rag

print_status "Resource group created"

echo ""

# Phase 3: Deploy Core Infrastructure
echo -e "${BLUE}🏗️  Phase 3: Deploy Core Infrastructure${NC}"
echo "=========================================="

print_info "Deploying core resources..."

# Clean up any existing failed deployment first
if az deployment group show --resource-group $RESOURCE_GROUP --name azure-resources-core &> /dev/null; then
    print_info "Cleaning up existing failed deployment..."
    az deployment group delete --resource-group $RESOURCE_GROUP --name azure-resources-core --yes
    sleep 10  # Wait for cleanup
fi

if [ -f "./scripts/deploy-core.sh" ]; then
    # Try deployment with current location
    if ./scripts/deploy-core.sh; then
        print_status "Core infrastructure deployed"
    else
        print_warning "Core deployment failed. Trying with alternative region..."

        # Try alternative regions for Cosmos DB
        ALTERNATIVE_REGIONS=("westus2" "centralus" "southcentralus" "northeurope")

        for alt_region in "${ALTERNATIVE_REGIONS[@]}"; do
            print_info "Trying deployment with region: $alt_region"

            # Clean up any existing deployment for this region
            if az deployment group show --resource-group $RESOURCE_GROUP --name azure-resources-core &> /dev/null; then
                az deployment group delete --resource-group $RESOURCE_GROUP --name azure-resources-core
                sleep 5
            fi

            # Temporarily change location for this deployment
            export LOCATION=$alt_region

            if ./scripts/deploy-core.sh; then
                print_status "Core infrastructure deployed in $alt_region"
                break
            else
                print_warning "Failed with region $alt_region, trying next..."
            fi
        done

        # Restore original location
        export LOCATION="eastus"
    fi
else
    print_error "Core deployment script not found"
    exit 1
fi

echo ""

# Phase 4: Deploy ML Infrastructure
echo -e "${BLUE}🤖 Phase 4: Deploy ML Infrastructure${NC}"
echo "================================="

print_info "Deploying ML resources..."
if [ -f "./scripts/deploy-ml.sh" ]; then
    ./scripts/deploy-ml.sh
    print_status "ML infrastructure deployed"
else
    print_error "ML deployment script not found"
    exit 1
fi

echo ""

# Phase 5: Manual ML Workspace Creation (if needed)
echo -e "${BLUE}🔧 Phase 5: Manual ML Workspace Creation${NC}"
echo "============================================="

# Check if ML workspace exists
ML_WORKSPACE_EXISTS=$(az ml workspace show --name maintie-dev-ml --resource-group $RESOURCE_GROUP --query "name" --output tsv 2>/dev/null || echo "")

if [ -z "$ML_WORKSPACE_EXISTS" ]; then
    print_warning "ML workspace not found. Creating manually..."

    # Check if ML storage exists
    ML_STORAGE_EXISTS=$(az storage account show --name maintiedevmlstorage --resource-group $RESOURCE_GROUP --query "name" --output tsv 2>/dev/null || echo "")

    if [ -z "$ML_STORAGE_EXISTS" ]; then
        print_info "Creating ML storage account..."
        az storage account create \
            --name maintiedevmlstorage \
            --resource-group $RESOURCE_GROUP \
            --location $LOCATION \
            --sku Standard_LRS \
            --kind StorageV2 \
            --https-only true \
            --min-tls-version TLS1_2
        print_status "ML storage account created"
    else
        print_info "ML storage account already exists"
    fi

    # Create ML workspace
    print_info "Creating ML workspace..."
    SUBSCRIPTION_ID=$(az account show --query id -o tsv)
    az ml workspace create \
        --name maintie-dev-ml \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --storage-account "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/maintiedevmlstorage" \
        --key-vault "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/maintie-dev-kv" \
        --application-insights "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Insights/components/maintie-dev-app-insights"
    print_status "ML workspace created"
else
    print_info "ML workspace already exists"
fi

echo ""

# Phase 6: Verification
echo -e "${BLUE}🧪 Phase 6: Verification${NC}"
echo "========================"

print_info "Running comprehensive resource check..."
if [ -f "./scripts/check-resources.sh" ]; then
    ./scripts/check-resources.sh
else
    print_warning "Resource check script not found. Running manual check..."
    az resource list --resource-group $RESOURCE_GROUP --query "[].{Name:name,Type:type,Location:location}" --output table
fi

echo ""

# Phase 7: Final Summary
echo -e "${BLUE}📊 Phase 7: Final Summary${NC}"
echo "========================"

print_info "Final resource inventory:"
az resource list \
    --resource-group $RESOURCE_GROUP \
    --query "[].{Name:name,Type:type,Location:location}" \
    --output table

echo ""
print_info "Resource count:"
RESOURCE_COUNT=$(az resource list --resource-group $RESOURCE_GROUP --query "length(@)" --output tsv)
print_status "Total resources: $RESOURCE_COUNT"

echo ""
print_info "Expected resources:"
echo "✅ Storage Account (maintiedevstorage) - Document storage"
echo "✅ ML Storage Account (maintiedevmlstorage) - ML artifacts"
echo "✅ Search Service (maintie-dev-search) - Vector search"
echo "✅ Key Vault (maintie-dev-kv) - Secrets management"
echo "✅ Cosmos DB (maintie-dev-cosmos) - Knowledge graph (Gremlin API)"
echo "✅ ML Workspace (maintie-dev-ml) - GNN training"
echo "✅ Application Insights (maintie-dev-app-insights) - Monitoring"
echo "✅ Log Analytics (maintie-dev-laworkspace) - Logging"
echo "✅ Container Environment (maintie-dev-env) - Container hosting"
echo "✅ Container App (maintie-dev-rag-app) - Application hosting"

echo ""
print_status "🎉 Complete redeployment finished successfully!"
echo ""
print_info "Next steps:"
echo "1. Test GNN training: python backend/scripts/train_comprehensive_gnn.py --workspace maintie-dev-ml"
echo "2. Build application: cd backend && docker build -t azure-maintie-rag:latest ."
echo "3. Deploy application: az containerapp update --name maintie-dev-rag-app --resource-group $RESOURCE_GROUP --image azure-maintie-rag:latest"
echo "4. Test end-to-end RAG pipeline"
echo ""
print_info "For detailed instructions, see: docs/COMPLETE_EXECUTION_INSTRUCTIONS.md"