#!/bin/bash
# Azure Universal RAG Status Script
# Check current deployment status and resource health

set -euo pipefail

# Import enterprise deployment modules
source "$(dirname "$0")/azure-deployment-manager.sh"
source "$(dirname "$0")/azure-service-validator.sh"

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-maintie-rag-rg}"
ENVIRONMENT="${AZURE_ENVIRONMENT:-dev}"

# Color coding for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_header() { echo -e "${BLUE}🏗️  $1${NC}"; }

check_deployment_status() {
    print_header "📊 Azure Universal RAG Deployment Status"
    print_info "Resource Group: $RESOURCE_GROUP"
    print_info "Environment: $ENVIRONMENT"
    echo ""

    # Check if resource group exists
    if ! az group show --name "$RESOURCE_GROUP" --output none 2>/dev/null; then
        print_error "Resource group '$RESOURCE_GROUP' does not exist"
        print_info "No deployment found. Run './scripts/deploy.sh' to deploy."
        return 1
    fi

    print_status "Resource group exists"

    # Get resource group location
    local location=$(az group show --name "$RESOURCE_GROUP" --query "location" --output tsv)
    print_info "Location: $location"

    # List all resources
    print_info "Current resources:"
    echo ""

    local resources=$(az resource list --resource-group "$RESOURCE_GROUP" --query "[].{Name:name,Type:type,Status:properties.provisioningState}" --output table 2>/dev/null)
    if [ ! -z "$resources" ]; then
        echo "$resources"
    else
        print_warning "No resources found in resource group"
    fi

    echo ""

    # Check specific resource types
    check_specific_resources

    # Check deployments
    check_deployments

    # Check soft-deleted resources
    check_soft_deleted_resources "$location"
}

check_specific_resources() {
    print_info "Checking specific resource types..."

    # Check Storage Account
    local storage_accounts=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.Storage/storageAccounts" --query "[].name" --output tsv 2>/dev/null)
    if [ ! -z "$storage_accounts" ]; then
        print_status "Storage Accounts: $(echo "$storage_accounts" | wc -l)"
        echo "$storage_accounts" | while read -r sa; do
            if [ ! -z "$sa" ]; then
                local status=$(az storage account show --name "$sa" --resource-group "$RESOURCE_GROUP" --query "statusOfPrimary" --output tsv 2>/dev/null || echo "Unknown")
                print_info "  - $sa ($status)"
            fi
        done
    else
        print_warning "No Storage Accounts found"
    fi

    # Check Search Services
    local search_services=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.Search/searchServices" --query "[].name" --output tsv 2>/dev/null)
    if [ ! -z "$search_services" ]; then
        print_status "Search Services: $(echo "$search_services" | wc -l)"
        echo "$search_services" | while read -r ss; do
            if [ ! -z "$ss" ]; then
                local status=$(az search service show --name "$ss" --resource-group "$RESOURCE_GROUP" --query "status" --output tsv 2>/dev/null || echo "Unknown")
                print_info "  - $ss ($status)"
            fi
        done
    else
        print_warning "No Search Services found"
    fi

    # Check Key Vaults
    local key_vaults=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.KeyVault/vaults" --query "[].name" --output tsv 2>/dev/null)
    if [ ! -z "$key_vaults" ]; then
        print_status "Key Vaults: $(echo "$key_vaults" | wc -l)"
        echo "$key_vaults" | while read -r kv; do
            if [ ! -z "$kv" ]; then
                local status=$(az keyvault show --name "$kv" --resource-group "$RESOURCE_GROUP" --query "properties.provisioningState" --output tsv 2>/dev/null || echo "Unknown")
                print_info "  - $kv ($status)"
            fi
        done
    else
        print_warning "No Key Vaults found"
    fi

    # Check ML Workspaces
    local ml_workspaces=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.MachineLearningServices/workspaces" --query "[].name" --output tsv 2>/dev/null)
    if [ ! -z "$ml_workspaces" ]; then
        print_status "ML Workspaces: $(echo "$ml_workspaces" | wc -l)"
        echo "$ml_workspaces" | while read -r ml; do
            if [ ! -z "$ml" ]; then
                local status=$(az ml workspace show --name "$ml" --resource-group "$RESOURCE_GROUP" --query "properties.provisioningState" --output tsv 2>/dev/null || echo "Unknown")
                print_info "  - $ml ($status)"
            fi
        done
    else
        print_warning "No ML Workspaces found"
    fi
}

check_deployments() {
    echo ""
    print_info "Checking deployments..."

    local deployments=$(az deployment group list --resource-group "$RESOURCE_GROUP" --query "[].{Name:name,State:properties.provisioningState,Timestamp:properties.timestamp}" --output table 2>/dev/null)
    if [ ! -z "$deployments" ]; then
        echo "$deployments"
    else
        print_warning "No deployments found"
    fi
}

check_soft_deleted_resources() {
    local location=$1

    echo ""
    print_info "Checking soft-deleted resources in $location..."

    # Check soft-deleted Key Vaults
    local deleted_keyvaults=$(az keyvault list-deleted --query "[?properties.location=='$location'].name" --output tsv 2>/dev/null)
    if [ ! -z "$deleted_keyvaults" ]; then
        print_warning "Found soft-deleted Key Vaults:"
        echo "$deleted_keyvaults" | while read -r kv; do
            if [ ! -z "$kv" ]; then
                print_info "  - $kv"
            fi
        done
    else
        print_status "No soft-deleted Key Vaults found"
    fi
}

check_quota_usage() {
    echo ""
    print_info "Checking subscription quota usage..."

    local location=$(az group show --name "$RESOURCE_GROUP" --query "location" --output tsv 2>/dev/null || echo "eastus")

    # Check core quotas
    local quotas=("Microsoft.Storage/storageAccounts" "Microsoft.Search/searchServices" "Microsoft.KeyVault/vaults")

    for quota in "${quotas[@]}"; do
        local current_usage=$(az vm list-usage --location "$location" --query "[?name.value=='${quota#*/}'].currentValue" --output tsv 2>/dev/null || echo "0")
        local limit=$(az vm list-usage --location "$location" --query "[?name.value=='${quota#*/}'].limit" --output tsv 2>/dev/null || echo "100")
        local usage_percent=$((current_usage * 100 / limit))

        if [ $usage_percent -gt 80 ]; then
            print_warning "${quota#*/}: $current_usage/$limit (${usage_percent}%)"
        else
            print_status "${quota#*/}: $current_usage/$limit (${usage_percent}%)"
        fi
    done
}

main() {
    # Check Azure CLI authentication
    if ! az account show --output none 2>/dev/null; then
        print_error "Azure CLI not authenticated. Please run 'az login'"
        exit 1
    fi

    # Check deployment status
    if check_deployment_status; then
        # Check quota usage
        check_quota_usage

        echo ""
        print_status "Status check completed"
    else
        print_error "Status check failed"
        exit 1
    fi
}

# Execute status check
main "$@"