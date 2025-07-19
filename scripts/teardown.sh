#!/bin/bash
# Azure Universal RAG Teardown Script
# Safely removes all resources and handles soft-delete conflicts

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

teardown_resources() {
    print_header "🔥 Azure Universal RAG Teardown"
    print_info "Resource Group: $RESOURCE_GROUP"
    print_info "Environment: $ENVIRONMENT"

    # Check if resource group exists
    if ! az group show --name "$RESOURCE_GROUP" --output none 2>/dev/null; then
        print_info "Resource group '$RESOURCE_GROUP' does not exist"
        return 0
    fi

    print_info "Resource group exists. Starting teardown..."

    # List current resources
    print_info "Current resources:"
    az resource list --resource-group "$RESOURCE_GROUP" --query "[].{Name:name,Type:type}" --output table 2>/dev/null || print_warning "Could not list resources"

    # Clean up soft-deleted resources first
    print_info "Cleaning up soft-deleted resources..."
    cleanup_soft_deleted_resources "$AZURE_LOCATION" "$ENVIRONMENT"

    # Delete resource group
    print_info "Deleting resource group and all resources..."
    if az group delete --name "$RESOURCE_GROUP" --yes --no-wait; then
        print_status "Resource group deletion initiated"

        # Wait for deletion
        print_info "Waiting for resource group deletion to complete..."
        while az group show --name "$RESOURCE_GROUP" --output none 2>/dev/null; do
            echo -n "."
            sleep 10
        done
        echo ""
        print_status "Resource group deletion completed"
    else
        print_error "Failed to delete resource group"
        return 1
    fi

    # Additional cleanup for soft-deleted resources
    print_info "Performing additional cleanup for soft-deleted resources..."

    # Clean up any remaining soft-deleted Key Vaults
    local deleted_keyvaults=$(az keyvault list-deleted --query "[?contains(name, '$ENVIRONMENT')].name" --output tsv 2>/dev/null || echo "")
    if [ ! -z "$deleted_keyvaults" ]; then
        print_info "Purging remaining soft-deleted Key Vaults..."
        echo "$deleted_keyvaults" | while read -r kv; do
            if [ ! -z "$kv" ]; then
                print_info "Purging Key Vault: $kv"
                az keyvault purge --name "$kv" --location "$AZURE_LOCATION" 2>/dev/null || true
            fi
        done
    fi

    print_success "✅ Teardown completed successfully"
}

main() {
    # Check Azure CLI authentication
    if ! az account show --output none 2>/dev/null; then
        print_error "Azure CLI not authenticated. Please run 'az login'"
        exit 1
    fi

    # Get current location or use default
    if [ -z "${AZURE_LOCATION:-}" ]; then
        AZURE_LOCATION="eastus"
        print_info "Using default location: $AZURE_LOCATION"
    fi

    # Confirm teardown
    print_warning "This will delete ALL resources in resource group: $RESOURCE_GROUP"
    print_warning "This action cannot be undone!"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        print_info "Teardown cancelled"
        exit 0
    fi

    # Execute teardown
    teardown_resources
}

# Execute teardown
main "$@"