#!/bin/bash
# Enhanced Azure Universal RAG Complete Redeployment
# Enterprise-grade deployment with resilience patterns and conflict resolution

set -euo pipefail

# Import enterprise deployment modules
source "$(dirname "$0")/azure-deployment-manager.sh"
source "$(dirname "$0")/azure-service-validator.sh"
source "$(dirname "$0")/azure-extension-manager.sh"
source "$(dirname "$0")/azure-naming-service.sh"
source "$(dirname "$0")/azure-deployment-orchestrator.sh"
source "$(dirname "$0")/azure-service-health-validator.sh"

# Configuration from environment
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-maintie-rag-rg}"
ENVIRONMENT="${AZURE_ENVIRONMENT:-dev}"
AZURE_LOCATION="${AZURE_LOCATION:-eastus}"  # Default location
DEPLOYMENT_TIMESTAMP=$(date +%Y%m%d-%H%M%S)

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
print_success() { echo -e "${GREEN}🎉 $1${NC}"; }

validate_deployment_prerequisites() {
    print_header "Phase 1: Pre-deployment Validation"

    # Check Azure CLI authentication
    if ! az account show --output none 2>/dev/null; then
        print_error "Azure CLI not authenticated. Please run 'az login'"
        return 1
    fi

    # Use enterprise extension manager
    validate_and_install_extensions

    # Comprehensive health check
    if ! comprehensive_health_check "$RESOURCE_GROUP" "$AZURE_LOCATION"; then
        print_error "Comprehensive health check failed"
        return 1
    fi

    print_status "Azure CLI authentication and extensions validated"
    return 0
}

execute_clean_deployment() {
    print_header "Phase 3: Clean Deployment Preparation"

    # Clean up any existing failed deployments
    print_info "Cleaning up failed deployments..."
    az deployment group list \
        --resource-group "$RESOURCE_GROUP" \
        --query "[?properties.provisioningState=='Failed'].name" \
        --output tsv 2>/dev/null | while read -r failed_deployment; do
        if [ ! -z "$failed_deployment" ]; then
            print_info "Deleting failed deployment: $failed_deployment"
            az deployment group delete \
                --resource-group "$RESOURCE_GROUP" \
                --name "$failed_deployment" \
                --yes 2>/dev/null || true
        fi
    done

    # Clean up soft-deleted resources
    cleanup_soft_deleted_resources "eastus" "$ENVIRONMENT"

    print_status "Clean deployment preparation completed"
    return 0
}

deploy_core_infrastructure_resilient() {
    print_header "Phase 4: Resilient Core Infrastructure Deployment"

    # Generate unique resource names using enterprise naming service
    print_info "Generating globally unique resource names..."
    local unique_storage_name=$(generate_globally_unique_storage_name "maintie" "$ENVIRONMENT")
    local unique_search_name=$(generate_unique_search_name "maintie" "$ENVIRONMENT" "$AZURE_LOCATION")
    local unique_keyvault_name=$(generate_unique_keyvault_name "maintie" "$ENVIRONMENT")

    if [ -z "$unique_storage_name" ] || [ -z "$unique_search_name" ] || [ -z "$unique_keyvault_name" ]; then
        print_error "Failed to generate unique resource names"
        return 1
    fi

    print_info "Generated unique resource names:"
    print_info "  - Storage Account: $unique_storage_name"
    print_info "  - Search Service: $unique_search_name"
    print_info "  - Key Vault: $unique_keyvault_name"

    # Use enterprise deployment orchestrator with unique names
    if orchestrate_resilient_deployment "$RESOURCE_GROUP" "$ENVIRONMENT" "$AZURE_LOCATION"; then
        print_status "Core infrastructure deployment completed successfully"

        # Store generated names for verification
        echo "$unique_storage_name" > ".deployment_storage_name"
        echo "$unique_search_name" > ".deployment_search_name"
        echo "$unique_keyvault_name" > ".deployment_keyvault_name"

        return 0
    else
        print_error "Core infrastructure deployment failed"
        return 1
    fi
}

deploy_ml_infrastructure_conditional() {
    print_header "Phase 5: Conditional ML Infrastructure Deployment"

    # Check if core infrastructure was deployed successfully
    local core_resources_exist=true

    # Check for Search service using stored name
    local search_service_name
    if [ -f ".deployment_search_name" ]; then
        search_service_name=$(cat ".deployment_search_name")
    else
        search_service_name="maintie-${ENVIRONMENT}-search-${DEPLOYMENT_TIMESTAMP}"
    fi

    if ! az search service show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$search_service_name" \
        --output none 2>/dev/null; then
        print_warning "Core Search service not found, skipping ML deployment"
        core_resources_exist=false
    fi

    # Check for Storage account using stored name
    local storage_account_name
    if [ -f ".deployment_storage_name" ]; then
        storage_account_name=$(cat ".deployment_storage_name")
    else
        storage_account_name="maintie${ENVIRONMENT}stor${DEPLOYMENT_TIMESTAMP}"
    fi

    if ! az storage account show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$storage_account_name" \
        --output none 2>/dev/null; then
        print_warning "Core Storage account not found, skipping ML deployment"
        core_resources_exist=false
    fi

    if [ "$core_resources_exist" = false ]; then
        print_warning "ML infrastructure deployment skipped due to missing core resources"
        return 0
    fi

    # Deploy ML infrastructure if core resources exist
    print_info "Core resources verified, proceeding with ML infrastructure deployment"

    # Check if ML infrastructure template exists
    if [ ! -f "infrastructure/azure-resources-ml.bicep" ]; then
        print_warning "ML infrastructure template not found, skipping ML deployment"
        return 0
    fi

    # Deploy ML infrastructure
    local ml_deployment_parameters="environment=$ENVIRONMENT location=$AZURE_LOCATION deploymentTimestamp=$DEPLOYMENT_TIMESTAMP"

    if deploy_with_exponential_backoff \
        "azure-resources-ml-${DEPLOYMENT_TIMESTAMP}" \
        "infrastructure/azure-resources-ml.bicep" \
        "$ml_deployment_parameters"; then

        print_status "ML infrastructure deployment completed successfully"
        return 0
    else
        print_warning "ML infrastructure deployment failed, but core deployment succeeded"
        return 0  # Don't fail the entire deployment
    fi
}

verify_deployment_success() {
    print_header "Phase 6: Deployment Verification"

    # Verify core resources
    local required_resources=(
        "Microsoft.Storage/storageAccounts"
        "Microsoft.Search/searchServices"
        "Microsoft.KeyVault/vaults"
    )

    local verification_failed=false

    for resource_type in "${required_resources[@]}"; do
        local resource_count=$(az resource list \
            --resource-group "$RESOURCE_GROUP" \
            --resource-type "$resource_type" \
            --query "length(@)" \
            --output tsv 2>/dev/null || echo "0")

        if [ "$resource_count" -eq 0 ]; then
            print_error "Required resource type $resource_type not found"
            verification_failed=true
        else
            print_status "Verified $resource_type: $resource_count resources found"
        fi
    done

    # Verify specific resource names using stored names
    local expected_search_service
    local expected_storage_account
    local expected_key_vault

    if [ -f ".deployment_search_name" ]; then
        expected_search_service=$(cat ".deployment_search_name")
    else
        expected_search_service="maintie-${ENVIRONMENT}-search-${DEPLOYMENT_TIMESTAMP}"
    fi

    if [ -f ".deployment_storage_name" ]; then
        expected_storage_account=$(cat ".deployment_storage_name")
    else
        expected_storage_account="maintie${ENVIRONMENT}stor${DEPLOYMENT_TIMESTAMP}"
    fi

    if [ -f ".deployment_keyvault_name" ]; then
        expected_key_vault=$(cat ".deployment_keyvault_name")
    else
        expected_key_vault="maintie-${ENVIRONMENT}-kv-${DEPLOYMENT_TIMESTAMP:0:8}"
    fi

    # Check Search service
    if az search service show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$expected_search_service" \
        --output none 2>/dev/null; then
        print_status "Verified Search service: $expected_search_service"
    else
        print_error "Search service verification failed: $expected_search_service"
        verification_failed=true
    fi

    # Check Storage account
    if az storage account show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$expected_storage_account" \
        --output none 2>/dev/null; then
        print_status "Verified Storage account: $expected_storage_account"
    else
        print_error "Storage account verification failed: $expected_storage_account"
        verification_failed=true
    fi

    # Check Key Vault
    if az keyvault show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$expected_key_vault" \
        --output none 2>/dev/null; then
        print_status "Verified Key Vault: $expected_key_vault"
    else
        print_error "Key Vault verification failed: $expected_key_vault"
        verification_failed=true
    fi

    if [ "$verification_failed" = true ]; then
        print_error "Deployment verification failed"
        return 1
    fi

    print_status "All required Azure resources verified successfully"
    return 0
}

main() {
    print_header "🚀 Azure Universal RAG Enterprise Deployment"
    print_info "Resource Group: $RESOURCE_GROUP"
    print_info "Environment: $ENVIRONMENT"
    print_info "Deployment ID: $DEPLOYMENT_TIMESTAMP"

    # Phase 1: Pre-deployment validation
    if ! validate_deployment_prerequisites; then
        print_error "Pre-deployment validation failed"
        exit 1
    fi

    # Phase 2: Optimal region selection
    local optimal_region
    optimal_region=$(get_optimal_deployment_region "latency")
    if [ $? -ne 0 ] || [ -z "$optimal_region" ]; then
        print_error "Could not determine optimal Azure region"
        exit 1
    fi

    export AZURE_LOCATION="$optimal_region"
    print_status "Selected Azure region: $AZURE_LOCATION"

    # Phase 3: Clean deployment with conflict resolution
    if ! execute_clean_deployment; then
        print_error "Clean deployment failed"
        exit 1
    fi

    # Phase 4: Core infrastructure with resilience
    if ! deploy_core_infrastructure_resilient; then
        print_error "Core infrastructure deployment failed"
        exit 1
    fi

    # Phase 5: ML infrastructure with dependencies
    if ! deploy_ml_infrastructure_conditional; then
        print_error "ML infrastructure deployment failed"
        exit 1
    fi

    # Phase 6: Deployment verification
    if ! verify_deployment_success; then
        print_error "Deployment verification failed"
        exit 1
    fi

    print_success "✅ Azure Universal RAG deployment completed successfully"
    print_info "Deployment Summary:"
    print_info "  - Resource Group: $RESOURCE_GROUP"
    print_info "  - Region: $AZURE_LOCATION"
    print_info "  - Environment: $ENVIRONMENT"
    print_info "  - Deployment ID: $DEPLOYMENT_TIMESTAMP"
    # Display actual deployed resource names
    local deployed_search_service="maintie-${ENVIRONMENT}-search-${DEPLOYMENT_TIMESTAMP}"
    local deployed_storage_account="maintie${ENVIRONMENT}stor${DEPLOYMENT_TIMESTAMP}"
    local deployed_key_vault="maintie-${ENVIRONMENT}-kv-${DEPLOYMENT_TIMESTAMP:0:8}"

    if [ -f ".deployment_search_name" ]; then
        deployed_search_service=$(cat ".deployment_search_name")
    fi
    if [ -f ".deployment_storage_name" ]; then
        deployed_storage_account=$(cat ".deployment_storage_name")
    fi
    if [ -f ".deployment_keyvault_name" ]; then
        deployed_key_vault=$(cat ".deployment_keyvault_name")
    fi

    print_info "  - Search Service: $deployed_search_service"
    print_info "  - Storage Account: $deployed_storage_account"
    print_info "  - Key Vault: $deployed_key_vault"

    # Clean up temporary deployment files
    cleanup_deployment_files
}

cleanup_deployment_files() {
    print_info "Cleaning up temporary deployment files..."
    rm -f ".deployment_storage_name" ".deployment_search_name" ".deployment_keyvault_name"
    print_status "Temporary files cleaned up"
}

# Execute main deployment orchestration
main "$@"