#!/bin/bash
# Azure Universal RAG Deployment Script
# Simple deployment using enterprise architecture

set -euo pipefail

# Import enterprise deployment modules
source "$(dirname "$0")/azure-deployment-manager.sh"
source "$(dirname "$0")/azure-service-validator.sh"

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-maintie-rag-rg}"
ENVIRONMENT="${AZURE_ENVIRONMENT:-dev}"
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

main() {
    print_header "🚀 Azure Universal RAG Deployment"
    print_info "Resource Group: $RESOURCE_GROUP"
    print_info "Environment: $ENVIRONMENT"
    print_info "Deployment ID: $DEPLOYMENT_TIMESTAMP"

    # Check Azure CLI authentication
    if ! az account show --output none 2>/dev/null; then
        print_error "Azure CLI not authenticated. Please run 'az login'"
        exit 1
    fi

    # Get optimal region
    local optimal_region
    optimal_region=$(get_optimal_deployment_region "latency")
    if [ $? -ne 0 ] || [ -z "$optimal_region" ]; then
        print_error "Could not determine optimal Azure region"
        exit 1
    fi

    export AZURE_LOCATION="$optimal_region"
    print_status "Selected Azure region: $AZURE_LOCATION"

    # Create resource group if needed
    if ! create_resource_group_if_needed "$RESOURCE_GROUP" "$AZURE_LOCATION"; then
        print_error "Resource group creation failed"
        exit 1
    fi

    # Deploy core infrastructure
    local deployment_parameters="environment=$ENVIRONMENT location=$AZURE_LOCATION deploymentTimestamp=$DEPLOYMENT_TIMESTAMP"

    if deploy_with_exponential_backoff \
        "azure-resources-core-${DEPLOYMENT_TIMESTAMP}" \
        "infrastructure/azure-resources-core.bicep" \
        "$deployment_parameters"; then

        print_success "✅ Azure Universal RAG deployment completed successfully"
        print_info "Deployment Summary:"
        print_info "  - Resource Group: $RESOURCE_GROUP"
        print_info "  - Region: $AZURE_LOCATION"
        print_info "  - Environment: $ENVIRONMENT"
        print_info "  - Deployment ID: $DEPLOYMENT_TIMESTAMP"
#!/bin/bash
# Azure Universal RAG Deployment Script
# Simple deployment using enterprise architecture

set -euo pipefail

# Import enterprise deployment modules
source "$(dirname "$0")/azure-deployment-manager.sh"
source "$(dirname "$0")/azure-service-validator.sh"

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-maintie-rag-rg}"
ENVIRONMENT="${AZURE_ENVIRONMENT:-dev}"
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

main() {
    print_header "🚀 Azure Universal RAG Deployment"
    print_info "Resource Group: $RESOURCE_GROUP"
    print_info "Environment: $ENVIRONMENT"
    print_info "Deployment ID: $DEPLOYMENT_TIMESTAMP"

    # Check Azure CLI authentication
    if ! az account show --output none 2>/dev/null; then
        print_error "Azure CLI not authenticated. Please run 'az login'"
        exit 1
    fi

    # Get optimal region
    local optimal_region
    optimal_region=$(get_optimal_deployment_region "latency")
    if [ $? -ne 0 ] || [ -z "$optimal_region" ]; then
        print_error "Could not determine optimal Azure region"
        exit 1
    fi

    export AZURE_LOCATION="$optimal_region"
    print_status "Selected Azure region: $AZURE_LOCATION"

    # Create resource group if needed
    if ! create_resource_group_if_needed "$RESOURCE_GROUP" "$AZURE_LOCATION"; then
        print_error "Resource group creation failed"
        exit 1
    fi

    # Deploy core infrastructure
    local deployment_parameters="environment=$ENVIRONMENT location=$AZURE_LOCATION deploymentTimestamp=$DEPLOYMENT_TIMESTAMP"

    if deploy_with_exponential_backoff \
        "azure-resources-core-${DEPLOYMENT_TIMESTAMP}" \
        "infrastructure/azure-resources-core.bicep" \
        "$deployment_parameters"; then

        print_success "✅ Azure Universal RAG deployment completed successfully"
        print_info "Deployment Summary:"
        print_info "  - Resource Group: $RESOURCE_GROUP"
        print_info "  - Region: $AZURE_LOCATION"
        print_info "  - Environment: $ENVIRONMENT"
        print_info "  - Deployment ID: $DEPLOYMENT_TIMESTAMP"
    else
        print_error "Deployment failed"
        exit 1
    fi
}

# Execute deployment
main "$@"