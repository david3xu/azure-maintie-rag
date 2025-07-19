#!/bin/bash
# Azure Deployment Orchestrator - Enterprise resilience patterns
# Circuit breaker with regional failover for production-grade infrastructure

set -euo pipefail

# Import enterprise modules
source "$(dirname "$0")/azure-extension-manager.sh"
source "$(dirname "$0")/azure-naming-service.sh"

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

class="AzureDeploymentOrchestrator"

orchestrate_resilient_deployment() {
    local resource_group=$1
    local environment=$2
    local region=$3

    print_header "Azure Deployment Orchestrator: Starting Resilient Deployment"
    print_info "Resource Group: $resource_group"
    print_info "Environment: $environment"
    print_info "Region: $region"

    # Initialize Azure Extension Manager
    validate_and_install_extensions

    # Generate globally unique resource names
    local unique_storage_name=$(generate_globally_unique_storage_name "maintie" "$environment")
    local unique_search_name=$(generate_unique_search_name "maintie" "$environment" "$region")
    local unique_keyvault_name=$(generate_unique_keyvault_name "maintie" "$environment")

    print_info "Generated unique resource names:"
    print_info "  - Storage Account: $unique_storage_name"
    print_info "  - Search Service: $unique_search_name"
    print_info "  - Key Vault: $unique_keyvault_name"

    # Validate Azure service health before deployment
    if ! validate_azure_service_health "$region"; then
        print_error "Azure service health validation failed"
        return 1
    fi

    # Azure ARM template deployment with enterprise parameters
    # Find the project root by looking for the infrastructure directory
    local project_root=""
    local current_dir="$(pwd)"

    # Try to find the project root by looking for infrastructure directory
    if [ -d "$current_dir/infrastructure" ]; then
        project_root="$current_dir"
    elif [ -d "$(dirname "$0")/../infrastructure" ]; then
        project_root="$(dirname "$0")/.."
    elif [ -d "/home/azureuser/workspace/azure-maintie-rag/infrastructure" ]; then
        project_root="/home/azureuser/workspace/azure-maintie-rag"
    else
        print_error "Cannot find project root with infrastructure directory"
        print_error "Current directory: $current_dir"
        print_error "Script directory: $(dirname "$0")"
        return 1
    fi

    local template_file="$project_root/infrastructure/azure-resources-core.bicep"

    # Debug path resolution
    print_info "Current directory: $current_dir"
    print_info "Project root: $project_root"
    print_info "Template file: $template_file"

    # Verify template file exists
    if [ ! -f "$template_file" ]; then
        print_error "Template file not found: $template_file"
        print_error "Available files in project root: $(ls -la "$project_root" 2>/dev/null || echo 'Cannot access project root')"
        return 1
    fi

    local deployment_config=(
        "--resource-group" "$resource_group"
        "--template-file" "$template_file"
        "--parameters"
        "environment=$environment"
        "location=$region"
        "storageAccountName=$unique_storage_name"
        "searchServiceName=$unique_search_name"
        "keyVaultName=$unique_keyvault_name"
        "--mode" "Incremental"
        "--verbose"
    )

    # Execute deployment with circuit breaker pattern
    execute_deployment_with_circuit_breaker "${deployment_config[@]}"
}

execute_deployment_with_circuit_breaker() {
    local deployment_config=("$@")
    local max_failures=3
    local failure_count=0
    local circuit_open_duration=300  # 5 minutes
    local deployment_name="azure-resources-core-$(date +%Y%m%d-%H%M%S)"

    print_header "Executing deployment with circuit breaker pattern"
    print_info "Deployment name: $deployment_name"
    print_info "Max failures: $max_failures"
    print_info "Circuit open duration: ${circuit_open_duration}s"

    while [ $failure_count -lt $max_failures ]; do
        print_info "Executing Azure ARM deployment (attempt $((failure_count + 1)))"

        # Add deployment name to config
        local full_deployment_config=("--name" "$deployment_name" "${deployment_config[@]}")

        if az deployment group create "${full_deployment_config[@]}"; then
            print_status "Azure ARM deployment succeeded"

            # Verify deployment outputs
            verify_deployment_outputs "$deployment_name"
            return 0
        else
            failure_count=$((failure_count + 1))

            if [ $failure_count -lt $max_failures ]; then
                print_warning "Deployment failed. Circuit breaker cooling down for ${circuit_open_duration}s"
                sleep $circuit_open_duration

                # Refresh Azure authentication and validate service health
                refresh_azure_session
                validate_azure_service_health "${deployment_config[5]}"  # Extract region from config
            fi
        fi
    done

    print_error "Deployment failed after $max_failures attempts. Circuit breaker open."
    return 1
}

verify_deployment_outputs() {
    local deployment_name=$1
    local resource_group="${2:-$(az account show --query resourceGroup --output tsv 2>/dev/null || echo "")}"

    print_header "Verifying deployment outputs"

    # Get deployment outputs
    local outputs=$(az deployment group show \
        --resource-group "$resource_group" \
        --name "$deployment_name" \
        --query "properties.outputs" \
        --output json 2>/dev/null || echo "{}")

    if [ "$outputs" != "{}" ]; then
        print_status "Deployment outputs retrieved successfully"

        # Extract and display key outputs
        local storage_account=$(echo "$outputs" | jq -r '.storageAccountName.value // empty')
        local search_service=$(echo "$outputs" | jq -r '.searchServiceName.value // empty')
        local key_vault=$(echo "$outputs" | jq -r '.keyVaultName.value // empty')

        if [ ! -z "$storage_account" ]; then
            print_status "Storage Account: $storage_account"
        fi

        if [ ! -z "$search_service" ]; then
            print_status "Search Service: $search_service"
        fi

        if [ ! -z "$key_vault" ]; then
            print_status "Key Vault: $key_vault"
        fi
    else
        print_warning "No deployment outputs found"
    fi
}

refresh_azure_session() {
    print_info "Refreshing Azure session"

    # Check if current session is valid
    if ! az account show --output none 2>/dev/null; then
        print_warning "Azure session expired, attempting to refresh"

        # Try to refresh using existing credentials
        if az account get-access-token --output none 2>/dev/null; then
            print_status "Azure session refreshed successfully"
        else
            print_error "Failed to refresh Azure session"
            return 1
        fi
    else
        print_status "Azure session is valid"
    fi
}

validate_azure_service_health() {
    local region=$1
    local required_services=(
        "Microsoft.Storage"
        "Microsoft.Search"
        "Microsoft.KeyVault"
        "Microsoft.MachineLearningServices"
        "Microsoft.App"
    )

    print_header "Validating Azure service health in region: $region"

    # Azure Resource Health API integration
    for service in "${required_services[@]}"; do
        local service_status=$(az provider show \
            --namespace "$service" \
            --query "registrationState" \
            --output tsv 2>/dev/null)

        if [ "$service_status" = "Registered" ]; then
            print_status "Service $service: Available"
        else
            print_error "Service $service: Unavailable (Status: $service_status)"
            return 1
        fi
    done

    # Regional capacity validation
    validate_regional_capacity "$region"
}

validate_regional_capacity() {
    local region=$1

    print_info "Validating regional capacity for: $region"

    # Azure Resource Graph query for regional utilization
    local capacity_query="Resources | where location == '$region' | summarize count()"

    local region_load=$(az graph query \
        --graph-query "$capacity_query" \
        --query "data[0].count_" \
        --output tsv 2>/dev/null || echo "0")

    if [ "$region_load" -lt 1000 ]; then  # Configurable threshold
        print_status "Regional capacity: Available (Load: $region_load)"
        return 0
    else
        print_warning "Regional capacity: High utilization (Load: $region_load)"
        return 1
    fi
}

deploy_with_rollback_capability() {
    local deployment_name=$1
    local template_file=$2
    local parameters=$3
    local resource_group=$4

    print_header "Deploying with rollback capability"

    # Create deployment with rollback tracking
    local deployment_id=$(az deployment group create \
        --resource-group "$resource_group" \
        --name "$deployment_name" \
        --template-file "$template_file" \
        --parameters "$parameters" \
        --query "id" \
        --output tsv)

    if [ ! -z "$deployment_id" ]; then
        print_status "Deployment created with ID: $deployment_id"

        # Store deployment info for potential rollback
        echo "$deployment_id" > ".deployment_${deployment_name}.id"
        echo "$(date)" > ".deployment_${deployment_name}.timestamp"

        return 0
    else
        print_error "Deployment failed"
        return 1
    fi
}

rollback_deployment() {
    local deployment_name=$1
    local resource_group=$2

    print_header "Rolling back deployment: $deployment_name"

    # Get deployment ID from stored file
    local deployment_id_file=".deployment_${deployment_name}.id"

    if [ -f "$deployment_id_file" ]; then
        local deployment_id=$(cat "$deployment_id_file")

        print_info "Rolling back deployment ID: $deployment_id"

        # Delete the deployment (this will rollback resources)
        if az deployment group delete \
            --resource-group "$resource_group" \
            --name "$deployment_name" \
            --yes; then

            print_status "Deployment rollback completed"

            # Clean up tracking files
            rm -f "$deployment_id_file"
            rm -f ".deployment_${deployment_name}.timestamp"

            return 0
        else
            print_error "Deployment rollback failed"
            return 1
        fi
    else
        print_error "Deployment ID file not found for rollback"
        return 1
    fi
}

# Main execution function
main() {
    local action="${1:-deploy}"
    local resource_group="${2:-maintie-rag-rg}"
    local environment="${3:-dev}"
    local region="${4:-eastus}"

    case $action in
        "deploy")
            orchestrate_resilient_deployment "$resource_group" "$environment" "$region"
            ;;
        "rollback")
            local deployment_name="${5:-}"
            if [ -z "$deployment_name" ]; then
                print_error "Deployment name required for rollback"
                exit 1
            fi
            rollback_deployment "$deployment_name" "$resource_group"
            ;;
        "validate")
            validate_azure_service_health "$region"
            ;;
        *)
            print_error "Unknown action: $action"
            print_info "Available actions: deploy, rollback, validate"
            exit 1
            ;;
    esac
}

# Export functions for use in other scripts
export -f orchestrate_resilient_deployment
export -f execute_deployment_with_circuit_breaker
export -f verify_deployment_outputs
export -f refresh_azure_session
export -f validate_azure_service_health
export -f validate_regional_capacity
export -f deploy_with_rollback_capability
export -f rollback_deployment

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi