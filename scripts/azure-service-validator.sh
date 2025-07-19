#!/bin/bash
# Azure Service Availability Validator
# Pre-deployment service availability checking and conflict resolution

set -euo pipefail

# Import deployment manager functions
source "$(dirname "$0")/azure-deployment-manager.sh"

validate_azure_service_prerequisites() {
    local region=$1
    local environment=$2

    print_header "Validating Azure service prerequisites for region: $region"

    # Service availability matrix - using known available regions
    local services=(
        "Microsoft.Storage/storageAccounts"
        "Microsoft.KeyVault/vaults"
        "Microsoft.DocumentDB/databaseAccounts"
        "Microsoft.MachineLearningServices/workspaces"
        "Microsoft.CognitiveServices/accounts"
    )

    local all_services_available=true

    # Check regional service availability
    for service in "${services[@]}"; do
        local namespace="${service%/*}"
        local resource_type="${service#*/}"

        print_info "Checking availability of $service in $region"

        if ! az provider show \
            --namespace "$namespace" \
            --query "resourceTypes[?resourceType=='$resource_type'].locations" \
            --output tsv 2>/dev/null | grep -q "$region"; then

            print_error "Service $service not available in region $region"
            all_services_available=false
        else
            print_status "Service $service available in region $region"
        fi
    done

    # Special handling for Search services - use known available regions
    local search_available_regions=("eastus" "westus2" "centralus" "northeurope" "uksouth" "eastus2" "westus" "southcentralus")
    local search_available=false

    for search_region in "${search_available_regions[@]}"; do
        if [ "$search_region" = "$region" ]; then
            search_available=true
            break
        fi
    done

    if [ "$search_available" = false ]; then
        print_error "Search service not available in region $region"
        all_services_available=false
    else
        print_status "Search service available in region $region"
    fi

    if [ "$all_services_available" = false ]; then
        print_error "Not all required services are available in region $region"
        return 1
    fi

    # Check subscription quotas
    if ! validate_subscription_quotas "$region"; then
        print_error "Subscription quota validation failed for region $region"
        return 1
    fi

    # Check for existing soft-deleted resources
    if ! validate_soft_deleted_resources "$region" "$environment"; then
        print_error "Soft-deleted resource validation failed for region $region"
        return 1
    fi

    print_status "All service prerequisites validated successfully for region: $region"
    return 0
}

validate_subscription_quotas() {
    local region=$1

    print_info "Checking subscription quotas for region: $region"

    # Core quotas to check
    local quotas=(
        "Microsoft.Search/searchServices"
        "Microsoft.Storage/storageAccounts"
        "Microsoft.KeyVault/vaults"
        "Microsoft.DocumentDB/databaseAccounts"
    )

    local quota_issues=false

    for quota in "${quotas[@]}"; do
        local namespace="${quota%/*}"
        local resource_type="${quota#*/}"

        # Get current usage and limits
        local current_usage=$(az vm list-usage \
            --location "$region" \
            --query "[?name.value=='$resource_type'].currentValue" \
            --output tsv 2>/dev/null || echo "0")

        local limit=$(az vm list-usage \
            --location "$region" \
            --query "[?name.value=='$resource_type'].limit" \
            --output tsv 2>/dev/null || echo "100")

        local usage_percent=$((current_usage * 100 / limit))

        print_info "Quota for $resource_type: $current_usage/$limit (${usage_percent}%)"

        if [ $usage_percent -gt 90 ]; then
            print_error "Critical quota usage for $resource_type: ${usage_percent}%"
            quota_issues=true
        elif [ $usage_percent -gt 80 ]; then
            print_warning "High quota usage for $resource_type: ${usage_percent}%"
        else
            print_status "Quota OK for $resource_type: ${usage_percent}%"
        fi
    done

    if [ "$quota_issues" = true ]; then
        return 1
    fi

    return 0
}

validate_soft_deleted_resources() {
    local region=$1
    local environment=$2

    print_info "Checking for soft-deleted resources that could cause conflicts..."

    local conflicts_found=false

    # Check Key Vault soft deletes
    local deleted_keyvaults=$(az keyvault list-deleted \
        --query "[?properties.location=='$region'].name" \
        --output tsv 2>/dev/null || echo "")

    if [ ! -z "$deleted_keyvaults" ]; then
        print_warning "Found soft-deleted Key Vaults in $region:"
        echo "$deleted_keyvaults" | while read -r kv; do
            if [[ "$kv" == *"$environment"* ]]; then
                print_info "Found environment-specific soft-deleted Key Vault: $kv"
                conflicts_found=true
            fi
        done
    fi

    # Check for existing resources with environment-specific names
    local existing_resources=$(az resource list \
        --resource-group "$RESOURCE_GROUP" \
        --query "[?contains(name, '$environment')].{name:name, type:type}" \
        --output tsv 2>/dev/null || echo "")

    if [ ! -z "$existing_resources" ]; then
        print_warning "Found existing resources with environment-specific names:"
        echo "$existing_resources" | while read -r resource; do
            print_info "Existing resource: $resource"
            conflicts_found=true
        done
    fi

    # Check for Search services with similar names
    local existing_search_services=$(az search service list \
        --resource-group "$RESOURCE_GROUP" \
        --query "[?contains(name, '$environment')].name" \
        --output tsv 2>/dev/null || echo "")

    if [ ! -z "$existing_search_services" ]; then
        print_warning "Found existing Search services with environment-specific names:"
        echo "$existing_search_services" | while read -r search; do
            print_info "Existing Search service: $search"
            conflicts_found=true
        done
    fi

    if [ "$conflicts_found" = true ]; then
        print_warning "Soft-deleted or conflicting resources found"
        return 1
    else
        print_status "No soft-deleted or conflicting resources found"
        return 0
    fi
}

cleanup_soft_deleted_resources() {
    local region=$1
    local environment=$2

    print_header "Cleaning up soft-deleted resources in region: $region"

    # Purge soft-deleted Key Vaults
    local deleted_keyvaults=$(az keyvault list-deleted \
        --query "[?properties.location=='$region' && contains(name, '$environment')].name" \
        --output tsv 2>/dev/null || echo "")

    if [ ! -z "$deleted_keyvaults" ]; then
        print_info "Purging soft-deleted Key Vaults..."
        echo "$deleted_keyvaults" | while read -r kv; do
            print_info "Purging soft-deleted Key Vault: $kv"
            if az keyvault purge --name "$kv" --location "$region" 2>/dev/null; then
                print_status "Successfully purged Key Vault: $kv"
            else
                print_warning "Failed to purge Key Vault: $kv (may already be purged)"
            fi
        done
    fi

    # Wait for purge operations to complete
    print_info "Waiting for purge operations to complete..."
    sleep 30

    print_status "Soft-deleted resource cleanup completed"
}

validate_resource_group_state() {
    local resource_group=$1
    local region=$2

    print_info "Validating resource group state: $resource_group in $region"

    # Check if resource group exists
    if ! az group show --name "$resource_group" --output none 2>/dev/null; then
        print_info "Resource group $resource_group does not exist, will be created"
        return 0
    fi

    # Check resource group location
    local rg_location=$(az group show \
        --name "$resource_group" \
        --query "location" \
        --output tsv 2>/dev/null || echo "")

    if [ "$rg_location" != "$region" ]; then
        print_warning "Resource group $resource_group exists in $rg_location, but deployment target is $region"
        print_warning "This may cause deployment issues"
        return 1
    fi

    # Check for existing resources that might conflict
    local resource_count=$(az resource list \
        --resource-group "$resource_group" \
        --query "length(@)" \
        --output tsv 2>/dev/null || echo "0")

    if [ "$resource_count" -gt 0 ]; then
        print_warning "Resource group $resource_group contains $resource_count existing resources"
        print_info "This may cause deployment conflicts"
        return 1
    fi

    print_status "Resource group $resource_group is ready for deployment"
    return 0
}

validate_deployment_parameters() {
    local environment=$1
    local region=$2
    local deployment_timestamp=$3

    print_info "Validating deployment parameters..."

    # Generate expected resource names
    local expected_search_service="maintie-${environment}-search-${deployment_timestamp}"
    local expected_storage_account="maintie${environment}stor${deployment_timestamp}"
    local expected_key_vault="maintie-${environment}-kv-${deployment_timestamp:0:8}"

    # Validate Search service name availability
    if ! check_azure_service_availability "search" "$expected_search_service" "$region"; then
        print_error "Search service name '$expected_search_service' is not available"
        return 1
    fi

    # Validate Storage account name availability
    if ! check_azure_service_availability "storage" "$expected_storage_account" "$region"; then
        print_error "Storage account name '$expected_storage_account' is not available"
        return 1
    fi

    # Validate Key Vault name availability
    if ! check_azure_service_availability "keyvault" "$expected_key_vault" "$region"; then
        print_error "Key Vault name '$expected_key_vault' is not available"
        return 1
    fi

    print_status "All deployment parameters validated successfully"
    return 0
}

create_resource_group_if_needed() {
    local resource_group=$1
    local region=$2

    print_info "Ensuring resource group exists: $resource_group in $region"

    if ! az group show --name "$resource_group" --output none 2>/dev/null; then
        print_info "Creating resource group: $resource_group in $region"
        if az group create --name "$resource_group" --location "$region"; then
            print_status "Successfully created resource group: $resource_group"
            return 0
        else
            print_error "Failed to create resource group: $resource_group"
            return 1
        fi
    else
        print_status "Resource group $resource_group already exists"
        return 0
    fi
}

# Export functions for use in other scripts
export -f validate_azure_service_prerequisites
export -f validate_subscription_quotas
export -f validate_soft_deleted_resources
export -f cleanup_soft_deleted_resources
export -f validate_resource_group_state
export -f validate_deployment_parameters
export -f create_resource_group_if_needed