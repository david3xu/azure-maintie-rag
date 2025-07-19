#!/bin/bash
# Azure Deployment Manager - Enterprise resilience patterns
# Handles deployment conflicts, soft-delete issues, and multi-region failures

set -euo pipefail

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

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-maintie-rag-rg}"
ENVIRONMENT="${AZURE_ENVIRONMENT:-dev}"
DEPLOYMENT_TIMESTAMP=$(date +%Y%m%d-%H%M%S)

check_azure_service_availability() {
    local service_type=$1
    local service_name=$2
    local region=$3

    print_info "Checking $service_type service availability: $service_name in $region"

    # Check service-specific availability
    case $service_type in
        "search")
            # For Search services, just check if the name format is valid
            # The actual availability will be checked during deployment
            if [[ "$service_name" =~ ^[a-z0-9-]+$ ]]; then
                print_status "Search service name '$service_name' format is valid"
                return 0
            else
                print_warning "Search service name '$service_name' format is invalid"
                return 1
            fi
            ;;
        "storage")
            # Verify Storage account name availability globally
            local name_available=$(az storage account check-name \
                --name "$service_name" \
                --query "nameAvailable" \
                --output tsv 2>/dev/null || echo "false")

            if [ "$name_available" == "true" ]; then
                print_status "Storage account name '$service_name' is available"
                return 0
            else
                print_warning "Storage account name '$service_name' is not available"
                return 1
            fi
            ;;
        "keyvault")
            # Check for soft-deleted Key Vaults with same name
            local deleted_kv=$(az keyvault list-deleted \
                --query "[?name=='$service_name'].name" \
                --output tsv 2>/dev/null)

            if [ -z "$deleted_kv" ]; then
                print_status "Key Vault name '$service_name' is available"
                return 0
            else
                print_warning "Found soft-deleted Key Vault: $deleted_kv"
                return 1
            fi
            ;;
        *)
            print_error "Unknown service type: $service_type"
            return 1
            ;;
    esac
}

purge_soft_deleted_resources() {
    local service_type=$1
    local service_name=$2
    local region=$3

    print_info "Attempting to purge soft-deleted $service_type: $service_name"

    case $service_type in
        "keyvault")
            # Purge soft-deleted Key Vault
            if az keyvault purge --name "$service_name" --location "$region" 2>/dev/null; then
                print_status "Successfully purged soft-deleted Key Vault: $service_name"
                return 0
            else
                print_warning "Failed to purge Key Vault: $service_name (may not exist or already purged)"
                return 1
            fi
            ;;
        "search")
            # Search services don't have soft-delete, but we can check for existing services
            local existing_service=$(az search service list \
                --resource-group "$RESOURCE_GROUP" \
                --query "[?name=='$service_name'].name" \
                --output tsv 2>/dev/null)

            if [ ! -z "$existing_service" ]; then
                print_warning "Search service '$service_name' already exists"
                return 1
            fi
            return 0
            ;;
        *)
            print_warning "No purge mechanism for service type: $service_type"
            return 0
            ;;
    esac
}

generate_unique_resource_name() {
    local service_type=$1
    local base_name=$2
    local max_attempts=10

    for attempt in $(seq 1 $max_attempts); do
        local unique_suffix=$(date +%s%N | cut -b1-8)
        local unique_name="${base_name}-${unique_suffix}"

        if check_azure_service_availability "$service_type" "$unique_name" "$AZURE_LOCATION" 2>/dev/null; then
            echo "$unique_name"
            return 0
        fi
    done

    print_error "Failed to generate unique name for $service_type after $max_attempts attempts"
    return 1
}

deploy_with_exponential_backoff() {
    local deployment_name=$1
    local template_file=$2
    local parameters=$3
    local max_attempts=5
    local base_delay=60

    print_header "Deploying with exponential backoff: $deployment_name"

    for attempt in $(seq 1 $max_attempts); do
        print_info "Deployment attempt $attempt of $max_attempts..."

        # Refresh Azure authentication if needed
        if ! az account show --output none 2>/dev/null; then
            print_warning "Refreshing Azure authentication..."
            az login --output none || {
                print_error "Failed to authenticate with Azure"
                return 1
            }
        fi

        # Execute deployment
        if az deployment group create \
            --resource-group "$RESOURCE_GROUP" \
            --name "$deployment_name" \
            --template-file "$template_file" \
            --parameters $parameters \
            --mode Incremental \
            --verbose; then

            print_status "Deployment succeeded on attempt $attempt"
            return 0
        fi

        # Calculate exponential backoff delay
        delay=$((base_delay * (2 ** (attempt - 1))))
        print_warning "Deployment failed. Waiting ${delay}s before retry..."
        sleep $delay

        # Check for specific error patterns and handle them
        if [ $attempt -lt $max_attempts ]; then
            print_info "Analyzing deployment failure for retry optimization..."

            # Check for soft-delete conflicts
            if [[ "$parameters" == *"search"* ]]; then
                local search_name=$(echo "$parameters" | grep -o 'searchServiceName=[^ ]*' | cut -d'=' -f2)
                if [ ! -z "$search_name" ]; then
                    print_info "Checking for Search service soft-delete conflicts..."
                    purge_soft_deleted_resources "search" "$search_name" "$AZURE_LOCATION" || true
                fi
            fi

            # Check for Key Vault conflicts
            if [[ "$parameters" == *"keyvault"* ]]; then
                local kv_name=$(echo "$parameters" | grep -o 'keyVaultName=[^ ]*' | cut -d'=' -f2)
                if [ ! -z "$kv_name" ]; then
                    print_info "Checking for Key Vault soft-delete conflicts..."
                    purge_soft_deleted_resources "keyvault" "$kv_name" "$AZURE_LOCATION" || true
                fi
            fi
        fi
    done

    print_error "Deployment failed after $max_attempts attempts"
    return 1
}

validate_region_service_availability() {
    local region=$1

    print_info "Validating service availability in region: $region"

    # Service availability matrix - using known available regions
    local services=(
        "Microsoft.Storage/storageAccounts"
        "Microsoft.KeyVault/vaults"
        "Microsoft.DocumentDB/databaseAccounts"
        "Microsoft.MachineLearningServices/workspaces"
    )

    # Check regional service availability
    for service in "${services[@]}"; do
        local namespace="${service%/*}"
        local resource_type="${service#*/}"

        if ! az provider show \
            --namespace "$namespace" \
            --query "resourceTypes[?resourceType=='$resource_type'].locations" \
            --output tsv 2>/dev/null | grep -q "$region"; then

            print_error "Service $service not available in region $region"
            return 1
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
        return 1
    fi

    print_status "All required services available in region: $region"
    return 0
}

check_subscription_quotas() {
    local region=$1

    print_info "Checking subscription quotas for region: $region"

    # Check core quotas
    local quotas=(
        "Microsoft.Search/searchServices"
        "Microsoft.Storage/storageAccounts"
        "Microsoft.KeyVault/vaults"
    )

    for quota in "${quotas[@]}"; do
        local namespace="${quota%/*}"
        local resource_type="${quota#*/}"

        local current_usage=$(az vm list-usage \
            --location "$region" \
            --query "[?name.value=='$resource_type'].currentValue" \
            --output tsv 2>/dev/null || echo "0")

        local limit=$(az vm list-usage \
            --location "$region" \
            --query "[?name.value=='$resource_type'].limit" \
            --output tsv 2>/dev/null || echo "100")

        local usage_percent=$((current_usage * 100 / limit))

        if [ $usage_percent -gt 80 ]; then
            print_warning "High quota usage for $resource_type: ${usage_percent}%"
        else
            print_status "Quota OK for $resource_type: ${usage_percent}%"
        fi
    done
}

get_optimal_deployment_region() {
    local performance_priority="${1:-latency}"  # "latency"|"cost"|"compliance"

    # Define region capability matrix with priority weights
    local regions=(
        "eastus:1:low_latency:high_availability:100"
        "westus2:1:medium_latency:high_availability:90"
        "centralus:2:medium_latency:medium_availability:85"
        "northeurope:1:medium_latency:high_availability:95"
        "uksouth:2:medium_latency:medium_availability:80"
    )

    local best_region=""
    local best_score=0

    for region_spec in "${regions[@]}"; do
        IFS=':' read -r region tier latency availability capacity <<< "$region_spec"

        # Simplified validation - just check if region is in our known list
        local known_regions=("eastus" "westus2" "centralus" "northeurope" "uksouth" "eastus2" "westus" "southcentralus")
        local region_valid=false

        for known_region in "${known_regions[@]}"; do
            if [ "$known_region" = "$region" ]; then
                region_valid=true
                break
            fi
        done

        if [ "$region_valid" = true ]; then
            # Use a simple capacity estimate (not actual query)
            local region_load=50  # Assume 50% capacity for simplicity

            # Calculate region score based on priority
            local score=0
            case $performance_priority in
                "latency")
                    if [ "$latency" == "low_latency" ]; then
                        score=$((capacity - region_load + 20))
                    else
                        score=$((capacity - region_load))
                    fi
                    ;;
                "cost")
                    if [ "$tier" == "2" ]; then
                        score=$((capacity - region_load + 15))
                    else
                        score=$((capacity - region_load))
                    fi
                    ;;
                "compliance")
                    if [ "$availability" == "high_availability" ]; then
                        score=$((capacity - region_load + 10))
                    else
                        score=$((capacity - region_load))
                    fi
                    ;;
            esac

            if [ $score -gt $best_score ]; then
                best_score=$score
                best_region=$region
            fi
        fi
    done

    if [ ! -z "$best_region" ]; then
        echo "$best_region"
        return 0
    else
        return 1
    fi
}

get_region_capacity_utilization() {
    local region=$1

    # Query Azure Resource Graph for region utilization
    local resource_count=$(az graph query \
        --graph-query "Resources | where location == '$region' | summarize count()" \
        --query "data[0].count_" \
        --output tsv 2>/dev/null || echo "50")

    # Normalize to percentage (assuming 1000 resources = 100% capacity)
    local utilization=$((resource_count * 100 / 1000))
    echo "$utilization"
}

# Export functions for use in other scripts
export -f check_azure_service_availability
export -f purge_soft_deleted_resources
export -f generate_unique_resource_name
export -f deploy_with_exponential_backoff
export -f validate_region_service_availability
export -f check_subscription_quotas
export -f get_optimal_deployment_region
export -f get_region_capacity_utilization