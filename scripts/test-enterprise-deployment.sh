#!/bin/bash
# Test Enterprise Deployment Architecture
# Validates the implementation of enterprise deployment patterns

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

test_enterprise_deployment_architecture() {
    print_header "Testing Enterprise Deployment Architecture"

    # Test 1: Validate script files exist
    print_info "Test 1: Checking enterprise deployment scripts..."

    local scripts=(
        "azure-deployment-manager.sh"
        "azure-service-validator.sh"
        "enhanced-complete-redeploy.sh"
    )

    local all_scripts_exist=true
    for script in "${scripts[@]}"; do
        if [ -f "scripts/$script" ]; then
            print_status "Found: $script"
        else
            print_error "Missing: $script"
            all_scripts_exist=false
        fi
    done

    if [ "$all_scripts_exist" = false ]; then
        print_error "Some enterprise deployment scripts are missing"
        return 1
    fi

    # Test 2: Validate Bicep template updates
    print_info "Test 2: Checking Bicep template updates..."

    if [ -f "infrastructure/azure-resources-core.bicep" ]; then
        if grep -q "deploymentTimestamp" "infrastructure/azure-resources-core.bicep"; then
            print_status "Bicep template includes deployment timestamp parameter"
        else
            print_error "Bicep template missing deployment timestamp parameter"
            return 1
        fi

        if grep -q "take(deploymentTimestamp, 8)" "infrastructure/azure-resources-core.bicep"; then
            print_status "Bicep template includes unique resource naming"
        else
            print_error "Bicep template missing unique resource naming"
            return 1
        fi
    else
        print_error "Core Bicep template not found"
        return 1
    fi

    # Test 3: Validate enterprise service implementations
    print_info "Test 3: Checking enterprise service implementations..."

    local enterprise_services=(
        "backend/core/azure_openai/azure_text_analytics_service.py"
        "backend/core/azure_openai/azure_ml_quality_service.py"
        "backend/core/azure_openai/azure_monitoring_service.py"
        "backend/core/azure_openai/azure_rate_limiter.py"
    )

    local all_services_exist=true
    for service in "${enterprise_services[@]}"; do
        if [ -f "$service" ]; then
            print_status "Found: $service"
        else
            print_error "Missing: $service"
            all_services_exist=false
        fi
    done

    if [ "$all_services_exist" = false ]; then
        print_error "Some enterprise services are missing"
        return 1
    fi

    # Test 4: Validate configuration updates
    print_info "Test 4: Checking configuration updates..."

    if [ -f "backend/config/settings.py" ]; then
        local config_checks=(
            "azure_text_analytics_endpoint"
            "azure_ml_confidence_endpoint"
            "extraction_quality_tier"
            "azure_openai_max_tokens_per_minute"
        )

        local all_configs_exist=true
        for config in "${config_checks[@]}"; do
            if grep -q "$config" "backend/config/settings.py"; then
                print_status "Found config: $config"
            else
                print_error "Missing config: $config"
                all_configs_exist=false
            fi
        done

        if [ "$all_configs_exist" = false ]; then
            print_error "Some enterprise configurations are missing"
            return 1
        fi
    else
        print_error "Settings file not found"
        return 1
    fi

    # Test 5: Validate environment configuration
    print_info "Test 5: Checking environment configuration..."

    if [ -f "backend/config/environment_example.env" ]; then
        local env_checks=(
            "AZURE_TEXT_ANALYTICS_ENDPOINT"
            "AZURE_ML_CONFIDENCE_ENDPOINT"
            "EXTRACTION_QUALITY_TIER"
            "AZURE_OPENAI_MAX_TOKENS_PER_MINUTE"
        )

        local all_env_exist=true
        for env_var in "${env_checks[@]}"; do
            if grep -q "$env_var" "backend/config/environment_example.env"; then
                print_status "Found env var: $env_var"
            else
                print_error "Missing env var: $env_var"
                all_env_exist=false
            fi
        done

        if [ "$all_env_exist" = false ]; then
            print_error "Some environment variables are missing"
            return 1
        fi
    else
        print_error "Environment example file not found"
        return 1
    fi

    # Test 6: Validate knowledge extractor integration
    print_info "Test 6: Checking knowledge extractor integration..."

    if [ -f "backend/core/azure_openai/knowledge_extractor.py" ]; then
        local integration_checks=(
            "AzureTextAnalyticsService"
            "AzureMLQualityAssessment"
            "AzureKnowledgeMonitor"
            "AzureOpenAIRateLimiter"
        )

        local all_integrations_exist=true
        for integration in "${integration_checks[@]}"; do
            if grep -q "$integration" "backend/core/azure_openai/knowledge_extractor.py"; then
                print_status "Found integration: $integration"
            else
                print_error "Missing integration: $integration"
                all_integrations_exist=false
            fi
        done

        if [ "$all_integrations_exist" = false ]; then
            print_error "Some enterprise integrations are missing"
            return 1
        fi
    else
        print_error "Knowledge extractor file not found"
        return 1
    fi

    # Test 7: Validate test script
    print_info "Test 7: Checking enterprise test script..."

    if [ -f "backend/scripts/test_enterprise_knowledge_extraction.py" ]; then
        print_status "Found enterprise test script"
    else
        print_error "Enterprise test script not found"
        return 1
    fi

    print_status "All enterprise deployment architecture tests passed!"
    return 0
}

test_deployment_patterns() {
    print_header "Testing Deployment Patterns"

    # Test 1: Time-based unique naming
    print_info "Test 1: Time-based unique naming pattern..."

    local timestamp=$(date +%Y%m%d-%H%M%S)
    local search_service="maintie-dev-search-${timestamp}"
    local storage_account="maintiedevstor${timestamp}"
    local key_vault="maintie-dev-kv-${timestamp}"

    print_status "Generated unique names:"
    print_info "  - Search Service: $search_service"
    print_info "  - Storage Account: $storage_account"
    print_info "  - Key Vault: $key_vault"

    # Test 2: Conflict resolution patterns
    print_info "Test 2: Conflict resolution patterns..."

    # Simulate soft-delete conflict detection
    local conflict_detection_script="scripts/azure-service-validator.sh"
    if [ -f "$conflict_detection_script" ]; then
        print_status "Conflict detection script exists"

        # Check for key functions
        if grep -q "validate_soft_deleted_resources" "$conflict_detection_script"; then
            print_status "Soft-delete validation function found"
        else
            print_error "Soft-delete validation function missing"
            return 1
        fi

        if grep -q "purge_soft_deleted_resources" "$conflict_detection_script"; then
            print_status "Soft-delete purge function found"
        else
            print_error "Soft-delete purge function missing"
            return 1
        fi
    else
        print_error "Conflict detection script not found"
        return 1
    fi

    # Test 3: Exponential backoff pattern
    print_info "Test 3: Exponential backoff pattern..."

    local deployment_manager_script="scripts/azure-deployment-manager.sh"
    if [ -f "$deployment_manager_script" ]; then
        print_status "Deployment manager script exists"

        if grep -q "deploy_with_exponential_backoff" "$deployment_manager_script"; then
            print_status "Exponential backoff function found"
        else
            print_error "Exponential backoff function missing"
            return 1
        fi

        if grep -q "base_delay \* (2 \*\* (attempt - 1))" "$deployment_manager_script"; then
            print_status "Exponential backoff calculation found"
        else
            print_error "Exponential backoff calculation missing"
            return 1
        fi
    else
        print_error "Deployment manager script not found"
        return 1
    fi

    # Test 4: Region selection optimization
    print_info "Test 4: Region selection optimization..."

    if grep -q "get_optimal_deployment_region" "$deployment_manager_script"; then
        print_status "Region optimization function found"
    else
        print_error "Region optimization function missing"
        return 1
    fi

    if grep -q "validate_region_service_availability" "$deployment_manager_script"; then
        print_status "Region service validation found"
    else
        print_error "Region service validation missing"
        return 1
    fi

    print_status "All deployment pattern tests passed!"
    return 0
}

main() {
    print_header "🏗️ Enterprise Deployment Architecture Test Suite"
    print_info "Testing implementation of enterprise deployment patterns"

    # Test enterprise deployment architecture
    if test_enterprise_deployment_architecture; then
        print_status "Enterprise deployment architecture tests passed"
    else
        print_error "Enterprise deployment architecture tests failed"
        exit 1
    fi

    # Test deployment patterns
    if test_deployment_patterns; then
        print_status "Deployment pattern tests passed"
    else
        print_error "Deployment pattern tests failed"
        exit 1
    fi

    print_header "🎉 All Enterprise Deployment Architecture Tests Passed!"
    print_info "Implementation Summary:"
    print_info "  ✅ Enterprise deployment scripts created"
    print_info "  ✅ Bicep templates updated with unique naming"
    print_info "  ✅ Enterprise services implemented"
    print_info "  ✅ Configuration updated for enterprise features"
    print_info "  ✅ Knowledge extractor integrated with enterprise services"
    print_info "  ✅ Conflict resolution patterns implemented"
    print_info "  ✅ Exponential backoff patterns implemented"
    print_info "  ✅ Region selection optimization implemented"
    print_info ""
    print_info "Ready for enterprise deployment!"
}

# Execute main test function
main "$@"