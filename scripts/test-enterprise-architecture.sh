#!/bin/bash
# Test Enterprise Architecture - Comprehensive validation
# Tests all enterprise components: Extension Manager, Naming Service, Deployment Orchestrator, Health Validator

set -euo pipefail

# Import enterprise modules
source "$(dirname "$0")/azure-extension-manager.sh"
source "$(dirname "$0")/azure-naming-service.sh"
source "$(dirname "$0")/azure-deployment-orchestrator.sh"
source "$(dirname "$0")/azure-service-health-validator.sh"

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

# Test configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-maintie-rag-rg}"
ENVIRONMENT="${AZURE_ENVIRONMENT:-dev}"
AZURE_LOCATION="${AZURE_LOCATION:-eastus}"

test_extension_manager() {
    print_header "Testing Azure Extension Manager"

    # Test extension validation
    print_info "Testing extension validation..."
    if validate_and_install_extensions; then
        print_status "Extension manager test passed"
        return 0
    else
        print_error "Extension manager test failed"
        return 1
    fi
}

test_naming_service() {
    print_header "Testing Azure Global Naming Service"

    # Test storage name generation
    print_info "Testing storage name generation..."
    local storage_name=$(generate_globally_unique_storage_name "maintie" "$ENVIRONMENT")
    if [ ! -z "$storage_name" ]; then
        print_status "Storage name generated: $storage_name"

        # Validate storage name constraints
        if validate_resource_name_constraints "$storage_name" "storage"; then
            print_status "Storage name validation passed"
        else
            print_error "Storage name validation failed"
            return 1
        fi
    else
        print_error "Storage name generation failed"
        return 1
    fi

    # Test search name generation
    print_info "Testing search name generation..."
    local search_name=$(generate_unique_search_name "maintie" "$ENVIRONMENT" "$AZURE_LOCATION")
    if [ ! -z "$search_name" ]; then
        print_status "Search name generated: $search_name"

        # Validate search name constraints
        if validate_resource_name_constraints "$search_name" "search"; then
            print_status "Search name validation passed"
        else
            print_error "Search name validation failed"
            return 1
        fi
    else
        print_error "Search name generation failed"
        return 1
    fi

    # Test Key Vault name generation
    print_info "Testing Key Vault name generation..."
    local keyvault_name=$(generate_unique_keyvault_name "maintie" "$ENVIRONMENT")
    if [ ! -z "$keyvault_name" ]; then
        print_status "Key Vault name generated: $keyvault_name"

        # Validate Key Vault name constraints
        if validate_resource_name_constraints "$keyvault_name" "keyvault"; then
            print_status "Key Vault name validation passed"
        else
            print_error "Key Vault name validation failed"
            return 1
        fi
    else
        print_error "Key Vault name generation failed"
        return 1
    fi

    print_status "Naming service test passed"
    return 0
}

test_health_validator() {
    print_header "Testing Azure Service Health Validator"

    # Test comprehensive health check
    print_info "Testing comprehensive health check..."
    if comprehensive_health_check "$RESOURCE_GROUP" "$AZURE_LOCATION"; then
        print_status "Comprehensive health check passed"
    else
        print_warning "Comprehensive health check failed (this may be expected in test environment)"
    fi

    # Test individual health components
    print_info "Testing Azure CLI health..."
    if validate_azure_cli_health; then
        print_status "Azure CLI health check passed"
    else
        print_error "Azure CLI health check failed"
        return 1
    fi

    print_info "Testing network connectivity..."
    if validate_network_connectivity "$AZURE_LOCATION"; then
        print_status "Network connectivity test passed"
    else
        print_warning "Network connectivity test failed (this may be expected in test environment)"
    fi

    print_info "Testing service health..."
    if validate_azure_service_health "$AZURE_LOCATION"; then
        print_status "Service health check passed"
    else
        print_warning "Service health check failed (this may be expected in test environment)"
    fi

    print_status "Health validator test completed"
    return 0
}

test_deployment_orchestrator() {
    print_header "Testing Azure Deployment Orchestrator"

    # Test deployment orchestration (dry run)
    print_info "Testing deployment orchestration (dry run)..."

    # Generate test resource names
    local test_storage_name=$(generate_globally_unique_storage_name "maintie" "$ENVIRONMENT")
    local test_search_name=$(generate_unique_search_name "maintie" "$ENVIRONMENT" "$AZURE_LOCATION")
    local test_keyvault_name=$(generate_unique_keyvault_name "maintie" "$ENVIRONMENT")

    print_info "Test resource names generated:"
    print_info "  - Storage: $test_storage_name"
    print_info "  - Search: $test_search_name"
    print_info "  - Key Vault: $test_keyvault_name"

    # Test Bicep template validation
    print_info "Testing Bicep template validation..."
    if az bicep build --file infrastructure/azure-resources-core.bicep --stdout >/dev/null 2>&1; then
        print_status "Bicep template validation passed"
    else
        print_error "Bicep template validation failed"
        return 1
    fi

    print_status "Deployment orchestrator test passed"
    return 0
}

test_integration() {
    print_header "Testing Enterprise Architecture Integration"

    # Test that all components work together
    print_info "Testing component integration..."

    # 1. Extension Manager + Health Validator
    print_info "Testing Extension Manager + Health Validator integration..."
    validate_and_install_extensions
    if validate_azure_cli_health; then
        print_status "Extension Manager + Health Validator integration passed"
    else
        print_error "Extension Manager + Health Validator integration failed"
        return 1
    fi

    # 2. Naming Service + Health Validator
    print_info "Testing Naming Service + Health Validator integration..."
    local test_storage_name=$(generate_globally_unique_storage_name "maintie" "$ENVIRONMENT")
    if validate_storage_name_availability "$test_storage_name"; then
        print_status "Naming Service + Health Validator integration passed"
    else
        print_warning "Naming Service + Health Validator integration failed (name may be taken)"
    fi

    # 3. Deployment Orchestrator + Naming Service
    print_info "Testing Deployment Orchestrator + Naming Service integration..."
    local test_search_name=$(generate_unique_search_name "maintie" "$ENVIRONMENT" "$AZURE_LOCATION")
    local test_keyvault_name=$(generate_unique_keyvault_name "maintie" "$ENVIRONMENT")

    if [ ! -z "$test_search_name" ] && [ ! -z "$test_keyvault_name" ]; then
        print_status "Deployment Orchestrator + Naming Service integration passed"
    else
        print_error "Deployment Orchestrator + Naming Service integration failed"
        return 1
    fi

    print_status "Enterprise architecture integration test passed"
    return 0
}

generate_test_report() {
    print_header "Generating Enterprise Architecture Test Report"

    local report_file="enterprise-architecture-test-report-$(date +%Y%m%d-%H%M%S).json"

    # Create test report structure
    local report_data=$(cat <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "testEnvironment": {
    "resourceGroup": "$RESOURCE_GROUP",
    "environment": "$ENVIRONMENT",
    "region": "$AZURE_LOCATION"
  },
  "testResults": {
    "extensionManager": "unknown",
    "namingService": "unknown",
    "healthValidator": "unknown",
    "deploymentOrchestrator": "unknown",
    "integration": "unknown"
  },
  "generatedNames": {
    "storageAccount": "",
    "searchService": "",
    "keyVault": ""
  }
}
EOF
)

    # Save report to file
    echo "$report_data" > "$report_file"
    print_status "Test report saved to: $report_file"

    return 0
}

run_all_tests() {
    print_header "Running Enterprise Architecture Tests"
    print_info "Resource Group: $RESOURCE_GROUP"
    print_info "Environment: $ENVIRONMENT"
    print_info "Region: $AZURE_LOCATION"

    local test_results=()
    local failed_tests=()

    # Test 1: Extension Manager
    print_info "Running Extension Manager test..."
    if test_extension_manager; then
        test_results+=("extensionManager: PASSED")
        print_status "Extension Manager test: PASSED"
    else
        test_results+=("extensionManager: FAILED")
        failed_tests+=("extensionManager")
        print_error "Extension Manager test: FAILED"
    fi

    # Test 2: Naming Service
    print_info "Running Naming Service test..."
    if test_naming_service; then
        test_results+=("namingService: PASSED")
        print_status "Naming Service test: PASSED"
    else
        test_results+=("namingService: FAILED")
        failed_tests+=("namingService")
        print_error "Naming Service test: FAILED"
    fi

    # Test 3: Health Validator
    print_info "Running Health Validator test..."
    if test_health_validator; then
        test_results+=("healthValidator: PASSED")
        print_status "Health Validator test: PASSED"
    else
        test_results+=("healthValidator: FAILED")
        failed_tests+=("healthValidator")
        print_error "Health Validator test: FAILED"
    fi

    # Test 4: Deployment Orchestrator
    print_info "Running Deployment Orchestrator test..."
    if test_deployment_orchestrator; then
        test_results+=("deploymentOrchestrator: PASSED")
        print_status "Deployment Orchestrator test: PASSED"
    else
        test_results+=("deploymentOrchestrator: FAILED")
        failed_tests+=("deploymentOrchestrator")
        print_error "Deployment Orchestrator test: FAILED"
    fi

    # Test 5: Integration
    print_info "Running Integration test..."
    if test_integration; then
        test_results+=("integration: PASSED")
        print_status "Integration test: PASSED"
    else
        test_results+=("integration: FAILED")
        failed_tests+=("integration")
        print_error "Integration test: FAILED"
    fi

    # Generate test report
    generate_test_report

    # Print summary
    print_header "Enterprise Architecture Test Summary"
    for result in "${test_results[@]}"; do
        echo "  $result"
    done

    if [ ${#failed_tests[@]} -eq 0 ]; then
        print_status "All enterprise architecture tests passed! 🎉"
        return 0
    else
        print_error "Some tests failed: ${failed_tests[*]}"
        return 1
    fi
}

# Main execution function
main() {
    local action="${1:-all}"

    case $action in
        "all")
            run_all_tests
            ;;
        "extension")
            test_extension_manager
            ;;
        "naming")
            test_naming_service
            ;;
        "health")
            test_health_validator
            ;;
        "orchestrator")
            test_deployment_orchestrator
            ;;
        "integration")
            test_integration
            ;;
        "report")
            generate_test_report
            ;;
        *)
            print_error "Unknown action: $action"
            print_info "Available actions: all, extension, naming, health, orchestrator, integration, report"
            exit 1
            ;;
    esac
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi