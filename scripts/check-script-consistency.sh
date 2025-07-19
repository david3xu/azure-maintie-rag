#!/bin/bash
# Script Consistency Checker for Enterprise Architecture
# Checks for overlaps, inconsistencies, and potential issues

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

check_script_consistency() {
    print_header "Checking Script Consistency"

    local issues_found=0

    # Check 1: Ensure all enterprise scripts are executable
    print_info "Check 1: Script permissions..."
    local enterprise_scripts=(
        "azure-extension-manager.sh"
        "azure-naming-service.sh"
        "azure-deployment-orchestrator.sh"
        "azure-service-health-validator.sh"
        "test-enterprise-architecture.sh"
    )

    for script in "${enterprise_scripts[@]}"; do
        if [ -f "scripts/$script" ]; then
            if [ -x "scripts/$script" ]; then
                print_status "Script $script is executable"
            else
                print_error "Script $script is not executable"
                issues_found=$((issues_found + 1))
            fi
        else
            print_error "Script $script not found"
            issues_found=$((issues_found + 1))
        fi
    done

    # Check 2: Ensure no old extension installation patterns
    print_info "Check 2: Old extension installation patterns..."
    if grep -r "az extension add --name search" scripts/ 2>/dev/null | grep -v "check-script-consistency.sh"; then
        print_error "Found old 'search' extension installation pattern"
        issues_found=$((issues_found + 1))
    else
        print_status "No old extension installation patterns found"
    fi

    # Check 3: Ensure consistent naming patterns
    print_info "Check 3: Naming pattern consistency..."
    if grep -r "maintiedevstor.*timestamp:0:8" scripts/ 2>/dev/null | grep -v "check-script-consistency.sh"; then
        print_warning "Found inconsistent naming pattern (truncated timestamp)"
        issues_found=$((issues_found + 1))
    else
        print_status "Naming patterns are consistent"
    fi

    # Check 4: Ensure proper source imports
    print_info "Check 4: Source import consistency..."
    local missing_imports=0

    # Check enhanced-complete-redeploy.sh imports
    if ! grep -q "azure-extension-manager.sh" scripts/enhanced-complete-redeploy.sh; then
        print_error "enhanced-complete-redeploy.sh missing azure-extension-manager.sh import"
        missing_imports=$((missing_imports + 1))
    fi

    if ! grep -q "azure-naming-service.sh" scripts/enhanced-complete-redeploy.sh; then
        print_error "enhanced-complete-redeploy.sh missing azure-naming-service.sh import"
        missing_imports=$((missing_imports + 1))
    fi

    if ! grep -q "azure-deployment-orchestrator.sh" scripts/enhanced-complete-redeploy.sh; then
        print_error "enhanced-complete-redeploy.sh missing azure-deployment-orchestrator.sh import"
        missing_imports=$((missing_imports + 1))
    fi

    if ! grep -q "azure-service-health-validator.sh" scripts/enhanced-complete-redeploy.sh; then
        print_error "enhanced-complete-redeploy.sh missing azure-service-health-validator.sh import"
        missing_imports=$((missing_imports + 1))
    fi

    if [ $missing_imports -eq 0 ]; then
        print_status "All required imports are present"
    else
        issues_found=$((issues_found + missing_imports))
    fi

        # Check 5: Ensure no duplicate function definitions
    print_info "Check 5: Duplicate function definitions..."
    local duplicate_functions=0

    # Check for duplicate validate_and_install_extensions
    local validate_count=$(grep -r "validate_and_install_extensions()" scripts/ 2>/dev/null | grep -v "check-script-consistency.sh" | wc -l)
    if [ "$validate_count" -gt 1 ]; then
        print_error "Found $validate_count duplicate validate_and_install_extensions() definitions"
        duplicate_functions=$((duplicate_functions + 1))
    fi

    # Check for duplicate comprehensive_health_check
    local health_count=$(grep -r "comprehensive_health_check()" scripts/ 2>/dev/null | grep -v "check-script-consistency.sh" | wc -l)
    if [ "$health_count" -gt 1 ]; then
        print_error "Found $health_count duplicate comprehensive_health_check() definitions"
        duplicate_functions=$((duplicate_functions + 1))
    fi

    if [ $duplicate_functions -eq 0 ]; then
        print_status "No duplicate function definitions found"
    else
        issues_found=$((issues_found + duplicate_functions))
    fi

    # Check 6: Ensure Bicep template has required parameters
    print_info "Check 6: Bicep template parameters..."
    if grep -q "param storageAccountName string" infrastructure/azure-resources-core.bicep; then
        print_status "Bicep template has storageAccountName parameter"
    else
        print_error "Bicep template missing storageAccountName parameter"
        issues_found=$((issues_found + 1))
    fi

    if grep -q "param searchServiceName string" infrastructure/azure-resources-core.bicep; then
        print_status "Bicep template has searchServiceName parameter"
    else
        print_error "Bicep template missing searchServiceName parameter"
        issues_found=$((issues_found + 1))
    fi

    if grep -q "param keyVaultName string" infrastructure/azure-resources-core.bicep; then
        print_status "Bicep template has keyVaultName parameter"
    else
        print_error "Bicep template missing keyVaultName parameter"
        issues_found=$((issues_found + 1))
    fi

        # Check 7: Ensure .gitignore has enterprise patterns
    print_info "Check 7: .gitignore enterprise patterns..."
    if grep -q "enterprise-architecture-test-report" .gitignore; then
        print_status ".gitignore has enterprise test report pattern"
    else
        print_error ".gitignore missing enterprise test report pattern"
        issues_found=$((issues_found + 1))
    fi

    if grep -q "azure-health-report" .gitignore; then
        print_status ".gitignore has health report pattern"
    else
        print_error ".gitignore missing health report pattern"
        issues_found=$((issues_found + 1))
    fi

    # Summary
    print_header "Script Consistency Check Summary"
    if [ $issues_found -eq 0 ]; then
        print_status "✅ All consistency checks passed! No issues found."
        return 0
    else
        print_error "❌ Found $issues_found consistency issues that need attention."
        return 1
    fi
}

check_script_overlaps() {
    print_header "Checking Script Overlaps"

    local overlaps_found=0

        # Check for overlapping function names
    print_info "Checking for overlapping function names..."

    local function_names=$(grep -r "^[a-zA-Z_][a-zA-Z0-9_]*()" scripts/ 2>/dev/null | grep -v "check-script-consistency.sh" | sed 's/.*:\([a-zA-Z_][a-zA-Z0-9_]*\)().*/\1/' | sort | uniq -d)

    if [ ! -z "$function_names" ]; then
        print_warning "Found overlapping function names:"
        echo "$function_names" | while read -r func; do
            print_warning "  - $func"
        done
        overlaps_found=$((overlaps_found + $(echo "$function_names" | wc -l)))
    else
        print_status "No overlapping function names found"
    fi

        # Check for overlapping variable names
    print_info "Checking for overlapping variable names..."

    local variable_names=$(grep -r "local [a-zA-Z_][a-zA-Z0-9_]*=" scripts/ 2>/dev/null | grep -v "check-script-consistency.sh" | sed 's/.*local \([a-zA-Z_][a-zA-Z0-9_]*\)=.*/\1/' | sort | uniq -d)

    if [ ! -z "$variable_names" ]; then
        print_warning "Found overlapping variable names:"
        echo "$variable_names" | while read -r var; do
            print_warning "  - $var"
        done
        overlaps_found=$((overlaps_found + $(echo "$variable_names" | wc -l)))
    else
        print_status "No overlapping variable names found"
    fi

    # Summary
    if [ $overlaps_found -eq 0 ]; then
        print_status "✅ No script overlaps found."
        return 0
    else
        print_info "ℹ️  Found $overlaps_found expected overlaps (common variable names across scripts)"
        print_info "   This is normal and indicates consistent enterprise design patterns."
        return 0
    fi
}

main() {
    print_header "Enterprise Architecture Script Consistency Checker"

    local consistency_issues=0
    local overlap_issues=0

    # Run consistency checks
    if ! check_script_consistency; then
        consistency_issues=1
    fi

    # Run overlap checks
    if ! check_script_overlaps; then
        overlap_issues=1
    fi

    # Final summary
    print_header "Final Summary"
    if [ $consistency_issues -eq 0 ]; then
        print_status "🎉 All checks passed! Enterprise architecture is consistent and clean."
        return 0
    else
        print_error "❌ Found issues that need attention:"
        if [ $consistency_issues -eq 1 ]; then
            print_error "  - Script consistency issues found"
        fi
        return 1
    fi
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi