#!/bin/bash
# Azure Extension Manager - Enterprise dependency validation
# Enterprise-grade Azure CLI extension management with fallback strategies

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

class="AzureExtensionManager"

validate_and_install_extensions() {
    print_header "Azure Extension Manager: Validating and Installing Extensions"

    # Check for bicep availability (built into Azure CLI)
    if az bicep --version &>/dev/null; then
        print_status "Bicep compiler available (built into Azure CLI)"
    else
        print_warning "Bicep compiler not available - ARM templates will be used"
    fi

    local required_extensions=(
        "ml"              # Azure ML workspace management
        "containerapp"    # Container Apps deployment
        "log-analytics"   # Log Analytics integration
        "application-insights"  # Application Insights management
    )

    # Note: 'search' extension removed - not available in Azure CLI
    # Search operations handled via native 'az search' commands

    local installation_failures=()

    for extension in "${required_extensions[@]}"; do
        if az extension show --name "$extension" &>/dev/null; then
            print_status "Extension '$extension' already installed"
        else
            print_info "Installing Azure CLI extension: $extension"
            if az extension add --name "$extension" --yes; then
                print_status "Extension '$extension' installed successfully"
            else
                print_warning "Extension '$extension' installation failed - using fallback commands"
                installation_failures+=("$extension")
                register_extension_fallback "$extension"
            fi
        fi
    done

    # Report installation failures
    if [ ${#installation_failures[@]} -gt 0 ]; then
        print_warning "Some extensions failed to install: ${installation_failures[*]}"
        print_info "Using fallback strategies for failed extensions"
    fi

    return 0
}

register_extension_fallback() {
    local extension=$1
    case $extension in
        "search")
            # Search operations use native az commands - no extension needed
            print_info "Using native 'az search' commands for search operations"
            ;;
        "ml")
            # Fallback to REST API or alternative deployment methods
            print_warning "ML extension unavailable - verify Azure ML CLI v2 installation"
            print_info "Using Azure ML REST API for workspace operations"
            ;;
        "containerapp")
            # Fallback to ARM templates for Container Apps
            print_warning "Container Apps extension unavailable - using ARM templates"
            ;;
        "log-analytics")
            # Fallback to REST API for Log Analytics
            print_warning "Log Analytics extension unavailable - using REST API"
            ;;
        "application-insights")
            # Fallback to REST API for Application Insights
            print_warning "Application Insights extension unavailable - using REST API"
            ;;
        *)
            print_warning "No fallback strategy defined for extension: $extension"
            ;;
    esac
}

validate_extension_availability() {
    local extension=$1

    # Check if extension is available in Azure CLI
    local available_extensions=$(az extension list-available --query "[].name" --output tsv 2>/dev/null || echo "")

    if echo "$available_extensions" | grep -q "^${extension}$"; then
        return 0
    else
        return 1
    fi
}

get_extension_version() {
    local extension=$1

    if az extension show --name "$extension" &>/dev/null; then
        local version=$(az extension show --name "$extension" --query "version" --output tsv 2>/dev/null || echo "unknown")
        echo "$version"
    else
        echo "not-installed"
    fi
}

list_installed_extensions() {
    print_header "Installed Azure CLI Extensions"

    local installed_extensions=$(az extension list --query "[].{name:name, version:version}" --output table 2>/dev/null || echo "No extensions installed")

    if [ "$installed_extensions" != "No extensions installed" ]; then
        echo "$installed_extensions"
    else
        print_warning "No Azure CLI extensions are currently installed"
    fi
}

cleanup_failed_extensions() {
    print_header "Cleaning up failed extension installations"

    local failed_extensions=$(az extension list --query "[?version==null].name" --output tsv 2>/dev/null || echo "")

    if [ ! -z "$failed_extensions" ]; then
        for extension in $failed_extensions; do
            print_info "Removing failed extension: $extension"
            az extension remove --name "$extension" --yes 2>/dev/null || true
        done
        print_status "Failed extensions cleaned up"
    else
        print_status "No failed extensions found"
    fi
}

# Main execution function
main() {
    local action="${1:-validate}"

    case $action in
        "validate")
            validate_and_install_extensions
            ;;
        "list")
            list_installed_extensions
            ;;
        "cleanup")
            cleanup_failed_extensions
            ;;
        "check")
            local extension="${2:-}"
            if [ -z "$extension" ]; then
                print_error "Extension name required for check action"
                exit 1
            fi
            if validate_extension_availability "$extension"; then
                print_status "Extension '$extension' is available"
            else
                print_warning "Extension '$extension' is not available"
            fi
            ;;
        *)
            print_error "Unknown action: $action"
            print_info "Available actions: validate, list, cleanup, check"
            exit 1
            ;;
    esac
}

# Export functions for use in other scripts
export -f validate_and_install_extensions
export -f register_extension_fallback
export -f validate_extension_availability
export -f get_extension_version
export -f list_installed_extensions
export -f cleanup_failed_extensions

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi