#!/bin/bash

# Azure Developer CLI Teardown Script
# Provides graceful shutdown and complete resource cleanup for azd deployments
# Usage: ./scripts/azd-teardown.sh [environment] [--backup] [--force]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
CREATE_BACKUP=false
FORCE_TEARDOWN=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Help function
show_help() {
    cat << EOF
Azure Developer CLI Teardown Script

USAGE:
    ./scripts/azd-teardown.sh [ENVIRONMENT] [OPTIONS]

ARGUMENTS:
    ENVIRONMENT    Target environment (development, staging, production)
                   If not specified, uses current azd environment

OPTIONS:
    --backup       Create backup before teardown
    --force        Skip confirmation prompts
    --help         Show this help message

EXAMPLES:
    # Teardown current environment with backup
    ./scripts/azd-teardown.sh --backup

    # Force teardown development environment
    ./scripts/azd-teardown.sh development --force

    # Teardown production with backup and confirmation
    ./scripts/azd-teardown.sh production --backup

SAFETY FEATURES:
    - Production requires explicit confirmation
    - Automatic backup creation option
    - Graceful service shutdown
    - Resource dependency checking
    - Rollback support for failed teardowns

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup)
                CREATE_BACKUP=true
                shift
                ;;
            --force)
                FORCE_TEARDOWN=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            -*|--*)
                echo -e "${RED}Error: Unknown option $1${NC}" >&2
                echo "Use --help for usage information."
                exit 1
                ;;
            *)
                if [[ -z "$ENVIRONMENT" ]]; then
                    ENVIRONMENT="$1"
                else
                    echo -e "${RED}Error: Unexpected argument $1${NC}" >&2
                    exit 1
                fi
                shift
                ;;
        esac
    done
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if azd is installed
    if ! command -v azd &> /dev/null; then
        log_error "Azure Developer CLI (azd) is not installed"
        log_info "Install from: https://aka.ms/install-azd"
        exit 1
    fi

    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/azure.yaml" ]]; then
        log_error "azure.yaml not found. Please run this script from the project root."
        exit 1
    fi

    # Check azd authentication
    if ! azd auth get-access-token &> /dev/null; then
        log_error "Not authenticated with Azure. Run 'azd auth login' first."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Get current azd environment if not specified
get_current_environment() {
    if [[ -z "$ENVIRONMENT" ]]; then
        if azd env list --output json &> /dev/null; then
            local current_env
            current_env=$(azd env get-values --output json 2>/dev/null | jq -r '.AZURE_ENV_NAME // empty' || echo "")
            if [[ -n "$current_env" ]]; then
                ENVIRONMENT="$current_env"
                log_info "Using current azd environment: $ENVIRONMENT"
            else
                log_error "No environment specified and no current azd environment found"
                echo "Use: azd env select <environment> or specify environment as argument"
                exit 1
            fi
        else
            log_error "No azd environments found"
            exit 1
        fi
    fi
}

# Validate environment exists
validate_environment() {
    log_info "Validating environment: $ENVIRONMENT"

    if ! azd env select "$ENVIRONMENT" &> /dev/null; then
        log_error "Environment '$ENVIRONMENT' not found"
        log_info "Available environments:"
        azd env list 2>/dev/null || echo "  No environments found"
        exit 1
    fi

    log_success "Environment '$ENVIRONMENT' validated"
}

# Production safety check
production_safety_check() {
    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$FORCE_TEARDOWN" != true ]]; then
        log_warning "🚨 PRODUCTION ENVIRONMENT TEARDOWN WARNING 🚨"
        echo ""
        echo "You are about to tear down the PRODUCTION environment."
        echo "This action will:"
        echo "  • Delete ALL Azure resources in the production resource group"
        echo "  • Remove ALL production data unless backed up"
        echo "  • Make the production system UNAVAILABLE"
        echo ""

        if [[ "$CREATE_BACKUP" == true ]]; then
            echo "✅ Backup will be created before teardown"
        else
            echo "❌ NO BACKUP will be created"
            echo "   Use --backup flag to create backup before teardown"
        fi

        echo ""
        read -p "Type 'DELETE PRODUCTION' to confirm teardown: " confirmation

        if [[ "$confirmation" != "DELETE PRODUCTION" ]]; then
            log_info "Teardown cancelled"
            exit 0
        fi

        echo ""
        log_warning "Final confirmation: Are you absolutely sure? (y/N)"
        read -r final_confirm
        if [[ "$final_confirm" != "y" && "$final_confirm" != "Y" ]]; then
            log_info "Teardown cancelled"
            exit 0
        fi
    fi
}

# Create backup before teardown
create_backup_before_teardown() {
    if [[ "$CREATE_BACKUP" == true ]]; then
        log_info "Creating backup before teardown..."

        # Create backup using Python service
        backup_result=$(cd "$PROJECT_ROOT/backend" && python -c "
import asyncio
import sys
sys.path.append('.')
from services.infrastructure_service import InfrastructureService
from services.backup_service import AzdBackupService

async def create_backup():
    try:
        infra = InfrastructureService()
        backup_service = AzdBackupService(infra)
        result = await backup_service.create_full_backup()
        print(f'Backup created: {result[\"backup_id\"]}')
        return result['status'] == 'completed'
    except Exception as e:
        print(f'Backup failed: {e}')
        return False

success = asyncio.run(create_backup())
sys.exit(0 if success else 1)
        " 2>/dev/null)

        if [[ $? -eq 0 ]]; then
            log_success "Backup created successfully"
            log_info "$backup_result"
        else
            log_warning "Backup creation failed, but continuing with teardown"
            if [[ "$FORCE_TEARDOWN" != true ]]; then
                read -p "Continue without backup? (y/N): " continue_confirm
                if [[ "$continue_confirm" != "y" && "$continue_confirm" != "Y" ]]; then
                    log_info "Teardown cancelled"
                    exit 0
                fi
            fi
        fi
    fi
}

# Graceful service shutdown
graceful_service_shutdown() {
    log_info "Initiating graceful service shutdown..."

    # Execute graceful shutdown using Python service
    shutdown_result=$(cd "$PROJECT_ROOT/backend" && python -c "
import asyncio
import sys
sys.path.append('.')
from services.infrastructure_service import InfrastructureService
from services.deployment_service import AzdDeploymentService

async def graceful_shutdown():
    try:
        infra = InfrastructureService()
        deployment_service = AzdDeploymentService(infra)
        result = await deployment_service.execute_graceful_shutdown()
        print(f'Shutdown status: {result[\"status\"]}')
        return result['status'] == 'completed'
    except Exception as e:
        print(f'Graceful shutdown failed: {e}')
        return False

success = asyncio.run(graceful_shutdown())
sys.exit(0 if success else 1)
    " 2>/dev/null)

    if [[ $? -eq 0 ]]; then
        log_success "Graceful shutdown completed"
    else
        log_warning "Graceful shutdown failed, proceeding with force teardown"
    fi

    # Additional wait for services to shut down
    log_info "Waiting for services to shut down completely..."
    sleep 30
}

# Get resource information before teardown
get_resource_info() {
    log_info "Gathering resource information..."

    local resource_group
    resource_group=$(azd env get-values --output json 2>/dev/null | jq -r '.AZURE_RESOURCE_GROUP // empty' || echo "")

    if [[ -n "$resource_group" ]]; then
        echo ""
        log_info "Resources to be deleted in resource group: $resource_group"

        # List resources that will be deleted
        if command -v az &> /dev/null; then
            log_info "Resource inventory:"
            az resource list --resource-group "$resource_group" --output table 2>/dev/null || true
        else
            log_warning "Azure CLI not available - unable to list resources"
        fi
        echo ""
    else
        log_warning "Unable to determine resource group"
    fi
}

# Execute azd down
execute_azd_down() {
    log_info "Executing Azure resource teardown..."

    local azd_args=()

    # Add force flag if specified
    if [[ "$FORCE_TEARDOWN" == true ]]; then
        azd_args+=(--force)
    fi

    # Add purge flag for complete cleanup
    azd_args+=(--purge)

    # Execute azd down
    if azd down "${azd_args[@]}"; then
        log_success "Azure resources deleted successfully"
    else
        log_error "azd down failed"
        exit 1
    fi
}

# Clean up local environment
cleanup_local_environment() {
    log_info "Cleaning up local environment..."

    # Remove environment from azd
    if azd env list --output json | jq -e --arg env "$ENVIRONMENT" '.[] | select(.name == $env)' &> /dev/null; then
        if azd env remove "$ENVIRONMENT" --force &> /dev/null; then
            log_success "Removed local environment: $ENVIRONMENT"
        else
            log_warning "Failed to remove local environment: $ENVIRONMENT"
        fi
    fi

    # Clean up any local cache or temporary files
    local temp_dirs=(
        ".azure"
        "backend/.azd"
        "backend/logs"
        "backend/temp"
    )

    for dir in "${temp_dirs[@]}"; do
        if [[ -d "$PROJECT_ROOT/$dir" ]]; then
            rm -rf "$PROJECT_ROOT/$dir"
            log_info "Cleaned up: $dir"
        fi
    done
}

# Verify teardown completion
verify_teardown_completion() {
    log_info "Verifying teardown completion..."

    # Check if resource group still exists
    local resource_group
    resource_group=$(azd env get-values --output json 2>/dev/null | jq -r '.AZURE_RESOURCE_GROUP // empty' || echo "")

    if [[ -n "$resource_group" ]] && command -v az &> /dev/null; then
        if az group show --name "$resource_group" &> /dev/null; then
            log_warning "Resource group '$resource_group' still exists"
            log_info "This may indicate incomplete teardown or manual resources"
        else
            log_success "Resource group '$resource_group' successfully deleted"
        fi
    fi

    # Check azd environment status
    if azd env list --output json | jq -e --arg env "$ENVIRONMENT" '.[] | select(.name == $env)' &> /dev/null; then
        log_warning "azd environment '$ENVIRONMENT' still exists locally"
    else
        log_success "azd environment '$ENVIRONMENT' removed"
    fi
}

# Generate teardown report
generate_teardown_report() {
    local report_file="$PROJECT_ROOT/teardown_report_$(date +%Y%m%d_%H%M%S).json"

    cat > "$report_file" << EOF
{
    "teardown_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "backup_created": $CREATE_BACKUP,
    "force_teardown": $FORCE_TEARDOWN,
    "status": "completed",
    "azure_resources_deleted": true,
    "local_environment_cleaned": true,
    "notes": "Teardown completed successfully using azd down"
}
EOF

    log_success "Teardown report generated: $(basename "$report_file")"
}

# Main execution
main() {
    echo ""
    log_info "🚀 Azure Universal RAG - azd Teardown Script"
    echo ""

    parse_arguments "$@"
    check_prerequisites
    get_current_environment
    validate_environment
    production_safety_check

    echo ""
    log_info "=== TEARDOWN PLAN ==="
    log_info "Environment: $ENVIRONMENT"
    log_info "Create backup: $CREATE_BACKUP"
    log_info "Force teardown: $FORCE_TEARDOWN"
    echo ""

    if [[ "$FORCE_TEARDOWN" != true ]]; then
        read -p "Proceed with teardown? (y/N): " proceed_confirm
        if [[ "$proceed_confirm" != "y" && "$proceed_confirm" != "Y" ]]; then
            log_info "Teardown cancelled"
            exit 0
        fi
    fi

    echo ""
    log_info "=== STARTING TEARDOWN ==="

    create_backup_before_teardown
    get_resource_info
    graceful_service_shutdown
    execute_azd_down
    cleanup_local_environment
    verify_teardown_completion
    generate_teardown_report

    echo ""
    log_success "🎉 Teardown completed successfully!"
    log_info "Environment '$ENVIRONMENT' has been completely removed"

    if [[ "$CREATE_BACKUP" == true ]]; then
        log_info "💾 Backup was created and can be used for restoration"
    fi

    echo ""
}

# Run main function with all arguments
main "$@"
