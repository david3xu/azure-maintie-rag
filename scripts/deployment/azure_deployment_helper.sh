#!/bin/bash
"""
Azure Deployment Helper Script
Easy-to-use wrapper for Azure Universal RAG deployment and management
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
ENVIRONMENT_NAME="${AZURE_ENV_NAME:-dev}"
LOCATION="${AZURE_LOCATION:-westus2}"
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"

print_banner() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}                         Azure Universal RAG Deployment Helper${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
    echo ""
}

print_help() {
    cat << EOF
Azure Universal RAG Deployment Helper

USAGE:
    $0 <command> [options]

COMMANDS:
    status          Test Azure services connectivity and status
    deploy          Deploy the Universal RAG system using 'azd up'
    cleanup         Clean up Azure resources (interactive)
    force-cleanup   Force cleanup without prompts (DANGEROUS)
    validate        Validate deployment configuration
    logs            Show deployment logs
    help            Show this help message

OPTIONS:
    --env <name>           Environment name (default: $ENVIRONMENT_NAME)
    --location <region>    Azure region (default: $LOCATION)
    --subscription <id>    Azure subscription ID
    --dry-run             Run in dry-run mode (for cleanup)
    --verbose             Enable verbose output

EXAMPLES:
    $0 status
    $0 deploy --env staging --location eastus
    $0 cleanup --dry-run
    $0 validate --env prod

ENVIRONMENT VARIABLES:
    AZURE_ENV_NAME         Environment name
    AZURE_LOCATION         Azure region
    AZURE_SUBSCRIPTION_ID  Azure subscription ID

NOTE: Ensure you are logged in with 'az login' before running commands.
EOF
}

check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        echo -e "${RED}❌ Azure CLI not found. Please install it first.${NC}"
        echo "   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi

    # Check if azd is installed
    if ! command -v azd &> /dev/null; then
        echo -e "${RED}❌ Azure Developer CLI (azd) not found. Please install it first.${NC}"
        echo "   https://docs.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd"
        exit 1
    fi

    # Check if Python is installed
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python not found. Please install Python 3.8+ first.${NC}"
        exit 1
    fi

    # Check if user is logged in to Azure
    if ! az account show &> /dev/null; then
        echo -e "${RED}❌ Not logged in to Azure. Please run 'az login' first.${NC}"
        exit 1
    fi

    # Get current subscription info
    local current_sub=$(az account show --query "id" -o tsv)
    local current_name=$(az account show --query "name" -o tsv)

    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
    echo -e "   Azure CLI: $(az version --query '"azure-cli"' -o tsv)"
    echo -e "   Azure Developer CLI: $(azd version --output json | grep -o '"version":"[^"]*' | cut -d'"' -f4)"
    echo -e "   Current subscription: $current_name ($current_sub)"

    # Set subscription ID if not provided
    if [ -z "$SUBSCRIPTION_ID" ]; then
        SUBSCRIPTION_ID="$current_sub"
    fi

    echo ""
}

test_azure_status() {
    echo -e "${BLUE}Testing Azure services connectivity...${NC}"

    cd "$PROJECT_ROOT"

    # Install required Python packages if missing
    python3 -c "import httpx" 2>/dev/null || {
        echo -e "${YELLOW}Installing required Python packages...${NC}"
        pip3 install httpx azure-mgmt-resource azure-mgmt-storage azure-mgmt-search azure-mgmt-cognitiveservices
    }

    # Run the status check
    python3 "$SCRIPT_DIR/test_azure_services_status.py"
    local status_code=$?

    case $status_code in
        0)
            echo -e "${GREEN}✅ All Azure services are healthy${NC}"
            ;;
        1)
            echo -e "${YELLOW}⚠️  Some Azure services have issues${NC}"
            ;;
        2)
            echo -e "${RED}❌ Major issues with Azure services${NC}"
            ;;
        *)
            echo -e "${RED}❌ Status check failed${NC}"
            ;;
    esac

    return $status_code
}

deploy_system() {
    echo -e "${BLUE}Deploying Azure Universal RAG system...${NC}"

    cd "$PROJECT_ROOT"

    # Check if azure.yaml exists
    if [ ! -f "azure.yaml" ]; then
        echo -e "${RED}❌ azure.yaml not found in project root${NC}"
        exit 1
    fi

    # Set environment variables
    export AZURE_ENV_NAME="$ENVIRONMENT_NAME"
    export AZURE_LOCATION="$LOCATION"
    export AZURE_SUBSCRIPTION_ID="$SUBSCRIPTION_ID"

    echo -e "${YELLOW}Deployment configuration:${NC}"
    echo -e "  Environment: $ENVIRONMENT_NAME"
    echo -e "  Location: $LOCATION"
    echo -e "  Subscription: $SUBSCRIPTION_ID"
    echo ""

    # Ask for confirmation unless in CI
    if [ -t 0 ] && [ -z "${CI}" ]; then
        read -p "Proceed with deployment? (y/N): " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "Deployment cancelled."
            exit 0
        fi
    fi

    echo -e "${BLUE}Starting deployment with 'azd up'...${NC}"

    # Run azd up
    if azd up --environment "$ENVIRONMENT_NAME" --location "$LOCATION"; then
        echo -e "${GREEN}✅ Deployment completed successfully${NC}"

        # Test the deployed services
        echo -e "${BLUE}Testing deployed services...${NC}"
        test_azure_status

    else
        echo -e "${RED}❌ Deployment failed${NC}"
        echo -e "${YELLOW}Check the logs above for details${NC}"
        exit 1
    fi
}

cleanup_resources() {
    local dry_run="$1"
    local force="$2"

    if [ "$dry_run" = "true" ]; then
        echo -e "${BLUE}Running Azure resources cleanup (DRY RUN)...${NC}"
    else
        echo -e "${RED}Running Azure resources cleanup (LIVE MODE)...${NC}"
    fi

    cd "$PROJECT_ROOT"

    # Install required Python packages if missing
    python3 -c "import azure.mgmt.resource" 2>/dev/null || {
        echo -e "${YELLOW}Installing required Python packages...${NC}"
        pip3 install azure-mgmt-resource azure-mgmt-storage azure-mgmt-search azure-mgmt-cognitiveservices azure-mgmt-cosmosdb azure-mgmt-keyvault azure-mgmt-applicationinsights azure-mgmt-containerregistry azure-mgmt-web
    }

    # Prepare arguments
    local args="--subscription-id $SUBSCRIPTION_ID"

    if [ "$dry_run" = "true" ]; then
        args="$args --dry-run"
    else
        args="$args --live"
    fi

    # Add output file
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_file="cleanup_report_${timestamp}.json"
    args="$args --output $output_file"

    # Run cleanup
    if [ "$force" = "true" ]; then
        # Force mode - skip confirmations
        echo "DELETE_RESOURCES" | python3 "$SCRIPT_DIR/cleanup_azure_services.py" $args
    else
        python3 "$SCRIPT_DIR/cleanup_azure_services.py" $args
    fi

    local cleanup_code=$?

    if [ $cleanup_code -eq 0 ]; then
        echo -e "${GREEN}✅ Cleanup completed successfully${NC}"
    elif [ $cleanup_code -eq 1 ]; then
        echo -e "${YELLOW}⚠️  Cleanup completed with some failures${NC}"
    else
        echo -e "${RED}❌ Cleanup failed${NC}"
    fi

    if [ -f "$output_file" ]; then
        echo -e "${BLUE}Cleanup report saved to: $output_file${NC}"
    fi

    return $cleanup_code
}

validate_configuration() {
    echo -e "${BLUE}Validating deployment configuration...${NC}"

    cd "$PROJECT_ROOT"

    local errors=0

    # Check azure.yaml
    if [ -f "azure.yaml" ]; then
        echo -e "${GREEN}✅ azure.yaml found${NC}"
    else
        echo -e "${RED}❌ azure.yaml missing${NC}"
        errors=$((errors + 1))
    fi

    # Check Dockerfile
    if [ -f "Dockerfile" ]; then
        echo -e "${GREEN}✅ Dockerfile found${NC}"
    else
        echo -e "${RED}❌ Dockerfile missing${NC}"
        errors=$((errors + 1))
    fi

    # Check requirements.txt
    if [ -f "requirements.txt" ]; then
        echo -e "${GREEN}✅ requirements.txt found${NC}"
    else
        echo -e "${RED}❌ requirements.txt missing${NC}"
        errors=$((errors + 1))
    fi

    # Check infra directory
    if [ -d "infra" ]; then
        echo -e "${GREEN}✅ infra/ directory found${NC}"

        # Check main.bicep
        if [ -f "infra/main.bicep" ]; then
            echo -e "${GREEN}✅ infra/main.bicep found${NC}"
        else
            echo -e "${RED}❌ infra/main.bicep missing${NC}"
            errors=$((errors + 1))
        fi
    else
        echo -e "${RED}❌ infra/ directory missing${NC}"
        errors=$((errors + 1))
    fi

    # Check API entry point
    if [ -f "api/main.py" ]; then
        echo -e "${GREEN}✅ api/main.py found${NC}"
    else
        echo -e "${RED}❌ api/main.py missing${NC}"
        errors=$((errors + 1))
    fi

    # Validate Python syntax
    if python3 -m py_compile api/main.py 2>/dev/null; then
        echo -e "${GREEN}✅ Python syntax validation passed${NC}"
    else
        echo -e "${RED}❌ Python syntax validation failed${NC}"
        errors=$((errors + 1))
    fi

    if [ $errors -eq 0 ]; then
        echo -e "${GREEN}✅ Configuration validation passed${NC}"
        return 0
    else
        echo -e "${RED}❌ Configuration validation failed ($errors errors)${NC}"
        return 1
    fi
}

show_logs() {
    echo -e "${BLUE}Showing deployment logs...${NC}"

    # Try to show azd logs
    if command -v azd &> /dev/null; then
        azd monitor --live
    else
        echo -e "${YELLOW}azd not found. Cannot show live logs.${NC}"
    fi
}

# Parse command line arguments
VERBOSE=false
DRY_RUN=true
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT_NAME="$2"
            shift 2
            ;;
        --location)
            LOCATION="$2"
            shift 2
            ;;
        --subscription)
            SUBSCRIPTION_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --live)
            DRY_RUN=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -*)
            echo "Unknown option $1"
            print_help
            exit 1
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Enable verbose output if requested
if [ "$VERBOSE" = true ]; then
    set -x
fi

# Main logic
print_banner

case "${COMMAND:-}" in
    status)
        check_prerequisites
        test_azure_status
        ;;
    deploy)
        check_prerequisites
        validate_configuration
        deploy_system
        ;;
    cleanup)
        check_prerequisites
        cleanup_resources "$DRY_RUN" "$FORCE"
        ;;
    force-cleanup)
        check_prerequisites
        cleanup_resources false true
        ;;
    validate)
        validate_configuration
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        print_help
        ;;
    "")
        echo -e "${RED}No command specified${NC}"
        print_help
        exit 1
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        print_help
        exit 1
        ;;
esac
