#!/bin/bash
# Azure Universal RAG Deployment Wrapper
# This script provides easy access to the definitive deployment script

set -euo pipefail

# Color coding for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}🎉 $1${NC}"; }

# Check if the definitive deployment script exists
if [ ! -f "scripts/enhanced-complete-redeploy.sh" ]; then
    echo "❌ Error: Definitive deployment script not found"
    echo "   Expected: scripts/enhanced-complete-redeploy.sh"
    exit 1
fi

print_info "🚀 Azure Universal RAG Enterprise Deployment"
print_info "Using definitive deployment script: scripts/enhanced-complete-redeploy.sh"
print_info ""
print_info "This script includes all enterprise architecture components:"
print_info "  ✅ Azure Extension Manager"
print_info "  ✅ Azure Global Naming Service"
print_info "  ✅ Azure Deployment Orchestrator"
print_info "  ✅ Azure Service Health Validator"
print_info "  ✅ Enterprise Conflict Resolution"
print_info ""

# Execute the definitive deployment script
exec ./scripts/enhanced-complete-redeploy.sh "$@"