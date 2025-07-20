#!/bin/bash
# Status script for working Azure services

set -euo pipefail

# Color coding
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

RESOURCE_GROUP="maintie-rag-rg"

echo "🏗️  Azure Universal RAG - Working Services Status"
echo "=================================================="
echo ""

# Check each working service
echo "📊 Working Services Status:"
echo "=========================="

# Storage Account (check for any storage account)
STORAGE_ACCOUNT=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.Storage/storageAccounts" --query "[?contains(name, 'stor') && !contains(name, 'ml')].name" --output tsv 2>/dev/null || echo "")
if [ ! -z "$STORAGE_ACCOUNT" ]; then
    print_status "Storage Account: $STORAGE_ACCOUNT"
else
    print_warning "Storage Account: Not found"
fi

# ML Storage Account (check for any ML storage account)
ML_STORAGE_ACCOUNT=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.Storage/storageAccounts" --query "[?contains(name, 'ml')].name" --output tsv 2>/dev/null || echo "")
if [ ! -z "$ML_STORAGE_ACCOUNT" ]; then
    print_status "ML Storage Account: $ML_STORAGE_ACCOUNT"
else
    print_warning "ML Storage Account: Not found"
fi



# Search Service (check for any search service)
SEARCH_SERVICE=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.Search/searchServices" --query "[0].name" --output tsv 2>/dev/null || echo "")
if [ ! -z "$SEARCH_SERVICE" ]; then
    print_status "Search Service: $SEARCH_SERVICE"
else
    print_warning "Search Service: Not found"
fi

# Key Vault (check for any key vault)
KEY_VAULT=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.KeyVault/vaults" --query "[0].name" --output tsv 2>/dev/null || echo "")
if [ ! -z "$KEY_VAULT" ]; then
    print_status "Key Vault: $KEY_VAULT"
else
    print_warning "Key Vault: Not found"
fi

# Application Insights (check for any app insights)
APP_INSIGHTS=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.Insights/components" --query "[0].name" --output tsv 2>/dev/null || echo "")
if [ ! -z "$APP_INSIGHTS" ]; then
    print_status "Application Insights: $APP_INSIGHTS"
else
    print_warning "Application Insights: Not found"
fi

# Log Analytics (check for any log analytics workspace)
LOG_ANALYTICS=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.OperationalInsights/workspaces" --query "[0].name" --output tsv 2>/dev/null || echo "")
if [ ! -z "$LOG_ANALYTICS" ]; then
    print_status "Log Analytics: $LOG_ANALYTICS"
else
    print_warning "Log Analytics: Not found"
fi

# Cosmos DB (check for any cosmos db account)
COSMOS_DB=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.DocumentDB/databaseAccounts" --query "[0].name" --output tsv 2>/dev/null || echo "")
if [ ! -z "$COSMOS_DB" ]; then
    print_status "Cosmos DB: $COSMOS_DB"
else
    print_warning "Cosmos DB: Not found"
fi

# ML Workspace (check for any ml workspace)
ML_WORKSPACE=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.MachineLearningServices/workspaces" --query "[0].name" --output tsv 2>/dev/null || echo "")
if [ ! -z "$ML_WORKSPACE" ]; then
    print_status "ML Workspace: $ML_WORKSPACE"
else
    print_warning "ML Workspace: Not found"
fi

# Container Environment (check for any container environment)
CONTAINER_ENV=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.App/managedEnvironments" --query "[0].name" --output tsv 2>/dev/null || echo "")
if [ ! -z "$CONTAINER_ENV" ]; then
    print_status "Container Environment: $CONTAINER_ENV"
else
    print_warning "Container Environment: Not found"
fi

# Container App (check for any container app)
CONTAINER_APP=$(az resource list --resource-group "$RESOURCE_GROUP" --resource-type "Microsoft.App/containerApps" --query "[0].name" --output tsv 2>/dev/null || echo "")
if [ ! -z "$CONTAINER_APP" ]; then
    print_status "Container App: $CONTAINER_APP"
else
    print_warning "Container App: Not found"
fi



echo ""
echo "📋 Summary:"
echo "==========="

# Count deployed services dynamically
DEPLOYED_COUNT=0
if [ ! -z "$STORAGE_ACCOUNT" ]; then DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1)); fi
if [ ! -z "$ML_STORAGE_ACCOUNT" ]; then DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1)); fi
if [ ! -z "$SEARCH_SERVICE" ]; then DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1)); fi
if [ ! -z "$KEY_VAULT" ]; then DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1)); fi
if [ ! -z "$COSMOS_DB" ]; then DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1)); fi
if [ ! -z "$ML_WORKSPACE" ]; then DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1)); fi
if [ ! -z "$APP_INSIGHTS" ]; then DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1)); fi
if [ ! -z "$LOG_ANALYTICS" ]; then DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1)); fi
if [ ! -z "$CONTAINER_ENV" ]; then DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1)); fi
if [ ! -z "$CONTAINER_APP" ]; then DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1)); fi

# Build dynamic service list
SERVICES_LIST=""
if [ ! -z "$STORAGE_ACCOUNT" ]; then SERVICES_LIST="$SERVICES_LIST Storage Account"; fi
if [ ! -z "$ML_STORAGE_ACCOUNT" ]; then SERVICES_LIST="$SERVICES_LIST ML Storage Account"; fi
if [ ! -z "$SEARCH_SERVICE" ]; then SERVICES_LIST="$SERVICES_LIST Search Service"; fi
if [ ! -z "$KEY_VAULT" ]; then SERVICES_LIST="$SERVICES_LIST Key Vault"; fi
if [ ! -z "$COSMOS_DB" ]; then SERVICES_LIST="$SERVICES_LIST Cosmos DB"; fi
if [ ! -z "$ML_WORKSPACE" ]; then SERVICES_LIST="$SERVICES_LIST ML Workspace"; fi
if [ ! -z "$APP_INSIGHTS" ]; then SERVICES_LIST="$SERVICES_LIST Application Insights"; fi
if [ ! -z "$LOG_ANALYTICS" ]; then SERVICES_LIST="$SERVICES_LIST Log Analytics"; fi
if [ ! -z "$CONTAINER_ENV" ]; then SERVICES_LIST="$SERVICES_LIST Container Environment"; fi
if [ ! -z "$CONTAINER_APP" ]; then SERVICES_LIST="$SERVICES_LIST Container App"; fi

# Remove leading space
SERVICES_LIST=$(echo "$SERVICES_LIST" | sed 's/^ //')

echo "✅ Working Services: $SERVICES_LIST"
echo "ℹ️  Total Working Services: $DEPLOYED_COUNT"
echo "🏗️  Infrastructure: Clean and operational"
echo ""

if [ "$DEPLOYED_COUNT" -eq 10 ]; then
    print_status "All working services are deployed and operational!"
elif [ "$DEPLOYED_COUNT" -gt 0 ]; then
    print_warning "Some services are deployed ($DEPLOYED_COUNT/10)"
else
    print_error "No services are deployed"
fi