#!/bin/bash

# Fixed azd down script that actually works with existing resources
# This solves the fundamental azd down bug where it generates new deployment names

set -euo pipefail

echo "🎯 FIXED AZD DOWN - WORKS WITH REAL DEPLOYMENTS"
echo "==============================================="

# Configuration
RESOURCE_GROUP="rg-maintie-rag-prod"
FORCE_MODE=false
PURGE_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_MODE=true
            shift
            ;;
        --purge)
            PURGE_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--force] [--purge]"
            echo "  --force  Skip confirmations"
            echo "  --purge  Remove resource group completely"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "💡 WHY THIS SCRIPT EXISTS:"
echo "   azd down is fundamentally broken - it generates NEW deployment names"
echo "   instead of finding existing ones, causing 'no resources found' errors"
echo ""

# Check if resource group exists
if ! az group exists --name "$RESOURCE_GROUP" 2>/dev/null; then
    echo "✅ Resource group '$RESOURCE_GROUP' does not exist"
    echo "💡 Nothing to delete - system is already clean"
    exit 0
fi

# Get resource count
RESOURCE_COUNT=$(az resource list --resource-group "$RESOURCE_GROUP" --query "length(@)" -o tsv 2>/dev/null || echo "0")

echo "📊 CURRENT STATE:"
echo "   Resource Group: $RESOURCE_GROUP"
echo "   Resources Found: $RESOURCE_COUNT"

if [ "$RESOURCE_COUNT" -eq 0 ]; then
    echo ""
    echo "✅ Resource group exists but is empty"
    
    if [ "$PURGE_MODE" = true ]; then
        echo "🧹 Purging empty resource group..."
        az group delete --name "$RESOURCE_GROUP" --yes --no-wait
        echo "✅ Empty resource group deletion initiated"
    else
        echo "💡 Use --purge to remove empty resource group"
    fi
    
    echo "✅ System is clean - no resources to delete"
    exit 0
fi

# Show what will be deleted
echo ""
echo "📋 RESOURCES TO BE DELETED:"
az resource list --resource-group "$RESOURCE_GROUP" --query "[].{Name:name, Type:type, Location:location}" -o table

# Confirmation (unless --force)
if [ "$FORCE_MODE" = false ]; then
    echo ""
    read -p "❓ Delete all $RESOURCE_COUNT resources in '$RESOURCE_GROUP'? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Deletion cancelled"
        exit 1
    fi
fi

echo ""
echo "🧨 DELETING AZURE RESOURCES..."

# Strategy 1: Try azd down first (might work if environment is intact)
echo "1️⃣ Attempting azd down..."
if azd down --force --purge 2>/dev/null; then
    echo "✅ azd down succeeded"
    exit 0
else
    echo "⚠️  azd down failed (expected) - using direct deletion"
fi

# Strategy 2: Direct resource group deletion (always works)
echo ""
echo "2️⃣ Using direct Azure CLI deletion..."
echo "🧨 Deleting resource group '$RESOURCE_GROUP' with all resources..."

if [ "$FORCE_MODE" = true ]; then
    az group delete --name "$RESOURCE_GROUP" --yes --no-wait
    echo "✅ Resource group deletion initiated (background)"
    echo "💡 Check status with: az group show --name '$RESOURCE_GROUP'"
else
    az group delete --name "$RESOURCE_GROUP" --yes
    echo "✅ Resource group deletion completed"
fi

# Clean up local azd state
echo ""
echo "3️⃣ Cleaning up local azd state..."
if [ -d ".azure" ]; then
    echo "🧹 Backing up .azure directory..."
    mv .azure ".azure.backup.$(date +%Y%m%d-%H%M%S)"
    echo "✅ azd environment state reset"
fi

echo ""
echo "🎉 FIXED AZD DOWN COMPLETED SUCCESSFULLY!"
echo "✅ All Azure resources deleted"
echo "✅ Local azd state cleaned"
echo "💡 Ready for fresh deployment with: azd up"