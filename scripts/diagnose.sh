#!/bin/bash
# diagnose.sh - Diagnostic script to check Azure resource issues

set -e

echo "🔍 Azure Resource Diagnostic"
echo "============================"

# Check Azure CLI
echo "1. Checking Azure CLI..."
if command -v az &> /dev/null; then
    echo "✅ Azure CLI found"
    az version --output table
else
    echo "❌ Azure CLI not found"
    exit 1
fi

echo ""

# Check authentication
echo "2. Checking authentication..."
if az account show &> /dev/null; then
    echo "✅ Authenticated to Azure"
    SUBSCRIPTION=$(az account show --query name --output tsv)
    echo "   Subscription: $SUBSCRIPTION"
else
    echo "❌ Not authenticated to Azure"
    exit 1
fi

echo ""

# Check resource group
echo "3. Checking resource group..."
RESOURCE_GROUP="maintie-rag-rg"
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo "✅ Resource group exists"
    LOCATION=$(az group show --name $RESOURCE_GROUP --query location --output tsv)
    echo "   Location: $LOCATION"
else
    echo "❌ Resource group not found"
fi

echo ""

# Check deployments
echo "4. Checking deployments..."
echo "   Deployments in resource group:"
az deployment group list --resource-group $RESOURCE_GROUP --query "[].{Name:name,State:properties.provisioningState,Timestamp:properties.timestamp}" --output table

echo ""

# Check specific resources
echo "5. Checking specific resources..."

# Storage accounts
echo "   Storage accounts:"
az storage account list --resource-group $RESOURCE_GROUP --query "[].{Name:name,SKU:sku.name,Kind:kind}" --output table

echo ""

# Search services
echo "   Search services:"
az search service list --resource-group $RESOURCE_GROUP --query "[].{Name:name,SKU:sku.name,Replicas:properties.replicaCount}" --output table

echo ""

# Key Vaults
echo "   Key Vaults:"
# Handle the Azure CLI Key Vault API version issue
if az keyvault list --resource-group $RESOURCE_GROUP --output table 2>/dev/null; then
    echo "   Key Vaults listed successfully"
else
    echo "   ⚠️  Key Vault list failed (API version issue), checking individually..."
    # Check specific Key Vault
    if az keyvault show --name maintie-dev-kv --resource-group $RESOURCE_GROUP &>/dev/null; then
        echo "   ✅ Key Vault 'maintie-dev-kv' exists"
    else
        echo "   ❌ Key Vault 'maintie-dev-kv' not found"
    fi
fi

echo ""

# Check deployment outputs
echo "6. Checking deployment outputs..."
if az deployment group show --resource-group $RESOURCE_GROUP --name azure-resources-core &> /dev/null; then
    echo "   azure-resources-core outputs:"
    az deployment group show --resource-group $RESOURCE_GROUP --name azure-resources-core --query "properties.outputs" --output json
else
    echo "   azure-resources-core deployment not found"
fi

echo ""

echo "🔍 Diagnostic complete!"