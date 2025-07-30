#!/bin/bash
# Azure Credentials Setup for Real Azure ML Training

echo "🔐 AZURE CREDENTIALS SETUP"
echo "================================"

echo "📋 You need to set these environment variables:"
echo ""
echo "# Get these values from Azure Portal or Azure CLI"
echo "export AZURE_SUBSCRIPTION_ID='your-subscription-id'"
echo "export AZURE_RESOURCE_GROUP='your-resource-group'"
echo "export AZURE_TENANT_ID='your-tenant-id'"
echo "export AZURE_CLIENT_ID='your-client-id'"
echo "export AZURE_CLIENT_SECRET='your-client-secret'"
echo ""

echo "🛠️  To create a Service Principal for Azure ML:"
echo "1. Install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
echo "2. Login: az login"
echo "3. Create Service Principal:"
echo "   az ad sp create-for-rbac --name 'gnn-training-sp' --role 'Contributor' --scopes '/subscriptions/YOUR_SUBSCRIPTION_ID'"
echo ""
echo "4. Assign Azure ML permissions:"
echo "   az role assignment create --assignee YOUR_CLIENT_ID --role 'AzureML Data Scientist'"
echo ""

echo "💡 After setting credentials, run:"
echo "   python scripts/setup_azure_ml_real.py"
echo "   python scripts/real_azure_ml_gnn_training.py --partial --wait"