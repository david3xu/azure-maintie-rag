#!/bin/bash
# Deploy Azure Universal RAG with infrastructure fixes

echo "🚀 Azure Universal RAG - Fixed Deployment"
echo "Addressing region capacity and naming issues"
echo "============================================"

# Clean previous state
echo "🧹 Cleaning previous deployment state..."
rm -rf .azure

# Upgrade azd to latest version
echo "⬆️ Upgrading azd to latest version (1.18.0)..."
curl -fsSL https://aka.ms/install-azd.sh | bash

# Source the updated azd
source ~/.bashrc

# Deploy with alternative region to avoid East US capacity issues
echo "🌍 Deploying to West US 2 (better availability)..."

# Set environment variables for deployment
export AZURE_LOCATION="westus2"

# Deploy with azd up
echo "🚀 Starting azd deployment..."
azd up --environment development

echo "✅ Deployment script complete"
echo ""
echo "If Cosmos DB still fails due to capacity:"
echo "1. Try West US 3 or Central US regions"
echo "2. Wait 10-15 minutes and retry"
echo "3. Use 'azd provision' to retry just infrastructure"
