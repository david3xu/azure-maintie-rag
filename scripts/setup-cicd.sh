#!/bin/bash

# Azure Universal RAG CI/CD Setup Script
# This script configures GitHub Actions for OIDC authentication with Azure

set -e

echo "🔧 Azure Universal RAG CI/CD Setup"
echo "=================================="

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "Please install it from: https://cli.github.com/"
    echo "Or run: curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg"
    exit 1
fi

# Check if logged in to GitHub
if ! gh auth status &> /dev/null; then
    echo "📱 Please authenticate with GitHub:"
    gh auth login
fi

# Get Azure details
AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)
REPO_NAME=$(gh repo view --json nameWithOwner -q .nameWithOwner)

echo ""
echo "📋 Configuration Details:"
echo "  Repository: $REPO_NAME"
echo "  Azure Subscription: $AZURE_SUBSCRIPTION_ID"
echo "  Azure Tenant: $AZURE_TENANT_ID"
echo ""

# Create Azure AD application for GitHub OIDC
echo "🔐 Creating Azure AD application for GitHub OIDC..."
APP_NAME="github-oidc-$REPO_NAME" | tr '/' '-'

# Check if app already exists
APP_ID=$(az ad app list --display-name "$APP_NAME" --query "[0].appId" -o tsv 2>/dev/null || echo "")

if [ -z "$APP_ID" ]; then
    # Create new app
    APP_ID=$(az ad app create --display-name "$APP_NAME" --query appId -o tsv)
    echo "✅ Created Azure AD app: $APP_ID"
else
    echo "ℹ️  Using existing Azure AD app: $APP_ID"
fi

# Create service principal
SP_ID=$(az ad sp list --all --query "[?appId=='$APP_ID'].id" -o tsv 2>/dev/null || echo "")
if [ -z "$SP_ID" ]; then
    az ad sp create --id $APP_ID
    echo "✅ Created service principal"
else
    echo "ℹ️  Service principal already exists"
fi

# Configure federated credential for GitHub Actions
echo "🔗 Configuring federated credential..."
CREDENTIAL_NAME="github-deploy"

# Check if credential exists
EXISTING_CRED=$(az ad app federated-credential list --id $APP_ID --query "[?name=='$CREDENTIAL_NAME'].name" -o tsv 2>/dev/null || echo "")

if [ -z "$EXISTING_CRED" ]; then
    cat > federated-credential.json <<EOF
{
    "name": "$CREDENTIAL_NAME",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:$REPO_NAME:ref:refs/heads/main",
    "description": "Deploy from main branch",
    "audiences": [
        "api://AzureADTokenExchange"
    ]
}
EOF

    az ad app federated-credential create --id $APP_ID --parameters @federated-credential.json
    rm federated-credential.json
    echo "✅ Created federated credential"
else
    echo "ℹ️  Federated credential already exists"
fi

# Assign Contributor role to the service principal
echo "🔑 Assigning Contributor role..."
ASSIGNEE_ID=$(az ad sp list --all --query "[?appId=='$APP_ID'].id" -o tsv)
EXISTING_ROLE=$(az role assignment list --assignee $APP_ID --role Contributor --scope /subscriptions/$AZURE_SUBSCRIPTION_ID --query "[0].id" -o tsv 2>/dev/null || echo "")

if [ -z "$EXISTING_ROLE" ]; then
    az role assignment create --assignee $APP_ID --role Contributor --scope /subscriptions/$AZURE_SUBSCRIPTION_ID
    echo "✅ Assigned Contributor role"
else
    echo "ℹ️  Contributor role already assigned"
fi

# Set GitHub repository variables
echo ""
echo "📝 Setting GitHub repository variables..."

gh variable set AZURE_CLIENT_ID --body "$APP_ID"
gh variable set AZURE_TENANT_ID --body "$AZURE_TENANT_ID"
gh variable set AZURE_SUBSCRIPTION_ID --body "$AZURE_SUBSCRIPTION_ID"

echo "✅ GitHub variables configured"

# Display summary
echo ""
echo "=========================================="
echo "✅ CI/CD Setup Complete!"
echo ""
echo "GitHub Actions is now configured to deploy to Azure using OIDC."
echo ""
echo "Repository Variables Set:"
echo "  AZURE_CLIENT_ID: $APP_ID"
echo "  AZURE_TENANT_ID: $AZURE_TENANT_ID"
echo "  AZURE_SUBSCRIPTION_ID: $AZURE_SUBSCRIPTION_ID"
echo ""
echo "Next Steps:"
echo "1. Commit and push any changes"
echo "2. The pipeline will trigger automatically on push to main"
echo "3. Or trigger manually: gh workflow run azure-dev.yml"
echo ""
echo "Monitor deployment:"
echo "  gh run watch"
echo "  gh run list --workflow=azure-dev.yml"
echo "=========================================="