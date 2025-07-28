#!/bin/bash
# Environment Setup Script for Azure Universal RAG System
# Creates and configures development, staging, and production environments

set -e

echo "🌍 Setting up Azure Universal RAG environments..."

# Check if azd is installed
if ! command -v azd &> /dev/null; then
    echo "❌ Azure Developer CLI (azd) is not installed."
    echo "Please install azd: https://docs.microsoft.com/azure/developer/azure-developer-cli/install-azd"
    exit 1
fi

# Check if user is logged in
if ! azd auth show &> /dev/null; then
    echo "🔐 Please log in to Azure..."
    azd auth login
fi

echo "✅ Azure Developer CLI is ready"

# Function to create environment
create_environment() {
    local env_name=$1
    local location=$2
    local resource_group_suffix=$3
    
    echo ""
    echo "🚀 Creating environment: $env_name"
    echo "   Location: $location"
    echo "   Resource Group: rg-maintie-rag-$resource_group_suffix"
    
    # Create environment if it doesn't exist
    if azd env list | grep -q "^$env_name "; then
        echo "   Environment $env_name already exists"
        azd env select $env_name
    else
        echo "   Creating new environment: $env_name"
        azd env new $env_name
    fi
    
    # Set environment variables
    azd env set AZURE_LOCATION $location
    azd env set AZURE_RESOURCE_GROUP rg-maintie-rag-$resource_group_suffix
    azd env set OPENAI_MODEL_DEPLOYMENT gpt-4
    azd env set EMBEDDING_MODEL_DEPLOYMENT text-embedding-ada-002
    azd env set SEARCH_INDEX_NAME maintie-$env_name-index
    azd env set COSMOS_DATABASE_NAME maintie-rag-$env_name
    azd env set COSMOS_GRAPH_NAME knowledge-graph-$env_name
    azd env set BACKEND_IMAGE_NAME azure-maintie-rag-backend
    azd env set BACKEND_PORT 8000
    
    # Environment-specific configurations
    case $env_name in
        "development")
            azd env set ENVIRONMENT_TYPE dev
            azd env set LOG_LEVEL DEBUG
            azd env set ENABLE_SWAGGER true
            azd env set CACHE_TTL 300
            azd env set MAX_WORKERS 2
            ;;
        "staging")
            azd env set ENVIRONMENT_TYPE staging
            azd env set LOG_LEVEL INFO
            azd env set ENABLE_SWAGGER true
            azd env set CACHE_TTL 900
            azd env set MAX_WORKERS 4
            ;;
        "production")
            azd env set ENVIRONMENT_TYPE prod
            azd env set LOG_LEVEL WARNING
            azd env set ENABLE_SWAGGER false
            azd env set CACHE_TTL 3600
            azd env set MAX_WORKERS 8
            ;;
    esac
    
    echo "   ✅ Environment $env_name configured"
}

# Create environments
echo ""
echo "🏗️ Creating Azure Universal RAG environments..."

# Development Environment
create_environment "development" "eastus" "dev"

# Staging Environment  
create_environment "staging" "westus2" "staging"

# Production Environment
create_environment "production" "centralus" "prod"

echo ""
echo "🎯 Environment setup complete!"
echo ""
echo "Available environments:"
azd env list
echo ""
echo "📋 Next steps:"
echo "  1. Select an environment: azd env select <environment>"
echo "  2. Deploy infrastructure: azd provision"
echo "  3. Deploy application: azd deploy"
echo "  4. Or deploy both: azd up"
echo ""
echo "🔍 Example deployment:"
echo "  azd env select development"
echo "  azd up"
echo ""
echo "🌟 Happy coding with Azure Universal RAG!"