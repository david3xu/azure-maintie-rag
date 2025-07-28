#!/bin/bash
# Update Environment Configuration from Azure Deployment
# Extracts service endpoints and connection strings from deployed Azure resources

set -e

echo "🔧 Updating configuration from Azure deployment..."

# Check if azd is available
if ! command -v azd &> /dev/null; then
    echo "❌ Azure Developer CLI (azd) not found"
    exit 1
fi

# Get current environment
CURRENT_ENV=$(azd env get-values | grep AZURE_ENV_NAME | cut -d'=' -f2 | tr -d '"')
if [ -z "$CURRENT_ENV" ]; then
    echo "❌ No active azd environment found"
    exit 1
fi

echo "📍 Environment: $CURRENT_ENV"

# Extract outputs from the last deployment
echo "📤 Extracting deployment outputs..."

# Function to safely extract output value
extract_output() {
    local output_name=$1
    local value=$(azd env get-values | grep "^$output_name=" | cut -d'=' -f2- | tr -d '"')
    if [ -n "$value" ]; then
        echo "  ✅ $output_name: $value"
        echo "$value"
    else
        echo "  ⚠️  $output_name: not found"
        echo ""
    fi
}

# Extract Azure service endpoints
echo ""
echo "🔍 Extracting Azure service configurations..."

AZURE_OPENAI_ENDPOINT=$(extract_output "AZURE_OPENAI_ENDPOINT")
AZURE_SEARCH_ENDPOINT=$(extract_output "AZURE_SEARCH_ENDPOINT")
AZURE_COSMOS_ENDPOINT=$(extract_output "AZURE_COSMOS_ENDPOINT")
AZURE_STORAGE_ACCOUNT=$(extract_output "AZURE_STORAGE_ACCOUNT")
AZURE_KEY_VAULT_NAME=$(extract_output "AZURE_KEY_VAULT_NAME")
AZURE_APP_INSIGHTS_CONNECTION_STRING=$(extract_output "AZURE_APP_INSIGHTS_CONNECTION_STRING")
AZURE_ML_WORKSPACE_NAME=$(extract_output "AZURE_ML_WORKSPACE_NAME")
SERVICE_BACKEND_URI=$(extract_output "SERVICE_BACKEND_URI")
AZURE_CLIENT_ID=$(extract_output "AZURE_CLIENT_ID")

# Create backend environment configuration
BACKEND_ENV_FILE="backend/config/environments/${CURRENT_ENV}.env"
echo ""
echo "📝 Creating backend environment file: $BACKEND_ENV_FILE"

# Create directory if it doesn't exist
mkdir -p "backend/config/environments"

# Write environment configuration
cat > "$BACKEND_ENV_FILE" << EOF
# Azure Universal RAG - Environment Configuration
# Generated automatically by azd deployment on $(date)
# Environment: $CURRENT_ENV

# === Azure Service Endpoints ===
AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT
AZURE_SEARCH_ENDPOINT=$AZURE_SEARCH_ENDPOINT
AZURE_COSMOS_ENDPOINT=$AZURE_COSMOS_ENDPOINT
AZURE_STORAGE_ACCOUNT=$AZURE_STORAGE_ACCOUNT
AZURE_KEY_VAULT_NAME=$AZURE_KEY_VAULT_NAME
AZURE_ML_WORKSPACE_NAME=$AZURE_ML_WORKSPACE_NAME

# === Identity & Security ===
AZURE_CLIENT_ID=$AZURE_CLIENT_ID
AZURE_TENANT_ID=$(extract_output "AZURE_TENANT_ID")
AZURE_SUBSCRIPTION_ID=$(extract_output "AZURE_SUBSCRIPTION_ID")

# === Application Configuration ===
ENVIRONMENT=$CURRENT_ENV
AZURE_LOCATION=$(extract_output "AZURE_LOCATION")
AZURE_RESOURCE_GROUP=$(extract_output "AZURE_RESOURCE_GROUP")

# === Model Deployments ===
OPENAI_MODEL_DEPLOYMENT=$(extract_output "AZURE_OPENAI_DEPLOYMENT_NAME")
EMBEDDING_MODEL_DEPLOYMENT=$(extract_output "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# === Database Configuration ===
COSMOS_DATABASE_NAME=$(extract_output "AZURE_COSMOS_DATABASE_NAME")
COSMOS_GRAPH_NAME=$(extract_output "AZURE_COSMOS_GRAPH_NAME")
SEARCH_INDEX_NAME=$(extract_output "AZURE_SEARCH_INDEX")
STORAGE_CONTAINER_NAME=$(extract_output "AZURE_STORAGE_CONTAINER")

# === Monitoring ===
APPLICATIONINSIGHTS_CONNECTION_STRING=$AZURE_APP_INSIGHTS_CONNECTION_STRING
LOG_ANALYTICS_WORKSPACE_ID=$(extract_output "AZURE_LOG_ANALYTICS_WORKSPACE_ID")

# === Hosting ===
SERVICE_BACKEND_URI=$SERVICE_BACKEND_URI
CONTAINER_REGISTRY_ENDPOINT=$(extract_output "AZURE_CONTAINER_REGISTRY_ENDPOINT")

# === Security Settings ===
USE_MANAGED_IDENTITY=true
ENABLE_RBAC=true
REQUIRE_HTTPS=true
EOF

echo "✅ Environment configuration written to: $BACKEND_ENV_FILE"

# Create .env symlink for local development
if [ "$CURRENT_ENV" = "development" ]; then
    echo ""
    echo "🔗 Creating .env symlink for local development..."
    cd backend
    rm -f .env
    ln -s "config/environments/development.env" .env
    cd ..
    echo "✅ Symlink created: backend/.env -> config/environments/development.env"
fi

# Validate configuration
echo ""
echo "🔍 Validating configuration..."

if [ -n "$AZURE_OPENAI_ENDPOINT" ] && [ -n "$AZURE_SEARCH_ENDPOINT" ] && [ -n "$AZURE_COSMOS_ENDPOINT" ]; then
    echo "✅ Core Azure services configured successfully"
else
    echo "⚠️  Some core services may not be properly configured"
fi

if [ -n "$SERVICE_BACKEND_URI" ]; then
    echo "✅ Backend service deployed successfully"
    echo "🌐 Backend URL: $SERVICE_BACKEND_URI"
else
    echo "⚠️  Backend service URL not available"
fi

echo ""
echo "🎉 Configuration update complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Test backend health: curl $SERVICE_BACKEND_URI/health"
echo "  2. Start local development: cd backend && make dev"
echo "  3. View logs: azd monitor --live"
echo ""
echo "📚 Environment file location: $BACKEND_ENV_FILE"