# Azure Lifecycle Execution Guide

## 🚀 Complete Azure Deployment Lifecycle - Command Line Automation

This document provides **command-line only** execution for the complete Azure Universal RAG lifecycle from initial setup to teardown.

---

## 📋 **Prerequisites & Initial Setup**

### **1. Azure CLI Installation & Authentication**
```bash
# Install Azure CLI (if not already installed)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Set default subscription
az account set --subscription "your-subscription-id"

# Verify login
az account show
```

### **2. Environment Setup**
```bash
# Clone repository (if not already done)
git clone https://github.com/your-org/azure-maintie-rag.git
cd azure-maintie-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Install Azure CLI extensions
az extension add --name bicep
az extension add --name application-insights
```

---

## 🏗️ **Phase 1: Infrastructure Deployment**

### **1.1 Create Resource Group**
```bash
# Set variables
RESOURCE_GROUP="maintie-rag-rg"
LOCATION="eastus"
ENVIRONMENT="dev"

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION \
  --tags Environment=$ENVIRONMENT Project="Universal-RAG"
```

### **1.2 Deploy Infrastructure with Cost Optimization**
```bash
# Deploy main infrastructure
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file infrastructure/azure-resources.bicep \
  --parameters environment=$ENVIRONMENT \
  --verbose

# Verify deployment
az resource list --resource-group $RESOURCE_GROUP --output table
```

### **1.3 Deploy Enterprise Infrastructure**
```bash
# Deploy enterprise features (Key Vault, Application Insights)
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file infrastructure/deploy_enterprise.bicep \
  --parameters environment=$ENVIRONMENT \
  --verbose
```

---

## 🔧 **Phase 2: Configuration & Secrets Management**

### **2.1 Get Deployment Outputs**
```bash
# Get resource names from deployment
STORAGE_ACCOUNT=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources \
  --query 'properties.outputs.storageAccountName.value' \
  --output tsv)

SEARCH_SERVICE=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name azure-resources \
  --query 'properties.outputs.searchServiceName.value' \
  --output tsv)

KEY_VAULT_NAME=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name deploy_enterprise \
  --query 'properties.outputs.keyVaultName.value' \
  --output tsv)

echo "Storage Account: $STORAGE_ACCOUNT"
echo "Search Service: $SEARCH_SERVICE"
echo "Key Vault: $KEY_VAULT_NAME"
```

### **2.2 Configure Azure Key Vault**
```bash
# Get current user object ID for Key Vault access
USER_OBJECT_ID=$(az ad signed-in-user show --query id --output tsv)

# Grant Key Vault access to current user
az keyvault set-policy \
  --name $KEY_VAULT_NAME \
  --object-id $USER_OBJECT_ID \
  --secret-permissions get set list delete

# Store secrets in Key Vault
az keyvault secret set \
  --vault-name $KEY_VAULT_NAME \
  --name "AzureStorageConnectionString" \
  --value "DefaultEndpointsProtocol=https;AccountName=$STORAGE_ACCOUNT;AccountKey=$(az storage account keys list --account-name $STORAGE_ACCOUNT --query '[0].value' --output tsv);EndpointSuffix=core.windows.net"

az keyvault secret set \
  --vault-name $KEY_VAULT_NAME \
  --name "AzureSearchAdminKey" \
  --value $(az search admin-key show --service-name $SEARCH_SERVICE --resource-group $RESOURCE_GROUP --query primaryKey --output tsv)
```

### **2.3 Generate Environment Configuration**
```bash
# Create environment file
cat > backend/.env << EOF
# Azure Environment
AZURE_ENVIRONMENT=$ENVIRONMENT
AZURE_RESOURCE_GROUP=$RESOURCE_GROUP
AZURE_REGION=$LOCATION

# Azure Storage
AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT
AZURE_BLOB_CONTAINER=universal-rag-data

# Azure Search
AZURE_SEARCH_SERVICE=$SEARCH_SERVICE
AZURE_SEARCH_INDEX=universal-rag-index
AZURE_SEARCH_API_VERSION=2023-11-01

# Azure Key Vault
AZURE_KEY_VAULT_URL=https://$KEY_VAULT_NAME.vault.azure.net/
AZURE_USE_MANAGED_IDENTITY=true

# Application Insights
AZURE_ENABLE_TELEMETRY=true

# OpenAI Configuration
OPENAI_API_TYPE=azure
OPENAI_API_BASE=https://your-openai-instance.openai.azure.com/
OPENAI_API_VERSION=2025-03-01-preview
OPENAI_DEPLOYMENT_NAME=gpt-4.1
OPENAI_MODEL=gpt-4.1

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

# Application Settings
ENVIRONMENT=development
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
EOF

echo "Environment file created: backend/.env"
```

---

## 🧪 **Phase 3: Validation & Testing**

### **3.1 Test Enterprise Integration**
```bash
# Navigate to backend
cd backend

# Activate virtual environment
source ../.venv/bin/activate

# Run enterprise integration tests
python scripts/test_enterprise_simple.py

# Test configuration
python -c "from config.settings import azure_settings; print('Configuration loaded successfully')"
```

### **3.2 Test Azure Services**
```bash
# Test storage connectivity
python -c "
from core.azure_storage.storage_client import AzureStorageClient
client = AzureStorageClient()
print('Storage client initialized successfully')
"

# Test search connectivity
python -c "
from core.azure_search.search_client import AzureCognitiveSearchClient
client = AzureCognitiveSearchClient()
print('Search client initialized successfully')
"
```

### **3.3 Test Service Health**
```bash
# Test unified service health
python -c "
from integrations.azure_services import AzureServicesManager
manager = AzureServicesManager()
health = manager.check_all_services_health()
print(f'Overall health: {health[\"overall_status\"]}')
"
```

---

## 🚀 **Phase 4: Application Deployment**

### **4.1 Build Application**
```bash
# Build Docker image
cd backend && docker build -t azure-maintie-rag:latest .

# Tag for Azure Container Registry (if using ACR)
# az acr build --registry your-acr-name --image azure-maintie-rag:latest .
```

### **4.2 Deploy to Azure Container Apps**
```bash
# Deploy to Container Apps
az containerapp create \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --environment maintie-dev-env \
  --image azure-maintie-rag:latest \
  --target-port 8000 \
  --ingress external \
  --env-vars \
    AZURE_ENVIRONMENT=$ENVIRONMENT \
    AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT \
    AZURE_SEARCH_SERVICE=$SEARCH_SERVICE \
    AZURE_KEY_VAULT_URL=https://$KEY_VAULT_NAME.vault.azure.net/ \
    AZURE_USE_MANAGED_IDENTITY=true \
    AZURE_ENABLE_TELEMETRY=true

# Get application URL
APP_URL=$(az containerapp show \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

echo "Application deployed at: https://$APP_URL"
```

### **4.3 Deploy with Azure ML (Alternative)**
```bash
# Deploy to Azure ML endpoint
az ml online-endpoint create \
  --name maintie-rag-endpoint \
  --resource-group $RESOURCE_GROUP \
  --workspace-name maintie-dev-ml \
  --auth-mode key

az ml online-deployment create \
  --name maintie-rag-deployment \
  --endpoint maintie-rag-endpoint \
  --resource-group $RESOURCE_GROUP \
  --workspace-name maintie-dev-ml \
  --model azure-maintie-rag:latest \
  --instance-count 1 \
  --instance-type Standard_DS3_v2
```

---

## 📊 **Phase 5: Monitoring & Operations**

### **5.1 Enable Monitoring**
```bash
# Get Application Insights connection string
APP_INSIGHTS_CONNECTION=$(az monitor app-insights component show \
  --app maintie-dev-app-insights \
  --resource-group $RESOURCE_GROUP \
  --query connectionString \
  --output tsv)

# Update environment with App Insights
echo "AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING=$APP_INSIGHTS_CONNECTION" >> backend/.env

# Restart application with monitoring
az containerapp update \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING="$APP_INSIGHTS_CONNECTION"
```

### **5.2 Health Monitoring Commands**
```bash
# Check application health
curl -f https://$APP_URL/health || echo "Application health check failed"

# Check Azure services health
cd backend
python -c "
from integrations.azure_services import AzureServicesManager
manager = AzureServicesManager()
health = manager.check_all_services_health()
print('Service Health:', health['overall_status'])
for service, status in health['services'].items():
    print(f'  {service}: {status.get(\"status\", \"unknown\")}')
"

# Monitor logs
az containerapp logs show \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --follow
```

### **5.3 Performance Monitoring**
```bash
# Get resource usage
az monitor metrics list \
  --resource-group $RESOURCE_GROUP \
  --resource-type Microsoft.Storage/storageAccounts \
  --resource $STORAGE_ACCOUNT \
  --metric "UsedCapacity" \
  --interval 1h

# Get search service metrics
az monitor metrics list \
  --resource-group $RESOURCE_GROUP \
  --resource-type Microsoft.Search/searchServices \
  --resource $SEARCH_SERVICE \
  --metric "SearchLatency" \
  --interval 1h
```

---

## 🔄 **Phase 6: Scaling & Updates**

### **6.1 Scale Application**
```bash
# Scale up for production
az containerapp revision set-mode \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --mode multiple

az containerapp update \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --min-replicas 2 \
  --max-replicas 10

# Scale down for development
az containerapp update \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --min-replicas 1 \
  --max-replicas 3
```

### **6.2 Update Application**
```bash
# Build new version
cd backend && docker build -t azure-maintie-rag:v2.0.0 .

# Deploy update
az containerapp update \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --image azure-maintie-rag:v2.0.0

# Verify update
az containerapp revision list \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --output table
```

### **6.3 Environment Promotion**
```bash
# Promote to production
PROD_RESOURCE_GROUP="maintie-rag-prod-rg"
PROD_ENVIRONMENT="prod"

# Deploy to production with higher SKUs
az deployment group create \
  --resource-group $PROD_RESOURCE_GROUP \
  --template-file infrastructure/azure-resources.bicep \
  --parameters environment=$PROD_ENVIRONMENT \
  --verbose
```

---

## 🧹 **Phase 7: Teardown & Cleanup**

### **7.1 Stop Application**
```bash
# Scale down to zero
az containerapp update \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --min-replicas 0 \
  --max-replicas 0

# Or delete the application
az containerapp delete \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --yes
```

### **7.2 Cleanup Resources**
```bash
# Delete specific resources
az storage account delete \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --yes

az search service delete \
  --name $SEARCH_SERVICE \
  --resource-group $RESOURCE_GROUP \
  --yes

az keyvault delete \
  --name $KEY_VAULT_NAME \
  --resource-group $RESOURCE_GROUP \
  --yes

# Delete entire resource group (nuclear option)
az group delete \
  --name $RESOURCE_GROUP \
  --yes \
  --no-wait
```

### **7.3 Cleanup Local Environment**
```bash
# Remove local environment
rm -rf .venv
rm backend/.env

# Clean Docker images
docker rmi azure-maintie-rag:latest
docker system prune -f
```

---

## 🔄 **Automated Lifecycle Scripts**

### **Complete Deployment Script**
```bash
#!/bin/bash
# deploy.sh - Complete deployment script

set -e

# Configuration
RESOURCE_GROUP="maintie-rag-rg"
LOCATION="eastus"
ENVIRONMENT="dev"

echo "🚀 Starting Azure Universal RAG deployment..."

# Phase 1: Infrastructure
echo "📦 Deploying infrastructure..."
az group create --name $RESOURCE_GROUP --location $LOCATION
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file infrastructure/azure-resources.bicep \
  --parameters environment=$ENVIRONMENT

# Phase 2: Configuration
echo "⚙️  Configuring services..."
# ... (configuration steps from Phase 2)

# Phase 3: Validation
echo "🧪 Validating deployment..."
cd backend
python scripts/test_enterprise_simple.py

# Phase 4: Deployment
echo "🚀 Deploying application..."
# ... (deployment steps from Phase 4)

echo "✅ Deployment completed successfully!"
```

### **Complete Teardown Script**
```bash
#!/bin/bash
# teardown.sh - Complete teardown script

set -e

RESOURCE_GROUP="maintie-rag-rg"

echo "🧹 Starting cleanup..."

# Stop application
az containerapp update \
  --name maintie-rag-app \
  --resource-group $RESOURCE_GROUP \
  --min-replicas 0 \
  --max-replicas 0

# Delete resource group
az group delete --name $RESOURCE_GROUP --yes --no-wait

echo "✅ Cleanup completed!"
```

---

## 📋 **Quick Reference Commands**

### **Daily Operations**
```bash
# Check health
curl -f https://$APP_URL/health

# View logs
az containerapp logs show --name maintie-rag-app --resource-group $RESOURCE_GROUP

# Scale up
az containerapp update --name maintie-rag-app --resource-group $RESOURCE_GROUP --min-replicas 2

# Scale down
az containerapp update --name maintie-rag-app --resource-group $RESOURCE_GROUP --min-replicas 1
```

### **Troubleshooting**
```bash
# Check resource status
az resource list --resource-group $RESOURCE_GROUP --output table

# Check deployment status
az deployment group list --resource-group $RESOURCE_GROUP --output table

# Get connection strings
az storage account show-connection-string --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP

# Check Key Vault secrets
az keyvault secret list --vault-name $KEY_VAULT_NAME
```

---

## 🎯 **Summary**

This lifecycle execution guide provides **complete command-line automation** for:

✅ **Infrastructure Deployment** - Automated resource creation with cost optimization
✅ **Configuration Management** - Automated secrets and environment setup
✅ **Application Deployment** - Automated container deployment
✅ **Monitoring Setup** - Automated health checks and telemetry
✅ **Scaling Operations** - Automated scaling based on environment
✅ **Cleanup Procedures** - Automated teardown and resource cleanup

**All operations can be executed via command line with no manual intervention required.**