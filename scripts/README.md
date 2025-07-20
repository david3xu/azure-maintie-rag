# Azure Universal RAG - Essential Scripts

This directory contains only the essential scripts for Azure Universal RAG deployment and management.

## 📁 Essential Scripts

### 🚀 `enhanced-complete-redeploy.sh`
**Main deployment script** - Deploys all working Azure services
- Deploys Storage Account, Search Service, Key Vault, Application Insights, Log Analytics
- Uses clean Bicep template with only working services
- Includes health checks and validation

### 📊 `status-working.sh`
**Status checker** - Shows current working services status
- Lists all 10 working Azure services
- Shows which services are operational
- Provides clear status summary

### 🧹 `teardown.sh`
**Cleanup script** - Removes all Azure resources
- Deletes all resources in the resource group
- Cleans up deployment artifacts
- Use with caution in production

## 🎯 Usage

```bash
# Deploy all working services
./scripts/enhanced-complete-redeploy.sh

# Check status of working services
./scripts/status-working.sh

# Clean up all resources (use with caution)
./scripts/teardown.sh
```

## ✅ Working Services (10/10 Complete)

The deployment creates these 10 essential Azure services:

### Core Infrastructure (via Bicep)
1. **Storage Account** - `maintiedevstor1cdd8e11` - For Universal RAG data
2. **Search Service** - `maintie-dev-search-1cdd8e` - For vector search and indexing
3. **Key Vault** - `maintie-dev-kv-1cdd8e` - For secrets management
4. **Application Insights** - `maintie-dev-appinsights` - For monitoring
5. **Log Analytics** - `maintie-dev-logs` - For logging

### ML Infrastructure (via CLI)
6. **ML Storage Account** - `maintiedevmlstor1cdd8e11` - For ML workspace data
7. **ML Workspace** - `maintie-dev-ml-1cdd8e11` - For ML model training and inference

### Container Infrastructure (via CLI)
8. **Container Environment** - `maintie-dev-env-1cdd8e11` - For container apps
9. **Container App** - `maintie-dev-app-1cdd8e11` - For application deployment

### Data Infrastructure (via CLI)
10. **Cosmos DB** - `maintie-dev-cosmos-1cdd8e11` & `maintie-dev-cosmos-alt` - For knowledge graphs

## 🛠️ Step-by-Step CLI Commands

If you need to recreate services manually, here are the exact CLI commands:

### 1. Core Infrastructure (Bicep)
```bash
# Deploy core infrastructure
az deployment group create \
  --resource-group maintie-rag-rg \
  --template-file infrastructure/azure-resources-core.bicep \
  --parameters "environment=dev" "location=eastus" "resourcePrefix=maintie" "deploymentToken=1cdd8e11" \
  --name "azure-resources-core-$(date +%Y%m%d-%H%M%S)" \
  --mode Incremental
```

### 2. ML Storage Account
```bash
az storage account create \
  --resource-group maintie-rag-rg \
  --name maintiedevmlstor1cdd8e11 \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2
```

### 3. Enable Key Vault for Template Deployment
```bash
az keyvault update \
  --name maintie-dev-kv-1cdd8e \
  --resource-group maintie-rag-rg \
  --enabled-for-template-deployment true
```

### 4. ML Workspace
```bash
# Get resource IDs
STORAGE_ID=$(az resource list --resource-group maintie-rag-rg --resource-type "Microsoft.Storage/storageAccounts" --query "[?name=='maintiedevmlstor1cdd8e11'].id" --output tsv)
KEYVAULT_ID=$(az resource list --resource-group maintie-rag-rg --resource-type "Microsoft.KeyVault/vaults" --query "[?name=='maintie-dev-kv-1cdd8e'].id" --output tsv)
APPINSIGHTS_ID=$(az resource list --resource-group maintie-rag-rg --resource-type "Microsoft.Insights/components" --query "[?name=='maintie-dev-appinsights'].id" --output tsv)

# Create ML workspace
az ml workspace create \
  --resource-group maintie-rag-rg \
  --name maintie-dev-ml-1cdd8e11 \
  --location eastus \
  --storage-account "$STORAGE_ID" \
  --key-vault "$KEYVAULT_ID" \
  --application-insights "$APPINSIGHTS_ID"
```

### 5. Container Environment
```bash
az containerapp env create \
  --resource-group maintie-rag-rg \
  --name maintie-dev-env-1cdd8e11 \
  --location eastus
```

### 6. Container App
```bash
az containerapp create \
  --resource-group maintie-rag-rg \
  --name maintie-dev-app-1cdd8e11 \
  --environment maintie-dev-env-1cdd8e11 \
  --image nginx:latest \
  --target-port 80 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 1
```

### 7. Cosmos DB (if region available)
```bash
az cosmosdb create \
  --resource-group maintie-rag-rg \
  --name maintie-dev-cosmos-1cdd8e11 \
  --locations regionName=eastus failoverPriority=0 isZoneRedundant=false \
  --capabilities EnableGremlin \
  --default-consistency-level Session
```

## 🏗️ Architecture

- **Core Services**: Deployed via Bicep templates for consistency
- **ML Services**: Deployed via CLI for better control and resource ID handling
- **Container Services**: Deployed via CLI for immediate availability
- **Data Services**: Deployed via CLI with region-specific considerations
- **Deterministic Naming**: Uses consistent naming patterns across all services
- **Environment-Driven**: Supports dev/staging/prod configurations
- **Production Ready**: All 10 services tested and operational