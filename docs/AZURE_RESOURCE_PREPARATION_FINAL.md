# 🚀 Azure Universal RAG - Complete Resource Preparation Guide

## 📋 Overview
This is the **FINAL VERSION** of the complete Azure Universal RAG resource preparation guide. It provides everything needed to deploy, manage, and maintain the Azure Universal RAG infrastructure.

## 🎯 What This Guide Covers
- **Complete Infrastructure Deployment** (10 Azure services)
- **Enterprise Security** (Key Vault, managed identities)
- **ML Training Environment** (Azure ML workspace)
- **Monitoring & Logging** (Application Insights)
- **Application Hosting** (Container Apps)
- **Complete Lifecycle Management** (deploy, teardown, redeploy)

---

## 🏗️ Architecture Overview

### Core Infrastructure (4 services)
- **Azure Storage Account** (`maintiedevstorage`) - Document storage
- **Azure Cognitive Search** (`maintie-dev-search`) - Vector search
- **Azure Key Vault** (`maintie-dev-kv`) - Secrets management
- **Azure Cosmos DB** (`maintie-dev-cosmos`) - Knowledge graph storage

### ML Infrastructure (3 services)
- **Azure ML Workspace** (`maintie-dev-ml`) - GNN training environment
- **ML Storage Account** (`maintiedevmlstorage`) - ML artifacts storage
- **Application Insights** (`maintie-dev-app-insights`) - Monitoring & telemetry

### Application Infrastructure (3 services)
- **Log Analytics** (`maintie-dev-laworkspace`) - Centralized logging
- **Container Environment** (`maintie-dev-env`) - Container hosting
- **Container App** (`maintie-dev-rag-app`) - Application hosting

---

## 🚀 Quick Start Deployment

### Option 1: Automated Complete Deployment
```bash
# Run complete automated deployment (recommended)
./scripts/complete-redeploy.sh
```

### Option 2: Step-by-Step Deployment
```bash
# 1. Create resource group
az group create --name maintie-rag-rg --location eastus

# 2. Deploy core infrastructure
./scripts/deploy-core.sh

# 3. Deploy ML infrastructure
./scripts/deploy-ml.sh

# 4. Verify deployment
./scripts/check-resources.sh
```

---

## 📦 Infrastructure as Code

### Core Infrastructure Template
**File**: `infrastructure/azure-resources-core.bicep`

**Deploys**:
- Storage Account (with HNS for document storage)
- Cognitive Search Service
- Key Vault (with managed identity)
- Cosmos DB Gremlin API

### ML Infrastructure Template
**File**: `infrastructure/azure-resources-ml.bicep`

**Deploys**:
- ML Storage Account (no HNS for ML compatibility)
- ML Workspace
- Application Insights
- Log Analytics
- Container Environment
- Container App

---

## 🔧 Deployment Scripts

### Core Deployment
**File**: `scripts/deploy-core.sh`
```bash
# Deploys core infrastructure
./scripts/deploy-core.sh
```

### ML Deployment
**File**: `scripts/deploy-ml.sh`
```bash
# Deploys ML infrastructure
./scripts/deploy-ml.sh
```

### Complete Redeployment
**File**: `scripts/complete-redeploy.sh`
```bash
# Complete teardown and redeployment
./scripts/complete-redeploy.sh
```

### Resource Verification
**File**: `scripts/check-resources.sh`
```bash
# Verify all resources are deployed
./scripts/check-resources.sh
```

### Diagnostics
**File**: `scripts/diagnose.sh`
```bash
# Run comprehensive diagnostics
./scripts/diagnose.sh
```

---

## 🛠️ Manual Resource Creation (If Needed)

### ML Storage Account (No HNS)
```bash
az storage account create \
  --name maintiedevmlstorage \
  --resource-group maintie-rag-rg \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2 \
  --https-only true \
  --min-tls-version TLS1_2
```

### ML Workspace (With Full ARM IDs)
```bash
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
az ml workspace create \
  --name maintie-dev-ml \
  --resource-group maintie-rag-rg \
  --location eastus \
  --storage-account "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/maintie-rag-rg/providers/Microsoft.Storage/storageAccounts/maintiedevmlstorage" \
  --key-vault "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/maintie-rag-rg/providers/Microsoft.KeyVault/vaults/maintie-dev-kv" \
  --application-insights "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/maintie-rag-rg/providers/Microsoft.Insights/components/maintie-dev-app-insights"
```

---

## 🧪 Testing & Verification

### 1. Resource Verification
```bash
# Check all resources
./scripts/check-resources.sh

# Expected output: All 10 resources found
```

### 2. ML Workspace Test
```bash
# Verify ML workspace
az ml workspace show \
  --name maintie-dev-ml \
  --resource-group maintie-rag-rg \
  --query "{name:name,location:location,provisioningState:provisioningState}"
```

### 3. GNN Training Test
```bash
# Test GNN training
python backend/scripts/train_comprehensive_gnn.py \
  --workspace maintie-dev-ml \
  --resource-group maintie-rag-rg
```

### 4. Application Deployment
```bash
# Build container
cd backend && docker build -t azure-maintie-rag:latest .

# Deploy to Container Apps
az containerapp update \
  --name maintie-dev-rag-app \
  --resource-group maintie-rag-rg \
  --image azure-maintie-rag:latest
```

---

## 🔥 Complete Teardown

### Option 1: Automated Teardown
```bash
# Complete teardown via redeployment script
./scripts/complete-redeploy.sh
```

### Option 2: Manual Teardown
```bash
# Delete entire resource group
az group delete --name maintie-rag-rg --yes --no-wait

# Wait for completion
az group show --name maintie-rag-rg
```

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### Issue 1: Storage Account HNS Conflict
**Problem**: ML workspace fails with "Cannot use storage with HNS enabled"
**Solution**: Use separate storage account for ML without HNS
```bash
# Create ML storage without HNS
az storage account create \
  --name maintiedevmlstorage \
  --resource-group maintie-rag-rg \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2 \
  --https-only true \
  --min-tls-version TLS1_2
```

#### Issue 2: Azure CLI Extension Conflicts
**Problem**: ML CLI extension conflicts
**Solution**: Remove conflicting extensions
```bash
az extension remove -n azure-cli-ml
az extension add -n ml
```

#### Issue 3: Resource Group Deletion Hangs
**Problem**: Resource group deletion takes too long
**Solution**: Force deletion
```bash
az group delete --name maintie-rag-rg --yes --no-wait
# Wait and check status
az group show --name maintie-rag-rg
```

#### Issue 4: Container App Deployment Fails
**Problem**: Container image not found
**Solution**: Use placeholder image first
```bash
az containerapp update \
  --name maintie-dev-rag-app \
  --resource-group maintie-rag-rg \
  --image mcr.microsoft.com/azuredocs/containerapps-helloworld:latest
```

---

## 📊 Resource Inventory

### Expected Final Resources (10 total)

| Resource | Name | Type | Purpose |
|----------|------|------|---------|
| Storage Account | `maintiedevstorage` | `Microsoft.Storage/storageAccounts` | Document storage |
| ML Storage Account | `maintiedevmlstorage` | `Microsoft.Storage/storageAccounts` | ML artifacts |
| Search Service | `maintie-dev-search` | `Microsoft.Search/searchServices` | Vector search |
| Key Vault | `maintie-dev-kv` | `Microsoft.KeyVault/vaults` | Secrets management |
| Cosmos DB | `maintie-dev-cosmos` | `Microsoft.DocumentDB/databaseAccounts` | Knowledge graph |
| ML Workspace | `maintie-dev-ml` | `Microsoft.MachineLearningServices/workspaces` | GNN training |
| Application Insights | `maintie-dev-app-insights` | `Microsoft.Insights/components` | Monitoring |
| Log Analytics | `maintie-dev-laworkspace` | `Microsoft.OperationalInsights/workspaces` | Logging |
| Container Environment | `maintie-dev-env` | `Microsoft.App/managedEnvironments` | Container hosting |
| Container App | `maintie-dev-rag-app` | `Microsoft.App/containerApps` | Application hosting |

### Verification Command
```bash
az resource list \
  --resource-group maintie-rag-rg \
  --query "[].{Name:name,Type:type,Location:location}" \
  --output table
```

---

## 📈 Monitoring & Maintenance

### Daily Operations
```bash
# Check resource health
./scripts/check-resources.sh

# Monitor costs
az consumption usage list \
  --billing-period-name $(az billing period list --query "[0].name" -o tsv)
```

### Weekly Maintenance
```bash
# Update Azure CLI extensions
az extension update --name ml

# Check for resource updates
az resource list \
  --resource-group maintie-rag-rg \
  --query "[].{Name:name,LastModified:lastModifiedTime}" \
  --output table
```

### Monthly Review
```bash
# Complete resource audit
./scripts/diagnose.sh

# Cost optimization review
az consumption usage list \
  --billing-period-name $(az billing period list --query "[0].name" -o tsv) \
  --query "[?contains(instanceName, 'maintie')].{Resource:instanceName,Cost:pretaxCost}" \
  --output table
```

---

## 🎯 Success Criteria

Your deployment is successful when:

1. ✅ **All 10 resources** are created and running
2. ✅ **Resource checker** shows all resources as "Found"
3. ✅ **ML workspace** can be accessed and used for training
4. ✅ **GNN training script** runs without errors
5. ✅ **Container app** is deployed and accessible
6. ✅ **All services** are properly connected and authenticated
7. ✅ **Monitoring and logging** are working
8. ✅ **Costs** are within expected budget

---

## 🏆 Final Architecture Benefits

Once deployed, your Azure Universal RAG infrastructure provides:

### 🔒 **Enterprise Security**
- **Key Vault integration** for secure credential management
- **Managed identities** for service-to-service authentication
- **HTTPS-only** storage and communication
- **TLS 1.2+** encryption

### 🧠 **Advanced ML Capabilities**
- **Azure ML workspace** for GNN training
- **Scalable compute** for model training
- **Model versioning** and management
- **Experiment tracking** and monitoring

### 📊 **Comprehensive Monitoring**
- **Application Insights** for application telemetry
- **Log Analytics** for centralized logging
- **Resource health** monitoring
- **Cost tracking** and optimization

### 🚀 **Scalable Architecture**
- **Container Apps** for application hosting
- **Blob Storage** for document storage
- **Cognitive Search** for vector search
- **Cosmos DB** for knowledge graph storage

### 🔄 **Complete Lifecycle Management**
- **Infrastructure as Code** with Bicep templates
- **Automated deployment** scripts
- **Resource verification** tools
- **Complete teardown** and redeployment capabilities

---

## 📞 Support & Documentation

### Primary Documentation
- **This Guide**: Complete resource preparation
- **Architecture Guide**: `docs/COMPLETE_RAG_ARCHITECTURE.md`
- **Lifecycle Guide**: `docs/AZURE_LIFECYCLE_EXECUTION.md`

### Scripts & Tools
- **Complete Redeployment**: `./scripts/complete-redeploy.sh`
- **Resource Checker**: `./scripts/check-resources.sh`
- **Diagnostics**: `./scripts/diagnose.sh`
- **Core Deployment**: `./scripts/deploy-core.sh`
- **ML Deployment**: `./scripts/deploy-ml.sh`

### Troubleshooting
- **Common Issues**: See troubleshooting section above
- **Azure Portal**: Check resource health and logs
- **Azure CLI**: Use diagnostic commands
- **Scripts**: Run verification and diagnostic scripts

---

## 🎉 Congratulations!

You now have a **complete, enterprise-ready Azure Universal RAG infrastructure** that provides:

- **🔒 Enterprise-grade security** with Key Vault and managed identities
- **🧠 Advanced ML capabilities** with Azure ML workspace
- **📊 Comprehensive monitoring** with Application Insights
- **🚀 Scalable architecture** with Container Apps and Blob Storage
- **🔄 Complete lifecycle management** with automated scripts

**Your Azure Universal RAG system is ready to power intelligent document processing and knowledge discovery!** 🚀

---

*This is the FINAL VERSION of the Azure Universal RAG resource preparation guide. All previous versions should be replaced with this comprehensive document.*