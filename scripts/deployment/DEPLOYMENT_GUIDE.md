# Azure Universal RAG Deployment Guide

## 🚀 Quick Start

The Azure Universal RAG system is now ready for deployment with `azd up`. All configuration files have been updated to work with the new directory structure.

### Prerequisites

1. **Azure CLI**: `az login` (must be logged in)
2. **Azure Developer CLI**: `azd` command available
3. **Python 3.8+**: For status testing and cleanup scripts
4. **Azure Subscription**: With appropriate permissions

### Deployment Commands

```bash
# Quick deployment
azd up

# Custom environment
azd up --environment staging --location eastus

# Test services status
./scripts/deployment/azure_deployment_helper.sh status

# Full deployment with testing
./scripts/deployment/azure_deployment_helper.sh deploy --env prod --location westus2
```

## 📋 What's Been Fixed

### 1. Azure.yaml Configuration ✅
- **Fixed**: Updated `project: .` (was `project: ./backend`)  
- **Fixed**: Updated prepackage hook for new structure
- **Status**: Ready for `azd up`

### 2. Directory Structure ✅
- **Backend code**: Moved from `backend/` to root level
- **API entry point**: `api/main.py` 
- **Dockerfile**: Updated for new structure
- **Infrastructure**: Bicep templates in `infra/`

### 3. Testing & Monitoring Tools ✅

#### Azure Services Status Test
```bash
python3 scripts/deployment/test_azure_services_status.py
```

**Features:**
- Tests all Azure services connectivity
- Concurrent testing for speed
- Detailed health reports
- JSON output for automation
- Exit codes for CI/CD integration

**Tests These Services:**
- Azure OpenAI (GPT-4, embeddings)
- Azure Cognitive Search  
- Azure Cosmos DB (Gremlin)
- Azure Storage Account
- Azure Key Vault
- Azure ML Workspace
- Application Insights

#### Azure Resources Cleanup
```bash
# Dry run (safe)
python3 scripts/deployment/cleanup_azure_services.py --subscription-id <sub-id> --dry-run

# Live cleanup (DANGEROUS)
python3 scripts/deployment/cleanup_azure_services.py --subscription-id <sub-id> --live
```

**Features:**
- Smart resource discovery by naming patterns
- Protected resource patterns (never deletes prod resources)
- Comprehensive safety checks
- Resource group or individual resource deletion
- Detailed cleanup reports

### 4. Deployment Helper Script ✅

All-in-one deployment management:

```bash
./scripts/deployment/azure_deployment_helper.sh <command>

Commands:
  status          Test Azure services connectivity
  deploy          Deploy with azd up + validation
  cleanup         Interactive cleanup (dry-run by default)  
  force-cleanup   Force cleanup without prompts
  validate        Validate deployment configuration
  logs            Show deployment logs
```

## 🏗️ Infrastructure Overview

### Bicep Templates Structure
```
infra/
├── main.bicep                 # Main deployment template
├── main.parameters.json       # Deployment parameters  
└── modules/
    ├── ai-services.bicep      # Azure OpenAI
    ├── core-services.bicep    # Search, Storage, KeyVault
    ├── data-services.bicep    # Cosmos DB, ML Workspace  
    └── hosting-services.bicep # Container Apps
```

### Deployed Resources
- **AI Services**: Azure OpenAI (GPT-4, embeddings)
- **Search**: Azure Cognitive Search with vector support
- **Database**: Cosmos DB with Gremlin API for knowledge graphs
- **Storage**: Blob storage for documents and models
- **Compute**: Container Apps for API hosting
- **ML**: Azure ML workspace for GNN training
- **Monitoring**: Application Insights for observability
- **Security**: Key Vault for secrets management

## 🔧 Environment Configuration

### Required Environment Variables

```bash
# Core Azure Configuration
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_LOCATION="westus2"
export AZURE_ENV_NAME="dev"

# Set by azd automatically:
export AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/"
export AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net"
export AZURE_COSMOS_ENDPOINT="https://your-cosmos.documents.azure.com:443/"
export AZURE_STORAGE_ACCOUNT="yourstorageaccount"
export AZURE_KEY_VAULT_NAME="your-keyvault"
```

### Configuration Files
- `azure.yaml` - azd configuration
- `infra/main.parameters.json` - Infrastructure parameters
- `config/settings.py` - Application settings
- `Dockerfile` - Container configuration

## 🚀 Deployment Workflow

### 1. Initial Setup
```bash
# Clone and navigate to project
cd azure-maintie-rag

# Login to Azure
az login

# Set subscription (if needed)
az account set --subscription "your-subscription-id"
```

### 2. Validate Configuration
```bash
# Check prerequisites and configuration
./scripts/deployment/azure_deployment_helper.sh validate
```

### 3. Deploy Infrastructure
```bash
# Deploy everything
azd up

# Or with custom settings
azd up --environment staging --location eastus
```

### 4. Test Deployment
```bash
# Test all services
./scripts/deployment/azure_deployment_helper.sh status

# Or manually
python3 scripts/deployment/test_azure_services_status.py
```

### 5. Monitor & Manage
```bash
# Show logs
azd monitor --live

# Check resource groups
az group list --query "[?contains(name, 'maintie-rag')]"
```

## 🧹 Cleanup & Maintenance

### Safe Cleanup (Recommended)
```bash
# Dry run first (always!)
./scripts/deployment/azure_deployment_helper.sh cleanup --dry-run

# Live cleanup if satisfied
./scripts/deployment/azure_deployment_helper.sh cleanup --live
```

### Emergency Cleanup
```bash
# Force cleanup without prompts (DANGEROUS)
./scripts/deployment/azure_deployment_helper.sh force-cleanup
```

### azd Cleanup
```bash
# azd managed cleanup
azd down --force --purge
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "Backend project not found"
- **Cause**: Old azure.yaml with `project: ./backend`
- **Fix**: Updated to `project: .` ✅

#### 2. "Dockerfile not found"  
- **Cause**: Looking in wrong directory
- **Fix**: Dockerfile in root directory ✅

#### 3. "Import errors in Python"
- **Cause**: Directory structure migration
- **Fix**: All imports updated for new structure ✅

#### 4. "Azure authentication failed"
- **Fix**: Run `az login` and ensure proper subscription access

#### 5. "Resource already exists"
- **Fix**: Use cleanup scripts or deploy to different environment

### Validation Commands
```bash
# Check azd configuration
azd config list

# Validate Bicep templates
az deployment sub validate --template-file infra/main.bicep --parameters infra/main.parameters.json

# Test Python imports
python3 -c "from api.main import app; print('✅ API imports working')"
```

## 📊 Monitoring & Observability

### Built-in Monitoring
- **Application Insights**: Automatic telemetry
- **Azure Monitor**: Infrastructure metrics
- **Container Apps logs**: Application logs
- **Health endpoints**: `/api/v1/health`

### Custom Monitoring
- **Status testing**: Automated with scripts
- **Performance metrics**: Sub-3s response time tracking
- **Error tracking**: Comprehensive error handling
- **Usage metrics**: Query and response analytics

## 🔐 Security Considerations

### Implemented Security
- **Managed Identity**: For Azure service authentication
- **Key Vault**: For secrets management
- **Network Security**: VNet integration in production
- **RBAC**: Principle of least privilege
- **HTTPS**: All endpoints secured

### Security Checklist
- [ ] Review Key Vault access policies
- [ ] Validate network security groups
- [ ] Check managed identity permissions
- [ ] Audit resource access logs
- [ ] Verify HTTPS enforcement

## 📈 Performance & Scaling

### Performance Targets
- **Response Time**: <3 seconds for all queries
- **Throughput**: 100+ concurrent requests
- **Availability**: 99.9% uptime
- **Scalability**: Auto-scaling based on demand

### Scaling Configuration
- **Container Apps**: Auto-scale 1-10 replicas
- **Azure Search**: Standard tier with replicas
- **Cosmos DB**: Provisioned throughput with auto-scale
- **Storage**: Hot tier for active data

## 🎯 Next Steps

1. **Deploy**: Run `azd up` to deploy infrastructure
2. **Test**: Use status scripts to validate deployment  
3. **Monitor**: Set up alerts and monitoring dashboards
4. **Scale**: Configure auto-scaling based on usage
5. **Secure**: Review and harden security settings

## 📚 Additional Resources

- [Azure Developer CLI Documentation](https://docs.microsoft.com/azure/developer/azure-developer-cli/)
- [Azure Container Apps](https://docs.microsoft.com/azure/container-apps/)
- [Azure OpenAI Service](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Project Architecture Guide](../../docs/architecture/SYSTEM_ARCHITECTURE.md)

---

**Status**: ✅ Ready for deployment with `azd up`
**Last Updated**: August 2, 2025
**Directory Structure**: Migrated and validated