# 🚀 Azure Universal RAG - Minimal Setup Guide

## ✅ **Your Architecture is Already Ready!**

Your `make dev` is already architected for Azure services integration. Based on your existing codebase analysis, here are the **minimal setup instructions**:

## 🔧 **Setup Options: Manual vs Automated**

### **Option 1: Automated Configuration** 🚀 **RECOMMENDED**
**Single command deployment with automatic configuration:**

```bash
# Deploy Azure infrastructure + auto-configure + start development
make azure-dev-auto
```

**Benefits:**
- ✅ **No manual .env copying**
- ✅ **No manual key configuration**
- ✅ **No manual endpoint configuration**
- ✅ **Uses Azure Key Vault/Managed Identity**
- ✅ **Data-driven from Bicep templates**

### **Option 2: Manual Configuration** (Your current approach)
**Step 1: Environment Configuration** ✅ COMPLETED
```bash
# ✅ Already done: Environment file created
cp backend/config/environment_example.env backend/.env
```

**Step 2: Azure Service Endpoints Configuration**
**Update `backend/.env` with values from your deployment output:**

```bash
# Run the automated update script
./scripts/update-env-from-deployment.sh
```

**Or manually update these values in `backend/.env`:**
```bash
# From your ./scripts/enhanced-complete-redeploy.sh output:
AZURE_STORAGE_ACCOUNT=maintiedevmlstor1cdd8e11
AZURE_SEARCH_SERVICE=maintie-dev-search-1cdd8e
AZURE_COSMOS_ENDPOINT=https://maintie-dev-cosmos-1cdd8e11.documents.azure.com:443/
AZURE_KEY_VAULT_URL=https://maintie-dev-kv-[token].vault.azure.net/

# Azure OpenAI (your existing configuration)
OPENAI_API_BASE=https://your-azure-openai-instance.openai.azure.com/
OPENAI_API_KEY=your-azure-openai-key
```

### **Step 3: API Keys Configuration** ⚠️ REQUIRED
**Get these from Azure Portal and update `backend/.env`:**
- `AZURE_STORAGE_KEY` - Storage account access key
- `AZURE_SEARCH_KEY` - Search service admin key
- `AZURE_COSMOS_KEY` - Cosmos DB primary key
- `OPENAI_API_KEY` - Your Azure OpenAI API key

## 🚀 **Start Development** (Your existing commands)
```bash
# Your existing workflow works:
make dev

# Services will be available at:
# Backend API: http://localhost:8000 (with Azure services)
# Frontend UI: http://localhost:5174 (connects to backend)
```

## 🔍 **Verification Commands** (Your existing targets)
```bash
# Verify setup works
make health

# Check Azure services connectivity
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/health/detailed

# Test workflow integration
make test-workflow
```

## 📋 **Configuration Checklist**

**Required environment variables in `backend/.env`:**
- ✅ `AZURE_STORAGE_ACCOUNT` and `AZURE_STORAGE_KEY`
- ✅ `AZURE_SEARCH_SERVICE` and `AZURE_SEARCH_KEY`
- ✅ `AZURE_COSMOS_ENDPOINT` and `AZURE_COSMOS_KEY`
- ✅ `OPENAI_API_BASE` and `OPENAI_API_KEY`
- ✅ `AZURE_KEY_VAULT_URL` (optional for enhanced security)

## 🎯 **What's Already Working**

### **Your Existing Architecture:**
1. **✅ Azure Services Manager** - Already initialized in `main.py`
2. **✅ Configuration Validation** - Added startup validation
3. **✅ Environment Settings** - Complete Azure settings support
4. **✅ Service Integration** - All Azure services integrated
5. **✅ Development Workflow** - `make dev` ready for Azure

### **Your Existing Features:**
- **✅ Three-layer progressive disclosure** - Workflow transparency
- **✅ Real-time streaming queries** - Azure-powered backend
- **✅ Universal RAG processing** - Domain-agnostic text processing
- **✅ Azure Cognitive Search** - Vector search integration
- **✅ Azure OpenAI processing** - LLM integration
- **✅ Azure Blob Storage (Multi-Account)** - RAG data, ML models, and app data storage
- **✅ Azure Cosmos DB** - Metadata storage

## 🔧 **Optional Enhancement: Startup Validation** ✅ ADDED

**Added to `backend/api/main.py` lifespan function:**
```python
# ADD: Validation check
validation_result = azure_settings.validate_azure_config()
logger.info(f"Azure configuration validation: {validation_result}")

# Log validation details
for service, configured in validation_result.items():
    status = "✅" if configured else "❌"
    logger.info(f"  {status} {service}: {'Configured' if configured else 'Not configured'}")
```

## 🚀 **Quick Start Commands**

### **Automated Approach (Recommended)**
```bash
# Single command: Deploy + Configure + Start Development
make azure-dev-auto
```

### **Manual Approach**
```bash
# 1. Update environment with Azure endpoints
./scripts/update-env-from-deployment.sh

# 2. Update API keys in backend/.env (manual step)

# 3. Start development
make dev

# 4. Verify everything works
make health
```

## 🔐 **Security Benefits**

### **Automated Approach**
- ✅ **Azure Key Vault**: Secrets stored securely in Azure
- ✅ **Managed Identity**: No keys in source code or files
- ✅ **Credential Chain**: Automatic authentication resolution
- ✅ **Enterprise Security**: Follows Azure security best practices

### **Manual Approach**
- ❌ **Manual Key Management**: Keys in plain text files
- ❌ **Security Risk**: Potential key exposure
- ❌ **Operational Overhead**: Manual configuration required

## 📊 **Expected Output**

When you run `make dev`, you should see:
```
🚀 Starting Universal RAG system:

📍 Backend API:   http://localhost:8000 (Universal RAG service)
📍 Frontend UI:   http://localhost:5174 (Workflow transparency)
📍 API Docs:      http://localhost:8000/docs
📍 Workflow API:  http://localhost:8000/api/v1/query/stream/{query_id}

🔄 Features: Three-layer progressive disclosure + real-time workflow tracking
```

## 🎉 **Result**

After setup, your `make dev` will:
1. **Start backend** with Azure services initialization
2. **Start frontend** connecting to Azure-powered backend
3. **Enable real-time workflow** with three-layer progressive disclosure
4. **Provide Azure service integration** through your existing architecture

**Your existing architecture requires minimal changes** - primarily environment configuration with deployed Azure service endpoints.

## 🔍 **Troubleshooting**

### **If services don't start:**
```bash
# Check Azure CLI authentication
az login

# Verify resource group exists
az group show --name maintie-rag-rg

# Check service status
./scripts/status-working.sh
```

### **If configuration validation fails:**
```bash
# Check .env file
cat backend/.env

# Re-run environment update
./scripts/update-env-from-deployment.sh
```

### **If API keys are missing:**
1. Go to Azure Portal
2. Navigate to each service (Storage, Search, Cosmos DB)
3. Copy the access keys
4. Update `backend/.env`

## 📝 **Summary**

**Your architecture is already perfect for Azure integration!** You now have **two approaches**:

### **Automated Approach (Recommended)**
```bash
make azure-dev-auto  # Deploy + Configure + Start Development
```

### **Manual Approach**
1. ✅ **Environment file created** (`backend/.env`)
2. ✅ **Startup validation added** (in `main.py`)
3. ⚠️ **Update service endpoints** (run `./scripts/update-env-from-deployment.sh`)
4. ⚠️ **Add API keys** (manual step from Azure Portal)

**The automated approach eliminates manual configuration and uses Azure security best practices!**