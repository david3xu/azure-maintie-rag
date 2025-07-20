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
- Lists all 6 working Azure services
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

## ✅ Working Services

The deployment creates these 7 essential Azure services:
1. **Storage Account** - For Universal RAG data
2. **ML Storage Account** - For ML workspace data
3. **Search Service** - For vector search and indexing
4. **Key Vault** - For secrets management
5. **Application Insights** - For monitoring
6. **Log Analytics** - For logging
7. **Smart Detector Alert Rule** - For failure anomaly detection

## 🏗️ Architecture

- **Clean Bicep Template**: `infrastructure/azure-resources-core.bicep`
- **Deterministic Naming**: Uses `uniqueString()` for consistent resource names
- **Environment-Driven**: Supports dev/staging/prod configurations
- **Production Ready**: Only includes services that actually work