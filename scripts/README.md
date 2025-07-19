# Azure Universal RAG Scripts

This directory contains the enterprise deployment scripts for Azure Universal RAG infrastructure.

## 📁 Script Structure

### **Core Deployment Scripts**

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy.sh` | **Simple deployment** using enterprise architecture | `./scripts/deploy.sh` |
| `enhanced-complete-redeploy.sh` | **Full enterprise deployment** with all phases | `./scripts/enhanced-complete-redeploy.sh` |
| `teardown.sh` | **Safe teardown** with soft-delete cleanup | `./scripts/teardown.sh` |
| `status.sh` | **Status check** for deployment and resources | `./scripts/status.sh` |

### **Enterprise Architecture Modules**

| Module | Purpose | Description |
|--------|---------|-------------|
| `azure-deployment-manager.sh` | **Enterprise resilience patterns** | Handles deployment conflicts, soft-delete issues, and multi-region failures |
| `azure-service-validator.sh` | **Pre-deployment validation** | Service availability checking and conflict resolution |
| `test-enterprise-deployment.sh` | **Implementation validation** | Tests enterprise deployment architecture |

## 🚀 Quick Start

### **1. Simple Deployment**
```bash
# Deploy with enterprise architecture
./scripts/deploy.sh
```

### **2. Full Enterprise Deployment**
```bash
# Deploy with all enterprise features
./scripts/enhanced-complete-redeploy.sh
```

### **3. Check Status**
```bash
# Check current deployment status
./scripts/status.sh
```

### **4. Teardown**
```bash
# Safely remove all resources
./scripts/teardown.sh
```

## 🔧 Configuration

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_RESOURCE_GROUP` | `maintie-rag-rg` | Azure resource group name |
| `AZURE_ENVIRONMENT` | `dev` | Environment (dev, staging, prod) |
| `AZURE_LOCATION` | Auto-detected | Azure region for deployment |

### **Example Configuration**
```bash
export AZURE_RESOURCE_GROUP="my-rag-rg"
export AZURE_ENVIRONMENT="prod"
export AZURE_LOCATION="eastus"
```

## 🏗️ Enterprise Features

### **1. Conflict Resolution**
- **Time-based unique naming**: Prevents soft-delete conflicts
- **Automatic cleanup**: Detects and purges soft-deleted resources
- **Service availability validation**: Pre-deployment resource checking

### **2. Multi-Region Resilience**
- **Intelligent region selection**: Capacity-aware region optimization
- **Failover capabilities**: Automated region failover for deployment failures
- **Quota management**: Subscription quota validation and optimization

### **3. Enterprise Monitoring**
- **Real-time telemetry**: Application Insights integration
- **Performance tracking**: Comprehensive performance metrics
- **Error tracking**: Detailed error tracking and business metrics

## 📊 Deployment Phases

### **Simple Deployment** (`deploy.sh`)
1. **Authentication check**: Verify Azure CLI authentication
2. **Region selection**: Get optimal deployment region
3. **Resource group creation**: Create if needed
4. **Core deployment**: Deploy with exponential backoff

### **Enterprise Deployment** (`enhanced-complete-redeploy.sh`)
1. **Pre-deployment validation**: Azure CLI auth, extensions, prerequisites
2. **Optimal region selection**: Capacity-aware region selection
3. **Clean deployment preparation**: Conflict resolution and cleanup
4. **Resilient core infrastructure**: Exponential backoff deployment
5. **Conditional ML infrastructure**: Dependency-aware ML deployment
6. **Deployment verification**: Comprehensive resource verification

## 🔍 Testing

### **Test Enterprise Architecture**
```bash
# Test implementation of enterprise deployment patterns
./scripts/test-enterprise-deployment.sh
```

### **Test Enterprise Knowledge Extraction**
```bash
# Test enterprise knowledge extraction with Azure services
cd backend
python scripts/test_enterprise_knowledge_extraction.py
```

## 🛠️ Troubleshooting

### **Common Issues**

#### **1. Authentication Issues**
```bash
# Ensure Azure CLI is authenticated
az login
az account show
```

#### **2. Resource Group Conflicts**
```bash
# Check current resources
./scripts/status.sh

# Clean up if needed
./scripts/teardown.sh
```

#### **3. Soft-Delete Conflicts**
```bash
# The enterprise scripts automatically handle soft-delete conflicts
# If manual cleanup is needed:
az keyvault list-deleted
az keyvault purge --name <vault-name> --location <location>
```

#### **4. Region Availability Issues**
```bash
# The enterprise scripts automatically select optimal regions
# Check available regions:
az account list-locations --query "[].name" --output table
```

### **Debug Mode**
```bash
# Enable verbose output
set -x
./scripts/deploy.sh
set +x
```

## 📈 Success Metrics

### **Deployment Success Rate**
- **Target**: >95% successful deployments
- **Current**: ~60% (before enterprise patterns)
- **Expected**: >95% (with enterprise patterns)

### **Conflict Resolution Time**
- **Target**: <5 minutes for conflict resolution
- **Current**: Manual intervention required
- **Expected**: Automated resolution

### **Multi-Region Success Rate**
- **Target**: >90% successful multi-region deployments
- **Current**: Manual region selection
- **Expected**: Intelligent region selection

## 🔮 Future Enhancements

### **1. Advanced Monitoring**
- Real-time deployment status dashboard
- ML-powered deployment failure prediction
- Advanced cost prediction and optimization

### **2. Enhanced Automation**
- Self-healing deployments
- Dynamic resource scaling
- Comprehensive automated testing

### **3. Enterprise Integration**
- Azure DevOps integration
- GitHub Actions workflows
- Advanced security and compliance features

---

## 📚 References

- [Enterprise Deployment Architecture](../docs/ENTERPRISE_DEPLOYMENT_ARCHITECTURE.md)
- [Azure Bicep Documentation](https://docs.microsoft.com/en-us/azure/azure-resource-manager/bicep/)
- [Azure CLI Documentation](https://docs.microsoft.com/en-us/cli/azure/)