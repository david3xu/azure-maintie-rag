# Azure Infrastructure Plan - Universal RAG System

**Azure Developer CLI (azd) Foundation for Maintenance RAG Platform**  
**OBJECTIVE**: Automated, enterprise-grade infrastructure provisioning using Infrastructure-as-Code

## 🎯 Executive Summary

This document outlines the complete Azure infrastructure plan for the Universal RAG system, following the proven [Azure Search OpenAI Demo](https://github.com/Azure-Samples/azure-search-openai-demo) pattern using Azure Developer CLI (azd) for automated provisioning.

## 📊 Current Infrastructure State

### **Existing Assets (infrastructure/)**
```
infrastructure/
├── azure-resources-core.bicep      ✅ WORKING - Core services (Search, Storage, KeyVault, AppInsights)
├── azure-resources-cosmos.bicep    ✅ WORKING - Cosmos DB with Gremlin API
├── azure-resources-ml-simple.bicep ✅ WORKING - ML workspace configuration
└── azure-resources-ml-simple.json  ✅ WORKING - ML workspace parameters
```

### **Infrastructure Gaps**
- ❌ **No azure.yaml** - Missing azd configuration
- ❌ **No main.bicep** - No unified entry point
- ❌ **No modular structure** - Monolithic bicep files
- ❌ **No environment separation** - Manual parameter management
- ❌ **No container hosting** - Missing backend deployment target
- ❌ **No secrets automation** - Manual Key Vault configuration

## 🏗️ Target Azure Infrastructure Architecture

### **Core RAG Services** (Production-Ready)
| Service | Purpose | SKU/Tier | Multi-Environment |
|---------|---------|----------|-------------------|
| **Azure OpenAI** | Text processing, embeddings, chat completions | Standard | ✅ Dev/Staging/Prod |
| **Azure Cognitive Search** | Vector search, full-text search, indexing | Basic → Standard | ✅ Environment-specific SKUs |
| **Azure Cosmos DB** | Knowledge graphs (Gremlin API) | Serverless → Provisioned | ✅ RU scaling by env |
| **Azure Blob Storage** | Data persistence, model storage | Standard_LRS → ZRS | ✅ Replication by env |
| **Azure ML Workspace** | GNN training, model management | Basic → Standard | ✅ Compute scaling |
| **Azure Application Insights** | Monitoring, telemetry, performance | Standard | ✅ Retention by env |

### **Container & Hosting Services** (New)
| Service | Purpose | Configuration | Scaling |
|---------|---------|---------------|---------|
| **Azure Container Apps** | Backend FastAPI hosting | Linux containers | Auto-scale 1-10 instances |
| **Azure Container Registry** | Docker image storage | Basic → Standard | Environment-specific |
| **Azure Key Vault** | Secrets, certificates, keys | Standard | RBAC + Managed Identity |
| **Azure Log Analytics** | Centralized logging | Per-GB pricing | Retention by environment |

## 🚀 Azure Developer CLI (azd) Integration

### **Project Structure Enhancement**
```
/workspace/azure-maintie-rag/
├── azure.yaml                     🆕 NEW - azd configuration
├── infrastructure/                # Infrastructure-as-Code
│   ├── main.bicep                 🆕 NEW - azd entry point
│   ├── main.parameters.json       🆕 NEW - Environment parameters
│   ├── abbreviations.json         🆕 NEW - Azure naming conventions
│   │
│   ├── **Enhanced Existing Files**
│   ├── azure-resources-core.bicep ✅ ENHANCE - Add Container Apps, ACR
│   ├── azure-resources-cosmos.bicep ✅ ENHANCE - Add managed identity
│   ├── azure-resources-ml-simple.bicep ✅ ENHANCE - Add compute instances
│   │
│   └── modules/                   🆕 NEW - Modular Bicep components
│       ├── openai.bicep          🆕 NEW - OpenAI service configuration
│       ├── search.bicep          🆕 NEW - Cognitive Search with indexes
│       ├── cosmos.bicep          🆕 NEW - Cosmos DB Gremlin API
│       ├── storage.bicep         🆕 NEW - Multi-container blob storage
│       ├── ml.bicep              🆕 NEW - ML workspace with compute
│       ├── monitoring.bicep      🆕 NEW - App Insights + Log Analytics
│       ├── keyvault.bicep        🆕 NEW - Key Vault with RBAC
│       ├── containerapp.bicep    🆕 NEW - Container Apps hosting
│       ├── registry.bicep        🆕 NEW - Container Registry
│       └── networking.bicep      🆕 NEW - Virtual networks (optional)
│
├── backend/                       # Application code
└── frontend/                      # React application
```

### **Azure.yaml Configuration**
```yaml
# azure.yaml - azd project configuration
name: azure-maintie-rag
metadata:
  template: azure-search-openai-demo@main

services:
  backend:
    project: ./backend
    language: py
    host: containerapp
    
  frontend:
    project: ./frontend
    language: js
    host: staticwebapp

infra:
  provider: bicep
  path: ./infrastructure

hooks:
  preprovision:
    shell: sh
    run: |
      echo "🏗️ Preparing Azure Universal RAG infrastructure..."
      echo "Environment: ${AZURE_ENV_NAME}"
      
  postprovision:
    shell: sh
    run: |
      echo "🔧 Configuring deployed services..."
      # Update backend configuration with deployed endpoints
      ./scripts/update-config-from-deployment.sh
      
  prepackage:
    shell: sh
    run: |
      echo "📦 Building backend container..."
      cd backend && docker build -t azure-maintie-rag-backend .
      
  postdeploy:
    shell: sh
    run: |
      echo "✅ Deployment complete!"
      echo "Backend URL: ${SERVICE_BACKEND_URI}"
      echo "Frontend URL: ${SERVICE_FRONTEND_URI}"
```

## 🌍 Multi-Environment Strategy

### **Environment Configuration Matrix**
| Environment | Purpose | Azure Location | Resource SKUs | Data Retention |
|-------------|---------|----------------|---------------|----------------|
| **development** | Local dev, testing | East US | Basic/Small | 7 days |
| **staging** | Integration testing | West US 2 | Standard/Medium | 30 days |
| **production** | Live system | Central US | Premium/Large | 90 days |

### **Environment-Specific Parameters**
```json
// main.parameters.json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "environmentName": {
      "value": "${AZURE_ENV_NAME}"
    },
    "location": {
      "value": "${AZURE_LOCATION}"
    },
    "principalId": {
      "value": "${AZURE_PRINCIPAL_ID}"
    },
    "resourceGroupName": {
      "value": "rg-maintie-rag-${AZURE_ENV_NAME}"
    }
  }
}
```

### **Environment Setup Commands**
```bash
# Development Environment
azd env new development
azd env set AZURE_LOCATION eastus
azd env set AZURE_RESOURCE_GROUP rg-maintie-rag-dev
azd env set OPENAI_MODEL_DEPLOYMENT gpt-4
azd env set SEARCH_INDEX_NAME maintie-dev-index

# Staging Environment
azd env new staging
azd env set AZURE_LOCATION westus2
azd env set AZURE_RESOURCE_GROUP rg-maintie-rag-staging
azd env set OPENAI_MODEL_DEPLOYMENT gpt-4-32k
azd env set SEARCH_INDEX_NAME maintie-staging-index

# Production Environment
azd env new production
azd env set AZURE_LOCATION centralus
azd env set AZURE_RESOURCE_GROUP rg-maintie-rag-prod
azd env set OPENAI_MODEL_DEPLOYMENT gpt-4-turbo
azd env set SEARCH_INDEX_NAME maintie-prod-index
```

## 🔧 Service Configuration Details

### **Azure OpenAI Service**
```bicep
// modules/openai.bicep
param location string
param environmentName string

resource openaiAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: 'openai-${environmentName}-${uniqueString(resourceGroup().id)}'
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: 'maintie-rag-${environmentName}'
    publicNetworkAccess: 'Enabled'
  }
}

// Model deployments
resource gpt4Deployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openaiAccount
  name: 'gpt-4'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4'
      version: '0613'
    }
    scaleSettings: {
      scaleType: 'Standard'
      capacity: environmentName == 'production' ? 20 : 10
    }
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openaiAccount
  name: 'text-embedding-ada-002'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-ada-002'
      version: '2'
    }
    scaleSettings: {
      scaleType: 'Standard'
      capacity: environmentName == 'production' ? 30 : 15
    }
  }
}
```

### **Azure Cognitive Search**
```bicep
// modules/search.bicep
param location string
param environmentName string

var searchSkuMap = {
  development: 'basic'
  staging: 'standard'
  production: 'standard'
}

resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: 'search-${environmentName}-${uniqueString(resourceGroup().id)}'
  location: location
  sku: {
    name: searchSkuMap[environmentName]
  }
  properties: {
    replicaCount: environmentName == 'production' ? 2 : 1
    partitionCount: environmentName == 'production' ? 2 : 1
    publicNetworkAccess: 'enabled'
    semanticSearch: 'standard'
  }
}

// Semantic search configuration
resource semanticConfig 'Microsoft.Search/searchServices/semanticConfigurations@2023-11-01' = {
  parent: searchService
  name: 'maintie-semantic-config'
  properties: {
    prioritizedFields: {
      titleField: {
        fieldName: 'title'
      }
      contentFields: [
        {
          fieldName: 'content'
        }
      ]
      keywordFields: [
        {
          fieldName: 'maintenance_type'
        }
      ]
    }
  }
}
```

### **Azure Container Apps**
```bicep
// modules/containerapp.bicep
param location string
param environmentName string
param containerRegistryName string
param openaiEndpoint string
param searchEndpoint string

resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: 'cae-${environmentName}-${uniqueString(resourceGroup().id)}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource backendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'ca-backend-${environmentName}'
  location: location
  properties: {
    managedEnvironmentId: containerAppEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
      }
      registries: [
        {
          server: '${containerRegistryName}.azurecr.io'
          identity: userAssignedIdentity.id
        }
      ]
    }
    template: {
      containers: [
        {
          image: '${containerRegistryName}.azurecr.io/azure-maintie-rag-backend:latest'
          name: 'backend'
          env: [
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: openaiEndpoint
            }
            {
              name: 'AZURE_SEARCH_ENDPOINT'
              value: searchEndpoint
            }
            {
              name: 'ENVIRONMENT'
              value: environmentName
            }
          ]
          resources: {
            cpu: environmentName == 'production' ? 1.0 : 0.5
            memory: environmentName == 'production' ? '2Gi' : '1Gi'
          }
        }
      ]
      scale: {
        minReplicas: environmentName == 'production' ? 2 : 1
        maxReplicas: environmentName == 'production' ? 10 : 3
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}
```

## 🔐 Security & Identity Management

### **Managed Identity Strategy**
```bicep
// User-assigned managed identity for all services
resource userAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-maintie-rag-${environmentName}'
  location: location
}

// Role assignments for each service
resource openaiRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: openaiAccount
  name: guid(openaiAccount.id, userAssignedIdentity.id, 'Cognitive Services OpenAI User')
  properties: {
    principalId: userAssignedIdentity.properties.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  }
}
```

### **Key Vault Configuration**
```bicep
// modules/keyvault.bicep
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${environmentName}-${uniqueString(resourceGroup().id)}'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    publicNetworkAccess: 'Enabled'
  }
}

// Store service endpoints as secrets
resource openaiEndpointSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'openai-endpoint'
  properties: {
    value: openaiAccount.properties.endpoint
  }
}
```

## 📋 Implementation Roadmap

### **Phase 1: azd Foundation Setup** (2-3 days)
1. **Create azure.yaml** - Configure azd project structure
2. **Create main.bicep** - Unified infrastructure entry point  
3. **Enhance existing Bicep** - Add Container Apps, ACR, enhanced networking
4. **Setup environments** - Dev, staging, production configurations
5. **Test azd workflow** - Verify `azd up` provisions all services

**Deliverables:**
- ✅ `azd up` creates complete infrastructure
- ✅ Multi-environment support working
- ✅ Backend deploys to Container Apps
- ✅ All services use managed identity

### **Phase 2: Modular Bicep Architecture** (1-2 days)
1. **Extract service modules** - Break monolithic bicep into focused modules
2. **Implement naming conventions** - Consistent Azure resource naming
3. **Add environment scaling** - SKU/size differences by environment
4. **Setup monitoring** - Application Insights, Log Analytics integration

**Deliverables:**
- ✅ Clean modular Bicep structure
- ✅ Environment-specific resource sizing
- ✅ Comprehensive monitoring setup
- ✅ Automated secret management

### **Phase 3: Backend Integration** (1 day)
1. **Update backend configuration** - Connect to azd-managed services
2. **Implement managed identity** - Remove hardcoded secrets
3. **Container optimization** - Multi-stage Docker builds
4. **Health check integration** - Container Apps health probes

**Deliverables:**
- ✅ Backend uses azd-provisioned services
- ✅ Zero manual configuration required
- ✅ Production-ready container deployment
- ✅ Health monitoring integrated

### **Phase 4: CI/CD Integration** (1 day)
1. **GitHub Actions** - Automated azd deployment
2. **Environment promotion** - Dev → Staging → Production
3. **Infrastructure drift detection** - Bicep validation
4. **Rollback procedures** - Safe deployment practices

**Deliverables:**
- ✅ Automated deployment pipeline
- ✅ Infrastructure as Code enforcement
- ✅ Environment consistency validation
- ✅ Production deployment safety

## ✅ Success Criteria & Validation

### **Infrastructure Automation**
- ✅ `azd up` provisions 8+ Azure services in < 15 minutes
- ✅ Zero manual Azure portal configuration required
- ✅ Environment creation is 100% reproducible
- ✅ All secrets managed through Key Vault + Managed Identity

### **Development Experience**
- ✅ Local development connects to real Azure services
- ✅ Environment switching with `azd env select <env>`
- ✅ Backend configuration auto-updates from infrastructure
- ✅ One-command deployment for any environment

### **Production Readiness**
- ✅ Multi-region deployment capability
- ✅ Auto-scaling based on load
- ✅ Comprehensive monitoring and alerting
- ✅ Security best practices (RBAC, managed identity, Key Vault)

## 🔄 Integration with Backend Refactoring

### **Dependencies**
**Backend refactoring depends on this infrastructure foundation:**
1. **All services/** will use azd-provisioned Azure clients
2. **Configuration management** driven by azd environment variables
3. **Deployment targets** provided by Container Apps
4. **Secrets management** through Key Vault (no .env files)

### **Timeline Coordination**
1. **Infrastructure Phase 1** → **Backend Refactoring Phase 1** (parallel)
2. **Infrastructure Phase 2-3** → **Backend Refactoring Phase 2-4** 
3. **Infrastructure Phase 4** → **End-to-end validation**

## 📊 Cost Optimization

### **Development Environment** (~$200-300/month)
- OpenAI: Basic deployment (10 TPM)
- Search: Basic SKU (1 unit)
- Cosmos: Serverless (pay per RU)
- Container Apps: Consumption tier
- Storage: LRS with Cool tier

### **Production Environment** (~$800-1200/month)
- OpenAI: Standard deployment (50 TPM)
- Search: Standard SKU (2 replicas)
- Cosmos: Provisioned throughput
- Container Apps: Dedicated tier with auto-scaling
- Storage: ZRS with Hot tier

### **Cost Monitoring**
```bicep
// Cost management alerts
resource costAlert 'Microsoft.Consumption/budgets@2023-05-01' = {
  name: 'budget-${environmentName}'
  properties: {
    category: 'Cost'
    amount: environmentName == 'production' ? 1500 : 500
    timeGrain: 'Monthly'
    notifications: {
      actual: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 80
        contactEmails: ['admin@maintie-rag.com']
      }
    }
  }
}
```

---

## 🎯 Next Steps

1. **Review & Approve** this infrastructure plan
2. **Begin Phase 1** - Azure.yaml + main.bicep creation
3. **Parallel development** - Continue backend refactoring on azd foundation
4. **Integration testing** - Validate end-to-end azd workflow
5. **Production deployment** - Multi-environment validation

**This infrastructure foundation enables the backend refactoring to succeed with enterprise-grade Azure automation.**