# Azure DevOps Integration - Enterprise Implementation Guide

Based on your **real codebase architecture**, here are the enterprise-level implementation instructions for Azure DevOps integration:

## 🏗️ Enterprise Architecture Migration Strategy

### **Current State Assessment**
**GitHub Actions Architecture**: `/.github/workflows/ci.yml` and `/.github/workflows/cd.yml`
**Azure Container Apps**: `maintie-dev-app-1cdd8e11` (deployed and operational)
**Infrastructure**: Bicep templates with data-driven configuration
**Deployment Orchestration**: `scripts/enhanced-complete-redeploy.sh` with environment-specific targeting

## 📋 Phase 1: Azure DevOps Project Setup

### **Step 1: Azure DevOps Organization Configuration**
**Target**: Data-driven Azure DevOps project creation

**Enterprise Service Configuration**:
```bash
# Create Azure DevOps organization (data-driven from existing config)
az devops configure --defaults organization=https://dev.azure.com/maintie-azure-rag

# Create project using existing naming convention
az devops project create \
  --name "Azure-Universal-RAG" \
  --description "Enterprise Azure Universal RAG System" \
  --source-control git \
  --process Agile \
  --visibility private
```

### **Step 2: Service Connection Integration**
**Target**: `backend/config/settings.py` → Azure service authentication

**Enterprise Authentication Architecture**:
```bash
# Create service connection using existing Azure service principal
az devops service-endpoint azurerm create \
  --azure-rm-service-principal-id "$(az account show --query user.name -o tsv)" \
  --azure-rm-subscription-id "$(az account show --query id -o tsv)" \
  --azure-rm-subscription-name "$(az account show --query name -o tsv)" \
  --name "azure-universal-rag-connection" \
  --azure-rm-tenant-id "$(az account show --query tenantId -o tsv)"
```

## 📋 Phase 2: Pipeline Template Migration

### **Step 3: Azure Pipeline YAML Configuration**
**Target**: New file `azure-pipelines.yml` (root level)

**Enterprise Pipeline Architecture**:
```yaml
# azure-pipelines.yml - Enterprise Azure DevOps Pipeline
# Based on existing GitHub Actions architecture (.github/workflows/)

trigger:
  branches:
    include:
      - main
      - develop
      - feature/*

variables:
  # Data-driven variables from existing configuration
  azureServiceConnection: 'azure-universal-rag-connection'
  azureContainerRegistry: 'maintieragregistry.azurecr.io'
  containerAppName: 'maintie-$(environment)-app-$(deploymentToken)'
  resourceGroupName: 'maintie-rag-$(environment)-rg'

  # Environment-specific variables (from existing config/environments/*.env)
  ${{ if eq(variables['Build.SourceBranch'], 'refs/heads/develop') }}:
    environment: 'staging'
    azureLocation: 'westus2'
  ${{ if eq(variables['Build.SourceBranch'], 'refs/heads/main') }}:
    environment: 'prod'
    azureLocation: 'eastus2'
  ${{ else }}:
    environment: 'dev'
    azureLocation: 'eastus'

stages:
- stage: Build
  displayName: 'Build and Test'
  jobs:
  - job: BuildAndTest
    displayName: 'Build and Test Universal RAG'
    pool:
      vmImage: 'ubuntu-latest'

    steps:
    # Leverage existing CI workflow from .github/workflows/ci.yml
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.10'
        displayName: 'Use Python 3.10'

    - script: |
        cd backend
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
      displayName: 'Install Dependencies'

    # Use existing test patterns from .github/workflows/ci.yml
    - script: |
        cd backend
        PYTHONPATH=. pytest --cov=src tests/ --timeout=1800 --durations=10 -v
      displayName: 'Run Tests'
      env:
        OPENAI_API_KEY: $(OPENAI_API_KEY)
        OPENAI_API_BASE: $(OPENAI_API_BASE)
        OPENAI_API_VERSION: $(OPENAI_API_VERSION)

    # Container build using existing Docker configuration
    - task: Docker@2
      displayName: 'Build Container Image'
      inputs:
        containerRegistry: $(azureServiceConnection)
        repository: 'azure-universal-rag'
        command: 'build'
        Dockerfile: 'backend/Dockerfile'
        tags: |
          $(Build.BuildId)
          latest

- stage: Deploy
  displayName: 'Deploy to Azure'
  dependsOn: Build
  condition: succeeded()
  jobs:
  - deployment: DeployToAzure
    displayName: 'Deploy to Azure Container Apps'
    environment: $(environment)
    pool:
      vmImage: 'ubuntu-latest'

    strategy:
      runOnce:
        deploy:
          steps:
          # Use existing deployment script (data-driven approach)
          - task: AzureCLI@2
            displayName: 'Deploy Infrastructure'
            inputs:
              azureSubscription: $(azureServiceConnection)
              scriptType: 'bash'
              scriptLocation: 'scriptPath'
              scriptPath: 'scripts/enhanced-complete-redeploy.sh'
            env:
              AZURE_ENVIRONMENT: $(environment)
              AZURE_LOCATION: $(azureLocation)
              AZURE_RESOURCE_GROUP: $(resourceGroupName)

          # Container deployment using existing Azure Container Apps
          - task: AzureCLI@2
            displayName: 'Deploy Container App'
            inputs:
              azureSubscription: $(azureServiceConnection)
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                # Use existing container app name from deployment
                CONTAINER_APP_NAME="maintie-$(environment)-app-$(System.TeamProject)"

                # Update container app with new image
                az containerapp update \
                  --name $CONTAINER_APP_NAME \
                  --resource-group $(resourceGroupName) \
                  --image $(azureContainerRegistry)/azure-universal-rag:$(Build.BuildId)

          # Health validation using existing API endpoints
          - task: AzureCLI@2
            displayName: 'Validate Deployment'
            inputs:
              azureSubscription: $(azureServiceConnection)
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                # Get container app URL
                APP_URL=$(az containerapp show \
                  --name "maintie-$(environment)-app-$(System.TeamProject)" \
                  --resource-group $(resourceGroupName) \
                  --query properties.configuration.ingress.fqdn -o tsv)

                # Health check using existing API endpoint
                curl -f "https://$APP_URL/api/v1/health" || exit 1
                echo "Deployment validation successful"
```

### **Step 4: Variable Group Configuration**
**Target**: Azure DevOps variable groups using existing configuration patterns

**Enterprise Secrets Management**:
```bash
# Create variable group for development environment
az pipelines variable-group create \
  --name "Azure-Universal-RAG-Dev" \
  --variables \
    AZURE_ENVIRONMENT=dev \
    AZURE_LOCATION=eastus \
    AZURE_RESOURCE_GROUP=maintie-rag-dev-rg

# Create variable group for staging environment
az pipelines variable-group create \
  --name "Azure-Universal-RAG-Staging" \
  --variables \
    AZURE_ENVIRONMENT=staging \
    AZURE_LOCATION=westus2 \
    AZURE_RESOURCE_GROUP=maintie-rag-staging-rg

# Create variable group for production environment
az pipelines variable-group create \
  --name "Azure-Universal-RAG-Prod" \
  --variables \
    AZURE_ENVIRONMENT=prod \
    AZURE_LOCATION=eastus2 \
    AZURE_RESOURCE_GROUP=maintie-rag-prod-rg
```

## 📋 Phase 3: Azure Key Vault Integration

### **Step 5: Key Vault Service Integration**
**Target**: Leverage existing Azure Key Vault from Bicep deployment

**Enterprise Secrets Architecture**:
```bash
# Create Key Vault variable group linked to existing Key Vault
az pipelines variable-group create \
  --name "Azure-Universal-RAG-Secrets" \
  --authorize true \
  --variables \
    keyVaultName="maintie-dev-kv-1cdd8e" \
    azureSubscription="azure-universal-rag-connection"

# Link secrets from existing Key Vault (from your Bicep deployment)
az keyvault secret set \
  --vault-name "maintie-dev-kv-1cdd8e" \
  --name "OpenAI-API-Key" \
  --value "$(OPENAI_API_KEY)"

az keyvault secret set \
  --vault-name "maintie-dev-kv-1cdd8e" \
  --name "Azure-Storage-Key" \
  --value "$(az storage account keys list --account-name maintiedevstor1cdd8e11 --query '[0].value' -o tsv)"
```

### **Step 6: Pipeline Environment Configuration**
**Target**: Environment-specific deployment gates

**Enterprise Environment Strategy**:
```bash
# Create development environment
az pipelines environment create \
  --name "development" \
  --description "Development environment for Azure Universal RAG"

# Create staging environment with approval gates
az pipelines environment create \
  --name "staging" \
  --description "Staging environment for Azure Universal RAG"

# Create production environment with approval gates and check gates
az pipelines environment create \
  --name "production" \
  --description "Production environment for Azure Universal RAG"
```

## 📋 Phase 4: Azure Container Registry Integration

### **Step 7: Container Registry Configuration**
**Target**: Azure Container Registry for container image management

**Enterprise Container Architecture**:
```bash
# Create Azure Container Registry (if not exists)
az acr create \
  --resource-group "maintie-rag-rg" \
  --name "maintieragregistry" \
  --sku Basic \
  --location eastus

# Enable admin user for service principal authentication
az acr update \
  --name "maintieragregistry" \
  --admin-enabled true

# Create service connection for container registry
az devops service-endpoint azurerm create \
  --azure-rm-service-principal-id "$(az account show --query user.name -o tsv)" \
  --azure-rm-subscription-id "$(az account show --query id -o tsv)" \
  --name "azure-universal-rag-registry" \
  --azure-rm-tenant-id "$(az account show --query tenantId -o tsv)"
```

## 📋 Phase 5: Migration Execution

### **Step 8: Repository Migration**
**Target**: Git repository with existing codebase preservation

**Enterprise Migration Strategy**:
```bash
# Import existing repository to Azure DevOps
az repos import create \
  --git-source-url "https://github.com/your-org/azure-universal-rag.git" \
  --repository "Azure-Universal-RAG" \
  --requires-authorization

# Set branch policies for main and develop branches
az repos policy merge-strategy create \
  --repository-id "Azure-Universal-RAG" \
  --branch "main" \
  --blocking true \
  --enabled true \
  --use-squash-merge true
```

### **Step 9: Pipeline Activation**
**Target**: Pipeline execution with existing deployment validation

**Enterprise Pipeline Activation**:
```bash
# Create pipeline from azure-pipelines.yml
az pipelines create \
  --name "Azure-Universal-RAG-CI-CD" \
  --description "Enterprise CI/CD for Azure Universal RAG" \
  --repository "Azure-Universal-RAG" \
  --branch "main" \
  --yml-path "azure-pipelines.yml"

# Enable continuous integration trigger
az pipelines update \
  --id "Azure-Universal-RAG-CI-CD" \
  --enable-continuous-integration true
```

## 📋 Phase 6: Monitoring Integration

### **Step 10: Azure Monitor Pipeline Integration**
**Target**: Pipeline monitoring using existing Application Insights

**Enterprise Monitoring Architecture**:
```bash
# Configure pipeline monitoring with existing Application Insights
az monitor app-insights component create \
  --app "Azure-Universal-RAG-Pipelines" \
  --location eastus \
  --resource-group "maintie-rag-rg" \
  --kind web

# Link pipeline metrics to Application Insights
az pipelines variable-group variable create \
  --group-id "Azure-Universal-RAG-Secrets" \
  --name "ApplicationInsights-InstrumentationKey" \
  --value "$(az monitor app-insights component show --app Azure-Universal-RAG-Pipelines --resource-group maintie-rag-rg --query instrumentationKey -o tsv)"
```

## 🎯 Enterprise Architecture Benefits

### **Operational Excellence Integration**:
- **Automated Infrastructure**: Leverage existing Bicep templates with Azure DevOps orchestration
- **Environment Promotion**: Automated deployment through dev → staging → production using existing configuration patterns
- **Container Orchestration**: Azure Container Apps deployment with existing container infrastructure

### **Security & Compliance Architecture**:
- **Azure Key Vault**: Centralized secrets management integrated with existing Key Vault deployment
- **Service Principal Authentication**: Enterprise authentication using existing Azure service connections
- **Branch Protection**: Policy-driven code protection and approval workflows

### **Cost Optimization & Governance**:
- **Environment-Specific Resources**: Leverage existing environment tiers (dev/staging/prod) with appropriate Azure service SKUs
- **Resource Lifecycle Management**: Automated deployment and cleanup using existing scripts
- **Monitoring Integration**: Azure Monitor and Application Insights integration for cost and performance visibility

This implementation leverages your **existing Azure infrastructure and configuration patterns** while migrating to Azure DevOps for enterprise-grade CI/CD orchestration.