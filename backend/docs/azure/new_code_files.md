Based on your **real codebase analysis**, here are the essential NEW files required for Azure Universal RAG migration:

## **Critical Azure Integration Files**

### **1. `backend/config/azure_settings.py`**
**Purpose**: Azure service configuration management following existing `settings.py` pattern
**Dependencies**: Extends current `backend/config/settings.py` structure
```python
# Based on existing settings.py pydantic pattern
# Manages Azure service endpoints, keys, resource names
# Environment-driven configuration (no hardcoded values)
# Validation methods for Azure service connectivity
```

### **2. `backend/azure/storage_client.py`**
**Purpose**: Azure Blob Storage integration for Universal RAG data
**Dependencies**: Uses existing `azure_openai.py` error handling patterns
```python
# File operations: upload/download/list raw text files
# Replaces local backend/data/raw/ with Azure Blob Storage
# Follows existing AzureOpenAIClient initialization pattern
# Connection status validation matching azure_openai.py format
```

### **3. `backend/azure/search_client.py`**
**Purpose**: Azure Cognitive Search for vector indices
**Dependencies**: Replaces current `universal_vector_search.py` FAISS functionality
```python
# Vector search operations maintaining existing search interface
# Index creation/management for Universal RAG domains
# Document upload/search matching current FAISS capabilities
# Service health monitoring following azure_openai.py pattern
```

### **4. `backend/azure/cosmos_client.py`**
**Purpose**: Azure Cosmos DB for knowledge graph storage
**Dependencies**: Enhances current NetworkX graph processing
```python
# Graph operations: store/query entities and relations
# Gremlin API integration preserving NetworkX functionality
# Multi-domain graph support maintaining current capabilities
# Connection validation matching existing service patterns
```

### **5. `backend/integrations/azure_services.py`**
**Purpose**: Unified Azure services manager
**Dependencies**: Coordinates all Azure clients using existing integration patterns
```python
# Service health checking across all Azure components
# Data migration orchestration between local and Azure storage
# Configuration validation for complete Azure setup
# Error handling consistent with current integrations/
```

### **6. `infrastructure/azure-resources.bicep`**
**Purpose**: Infrastructure as Code for complete Azure resource provisioning
**Dependencies**: Creates all required Azure services automatically
```bicep
// Storage Account for Universal RAG data
// Cognitive Search service for vector indices
// Cosmos DB account for knowledge graphs
// Container Apps for deployment
// Key Vault for secrets management
```

### **7. `backend/utilities/azure_migrator.py`**
**Purpose**: Data migration from local to Azure services
**Dependencies**: Uses existing `config_loader.py` patterns for settings management
```python
# Migrate backend/data/raw/ to Azure Blob Storage
# Convert FAISS indices to Azure Cognitive Search format
# Transfer NetworkX graphs to Cosmos DB Gremlin format
# Validation and rollback capabilities
```

### **8. `.github/workflows/azure-deploy.yml`**
**Purpose**: Automated Azure deployment pipeline
**Dependencies**: Extends existing `.github/workflows/ci.yml` structure
```yaml
# Infrastructure provisioning via Bicep templates
# Container deployment to Azure Container Apps
# Environment-specific deployments (dev/staging/prod)
# Health checking and rollback procedures
```

## **Supporting Automation Files**

### **9. `backend/scripts/azure-provision.py`**
**Purpose**: Python-based Azure resource provisioning
**Dependencies**: Uses `azure_settings.py` configuration
```python
# Alternative to Bicep for Python-based infrastructure
# Resource creation with validation and error handling
# Integration with existing scripts/ directory patterns
```

### **10. `backend/scripts/azure-validate.py`**
**Purpose**: Azure setup validation and testing
**Dependencies**: Uses all Azure service clients for comprehensive testing
```python
# Connection testing for all Azure services
# Configuration validation across environments
# Integration testing for Universal RAG on Azure
```

## **Implementation Priority Order**

### **Phase 1: Core Azure Services (Week 1)**
1. `backend/config/azure_settings.py` - Configuration foundation
2. `backend/azure/storage_client.py` - Data storage capability
3. `backend/azure/search_client.py` - Vector search replacement

### **Phase 2: Advanced Services (Week 2)**
4. `backend/azure/cosmos_client.py` - Knowledge graph storage
5. `backend/integrations/azure_services.py` - Service coordination
6. `backend/utilities/azure_migrator.py` - Data migration

### **Phase 3: Infrastructure Automation (Week 3)**
7. `infrastructure/azure-resources.bicep` - Resource provisioning
8. `.github/workflows/azure-deploy.yml` - Deployment automation
9. `backend/scripts/azure-validate.py` - Validation framework

## **Quick-Start Implementation Guide**

### **Step 1: Configuration Foundation**
```bash
# Create Azure configuration following existing patterns
touch backend/config/azure_settings.py
# Implement using backend/config/settings.py structure
# Test: python -c "from config.azure_settings import azure_settings; print('✅')"
```

### **Step 2: Core Service Clients**
```bash
# Create Azure service directory and clients
mkdir -p backend/azure
touch backend/azure/{__init__.py,storage_client.py,search_client.py}
# Implement using backend/integrations/azure_openai.py patterns
# Test: python -c "from azure.storage_client import AzureStorageClient; print('✅')"
```

### **Step 3: Integration Validation**
```bash
# Create service manager
touch backend/integrations/azure_services.py
# Test unified service health: python scripts/azure-validate.py
# Verify: All services report "healthy" status
```

### **Step 4: Infrastructure Automation**
```bash
# Create infrastructure provisioning
mkdir -p infrastructure
touch infrastructure/azure-resources.bicep
# Deploy: az deployment group create --template-file azure-resources.bicep
# Validate: All Azure resources created successfully
```

## **File Dependencies Map**

```
azure_settings.py ← (foundation for all Azure files)
    ↓
storage_client.py, search_client.py, cosmos_client.py ← (use azure_settings)
    ↓
azure_services.py ← (coordinates all clients)
    ↓
azure_migrator.py ← (uses all services for migration)
    ↓
azure-resources.bicep ← (provisions infrastructure)
    ↓
azure-deploy.yml ← (automates deployment)
```

**Total New Files**: 10 critical files
**Implementation Time**: 3 weeks
**Risk Level**: Low (follows existing codebase patterns)
**Result**: Production-ready Azure Universal RAG with zero functionality loss

Each file leverages existing codebase patterns, ensuring consistency and maintainability while enabling complete Azure integration.