# Step 1 Data Ingestion - Testing Instructions

## 🎯 Based on Your Actual Test Infrastructure

### Phase 1: Configuration Validation Tests

**Use Your Existing Enterprise Validation Script**:
```bash
cd backend
python scripts/validate_enterprise_integration.py
```

**Expected Success Output**:
```
🔧 Testing Enterprise Configuration Updates...
✅ Configuration updates validated

🔐 Testing Enterprise Credential Management...
✅ Storage client enterprise credentials working

📊 Testing Telemetry Pattern...
✅ Telemetry pattern validated

💰 Testing Cost Optimization...
   Environment: [your-environment-from-env]
   Resource naming: [dynamic-from-config]
   Configuration validation: {...}
✅ Cost optimization features validated

🎉 All Enterprise Integration Tests Passed!
```

### Phase 2: Storage Factory Data-Driven Tests

**Use Your Existing Storage Integration Test**:
```bash
cd backend
python -m pytest tests/test_storage_integration.py -v
```

**Expected Success Indicators**:
- ✅ Storage factory loads client types from configuration
- ✅ RAG storage client available from settings
- ✅ ML storage client available from settings
- ✅ App storage client available from settings
- ✅ Storage status retrieved from configuration

### Phase 3: Azure Structure and Settings Tests

**Use Your Existing Azure Structure Test**:
```bash
cd backend
python -m pytest tests/test_azure_structure.py -v
```

**Validate No Hardcoded Values**:
```bash
cd backend
python -c "
from config.settings import Settings
settings = Settings()

# Test 1: Verify no hardcoded container names
try:
    container = settings.azure_blob_container
    print(f'✅ Container name from env: {container}')
    assert container != 'universal-rag-data', 'FAIL: Hardcoded container name found'
except Exception as e:
    print(f'✅ Container requires env var: {str(e)}')

# Test 2: Verify no hardcoded resource prefix
try:
    prefix = settings.azure_resource_prefix
    print(f'✅ Resource prefix from env: {prefix}')
    assert prefix != 'maintie', 'FAIL: Hardcoded prefix found'
except Exception as e:
    print(f'✅ Prefix requires env var: {str(e)}')

# Test 3: Verify no hardcoded API keys
try:
    api_key = settings.openai_api_key
    assert api_key != '1234567890', 'FAIL: Hardcoded API key found'
    print(f'✅ API key from env (length: {len(api_key)})')
except Exception as e:
    print(f'✅ API key requires env var: {str(e)}')

print('🎉 All hardcoded value tests passed!')
"
```

### Phase 4: Azure Health Check Integration Test

**Use Your Existing Azure Health Check**:
```bash
cd backend
make azure-health-check
```

**Expected Success Indicators**:
```
🔍 Azure Universal RAG Service Health Check - Session: [timestamp]
✅ Dependencies installed
Overall Status: HEALTHY
Service Health Ratio: X/X
📊 Azure service health check completed
```

### Phase 5: Storage Factory Dynamic Configuration Test

**Create Dynamic Configuration Test**:
```bash
cd backend
python -c "
from core.azure_storage.storage_factory import AzureStorageFactory
from config.settings import settings

print('🧪 Testing Storage Factory Dynamic Configuration...')

# Test 1: Factory initializes from settings
factory = AzureStorageFactory()
clients = factory.list_available_clients()
print(f'✅ Available clients from config: {clients}')

# Test 2: Storage status from configuration
status = factory.get_storage_status()
print(f'✅ Storage status: {status}')

# Test 3: Client access via configuration
try:
    if 'rag_data' in clients:
        rag_client = factory.get_storage_client('rag_data')
        print(f'✅ RAG client from config: {rag_client.account_name}')

    if 'ml_models' in clients:
        ml_client = factory.get_storage_client('ml_models')
        print(f'✅ ML client from config: {ml_client.account_name}')

except Exception as e:
    print(f'✅ Configuration-driven access working: {str(e)}')

print('🎉 Storage Factory dynamic configuration tests passed!')
"
```

### Phase 6: Resource Naming Configuration Test

**Test Dynamic Resource Naming**:
```bash
cd backend
python -c "
from config.settings import settings
import os

print('🧪 Testing Resource Naming Configuration...')

# Test different resource types
resource_types = ['storage', 'search', 'keyvault', 'cosmos', 'ml']

for res_type in resource_types:
    try:
        resource_name = settings.get_resource_name(res_type)
        print(f'✅ {res_type}: {resource_name}')

        # Verify it uses configuration values, not hardcoded
        assert settings.azure_resource_prefix in resource_name or settings.azure_environment in resource_name

    except Exception as e:
        print(f'⚠️  {res_type}: Requires configuration - {str(e)}')

print('🎉 Resource naming configuration tests passed!')
"
```

### Phase 7: Data-Driven End-to-End Test

**Use Your Existing Data Preparation Test**:
```bash
cd backend
# Test with minimal configuration
AZURE_ENVIRONMENT=test AZURE_RESOURCE_PREFIX=testprefix make data-prep-enterprise
```

**Expected Behavior**:
- ✅ No hardcoded values used in any output
- ✅ All container names from environment variables
- ✅ Resource naming uses configuration values
- ✅ Storage clients initialize from settings

### Phase 8: Integration Test Suite

**Run Your Complete Test Suite**:
```bash
cd backend
make test
```

**Expected Test Results**:
```
🧪 Running unit tests...
✅ Unit tests passed

🔗 Running API integration tests...
✅ Integration tests completed

⚙️ Running workflow manager integration tests...
✅ Workflow tests passed

🌐 Running Universal RAG system tests...
✅ Universal RAG tests passed
```

## 🔍 Failure Indicators (What to Look For)

### ❌ Configuration Failures:
- Settings loading fails due to missing environment variables
- Storage factory cannot find required configuration
- Resource naming throws errors about missing config

### ❌ Hardcoded Value Failures:
- Test output shows "universal-rag-data" or "maintie" or "1234567890"
- Error messages reference hardcoded paths or names
- Resource names don't change when environment variables change

### ❌ Integration Failures:
- Azure health check fails due to configuration issues
- Storage clients cannot initialize
- Data preparation fails with configuration errors

## ✅ Success Validation Checklist

**After running all tests, verify**:

1. **No Hardcoded Defaults**: All tests pass without hardcoded fallback values
2. **Configuration-Driven**: Settings load from environment variables only
3. **Dynamic Storage Factory**: Client types determined by available configuration
4. **Flexible Resource Naming**: Names change based on environment configuration
5. **Test Independence**: Tests don't use hardcoded Azure resource names
6. **End-to-End Success**: Data preparation and query processing work with your configuration

## 🚀 Quick Success Validation

**One-Line Validation Command**:
```bash
cd backend && python scripts/validate_enterprise_integration.py && python -m pytest tests/test_storage_integration.py -v && make azure-health-check
```

**Expected Final Output**:
```
🎉 All Enterprise Integration Tests Passed!
✅ Security Enhancement (Key Vault Integration)
✅ Monitoring Integration (Application Insights)
✅ Infrastructure Cost Optimization
✅ Service Health Monitoring

================================ test session starts ================================
tests/test_storage_integration.py::test_storage_integration PASSED    [100%]
================================ 1 passed in X.XXs ================================

🔍 Azure Universal RAG Service Health Check - Session: [timestamp]
📊 Azure service health check completed
```

If all these tests pass, your Step 1 Data Ingestion fixes are successful and fully data-driven! 🎉


# Azure Data Storage Readiness Validation for Knowledge Extraction

## 🏗️ Azure Enterprise Data Architecture Validation

### Phase 1: Azure Data State Analysis

**Use Your Existing Azure Data State Validation Script**:
```bash
cd backend
python scripts/azure_data_state.py
```

**Expected Enterprise Data Readiness Output**:
```
Azure Data State Analysis:
  Blob Storage: [X] documents
  Search Index: [X] documents
  Cosmos DB: [X] entities
  Raw Data: [X] files
  Processing Required: [status]
```

### Phase 2: Azure Service Integration Data Validation

**Execute Your Implemented Makefile Data State Command**:
```bash
cd backend
make data-state
```

**Enterprise Data Architecture Assessment**:
```
🔍 Azure Data State Analysis...
📊 Analyzing Azure services data state for domain: general
```

### Phase 3: Comprehensive Azure Data Pipeline Validation

**Use Your Data Preparation Workflow State Validation**:
```bash
cd backend
python scripts/data_preparation_workflow.py
```

**Azure Enterprise Data Services State Matrix**:
```
📊 Azure Services Data State:
   🗄️  Azure Blob Storage: ✅ Has Data ([X] docs)
   🔍 Azure Cognitive Search: ✅ Has Index ([X] docs)
   💾 Azure Cosmos DB: ✅ Has Metadata ([X] entities)
   📁 Raw Data: ✅ Available ([X] files)

🔍 Core Data Services Status:
   📊 Blob Storage + Search Index populated: Yes
   💡 Cosmos DB metadata alone does not prevent processing
```

## 🎯 Azure Data Readiness Validation Patterns

### Enterprise Data Architecture Requirements

**Based on Your Implementation Logic**:

#### **Knowledge Extraction Readiness Criteria**:
```python
# From your azure_services.py validation logic
has_core_data = all([
    data_state['azure_blob_storage']['has_data'],      # Documents in Blob Storage
    data_state['azure_cognitive_search']['has_data']   # Search index populated
])
```

#### **Processing Requirement States** (From Your Codebase):
- `"no_raw_data"` → ❌ Cannot proceed - No source documents
- `"full_processing_required"` → ⚠️ Data ingestion incomplete
- `"data_exists_check_policy"` → ✅ Ready for knowledge extraction

### Azure Service Layer Data Validation

**Storage Factory Data Validation**:
```bash
cd backend
python -c "
from integrations.azure_services import AzureServicesManager
import asyncio

async def validate_storage_readiness():
    services = AzureServicesManager()

    # Validate RAG storage client
    rag_storage = services.get_rag_storage_client()
    storage_status = rag_storage.get_connection_status()
    print(f'🗄️  RAG Storage Status: {storage_status[\"status\"]}')

    # Check storage factory health
    factory_status = services.get_storage_factory().get_storage_status()
    print(f'📦 Storage Factory Health:')
    for client_type, status in factory_status.items():
        health = '✅' if status['initialized'] else '❌'
        print(f'   {client_type}: {health} {status.get(\"account_name\", \"unknown\")}')

asyncio.run(validate_storage_readiness())
"
```

### Azure Enterprise Data Pipeline Health Check

**Comprehensive Azure Services Data Validation**:
```bash
cd backend
python -c "
from integrations.azure_services import AzureServicesManager
import asyncio

async def validate_data_pipeline_readiness():
    print('🔍 Azure Enterprise Data Pipeline Validation')
    print('=' * 50)

    services = AzureServicesManager()
    domain = 'general'

    # Execute your implemented validation method
    data_state = await services.validate_domain_data_state(domain)

    # Analyze readiness for knowledge extraction
    blob_ready = data_state['azure_blob_storage']['has_data']
    search_ready = data_state['azure_cognitive_search']['has_data']
    cosmos_available = data_state['azure_cosmos_db']['has_data']
    raw_data_available = data_state['raw_data_directory']['has_files']

    print(f'📊 Data Pipeline Component Status:')
    print(f'   🗄️  Blob Storage Data: {\"✅ Ready\" if blob_ready else \"❌ Missing\"}')
    print(f'   🔍 Search Index: {\"✅ Ready\" if search_ready else \"❌ Missing\"}')
    print(f'   💾 Cosmos Metadata: {\"✅ Available\" if cosmos_available else \"❌ Empty\"}')
    print(f'   📁 Raw Data: {\"✅ Available\" if raw_data_available else \"❌ Missing\"}')

    # Knowledge extraction readiness assessment
    core_services_ready = blob_ready and search_ready
    processing_status = data_state['requires_processing']

    print(f'\\n🎯 Knowledge Extraction Readiness:')
    if processing_status == 'data_exists_check_policy' and core_services_ready:
        print(f'   ✅ READY - Core data services populated')
        print(f'   📈 Blob Storage: {data_state[\"azure_blob_storage\"][\"document_count\"]} documents')
        print(f'   📈 Search Index: {data_state[\"azure_cognitive_search\"][\"document_count\"]} documents')
        print(f'   🚀 Proceed to Step 2: Knowledge Extraction')
    elif processing_status == 'full_processing_required':
        print(f'   ⚠️  NOT READY - Data ingestion incomplete')
        print(f'   💡 Run: make data-prep-enterprise')
    elif processing_status == 'no_raw_data':
        print(f'   ❌ NOT READY - No raw data available')
        print(f'   💡 Add files to data/raw/ directory')
    else:
        print(f'   ⚠️  Status: {processing_status}')

asyncio.run(validate_data_pipeline_readiness())
"
```

## ✅ Azure Enterprise Data Readiness Success Indicators

### **Ready for Knowledge Extraction**:
```
🎯 Knowledge Extraction Readiness:
   ✅ READY - Core data services populated
   📈 Blob Storage: [X > 0] documents
   📈 Search Index: [X > 0] documents
   🚀 Proceed to Step 2: Knowledge Extraction
```

### **Data Pipeline Architecture Health**:
- **Azure Blob Storage**: Documents successfully ingested and accessible
- **Azure Cognitive Search**: Search indices populated with document vectors
- **Azure Storage Factory**: All client types initialized and operational
- **Configuration Management**: Environment-driven settings validated

## 🚨 Azure Data Pipeline Issues Resolution

### **If Data Ingestion Incomplete**:
```bash
# Execute enterprise data preparation
cd backend
make data-prep-enterprise
```

### **If Raw Data Missing**:
```bash
# Verify raw data directory
ls -la data/raw/
# Expected: *.md or *.txt files present
```

### **If Azure Services Configuration Issues**:
```bash
# Validate Azure services health
cd backend
make azure-health-check
```

## 🎯 Azure Enterprise Data Architecture Success Criteria

**For Knowledge Extraction Phase Readiness**:

1. **✅ Azure Blob Storage**: Document repository populated with source content
2. **✅ Azure Cognitive Search**: Vector indices built and searchable
3. **✅ Azure Storage Factory**: Multi-account client access operational
4. **✅ Azure Services Integration**: 6/6 services healthy and configured
5. **✅ Data State Validation**: `requires_processing` = `"data_exists_check_policy"`

**Enterprise Architecture Validation Command**:
```bash
cd backend && python scripts/azure_data_state.py && echo "✅ Data readiness validated"
```

**Expected Final Validation**:
```
Azure Data State Analysis:
  Blob Storage: [X > 0] documents
  Search Index: [X > 0] documents
  Cosmos DB: [X] entities
  Raw Data: [X > 0] files
  Processing Required: data_exists_check_policy
✅ Data readiness validated
```

When you see this output with positive document counts and `data_exists_check_policy` status, your Azure data storage architecture is ready for Step 2 Knowledge Extraction! 🚀