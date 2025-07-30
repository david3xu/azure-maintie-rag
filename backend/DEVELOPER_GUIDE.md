# Azure Universal RAG Backend - Developer Guide

**Comprehensive development guide for Azure Universal RAG backend**

📖 **Related Documentation:**
- ⬅️ [Backend Overview](README.md)
- 🏗️ [Backend Architecture](ARCHITECTURE_OVERVIEW.md)
- 📊 [Development Status](DEVELOPMENT_STATUS.md)
- 🌐 [Main Project](../README.md) → [Setup Guide](../SETUP.md) → [API Reference](../API_REFERENCE.md)

---

## 🎯 **Developer Guide Overview**

This guide provides **comprehensive development workflows** for the Azure Universal RAG backend system.

### **Prerequisites Validation**
Before starting development, ensure:
- ✅ **Azure infrastructure deployed** via `azd up` (see [../DEPLOYMENT.md](../DEPLOYMENT.md))
- ✅ **Python 3.11+** installed with pip
- ✅ **Azure CLI** authenticated (`az login`)
- ✅ **Azure Developer CLI** authenticated (`azd auth login`)

---

## 🚀 **Development Workflows**

### **🔧 Environment Setup Workflow**

#### **Method 1: Automated Setup (Recommended)**
```bash
# Navigate to backend directory
cd backend

# Automated environment setup
make setup
# This creates venv, installs dependencies, configures environment

# Verify setup
make health
# Expected: ✅ All Azure services connected
```

#### **Method 2: Manual Setup**
```bash
cd backend

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify Azure configuration
python -c "from services.infrastructure_service import InfrastructureService; print('✅ Setup complete')"
```

### **🧪 Development Testing Workflow**

#### **1. Azure Service Integration Tests**
```bash
# Test individual Azure services
make test-azure-services

# Test complete infrastructure health
make health

# Expected outputs:
# ✅ OpenAI: Connected (GPT-4 + text-embedding-ada-002)
# ✅ Search: Connected (Basic SKU, eastus)
# ✅ Storage: Connected (4 containers)
# ✅ Cosmos: Connected (Gremlin API, centralus)
# ✅ ML: Connected (Basic SKU, centralus)
```

#### **2. Data Pipeline Testing**
```bash
# Test data processing pipeline
make data-prep-full

# Process flow:
# Raw Data (3,859 records) → Storage → Search (327 docs) → Cosmos (207 entities)

# Validate processing results
make data-validation
```

#### **3. API Endpoint Testing**
```bash
# Start development server
make run
# Server available at: http://localhost:8000

# Test health endpoint
curl http://localhost:8000/health

# Test universal query endpoint
curl -X POST http://localhost:8000/api/v1/query/universal \
  -H "Content-Type: application/json" \
  -d '{"query": "air conditioner maintenance issues"}'

# Expected: JSON response with answer, sources, confidence scores
```

### **🔬 Advanced Development Workflows**

#### **1. GNN Model Development**
```bash
# Train new GNN model
python scripts/gnn_trainer.py --domain maintenance

# Test GNN inference
python -c "
from services.ml_service import MLService
from services.infrastructure_service import InfrastructureService
import asyncio

async def test_gnn():
    ml_service = MLService(InfrastructureService())
    result = await ml_service.predict_entity_relationships(['air conditioner', 'thermostat'])
    print(f'GNN prediction: {result}')

asyncio.run(test_gnn())
"
```

#### **2. Knowledge Graph Operations**
```bash
# Explore knowledge graph
python -c "
from services.graph_service import GraphService
from services.infrastructure_service import InfrastructureService
import asyncio

async def explore_graph():
    graph_service = GraphService(InfrastructureService())
    entities = await graph_service.get_entities_by_type('maintenance_issue')
    print(f'Maintenance entities: {len(entities)}')

asyncio.run(explore_graph())
"
```

#### **3. Custom Workflow Development**
```bash
# Create custom workflow
python scripts/workflow_analyzer.py \
  --input "data/raw/custom_data.md" \
  --output "data/outputs/results/custom_analysis.json" \
  --domain "custom"
```

---

## 🏗️ **Development Architecture**

### **Service Layer Pattern**
The backend follows a **layered service architecture**:

```
📱 API Layer (FastAPI)
    ↓
🏗️ Business Logic Layer (Services)
    ↓
🧠 Infrastructure Layer (Core Azure Clients)
    ↓
☁️ Azure Services (OpenAI, Search, Cosmos, Storage, ML)
```

### **Key Development Patterns**

#### **1. Service Dependency Injection**
```python
# services/example_service.py
class ExampleService:
    def __init__(self, infrastructure_service: InfrastructureService):
        self.infra = infrastructure_service
        self.openai_client = infrastructure_service.openai_client
        self.search_client = infrastructure_service.search_client
```

#### **2. Async/Await Pattern**
```python
# All Azure operations use async/await
async def process_query(self, query: str):
    # Azure OpenAI call
    analysis = await self.openai_client.analyze_query(query)
    
    # Azure Search call
    search_results = await self.search_client.search(analysis.keywords)
    
    # Azure Cosmos call
    graph_results = await self.cosmos_client.traverse_graph(analysis.entities)
    
    return self._combine_results(search_results, graph_results)
```

#### **3. Configuration Management**
```python
# config/settings.py provides unified configuration
from config.settings import settings

# Access Azure endpoints
openai_endpoint = settings.azure_openai_endpoint
search_endpoint = settings.azure_search_endpoint

# Environment-specific settings
if settings.environment == 'development':
    batch_size = settings.dev_batch_size
```

---

## 🧪 **Testing Strategies**

### **Unit Testing**
```bash
# Run unit tests
pytest tests/unit/ -v

# Test specific module
pytest tests/unit/test_services.py::test_data_service -v

# Run with coverage
pytest tests/unit/ --cov=services --cov-report=html
```

### **Integration Testing**
```bash
# Run Azure integration tests
pytest tests/integration/ -v

# Test Azure services
pytest tests/integration/test_azure_integration.py -v

# Test workflow integration
pytest tests/integration/test_workflow_integration.py -v
```

### **Manual Testing Scenarios**
```bash
# Test complete data lifecycle
make test-lifecycle

# Test query processing with real data
make test-query-processing

# Test GNN training pipeline
make test-gnn-training

# Test multi-hop reasoning
make test-multihop-reasoning
```

---

## 🐛 **Debugging and Troubleshooting**

### **Common Development Issues**

#### **1. Azure Authentication Issues**
```bash
# Check Azure authentication
az account show
azd auth show

# Verify managed identity
python -c "
from azure.identity import DefaultAzureCredential
try:
    credential = DefaultAzureCredential()
    print('✅ Azure authentication working')
except Exception as e:
    print(f'❌ Auth error: {e}')
"
```

#### **2. Azure Service Connection Issues**
```bash
# Test individual service connections
python -c "
from services.infrastructure_service import InfrastructureService
infra = InfrastructureService()

# Test each service
services = ['openai_client', 'search_client', 'storage_client', 'cosmos_client']
for service_name in services:
    try:
        service = getattr(infra, service_name)
        print(f'✅ {service_name}: {type(service).__name__}')
    except Exception as e:
        print(f'❌ {service_name}: {e}')
"
```

#### **3. Configuration Issues**
```bash
# Validate configuration
python -c "
from config.settings import settings
print(f'Environment: {settings.environment}')
print(f'OpenAI: {bool(settings.azure_openai_endpoint)}')
print(f'Search: {bool(settings.azure_search_endpoint)}')
print(f'Cosmos: {bool(settings.azure_cosmos_endpoint)}')
print(f'Storage: {bool(settings.azure_storage_account)}')
"
```

### **Debug Mode Development**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
make run-debug

# Check application logs
tail -f logs/last_backend_session.log
```

---

## 📊 **Development Metrics and Monitoring**

### **Performance Monitoring**
```bash
# Monitor query performance
python -c "
import time
from services.query_service import QueryService
from services.infrastructure_service import InfrastructureService
import asyncio

async def benchmark_query():
    query_service = QueryService(InfrastructureService())
    
    start_time = time.time()
    result = await query_service.process_universal_query('air conditioner problems')
    duration = time.time() - start_time
    
    print(f'Query duration: {duration:.2f}s')
    print(f'Confidence: {result.get(\"confidence_score\", 0):.2f}')
    print(f'Sources: {len(result.get(\"sources\", []))}')

asyncio.run(benchmark_query())
"
```

### **Development Health Checks**
```bash
# Comprehensive health check
make health-detailed

# Memory usage check
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Azure service latency check
make test-service-latency
```

---

## 🚀 **Production Deployment Preparation**

### **Pre-Production Checklist**
```bash
# 1. Run full test suite
make test-all

# 2. Validate production configuration
make validate-prod-config

# 3. Performance testing
make load-test

# 4. Security validation
make security-check

# 5. Documentation update
make update-docs
```

### **Production Environment Variables**
```bash
# Required for production deployment
export AZURE_ENV_NAME=production
export AZURE_LOCATION=centralus
export LOG_LEVEL=INFO
export ENABLE_MONITORING=true
export ENABLE_CACHING=true
```

---

**📖 Navigation:**
- ⬅️ [Backend Overview](README.md)
- 🏗️ [Backend Architecture](ARCHITECTURE_OVERVIEW.md)  
- 📊 [Development Status](DEVELOPMENT_STATUS.md)
- 🌐 [Main Documentation](../README.md)

---

**Developer Guide Status**: ✅ **Complete** | **Last Updated**: July 29, 2025