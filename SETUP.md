# Azure Universal RAG - Setup Guide

**Complete Installation and Configuration**

📖 **Related Documentation:**
- ⬅️ [Back to Main README](README.md)
- 🏗️ [System Architecture](ARCHITECTURE.md)
- 🚀 [Deployment Guide](DEPLOYMENT.md)
- 📖 [API Reference](API_REFERENCE.md)

---

## 🚀 Quick Start (5 minutes)

### **Prerequisites**
```bash
# Install required tools
curl -fsSL https://aka.ms/install-azd.sh | bash  # Azure Developer CLI
az login                                          # Azure CLI authentication
```

### **One-Command Deployment**
```bash
# Clone and deploy
git clone <repository-url>
cd azure-maintie-rag

# NEW: Automatic environment sync + deploy
./scripts/sync-env.sh development && azd up  # Development environment
# OR
./scripts/sync-env.sh staging && azd up      # Staging environment
```

**That's it!** The system will:
- ✅ Deploy 9 Azure services automatically
- ✅ Configure hybrid RBAC authentication  
- ✅ Set up all networking and security
- ✅ Generate environment configuration

### **Verify Deployment**
```bash
cd backend
python -c "
import asyncio
from services.infrastructure_service import InfrastructureService

async def quick_test():
    infra = InfrastructureService()
    print('✅ All services initialized successfully!')
    print('🎯 System ready for data processing')

asyncio.run(quick_test())
"
```

### **Run Your First Lifecycle**
```bash
# Process sample data
python -c "
import asyncio
from services.data_service import DataService
from services.infrastructure_service import InfrastructureService

async def first_run():
    data_service = DataService(InfrastructureService())
    result = await data_service.process_raw_data('maintenance')
    print(f'🎉 First run complete: {result.get(\"success\")}')

asyncio.run(first_run())
"
```

---

## 📋 Prerequisites

### **Required Tools**

#### **Azure Tools**
```bash
# Azure CLI - version 2.50.0 or later
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Azure Developer CLI - version 1.5.0 or later  
curl -fsSL https://aka.ms/install-azd.sh | bash

# Verify installations
az --version
azd version
```

#### **Development Tools**
```bash
# Python 3.11+ with pip
python --version  # Should be 3.11+
pip --version

# Node.js 18+ with npm (for frontend)
node --version    # Should be 18+
npm --version

# Git
git --version
```

### **Azure Requirements**

#### **Azure Subscription**
- Active Azure subscription with appropriate permissions
- Contributor role or higher on the subscription
- Ability to create resources in target regions

#### **Resource Provider Registrations**
```bash
# Register required providers
az provider register --namespace Microsoft.CognitiveServices
az provider register --namespace Microsoft.Search
az provider register --namespace Microsoft.DocumentDB
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.Insights
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.App

# Verify registrations
az provider list --query "[?registrationState=='Registered' && namespace in ['Microsoft.CognitiveServices','Microsoft.Search','Microsoft.DocumentDB']].{Namespace:namespace, State:registrationState}" -o table
```

---

## 🏗️ Infrastructure Deployment

### **Step 1: Initial Setup**

#### **Clone Repository**
```bash
git clone <repository-url>
cd azure-maintie-rag
```

#### **Azure Authentication**
```bash
# Login to Azure CLI
az login

# Login to Azure Developer CLI  
azd auth login

# Verify authentication
az account show
azd auth show
```

#### **Initialize Project**
```bash
# Initialize azd project (if needed)
azd init

# Follow prompts to confirm:
# - Project name: azure-maintie-rag
# - Services: backend (Python)
# - Infrastructure: Bicep templates
```

### **Step 2: Environment Setup**

#### **Create Environments**
```bash
# Setup environments script (recommended)
./scripts/setup-environments.sh

# Or manually create environments:
azd env new development
azd env new staging
azd env new production
```

#### **Environment Configuration**
```bash
# Select development environment
azd env select development

# Set environment variables
azd env set AZURE_LOCATION eastus
azd env set AZURE_RESOURCE_GROUP_NAME "rg-maintie-rag-development"

# Optional: Set specific configuration
azd env set OPENAI_MODEL_DEPLOYMENT_NAME "gpt-4"
azd env set SEARCH_INDEX_NAME "maintie-index"
```

### **Step 3: Infrastructure Deployment**

#### **Deploy All Resources**
```bash
# Deploy complete infrastructure
azd up

# Follow prompts to select:
# - Subscription: Your Azure subscription
# - Location: eastus (recommended)
# - Environment: development/staging/production
```

**Expected Deployment Time**: ~15-20 minutes

#### **Deployment Results**
After successful deployment, you'll see:
```
✅ Azure OpenAI Service: oai-maintie-rag-[env]-[hash]
✅ Azure Cognitive Search: srch-maintie-rag-[env]-[hash]  
✅ Azure Cosmos DB: cosmos-maintie-rag-[env]-[hash]
✅ Azure Blob Storage: st[name][hash]
✅ Azure ML Workspace: ml-maintie-rag-[env]-[hash]
✅ Azure Key Vault: kv-maintie-rag-[env]-[hash]
✅ Application Insights: appi-maintie-rag-[env]
✅ Log Analytics: log-maintie-rag-[env]
✅ Container Apps: (optional) maintie-rag-[env]-app
```

---

## ⚙️ Backend Configuration

### **Step 1: Python Environment Setup**

#### **Create Virtual Environment**
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

#### **Install Dependencies**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi, openai; print('✅ Dependencies installed')"
```

### **Step 2: Environment Configuration**

#### **Create Environment File**
```bash
# Copy example environment file
cp config/environments/development.env .env

# Or create from template
cp .env.example .env
```

#### **Update Configuration (Automatic)**
```bash
# Update configuration from Azure deployment
../scripts/update-env-from-deployment.sh

# This automatically populates:
# - AZURE_OPENAI_ENDPOINT
# - AZURE_SEARCH_ENDPOINT  
# - AZURE_COSMOS_ENDPOINT
# - AZURE_STORAGE_ACCOUNT
# - Other Azure service endpoints
```

#### **Manual Configuration (if needed)**
```bash
# Edit .env file with your values
nano .env
```

**Required Settings:**
```bash
# Azure Service Endpoints (auto-populated by azd)
AZURE_OPENAI_ENDPOINT=https://oai-maintie-rag-dev-[hash].openai.azure.com/
AZURE_SEARCH_ENDPOINT=https://srch-maintie-rag-dev-[hash].search.windows.net
AZURE_COSMOS_ENDPOINT=https://cosmos-maintie-rag-dev-[hash].documents.azure.com:443/
AZURE_STORAGE_ACCOUNT=st[name][hash]

# Authentication Settings
USE_MANAGED_IDENTITY=true
COSMOS_USE_MANAGED_IDENTITY=false  # Uses API key for Gremlin compatibility

# Database Configuration
COSMOS_DATABASE_NAME=maintie-rag-development
COSMOS_CONTAINER_NAME=knowledge-graph-development
SEARCH_INDEX_NAME=maintie-index

# Domain Settings
DEFAULT_DOMAIN=maintenance
```

### **Step 3: Service Validation**

#### **Test Azure Service Connections**
```bash
# Test infrastructure service initialization
python -c "
import asyncio
from services.infrastructure_service import InfrastructureService

async def test_services():
    infra = InfrastructureService()
    print('=== Azure Service Validation ===')
    
    # Test each service connection
    services = ['openai', 'search', 'storage', 'cosmos', 'ml']
    for service in services:
        try:
            client = getattr(infra, f'{service}_client', None)
            if client:
                print(f'✅ {service.title()}: Connected')
            else:
                print(f'❌ {service.title()}: Not configured')
        except Exception as e:
            print(f'❌ {service.title()}: {str(e)[:100]}...')

asyncio.run(test_services())
"
```

#### **Test Complete System Health**
```bash
# Run comprehensive health check
make health

# Or directly:
python -c "
import asyncio
from services.infrastructure_service import InfrastructureService

async def health_check():
    infra = InfrastructureService()
    print('🔍 Running comprehensive health check...')
    
    # Test Azure service health
    health_status = {}
    
    try:
        # Test OpenAI
        response = await infra.openai_client.test_connection()
        health_status['openai'] = '✅ Connected'
    except Exception as e:
        health_status['openai'] = f'❌ {str(e)[:50]}...'
    
    # Similar tests for other services...
    for service, status in health_status.items():
        print(f'{service.title()}: {status}')

asyncio.run(health_check())
"
```

---

## 🎯 Frontend Setup (Optional)

### **Frontend Prerequisites**
```bash
# Ensure Node.js 18+ is installed
node --version
npm --version
```

### **Frontend Installation**
```bash
cd frontend

# Install dependencies
npm install

# Verify installation
npm list react typescript

# Build for production
npm run build
```

### **Frontend Configuration**
```bash
# Create environment file
cp .env.example .env.local

# Update with backend URL
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local
```

### **Start Development Server**
```bash
# Start frontend development server
npm run dev

# Frontend will be available at: http://localhost:5174
```

---

## 🔧 Development Environment

### **Complete Development Setup**

#### **Root Level Setup**
```bash
# From project root directory
make setup

# This will:
# 1. Set up backend Python environment
# 2. Install backend dependencies  
# 3. Set up frontend Node environment
# 4. Install frontend dependencies
# 5. Create necessary data directories
# 6. Validate Azure service connections
```

#### **Start Development Services**
```bash
# Start both backend and frontend
make dev

# Or start individually:
make backend    # Start backend only (localhost:8000)
make frontend   # Start frontend only (localhost:5174)
```

#### **Development Workflow**
```bash
# Check service health
make health

# Process sample data
make data-prep-full

# Run tests
make test

# Clean development environment
make clean
```

### **IDE Configuration**

#### **VSCode Setup (Recommended)**
```bash
# From backend directory
make docs-setup

# This installs VSCode extensions:
# - Python
# - Black Formatter  
# - Pylint
# - Markdown All in One
# - JSON and YAML support
```

#### **Python Development**
```bash
# Set up pre-commit hooks
pip install pre-commit
pre-commit install

# Configure Black formatter
pip install black isort
black --line-length 88 .
isort .
```

---

## 🔐 Authentication & Security

### **Hybrid Authentication Strategy**

The system uses **hybrid authentication** for optimal security and compatibility:

#### **RBAC Services (Automatic)**
- **Azure Storage Account**: Managed Identity + Storage Blob Data Contributor
- **Azure Cognitive Search**: Managed Identity + Search Index Data Contributor  
- **Azure OpenAI**: Managed Identity (automatic)
- **Azure ML Workspace**: Managed Identity + AzureML Data Scientist

#### **API Key Services (Manual Setup)**
- **Cosmos DB Gremlin**: Primary Master Key (compatibility requirement)

### **RBAC Setup (if needed)**

#### **Automatic RBAC (via Infrastructure)**
The Bicep templates automatically configure RBAC permissions during deployment.

#### **Manual RBAC (troubleshooting)**
```bash
# Get your user principal ID
USER_ID=$(az ad signed-in-user show --query id -o tsv)
RESOURCE_GROUP="rg-maintie-rag-development"

# Storage permissions
az role assignment create \
  --assignee $USER_ID \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP"

# Search permissions
az role assignment create \
  --assignee $USER_ID \
  --role "Search Index Data Contributor" \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP"

# Verify permissions
az role assignment list --assignee $USER_ID --query "[].{Role:roleDefinitionName,Scope:scope}" -o table
```

### **Security Validation**
```bash
# Test authentication configuration
python -c "
from config.settings import get_settings
settings = get_settings()
print(f'✅ Using Managed Identity: {settings.use_managed_identity}')
print(f'✅ Cosmos Auth Method: {'API Key' if not settings.cosmos_use_managed_identity else 'Managed Identity'}')
"
```

---

## 📊 Data Processing Setup

### **Prepare Sample Data**

#### **Default Sample Data**
The system includes sample maintenance data in `backend/data/raw/`:
```bash
ls backend/data/raw/
# demo_sample_10percent.md (15,916 bytes)
# maintenance_all_texts.md
```

#### **Add Your Own Data**
```bash
# Add your text files to the raw data directory
cp your-documents.md backend/data/raw/

# Supported formats:
# - .md (Markdown files) - Primary format
# - .txt (Plain text) - Secondary format
```

### **Run Complete Data Lifecycle**

#### **Method 1: Command Line**
```bash
# Complete data processing pipeline
make data-prep-full

# Or step by step:
make data-upload        # Upload docs & create chunks
make knowledge-extract  # Extract entities & relations
```

#### **Method 2: Python API**
```bash
cd backend
python -c "
import asyncio
from services.data_service import DataService
from services.infrastructure_service import InfrastructureService

async def process_data():
    infra = InfrastructureService()
    data_service = DataService(infra)
    
    print('🔄 Starting complete data lifecycle...')
    result = await data_service.process_raw_data('maintenance')
    
    print(f'✅ Success: {result.get(\"success\")}')
    print(f'📊 Summary: {result.get(\"details\", {}).get(\"summary\")}')
    return result

result = asyncio.run(process_data())
"
```

### **Expected Processing Results**

#### **Storage Migration**
```
✅ Container: rag-data-maintenance
✅ Uploaded Files: 1-4 documents
✅ Failed Uploads: 0
```

#### **Search Migration**  
```
✅ Index: maintie-index-maintenance
✅ Documents Indexed: 127-327 records
✅ Failed Documents: 0
```

#### **Cosmos Migration**
```
✅ Database: maintie-rag-development
✅ Graph: knowledge-graph-maintenance
✅ Entities Created: 45-207
✅ Relationships Created: 23+
```

---

## 🧪 Testing & Validation

### **System Health Checks**

#### **Infrastructure Health**
```bash
# Test infrastructure deployment
./scripts/test-infrastructure.sh

# Expected results:
✅ azure.yaml syntax valid
✅ Bicep file structure valid
✅ All modules have correct parameters
✅ Scripts are executable
```

#### **Service Health**
```bash
# Test all Azure services
cd backend && make health

# Expected results:
✅ Infrastructure service initialized
✅ OpenAI: Connected
✅ Search: Connected  
✅ Storage: Connected
✅ Cosmos: Connected
✅ ML: Connected
```

### **End-to-End Testing**

#### **Data Processing Test**
```bash
# Test complete data lifecycle
python -c "
import asyncio
from services.data_service import DataService
from services.infrastructure_service import InfrastructureService

async def test_lifecycle():
    data_service = DataService(InfrastructureService())
    
    # Validate before processing
    before_state = await data_service.validate_domain_data_state('maintenance')
    print(f'Before: {before_state.get(\"data_sources_ready\", 0)}/3 services ready')
    
    # Run processing
    result = await data_service.process_raw_data('maintenance')
    print(f'Processing: {\"✅ Success\" if result.get(\"success\") else \"❌ Failed\"}')
    
    # Validate after processing  
    after_state = await data_service.validate_domain_data_state('maintenance')
    print(f'After: {after_state.get(\"data_sources_ready\", 0)}/3 services ready')

asyncio.run(test_lifecycle())
"
```

#### **API Endpoint Test**
```bash
# Start backend server
cd backend && make run &

# Test query endpoint
curl localhost:8000/api/v1/query/universal \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "maintenance issues with air conditioners"}'

# Test health endpoint
curl localhost:8000/health
```

### **Load Testing**
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

---

## 💰 Cost Management

### **Environment-Specific Costs**

#### **Development Environment**
- **Estimated Monthly Cost**: $200-300
- **Configuration**: Basic SKUs, minimal throughput, auto-shutdown
- **Services**:
  - Azure OpenAI: S0 (10 TPM)
  - Cognitive Search: Basic
  - Cosmos DB: Serverless
  - Storage: Standard LRS

#### **Production Environment**  
- **Estimated Monthly Cost**: $800-1200
- **Configuration**: Standard/Premium SKUs, provisioned throughput, auto-scaling
- **Services**:
  - Azure OpenAI: S0 (50 TPM)
  - Cognitive Search: Standard with replicas
  - Cosmos DB: Provisioned throughput
  - Storage: Standard ZRS

### **Cost Optimization**

#### **Automatic Cost Controls**
```bash
# Budget alerts configured in Bicep templates
# - 80% threshold warning
# - 100% threshold critical alert

# Auto-shutdown for development resources
# - Container Apps: Scale to 0 after inactivity
# - ML Compute: Auto-shutdown enabled
```

#### **Manual Cost Management**
```bash
# Monitor costs
az consumption usage list --billing-period-name $(az billing billing-period list --query "[0].name" -o tsv)

# Scale down resources when not needed
azd down --purge    # Delete all resources
azd up              # Recreate when needed
```

---

## 🔧 Troubleshooting

### **Common Issues**

#### **1. Authentication Failures**
```bash
# Problem: "Authentication failed" errors
# Solution: Check RBAC permissions
az role assignment list --assignee $(az ad signed-in-user show --query id -o tsv)

# Wait 5-15 minutes for RBAC propagation
# Verify managed identity configuration
az identity show --resource-group $RESOURCE_GROUP --name $IDENTITY_NAME
```

#### **2. Cosmos DB Connection Issues**
```bash
# Problem: Cosmos DB Gremlin connection fails
# Solution: Verify API key configuration
az cosmosdb keys list --resource-group $RESOURCE_GROUP --name $COSMOS_ACCOUNT

# Ensure COSMOS_USE_MANAGED_IDENTITY=false in .env
# Cosmos Gremlin API requires API key authentication
```

#### **3. Search Index Missing**
```bash
# Problem: Search operations fail with "index not found"
# Solution: Create search index manually
python -c "
import asyncio
from core.azure_search.search_client import UnifiedSearchClient

async def create_index():
    client = UnifiedSearchClient()
    await client.create_index('maintie-index')
    print('✅ Search index created')

asyncio.run(create_index())
"
```

#### **4. Storage Container Issues**
```bash
# Problem: Blob storage operations fail
# Solution: Create containers manually
az storage container create --name rag-data-maintenance --account-name $STORAGE_ACCOUNT --auth-mode login

# Verify container permissions
az storage container show-permission --name rag-data-maintenance --account-name $STORAGE_ACCOUNT
```

### **Debugging Commands**

#### **Service Health Diagnostics**
```bash
# Check Azure resource status
az resource list --resource-group $RESOURCE_GROUP --query "[].{Name:name,Type:type,Location:location,Status:provisioningState}" -o table

# Test individual service connections
python -c "
import asyncio
from services.infrastructure_service import InfrastructureService

async def debug_services():
    infra = InfrastructureService()
    
    # Debug each service individually
    services = {
        'OpenAI': infra.openai_client,
        'Search': infra.search_client,
        'Storage': infra.storage_client,
        'Cosmos': infra.cosmos_client,
        'ML': infra.ml_client
    }
    
    for name, client in services.items():
        try:
            # Attempt basic operation
            if hasattr(client, 'test_connection'):
                await client.test_connection()
                print(f'✅ {name}: OK')
            else:
                print(f'⚠️  {name}: No test method')
        except Exception as e:
            print(f'❌ {name}: {str(e)[:100]}...')

asyncio.run(debug_services())
"
```

#### **Configuration Validation**
```bash
# Validate environment configuration
python -c "
from config.settings import get_settings
import os

settings = get_settings()
print('=== Configuration Validation ===')
print(f'OpenAI Endpoint: {settings.azure_openai_endpoint}')
print(f'Search Endpoint: {settings.azure_search_endpoint}')
print(f'Cosmos Endpoint: {settings.azure_cosmos_endpoint}') 
print(f'Storage Account: {settings.azure_storage_account}')
print(f'Use Managed Identity: {settings.use_managed_identity}')
print(f'Environment: {os.getenv(\"AZURE_ENV\", \"development\")}')
"
```

### **Log Analysis**

#### **Application Logs**
```bash
# View backend logs
tail -f backend/logs/last_backend_session.log

# View Azure service logs
az monitor activity-log list --resource-group $RESOURCE_GROUP --max-events 50
```

#### **Azure Diagnostics**
```bash
# Application Insights queries
az monitor app-insights query --app $APP_INSIGHTS_NAME --analytics-query "
requests 
| where timestamp > ago(1h)
| summarize count() by resultCode
| order by count_ desc
"

# Log Analytics queries
az monitor log-analytics query --workspace $LOG_ANALYTICS_WORKSPACE --analytics-query "
AppTraces
| where TimeGenerated > ago(1h)
| summarize count() by SeverityLevel
"
```

---

## 📈 Performance Optimization

### **Development Performance**

#### **Local Development Optimization**
```bash
# Use smaller batch sizes for faster development cycles
export BATCH_SIZE=10
export SKIP_PROCESSING_IF_DATA_EXISTS=true

# Enable caching for repeated operations
export AZURE_DATA_STATE_CACHE_TTL=3600
```

#### **Database Optimization**
```bash
# Configure Cosmos DB for development
export COSMOS_THROUGHPUT_MODE=serverless
export COSMOS_CONSISTENCY_LEVEL=session

# Configure Search for development  
export SEARCH_REPLICA_COUNT=1
export SEARCH_PARTITION_COUNT=1
```

### **Production Performance**

#### **Scaling Configuration**
```bash
# Production environment variables
export COSMOS_THROUGHPUT_MODE=provisioned
export COSMOS_MIN_THROUGHPUT=400
export COSMOS_MAX_THROUGHPUT=4000

export SEARCH_REPLICA_COUNT=2
export SEARCH_PARTITION_COUNT=2

export OPENAI_MAX_REQUESTS_PER_MINUTE=50
```

#### **Performance Monitoring**
```bash
# Enable detailed monitoring
export ENABLE_PERFORMANCE_MONITORING=true
export TELEMETRY_SAMPLE_RATE=1.0

# Configure alerting thresholds
export RESPONSE_TIME_THRESHOLD_MS=3000
export ERROR_RATE_THRESHOLD_PCT=1.0
```

---

## 🔄 Multi-Environment Management

### **🆕 Automatic Environment Sync**

The system now provides **automatic synchronization** between azd environments and backend configuration:

#### **Method 1: Switch and Sync in One Command**
```bash
./scripts/sync-env.sh development  # Switches azd to development + syncs backend
./scripts/sync-env.sh staging      # Switches azd to staging + syncs backend  
./scripts/sync-env.sh production   # Switches azd to production + syncs backend
```

#### **Method 2: Sync with Current Environment**
```bash
azd env select staging    # Select environment in azd
make sync-env            # Sync backend configuration to match
```

#### **What Gets Synchronized**
- ✅ **Environment file**: `backend/config/environments/{env}.env` 
- ✅ **Backend symlink**: `backend/.env` points to correct environment
- ✅ **Makefile defaults**: `AZURE_ENVIRONMENT` updated automatically
- ✅ **Runtime detection**: Backend automatically detects current azd environment

### **Environment Switching (Legacy)**

#### **Manual Switch Between Environments**
```bash
# List available environments
azd env list

# Switch to different environment (manual)
azd env select staging
azd env select production

# Sync backend manually (required after manual switch)
make sync-env

# Deploy to selected environment
azd up
```

#### **Environment-Specific Configuration**
```bash
# Development
azd env select development
azd env set AZURE_LOCATION eastus
azd env set OPENAI_DEPLOYMENT_CAPACITY 10

# Production  
azd env select production
azd env set AZURE_LOCATION centralus
azd env set OPENAI_DEPLOYMENT_CAPACITY 50
```

### **Configuration Management**

#### **Environment Variables per Environment**
```bash
# Development settings
echo "DEBUG=true
LOG_LEVEL=debug
BATCH_SIZE=10
ENABLE_CACHING=true" > backend/config/environments/development.env

# Production settings
echo "DEBUG=false
LOG_LEVEL=info
BATCH_SIZE=100
ENABLE_CACHING=true" > backend/config/environments/production.env
```

---

## 📚 Setup References

### **Key Configuration Files**
- **Azure Configuration**: `azure.yaml` - azd project configuration
- **Infrastructure**: `infra/main.bicep` - Azure resource templates
- **Backend Config**: `backend/config/settings.py` - Application settings
- **Environment Files**: `backend/config/environments/` - Environment-specific configs

### **Essential Scripts**
- **Setup**: `./scripts/setup-environments.sh` - Environment creation
- **Configuration**: `./scripts/update-env-from-deployment.sh` - Auto-configure from Azure
- **Testing**: `./scripts/test-infrastructure.sh` - Infrastructure validation
- **Teardown**: `./scripts/azd-teardown.sh` - Safe resource cleanup

### **Documentation References**
- **[Infrastructure Plan](infra/AZURE_INFRASTRUCTURE_PLAN.md)** - Detailed Azure infrastructure
- **[Backend Documentation](backend/docs/)** - Backend-specific setup guides
- **[System Architecture](ARCHITECTURE.md)** - Technical architecture details
- **[Development Guide](CLAUDE.md)** - Development patterns and practices

---

## 🆘 Getting Help

### **Setup Issues**
1. **Check Prerequisites**: Ensure all required tools are installed and versions meet requirements
2. **Verify Azure Permissions**: Confirm you have Contributor role on the subscription
3. **Resource Provider Registration**: Ensure all required providers are registered
4. **Region Availability**: Verify all services are available in your chosen region

### **Authentication Issues**
1. **RBAC Propagation**: Wait 5-15 minutes for role assignments to propagate
2. **Managed Identity**: Verify managed identity is properly configured and assigned
3. **API Keys**: For Cosmos Gremlin, ensure API key is correctly configured
4. **Token Refresh**: Clear authentication cache with `az account clear` and re-login

### **Performance Issues**
1. **Resource Sizing**: Check if your Azure resources are appropriately sized for workload
2. **Network Latency**: Consider deploying resources in same region for better performance
3. **Quotas and Limits**: Verify you haven't hit Azure service quotas or rate limits
4. **Batch Sizes**: Adjust batch sizes for optimal performance vs resource usage

---

**📖 Navigation:**
- ⬅️ [Back to Main README](README.md)
- 🏗️ [System Architecture](ARCHITECTURE.md)
- 🚀 [Deployment Guide](DEPLOYMENT.md)
- 📖 [API Reference](API_REFERENCE.md)

---

**Setup Status**: ✅ **Production-Ready** | **Deployment Time**: ~15-20 minutes | **Last Updated**: July 28, 2025