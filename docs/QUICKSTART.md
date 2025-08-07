# Azure Universal RAG - Quick Start Guide

**Universal RAG Setup - Based on Actual Implementation**

Get the Azure Universal RAG system with **zero hardcoded domain bias** running based on the actual codebase structure and Universal RAG philosophy.

## 🔍 Prerequisites

Based on the Universal RAG implementation:

### **Required Tools**
```bash
# Python 3.11+ (required for PydanticAI)
python --version  # Should be Python 3.11+

# Azure CLI for authentication  
az --version      # Verify Azure CLI installed
az login          # Authenticate with Azure

# Azure Developer CLI for infrastructure
curl -fsSL https://aka.ms/install-azd.sh | bash
azd --version     # Verify azd installed

# Node.js for React frontend
node --version    # Should be Node.js 18+ for React 19.1.0
```

### **Azure Services Required**
Universal RAG implementation uses real Azure services (no mocks):
- **Azure OpenAI** (AsyncAzureOpenAI with GPT-4 deployment)
- **Azure Cognitive Search** (vector search with 1536D embeddings)
- **Azure Cosmos DB** (Gremlin API for knowledge graphs)
- **Azure ML** (GNN training/inference with PyTorch)
- **Azure Blob Storage** (document management)
- **Azure Key Vault** (secrets management with DefaultAzureCredential)

## ⚡ Universal RAG Implementation Setup

### 1. Environment Configuration (2 minutes)

```bash
# Clone and navigate
git clone <repository-url>
cd azure-maintie-rag

# Install Python dependencies (includes PydanticAI)
pip install -r requirements.txt

# Environment synchronization (critical for multi-environment)
./scripts/deployment/sync-env.sh development    # Switch to development environment
make sync-env                                   # Sync backend configuration

# Set up Azure environment variables (DefaultAzureCredential)
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"  
export OPENAI_MODEL_DEPLOYMENT="your-deployment-name"
export USE_MANAGED_IDENTITY="false"  # Development mode
```

### 2. Deploy Azure Infrastructure (5 minutes)

```bash
# One-command Azure infrastructure deployment
azd up  # Deploys all 9 Azure services automatically

# Expected results:
# ✅ Azure OpenAI (GPT-4 deployment)
# ✅ Azure Cognitive Search (vector search)
# ✅ Azure Cosmos DB (Gremlin API)
# ✅ Azure ML (GNN training)  
# ✅ Azure Blob Storage (documents)
# ✅ Azure Key Vault (secrets)
# ✅ Azure Application Insights (monitoring)
# ✅ Azure Log Analytics (logging)
# ✅ Azure Container Apps (hosting)
```

### 3. Verify Universal RAG Implementation (1 minute)

Test the actual Universal RAG components:

```bash
# Verify universal models (domain-agnostic)
python -c "
from agents.core.universal_models import UniversalDomainAnalysis
print('✅ Universal models loaded (work for ANY domain)')

# Test Domain Intelligence Agent (content discovery, not classification)
cd agents/domain_intelligence && python agent.py
# Expected: Discovers content characteristics without domain assumptions
from agents.core.data_models import ExtractionQualityOutput, ValidatedEntity
print('✅ Centralized data models loaded (80+ Pydantic models)')

# Check PydanticAI agents
from agents.domain_intelligence.agent import get_domain_intelligence_agent
from agents.knowledge_extraction.agent import get_knowledge_extraction_agent
from agents.universal_search.agent import get_universal_search_agent
print('✅ Three PydanticAI agents available')
"
```

### 3. Test Real Azure Integration (2 minutes)

```bash
# Test actual Azure service initialization
python -c "
import asyncio
from agents.core.azure_service_container import ConsolidatedAzureServices

async def test_services():
    container = ConsolidatedAzureServices()
    print('🔧 Testing real Azure service initialization...')
    
    try:
        status = await container.initialize_all_services()
        print('Azure Service Status:')
        for service, success in status.items():
            icon = '✅' if success else '❌'
            print(f'{icon} {service}: {\"CONNECTED\" if success else \"FAILED\"}')
        
        health = container.get_service_status()
        print(f'Overall Health: {health[\"overall_health\"]}')
        
    except Exception as e:
        print(f'⚠️ Service initialization: {e}')
        print('Note: Requires real Azure service endpoints configured')

asyncio.run(test_services())
"
```

## 🚀 Running the System

### Start the FastAPI Backend

Based on `api/main.py` (42 lines):

```bash
# Start the FastAPI application
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or use the make command if available
make dev
```

**What you'll get:**
- FastAPI app titled "Azure Universal RAG API"
- CORS middleware with wildcard origins
- Root endpoint at `/` showing version and available endpoints
- Health check at `/health`
- Search endpoints at `/api/v1/search`

### Test Real Endpoints

```bash
# Test root endpoint
curl http://localhost:8000/

# Test health check  
curl http://localhost:8000/health

# Test search endpoint (if implemented)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'
```

## 🔧 Real Agent Testing

### Domain Intelligence Agent (122 lines)

```bash
python -c "
from agents.domain_intelligence.agent import create_domain_intelligence_agent

try:
    agent = create_domain_intelligence_agent()
    print('✅ Domain Intelligence Agent created successfully')
    print(f'   - Model: Uses get_azure_openai_model() from environment')
    print(f'   - Deps: DomainDeps with lazy initialization')
    print(f'   - Toolsets: domain_intelligence_toolset (FunctionToolset pattern)')
except Exception as e:
    print(f'❌ Domain Intelligence Agent: {e}')
"
```

### Knowledge Extraction Agent (368 lines)

```bash
python -c "
from agents.knowledge_extraction.agent import get_knowledge_extraction_agent

try:
    agent = get_knowledge_extraction_agent()
    print('✅ Knowledge Extraction Agent created successfully')
    print(f'   - Implementation: _create_agent_with_toolset()')
    print(f'   - Model: OpenAIModel with AzureProvider')
    print(f'   - Features: Multi-strategy extraction with unified processor')
except Exception as e:
    print(f'❌ Knowledge Extraction Agent: {e}')
"
```

### Universal Search Agent (271 lines)

```bash
python -c "
from agents.universal_search.agent import get_universal_search_agent

try:
    agent = get_universal_search_agent()
    print('✅ Universal Search Agent created successfully')
    print(f'   - Implementation: _create_agent_with_consolidated_orchestrator()')
    print(f'   - Features: Tri-modal search with consolidated orchestrator')
    print(f'   - Dependencies: UniversalSearchDeps')
except Exception as e:
    print(f'❌ Universal Search Agent: {e}')
"
```

## 📊 Real Data Verification

If the `data/raw/Programming-Language/` directory exists:

```bash
# Check for real data files
ls -la data/raw/Programming-Language/ | head -10

# Count actual data files
find data/raw/Programming-Language/ -name "*.md" | wc -l

# Sample content from actual files
head -5 data/raw/Programming-Language/*.md | head -20
```

## 🧪 Shared Infrastructure Testing  

Test the actual shared utilities:

```bash
python -c "
# Test text statistics utility (PydanticAI-enhanced)
from agents.shared.text_statistics import TextStatistics, calculate_text_statistics

text = 'This is a sample text for statistical analysis testing.'
stats = calculate_text_statistics(text)
print('✅ Text Statistics utility working')
print(f'   Words: {stats.total_words}, Readability: {stats.readability_score:.1f}')

# Test content preprocessing
from agents.shared.content_preprocessing import clean_text_content, TextCleaningOptions

options = TextCleaningOptions(remove_html=True, normalize_whitespace=True)  
result = clean_text_content('<p>Sample HTML content</p>', options)
print('✅ Content preprocessing working')
print(f'   Quality score: {result.cleaning_quality_score:.2f}')

print('✅ Shared infrastructure operational')
"
```

## 🔍 Configuration Verification

Test the real configuration system:

```bash
python -c "
# Test centralized configuration functions
from config.universal_config import get_universal_config
from agents.core.simple_config_manager import SimpleConfigManager

try:
    system_config = get_system_config()
    print('✅ System configuration loaded')
    
    model_config = get_model_config_bootstrap()
    print('✅ Model configuration (bootstrap) loaded')
    print(f'   API Version: {model_config.api_version}')
    
except Exception as e:
    print(f'⚠️ Configuration: {e}')
    
# Test Azure settings
from config.settings import azure_settings
print('✅ Azure settings loaded')
print(f'   OpenAI Endpoint: {azure_settings.azure_openai_endpoint or \"Not configured\"}')
"
```

## ❌ Troubleshooting

### Common Issues

**1. Missing Azure Credentials:**
```bash
az login
# Set environment variables for your Azure services
```

**2. Missing Dependencies:**
```bash
pip install -r requirements.txt
# Ensure PydanticAI and Azure SDKs are installed
```

**3. Configuration Errors:**
```bash
# Check that azure_settings can load your endpoints
python -c "from config.settings import azure_settings; print(azure_settings.azure_openai_endpoint)"
```

## ✅ Success Indicators

When everything is working correctly:

- **Azure Service Container**: ConsolidatedAzureServices initializes without errors  
- **PydanticAI Agents**: All three agents create successfully with proper dependencies
- **FastAPI Server**: Starts on port 8000 with health check responding
- **Configuration**: Centralized config functions load without circular dependency issues
- **Shared Utilities**: Text statistics and content preprocessing work correctly

## 🎯 Next Steps

1. **Explore the Architecture**: Review `docs/ARCHITECTURE.md` for detailed implementation analysis
2. **Development Workflow**: See `docs/DEVELOPMENT_GUIDE.md` for real development patterns  
3. **Azure Services**: Check `docs/TROUBLESHOOTING.md` for Azure service integration details
4. **Frontend Integration**: See `docs/FRONTEND.md` if planning to use the React frontend

This system represents a **real, production-ready implementation** with genuine Azure service integration, PydanticAI agents, and comprehensive data models - no mock components.