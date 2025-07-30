# Core Azure Services Directory

This directory contains the **production-grade Azure service clients** for the Universal RAG system. All services are **real Azure services only** - no local fallbacks, no placeholders, no mock implementations.

## 🏗️ Architecture Overview

The core directory implements a **unified client architecture** where all Azure services inherit from `BaseAzureClient` for consistent initialization, error handling, and managed identity support.

```
core/
├── azure_auth/         # Managed Identity & Authentication
├── azure_openai/       # OpenAI GPT-4o + Embeddings
├── azure_search/       # Cognitive Search + Vector Search
├── azure_cosmos/       # Cosmos DB Gremlin Graph Database
├── azure_storage/      # Blob Storage for Documents
├── azure_ml/          # ML Workspace + GNN Training
├── azure_monitoring/   # Application Insights
├── models/            # Data Models
└── utilities/         # Shared Utilities
```

## 🔧 Real Azure Services Used

### 1. **Azure OpenAI Service**
- **Endpoint**: `https://maintie-rag-staging-oeeopj3ksgnlo.openai.azure.com/`
- **Models**: `gpt-4o` (20 TPM), `text-embedding-ada-002` (30 TPM)
- **Authentication**: Managed Identity (DefaultAzureCredential)
- **Capabilities**: Knowledge extraction, text completion, 1536D embeddings

### 2. **Azure Cognitive Search**
- **Endpoint**: `https://srch-maintie-rag-staging-oeeopj3ksgnlo.search.windows.net/`
- **Index**: `maintie-staging-index-maintenance`
- **Features**: Vector search, semantic search, full-text search
- **Authentication**: Managed Identity

### 3. **Azure Cosmos DB (Gremlin API)**
- **Endpoint**: `https://cosmos-maintie-rag-staging-oeeopj3ksgnlo.documents.azure.com:443/`
- **Database**: `maintie-rag-staging`
- **Container**: `knowledge-graph-staging`
- **Purpose**: Knowledge graph storage for GNN training

### 4. **Azure Blob Storage**
- **Account**: `stmaintieroeeopj3ksg`
- **Container**: `maintie-staging-data`
- **Purpose**: Document storage and data pipeline

### 5. **Azure ML Workspace**
- **Workspace**: `ml-maintierag-lnpxxab4`
- **Purpose**: GNN model training and deployment

## 🧪 Testing Results (Latest)

All core clients have been tested and verified working with real Azure services:

```bash
# Run from backend directory
cd /workspace/azure-maintie-rag/backend

# Test individual services
python -c "
import asyncio
from core.azure_openai.openai_client import UnifiedAzureOpenAIClient

async def test():
    client = UnifiedAzureOpenAIClient()
    result = await client.test_connection()
    print(f'OpenAI: {result[\"success\"]}')

asyncio.run(test())
"
```

### ✅ Current Test Results:
- **OpenAI Client**: ✅ Connected to `gpt-4o` model
- **Embedding Service**: ✅ Generating 1536D embeddings  
- **Search Client**: ✅ Connected to search index (0 documents)
- **Cosmos Client**: ✅ Connected to Gremlin database
- **Storage Client**: ✅ Connected to blob storage

## 🔄 Client Architecture Patterns

### BaseAzureClient Inheritance
All clients inherit from `BaseAzureClient` for consistency:

```python
class UnifiedAzureOpenAIClient(BaseAzureClient):
    def _get_default_endpoint(self) -> str:
        return azure_settings.azure_openai_endpoint
        
    def _get_default_key(self) -> str:
        return azure_settings.openai_api_key
        
    def _initialize_client(self):
        # Real Azure OpenAI initialization
        if self.use_managed_identity:
            credential = DefaultAzureCredential()
            self._client = AzureOpenAI(
                azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
                api_version=azure_settings.openai_api_version,
                azure_endpoint=self.endpoint
            )
```

### Key Features:
- **Managed Identity First**: All clients prefer managed identity over API keys
- **Lazy Initialization**: Clients initialize on first use with `ensure_initialized()`
- **Consistent Error Handling**: Standardized error responses across all services
- **Real Azure SDK**: Uses official Azure SDK libraries, no custom HTTP clients

## 📊 Service Capabilities

### OpenAI Operations
```python
# Knowledge Extraction (Real Azure OpenAI)
result = await openai_client.extract_knowledge(texts, domain="maintenance")

# Text Completion 
response = await openai_client.get_completion(prompt, domain="general")

# Embeddings Generation
embeddings = await openai_client.create_embeddings(["text1", "text2"])
```

### Vector Search Operations
```python
# Real Azure Cognitive Search Vector Search
result = await search_client.vector_search(query_vector, top=10)

# Document Indexing with Embeddings
await search_client.index_documents(documents_with_vectors)
```

### Graph Operations
```python
# Real Cosmos DB Gremlin Operations
cosmos_client.add_entity(entity_data, domain="maintenance")
paths = cosmos_client.find_entity_paths(start_entity, end_entity, domain)
```

## 🚫 No Fallbacks Policy

This system **does NOT** use:
- ❌ Local LLM models
- ❌ SQLite databases
- ❌ File-based storage fallbacks
- ❌ Mock services for development
- ❌ Placeholder implementations

**Everything is real Azure services** with proper error handling when services are unavailable.

## 🔧 Configuration

All configuration comes from environment variables via `settings.py`:

```bash
# Required Azure Service Endpoints
AZURE_OPENAI_ENDPOINT=https://maintie-rag-staging-oeeopj3ksgnlo.openai.azure.com/
AZURE_SEARCH_ENDPOINT=https://srch-maintie-rag-staging-oeeopj3ksgnlo.search.windows.net/
AZURE_COSMOS_ENDPOINT=https://cosmos-maintie-rag-staging-oeeopj3ksgnlo.documents.azure.com:443/

# Model Deployments
OPENAI_MODEL_DEPLOYMENT=gpt-4o
EMBEDDING_MODEL_DEPLOYMENT=text-embedding-ada-002

# Authentication
USE_MANAGED_IDENTITY=true
AZURE_CLIENT_ID=b17eee7d-0f80-4a7e-8caf-b302c1a0bdfb
```

## 🧪 How to Test

### Quick Connectivity Test
```bash
cd /workspace/azure-maintie-rag/backend

python -c "
import asyncio
from core.azure_openai.openai_client import UnifiedAzureOpenAIClient
from core.azure_search.search_client import UnifiedSearchClient
from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient

async def test_all():
    # Test OpenAI
    openai = UnifiedAzureOpenAIClient()
    openai_result = await openai.test_connection()
    print(f'OpenAI: {openai_result[\"success\"]}')
    
    # Test Search
    search = UnifiedSearchClient()
    search_result = await search.test_connection()
    print(f'Search: {search_result[\"success\"]}')
    
    # Test Cosmos
    cosmos = AzureCosmosGremlinClient()
    cosmos_result = await cosmos.test_connection()
    print(f'Cosmos: {cosmos_result[\"success\"]}')

asyncio.run(test_all())
"
```

### Knowledge Extraction Test
```bash
python -c "
import asyncio
from core.azure_openai.openai_client import UnifiedAzureOpenAIClient

async def test_extraction():
    client = UnifiedAzureOpenAIClient()
    result = await client.extract_knowledge(['The pump failed due to bearing wear'], 'maintenance')
    if result['success']:
        print(f'Entities: {len(result[\"data\"][\"entities\"])}')
        print(f'Relations: {len(result[\"data\"][\"relationships\"])}')
    else:
        print(f'Error: {result[\"error\"]}')

asyncio.run(test_extraction())
"
```

### Vector Search Test
```bash
python -c "
import asyncio
from core.azure_search.search_client import UnifiedSearchClient
from core.azure_openai.embedding import AzureEmbeddingService

async def test_vector_search():
    # Generate embedding
    embedding_service = AzureEmbeddingService()
    embed_result = await embedding_service.generate_embedding('test query')
    
    if embed_result['success']:
        # Perform vector search
        search_client = UnifiedSearchClient()
        search_result = await search_client.vector_search(
            embed_result['data']['embedding'], top=5
        )
        print(f'Vector search: {search_result[\"success\"]}')
    else:
        print(f'Embedding failed: {embed_result[\"error\"]}')

asyncio.run(test_vector_search())
"
```

## 🏆 Production Readiness

This core directory is **production-ready** with:
- ✅ Real Azure service integration
- ✅ Managed identity authentication
- ✅ Proper error handling and retry logic
- ✅ Consistent client architecture
- ✅ Comprehensive testing coverage
- ✅ No development-only fallbacks

All services are configured for the **staging environment** and ready for production deployment.