# Azure Universal RAG Backend

**Production-grade FastAPI backend for Azure Universal RAG system**

📖 **Related Documentation:**
- ⬅️ [Back to Main README](../README.md)
- 🏗️ [System Architecture](../ARCHITECTURE.md)
- 🧠 [PyTorch Geometric Guide](../PYTORCH_GEOMETRIC_GUIDE.md) - Graph Neural Network integration
- ⚙️ [Setup Guide](../SETUP.md)
- 🚀 [Deployment Guide](../DEPLOYMENT.md)
- 📖 [API Reference](../API_REFERENCE.md)
- 🔄 [Lifecycle Execution](../LIFECYCLE_EXECUTION_GUIDE.md)

---

## 📁 Actual Directory Structure

**Based on current refactored architecture** (79 Python files across core components):

```
backend/
├── 📄 Configuration Files
│   ├── .env -> config/environments/{current}.env    # Auto-synced with azd environment
│   ├── Dockerfile                    # Container configuration
│   ├── Makefile                     # Development commands (13,564 lines)
│   ├── pyproject.toml               # Python project metadata
│   ├── requirements.txt             # Dependencies (PyTorch, Azure SDKs)
│   └── pytest.ini                  # Testing configuration
│
├── 🚀 FastAPI Application Layer
│   ├── api/                         # REST API endpoints
│   │   ├── main.py                  # FastAPI application entry
│   │   ├── dependencies.py          # Dependency injection
│   │   ├── middleware.py            # Request/response middleware
│   │   ├── endpoints/               # REST endpoints (7 files)
│   │   │   ├── query_endpoint.py    # Universal query processing
│   │   │   ├── health_endpoint.py   # Health checks
│   │   │   ├── gnn_endpoint.py      # GNN training/inference
│   │   │   ├── graph_endpoint.py    # Knowledge graph operations
│   │   │   └── workflow_endpoint.py # Workflow management
│   │   ├── models/                  # Pydantic request/response models
│   │   └── streaming/               # Server-sent events
│
├── 🏗️ Business Logic Layer
│   ├── services/                    # High-level business services (14 files)
│   │   ├── infrastructure_service.py  # Azure service management
│   │   ├── data_service.py            # Data processing workflows
│   │   ├── query_service.py           # Query orchestration
│   │   ├── ml_service.py              # Machine learning operations
│   │   ├── graph_service.py           # Knowledge graph operations
│   │   ├── workflow_service.py        # Workflow management
│   │   └── cleanup_service.py         # Resource cleanup
│
├── 🧠 Infrastructure Layer
│   ├── core/                        # Azure service clients (42 files)
│   │   ├── azure_openai/            # GPT-4 + text-embedding-ada-002
│   │   ├── azure_search/            # Vector search operations
│   │   ├── azure_cosmos/            # Gremlin graph database
│   │   ├── azure_storage/           # Multi-account blob storage
│   │   ├── azure_ml/                # GNN training + inference
│   │   │   └── gnn/                 # PyTorch Geometric GNN models (8 files)
│   │   ├── azure_monitoring/        # Application Insights
│   │   ├── models/                  # Data models
│   │   └── utilities/               # Shared utilities
│
├── ⚙️ Configuration Management
│   ├── config/                      # Application configuration
│   │   ├── settings.py              # Unified settings (487 lines)
│   │   ├── environments/            # Environment-specific configs
│   │   └── azure_config_validator.py # Azure service validation
│
├── 💾 Data & Processing
│   ├── data/                        # Training data and results
│   │   ├── raw/                     # Source documents (3,859 records)
│   │   ├── gnn_models/              # Trained PyTorch models
│   │   ├── extraction_outputs/      # Knowledge extraction results
│   │   └── demo_outputs/            # Demonstration results
│   ├── scripts/                     # Operational tools (10 files)
│   └── prompt_flows/                # Azure ML prompt flows
│
├── 🧪 Testing & Validation
│   ├── tests/                       # Test suites
│   │   ├── integration/             # Azure integration tests
│   │   ├── unit/                    # Unit tests
│   │   └── fixtures/                # Test data and mocks
│   ├── data/
│   │   ├── outputs/                # Model outputs and results (reorganized)
│   └── logs/                        # Application logs
│
└── 📚 Documentation
    ├── docs/                        # Backend-specific documentation
    │   ├── BACKEND_REFACTORING_PLAN.md   # Architecture planning
    │   ├── core/                         # Core system documentation
    │   └── execution/                    # RAG lifecycle guides
    ├── BACKEND_QUICKSTART.md             # Quick start guide
    ├── BACKEND_STRUCTURE.md              # Directory structure
    ├── DOCUMENTATION_TABLE_OF_CONTENTS.md # Doc index
    └── STRUCTURE_VALIDATION.md           # Structure compliance
```

## 🚀 Quick Start

### **Prerequisites**
Ensure Azure Universal RAG infrastructure is deployed (see [Setup Guide](../SETUP.md)):
```bash
# From project root - deploy infrastructure first  
azd auth login

# NEW: Use automatic environment sync
../scripts/sync-env.sh development && azd up  # Development
# OR
../scripts/sync-env.sh staging && azd up      # Staging
```

### **Backend Development Setup**
```bash
# 1. Navigate to backend directory
cd backend

# 2. Set up Python environment
make setup              # Creates venv + installs dependencies

# 3. Verify Azure service connections
make health             # Tests all Azure services

# 4. Run complete data lifecycle (optional)
make data-prep-full     # Processes sample data through pipeline

# 5. Start FastAPI development server
make run                # Starts server on localhost:8000
```

### **Alternative Manual Setup**
```bash
# Manual Python environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Start FastAPI server manually
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 🔄 Environment Management

### **🆕 Automatic Environment Sync**
The backend automatically syncs with whatever azd environment you select:

```bash
# From project root - sync environment and backend config
../scripts/sync-env.sh development  # Switch to development
../scripts/sync-env.sh staging      # Switch to staging
../scripts/sync-env.sh production   # Switch to production

# From backend directory - sync with current azd environment
make sync-env                       # Sync backend with current azd env
```

### **What Gets Synchronized**
- ✅ **Environment file**: `config/environments/{env}.env` created/updated
- ✅ **Symlink**: `.env` points to correct environment file
- ✅ **Makefile**: Default environment updated
- ✅ **Runtime**: Backend automatically detects current environment

## 📊 Current System Status

**✅ Production-Ready Architecture** (Validated July 28, 2025):

| Component | Status | Details |
|-----------|--------|---------|
| **🏗️ Infrastructure** | ✅ **Deployed** | 9 Azure services across 3 regions |
| **🔌 Azure Integration** | ✅ **Operational** | OpenAI, Search, Cosmos, Storage, ML |
| **📊 Data Processing** | ✅ **Functional** | 3,859 records → 326 indexed → 540 entities + 597 relationships |
| **🤖 GNN Training** | ✅ **Complete** | PyTorch Geometric models (540 nodes, 1178 edges, 10 classes) |
| **🚀 API Endpoints** | ✅ **Ready** | Universal query + streaming progress |
| **🔄 Lifecycle** | ✅ **Validated** | Complete data pipeline working |
| **🧪 Testing** | ✅ **Passing** | Integration + unit tests |
| **📈 Performance** | ✅ **Optimized** | Sub-3s query processing |

### **Key Capabilities**
- ✅ **Universal Query Processing**: Multi-source search (Vector + Graph + GNN)
- ✅ **Real-time Streaming**: Server-sent events with 3-layer UI transparency  
- ✅ **Knowledge Graphs**: Entity/relationship extraction with Gremlin traversal
- ✅ **PyTorch Geometric Integration**: Graph Neural Network training with 64D node features and 32D edge features
- ✅ **GNN Enhancement**: Graph neural network training and inference for intelligent relationship reasoning
- ✅ **Multi-hop Reasoning**: Semantic path discovery across knowledge graphs
- ✅ **Enterprise Security**: Managed identity + RBAC across all services