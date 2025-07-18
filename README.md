# Azure Universal RAG

**Enterprise-Ready Azure-Powered Universal RAG System**

---

## 🚀 Overview

Azure Universal RAG is a **production-grade backend system** for advanced universal intelligence, combining:

- **Azure Services Integration** with complete cloud-native architecture
- **Universal RAG System** with 100% functional Azure components
- **Knowledge graph extraction** from any domain text data
- **Unified retrieval** (Azure Cognitive Search + Azure Cosmos DB)
- **Domain-agnostic LLM response generation** with Azure OpenAI
- **Progressive real-time workflow** with streaming UI
- **Azure Infrastructure as Code** with automated deployment
- **FastAPI API** with health, metrics, and streaming endpoints
- **Clean Service Architecture** with a dedicated frontend UI

---

## ✨ Features

### Core Azure RAG Capabilities
- Universal text-based knowledge extraction and processing
- Advanced query analysis and concept expansion
- Unified retrieval (Azure Cognitive Search + Azure Cosmos DB)
- Azure OpenAI-powered, safety-aware response generation
- Domain-agnostic processing (no hard-coded rules)
- Health, metrics, and system status endpoints

### Enhanced User Experience
- **Progressive Real-Time Workflow**: Step-by-step visual progress during query processing
- **Streaming API**: Server-sent events for real-time workflow updates
- **Smart Disclosure UI**: Three-layer information depth (user-friendly → technical → diagnostic)
- **Separated Backend API and Frontend UI services**

### Azure Infrastructure & Deployment
- Azure Blob Storage for document storage
- Azure Cognitive Search for vector search
- Azure Cosmos DB for knowledge graphs
- Azure OpenAI for processing and generation
- Azure Machine Learning for advanced analytics
- Docker and virtualenv support
- Comprehensive Azure service integration

---

## 🛠️ Technology Stack

```
Frontend Stack:
├─ React 19.1.0 + TypeScript 5.8.3
├─ Vite 7.0.4 (build tool)
├─ axios 1.10.0 (HTTP client)
├─ Server-Sent Events (real-time updates)
└─ CSS custom styling with progressive disclosure

Backend Stack:
├─ FastAPI + uvicorn (streaming endpoints)
├─ Azure OpenAI integration (openai>=1.13.3)
├─ Azure Cognitive Search (vector search)
├─ Azure Cosmos DB (knowledge graphs)
├─ Azure Blob Storage (document storage)
├─ Azure Machine Learning (advanced analytics + GNN training)
├─ NetworkX 3.2.0 graph processing
├─ PyTorch 2.0.0 + torch-geometric 2.3.0 (GNN)
├─ Optuna 3.0.0 + Weights & Biases 0.16.0 (experiment tracking)
└─ Comprehensive ML/AI pipeline with Universal GNN
```

## 🏗️ Architecture

```
backend/
├── core/
│   ├── azure_ml/           # ✅ Azure ML + GNN training
│   │   ├── ml_client.py    # Azure ML integration
│   │   ├── gnn/            # Universal GNN components
│   │   │   ├── model.py    # GNN architecture
│   │   │   ├── trainer.py  # Training logic
│   │   │   └── data_loader.py # Graph data loading
│   │   └── classification_service.py
│   ├── azure_cosmos/       # ✅ Azure Cosmos DB
│   │   └── cosmos_gremlin_client.py
│   ├── azure_search/       # ✅ Azure Cognitive Search
│   │   └── search_client.py
│   ├── azure_storage/      # ✅ Azure Blob Storage
│   │   └── storage_client.py
│   ├── azure_openai/       # ✅ Azure OpenAI
│   ├── models/             # ✅ Universal data models
│   ├── orchestration/      # ✅ RAG orchestration
│   └── workflow/           # ✅ Workflow management
├── scripts/                # ✅ Utility scripts
│   └── train_comprehensive_gnn.py # GNN training
├── api/                    # ✅ FastAPI endpoints
├── config/                 # ✅ Configuration
└── tests/                  # ✅ Test suite
```

---

## 🚦 Quick Commands

This project uses a root `Makefile` to simplify common tasks for both backend and frontend services.

### Using Makefile

```bash
make help               # See all available commands
make setup              # Full project setup (backend and frontend)
make dev                # Start both backend API and frontend UI services
make backend            # Start backend API service only
make frontend           # Start frontend UI service only
make test               # Run all tests (backend and frontend)
make health             # Check health of both services
make docker-up          # Build and run Docker containers for both services via docker-compose
make docker-down        # Stop and remove Docker containers
make clean              # Clean ALL generated files - reset to raw text data
```

---

## 📝 Documentation Setup

### VSCode Environment (Recommended)

For the best development experience with enhanced markdown preview:

```bash
# From backend directory
make docs-setup    # Sets up VSCode environment with extensions
make docs-status   # Shows documentation setup status
make docs-preview  # Opens markdown preview (if VSCode CLI available)
```

**For SSH Development (Azure ML):**
- Use VSCode Remote-SSH extension for best experience
- All extensions auto-install when you connect
- Markdown preview works perfectly with `Ctrl+Shift+V`

**Configured Extensions:**
- Markdown All in One
- Markdown Preview Enhanced
- Markdown Mermaid
- Python, Black, Pylint
- JSON and YAML support

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/david3xu/azure-maintie-rag.git
cd azure-maintie-rag
```

### 2. Configure Azure environment

- Copy `backend/config/environment_example.env` to `backend/.env`
- Set your Azure service credentials and endpoints
- Configure Azure OpenAI, Cognitive Search, Cosmos DB, and Blob Storage settings

### 3. Full Project Setup

This command will:

- Create Python virtual environments for the backend
- Install all Python dependencies for the backend
- Create necessary data directories within `backend/data/`
- Install Node.js dependencies for the frontend

```bash
make setup
```

### 4. Start from Raw Text Data

To begin purely from raw text data, ensure you have text files in `backend/data/raw/` then:

```bash
make clean              # Reset to raw data state
make setup              # Ensure dependencies installed
make dev                # Start both services
```

The system will automatically process raw text through the complete Azure Universal RAG pipeline.

---

## 🌟 Azure Universal RAG System

### System Status: 100% Functional ✅

Our Azure Universal RAG system has achieved **complete functionality** through comprehensive Azure services integration:

### 📖 **Azure Resource Preparation**
**📖 [Complete Azure Resource Preparation Guide](docs/AZURE_RESOURCE_PREPARATION_FINAL.md)**

For Azure infrastructure deployment and management, see our comprehensive guide covering:
- Complete infrastructure deployment (10 Azure services)
- Enterprise security (Key Vault, managed identities)
- ML training environment (Azure ML workspace)
- Monitoring & logging (Application Insights)
- Application hosting (Container Apps)
- Complete lifecycle management (deploy, teardown, redeploy)

| **Component** | **Status** | **Azure Service** |
|---------------|------------|-------------------|
| Knowledge Extraction | ✅ Working | Azure OpenAI GPT-4 |
| Vector Indexing | ✅ Working | Azure Cognitive Search |
| Query Processing | ✅ Working | Azure OpenAI + Azure Services |
| Vector Search | ✅ Working | Azure Cognitive Search retrieval |
| Response Generation | ✅ Working | Azure OpenAI GPT-4 |
| Document Storage | ✅ Working | Azure Blob Storage |
| Knowledge Graphs | ✅ Working | Azure Cosmos DB |
| Metadata Storage | ✅ Working | Azure Cosmos DB |
| **GNN Training** | ✅ **NEW** | Azure Machine Learning |

### Key Azure Services Integration

**Azure OpenAI Integration**
- GPT-4 for knowledge extraction and response generation
- Text embeddings for vector search
- Domain-agnostic processing

**Azure Cognitive Search**
- Vector search with semantic capabilities
- Multi-language support
- Real-time indexing and search

**Azure Cosmos DB**
- Knowledge graph storage with Gremlin API
- Native graph traversal and analytics
- Multi-domain support

**Azure Blob Storage**
- Document storage and retrieval
- Hierarchical namespace for data organization
- Version control for data updates

---

## 🎯 Progressive Real-Time Workflow

### Three-Layer Smart Disclosure

Our frontend provides **progressive disclosure** for different user types:

**Layer 1: User-Friendly** (90% of users)
```
🔍 Understanding your question...
☁️ Searching Azure services...
📝 Generating comprehensive answer...
```

**Layer 2: Technical Workflow** (power users)
```
📊 Knowledge Extraction (Azure OpenAI): 15 entities, 10 relations
🔧 Vector Indexing (Azure Cognitive Search): 7 documents, 1536D vectors
🔍 Query Processing: Troubleshooting type, 18 concepts
⚡ Vector Search: 3 results, top score 0.826
📝 Response Generation (Azure OpenAI): 2400+ chars, 3 citations
```

**Layer 3: System Diagnostics** (administrators)
```json
{
  "step": "azure_cognitive_search",
  "status": "completed",
  "duration": 2.7,
  "azure_service": "cognitive_search",
  "details": { "documents_found": 15, "search_score": 0.826 }
}
```

### Streaming API Endpoints

- `GET /api/v1/query/stream/{query_id}`: Server-sent events for real-time Azure service updates
- `POST /api/v1/query/universal`: Submit query with Azure services processing
- Real-time progress updates with detailed Azure service information

---

## 🔬 Comprehensive Azure ML Integration

Azure Universal RAG includes **research-level, end-to-end Azure ML integration** for advanced experimentation and production deployment, including **Universal GNN training**.

### Features:
- Azure ML workspace integration
- **Universal GNN training** with multiple architectures (GCN, GAT, GraphSAGE)
- Hyperparameter optimization (Optuna)
- Cross-validation (k-fold)
- Advanced training: schedulers, early stopping, gradient clipping, label smoothing, class weighting
- Comprehensive evaluation: accuracy, precision, recall, F1, AUC, confusion matrix, per-class analysis
- Ablation studies
- Experiment tracking (Azure ML + Weights & Biases)
- Model checkpointing and result saving
- **Graph data loading** from Azure Cosmos DB Gremlin

### How to Use:

**CLI:**
```bash
# Create environment and config files
python backend/scripts/train_comprehensive_gnn.py --create-env
python backend/scripts/train_comprehensive_gnn.py --create-config

# Train GNN with default config
python backend/scripts/train_comprehensive_gnn.py

# Train with custom config
python backend/scripts/train_comprehensive_gnn.py \
    --config example_comprehensive_gnn_config.json

# Train in Azure ML
python backend/scripts/train_comprehensive_gnn.py \
    --workspace my-workspace \
    --experiment universal-rag-gnn
```

**Config:** Edit `backend/scripts/example_comprehensive_gnn_config.json` or provide your own.

**API:** Import and call `run_comprehensive_gnn_training()` from `src.gnn.comprehensive_trainer`.

### Documentation:
- See `backend/scripts/README_comprehensive_gnn.md` for CLI/config details
- See module docstring in `backend/src/gnn/comprehensive_trainer.py` for full feature list and integration points

### CI/CD:
- The pipeline is smoke-tested in CI to ensure research code health

---

## 🐳 Docker

To build and run both backend and frontend services using Docker:

```bash
make docker-up
```

---

## 📂 Project Structure

```
Project Root:
├─ backend/                    # Complete Azure Universal RAG API service
│  ├─ core/                   # Azure Universal RAG core components
│  │  ├─ azure_ml/            # ✅ Azure ML + GNN training
│  │  │  ├─ ml_client.py      # Azure ML integration
│  │  │  ├─ gnn/              # Universal GNN components
│  │  │  │  ├─ model.py       # GNN architecture
│  │  │  │  ├─ trainer.py     # Training logic
│  │  │  │  └─ data_loader.py # Graph data loading
│  │  │  └─ classification_service.py
│  │  ├─ azure_cosmos/        # ✅ Azure Cosmos DB
│  │  │  └─ cosmos_gremlin_client.py
│  │  ├─ azure_search/        # ✅ Azure Cognitive Search
│  │  │  └─ search_client.py
│  │  ├─ azure_storage/       # ✅ Azure Blob Storage
│  │  │  └─ storage_client.py
│  │  ├─ azure_openai/        # ✅ Azure OpenAI
│  │  ├─ models/              # ✅ Universal data models
│  │  ├─ orchestration/       # Main RAG orchestration logic
│  │  ├─ workflow/            # Three-layer workflow transparency
│  │  └─ utilities/           # Core utility functions
│  ├─ scripts/                # ✅ Utility and demo scripts
│  │  └─ train_comprehensive_gnn.py # GNN training
│  ├─ api/                    # FastAPI endpoints + streaming
│  │  ├─ endpoints/           # Individual endpoint files
│  │  ├─ models/              # API request/response models
│  │  └─ main.py              # FastAPI application
│  ├─ config/                 # Configuration files
│  ├─ docs/                   # Backend documentation
│  ├─ integrations/           # External service integrations
│  ├─ utilities/              # Shared utility functions
│  ├─ tests/                  # Comprehensive test suite
│  └─ debug/                  # Debug and development tools
├─ frontend/                  # Pure UI consumer service
│  ├─ src/                    # React components + workflow transparency
│  ├─ public/                 # Static assets
│  └─ package.json            # Node.js dependencies
├─ infrastructure/            # Azure Infrastructure as Code
│  ├─ azure-resources.bicep   # Azure resource templates
│  ├─ parameters.json         # Environment parameters
│  └─ provision.py            # Python automation script
├─ docs/                      # Project documentation
├─ .vscode/                   # VSCode configuration
├─ .env                       # Environment variables
├─ docker-compose.yml         # Docker Compose configuration
└─ Makefile                   # Root Makefile for orchestrating services
```

---

## 🔄 Service Architecture

### Complete Azure Universal RAG Workflow

The Azure Universal RAG system implements a comprehensive workflow from raw text data to final answers with real-time progress tracking:

```mermaid
graph TD
    A[Raw Text Data] --> B[Azure Blob Storage]
    B --> C[Knowledge Extraction Azure OpenAI]
    C --> D[Azure Cognitive Search Vectors]
    C --> E[Entity/Relation Graph]
    D --> F[Vector Index 1536D]
    E --> G[Azure Cosmos DB Gremlin Graph]
    G --> H[GNN Training Azure ML]
    H --> I[Trained GNN Model]
    J[User Query] --> K[Query Analysis Azure OpenAI]
    K --> L[Unified Search - Azure Cognitive Search + Cosmos DB + GNN]
    G --> L
    F --> L
    I --> L
    L --> M[Context Retrieval]
    M --> N[Azure OpenAI Response Generation]
    N --> O[Final Answer with Citations]

    %% Real-time streaming
    K --> P[Streaming Progress Events]
    L --> P
    N --> P
    P --> Q[Frontend Progressive UI]
```

### Enhanced Service Flow with Azure Services

```
User Input → React Frontend → Streaming API → Azure Universal RAG → Real-time Updates → AI Response
     ↓             ↓              ↓               ↓               ↓              ↓
"pump failure"  handleSubmit()  POST /streaming  Azure Services   SSE Progress   JSON response
```

### Workflow Components (Azure Enhanced)

| **Phase** | **Component** | **Azure Service** | **Function** | **Streaming** |
|-----------|---------------|-------------------|--------------|---------------|
| **Data Ingestion** | Text Processor | Azure Blob Storage | Raw text → Clean documents | ✅ Progress |
| **Knowledge Extraction** | LLM Extractor | Azure OpenAI GPT-4 | Text → Entities + Relations | ✅ Progress |
| **Vector Indexing** | Azure Cognitive Search | Embedding + Vector DB | Documents → Searchable vectors | ✅ Progress |
| **Graph Construction** | Azure Cosmos DB Gremlin | Native graph algorithms | Entities → Knowledge graph | ✅ Progress |
| **Query Processing** | Query Analyzer | Azure OpenAI + Azure Services | User query → Enhanced query | ✅ Progress |
| **GNN Training** | GNN Trainer | Azure Machine Learning | Graph data → Trained GNN model | ✅ Progress |
| **Retrieval** | Unified Search | Azure Cognitive Search + Cosmos DB Gremlin + GNN | Query → Relevant context | ✅ Progress |
| **Generation** | LLM Interface | Azure OpenAI GPT-4 | Context → Final answer | ✅ Progress |

---

## 📚 Documentation

### Available Documentation

- **API Documentation**: Available at `http://localhost:8000/docs` when backend is running
- **🌟 Azure Universal RAG Capabilities Guide**: See `backend/docs/AZURE_UNIVERSAL_RAG_CAPABILITIES.md` for complete system capabilities, API reference, and integration examples
- **Comprehensive Azure ML Training**: See `backend/scripts/README_comprehensive_gnn.md`
- **System Architecture**: This README provides complete system overview
- **Streaming API**: Real-time workflow documentation in API docs

**📖 Key Documentation:**
- **[Complete Azure Capabilities Guide](backend/docs/AZURE_UNIVERSAL_RAG_CAPABILITIES.md)** - Full system capabilities, API endpoints, progressive workflow system
- **[Documentation Index](backend/docs/README.md)** - All documentation organized by purpose
- **[System Status](backend/docs/AZURE_UNIVERSAL_RAG_FINAL_STATUS.md)** - Current system health and performance metrics

### Key Scripts for Testing

```bash
# Azure Universal RAG workflow test
python backend/scripts/azure-rag-demo-script.py

# Real query processing with Azure services
python backend/scripts/query_processing_workflow.py

# Azure Universal smart RAG flow
python backend/scripts/azure-rag-workflow-demo.py

# GNN training with Azure ML
python backend/scripts/train_comprehensive_gnn.py
```

### Generated Output Files

The system generates comprehensive analysis in `backend/data/output/`:
- Query analysis results
- Response generation outputs
- Performance metrics
- Azure service diagnostics

---

## 🚀 Getting Started

### Quick Start (from Raw Text)

1. **Clean Setup**:
   ```bash
   make clean    # Reset to raw data state
   make setup    # Install dependencies
   ```

2. **Configure Azure Services**:
   - Set up Azure OpenAI, Cognitive Search, Cosmos DB, Blob Storage
   - Update `backend/.env` with Azure service credentials
   - Place text files in `backend/data/raw/`

3. **Start Services**:
   ```bash
   make dev      # Start backend + frontend
   ```

4. **Access System**:
   - Frontend UI: `http://localhost:5174`
   - Backend API: `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`

### Progressive Workflow Experience

1. Submit a question through the frontend
2. Watch real-time progress through Azure services
3. Receive comprehensive answers with citations
4. Explore technical details and Azure service diagnostics

The system provides a complete Azure-powered RAG experience from raw text data to intelligent responses with full visibility into the processing workflow.