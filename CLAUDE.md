# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

```bash
# Always work from project root
cd /workspace/azure-maintie-rag

# Quick Start Development
make setup                    # Install all dependencies (backend + frontend)  
make dev                      # Start API (port 8000) + Frontend (port 5174)
make health                   # System health check with Azure service status
make session-report           # View current session performance metrics
make clean                    # Clean current session with log replacement

# Testing with Real Azure Services (no mocks)
pytest                                                         # All tests with auto asyncio
pytest -m unit                                                 # Unit tests for agent logic
pytest -m integration                                          # Integration tests with Azure
pytest -m azure_validation                                     # Azure service health tests
pytest tests/test_agents.py::TestDomainIntelligenceAgent -v   # Test specific agent
pytest -x -vvv                                                 # Stop on first failure, verbose

# Code Quality (MUST pass before commits)
black . --check && isort . --check-only                     # Python formatting validation
cd frontend && npm run lint && npx tsc --noEmit             # Frontend lint + TypeScript check
./scripts/hooks/pre-commit-domain-bias-check.sh             # Domain bias validation (ENFORCED)

# 6-Phase Data Pipeline
make dataflow-full          # Execute all phases (0→1→2→3→4→5→6)
make dataflow-validate      # Phase 1: Validate all 3 PydanticAI agents
make dataflow-extract       # Phase 3: Knowledge extraction with unified templates
make dataflow-query         # Phase 4: Query analysis + universal search

# Direct Agent Testing (requires PYTHONPATH)
PYTHONPATH=/workspace/azure-maintie-rag python agents/domain_intelligence/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/knowledge_extraction/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/universal_search/agent.py

# Azure Deployment & Container Management
./scripts/deployment/sync-env.sh prod        # Switch to production environment
azd up                                        # Deploy complete Azure infrastructure
azd pipeline config                           # Setup GitHub Actions CI/CD with OIDC
./scripts/show-deployment-urls.sh             # Display frontend and backend URLs after deployment

# Container Build & Deployment (Streamlined)
./scripts/acr-cloud-build.sh                 # Build images via Azure Container Registry Cloud Build
./scripts/complete-deployment.sh             # Complete deployment workflow 
docker-compose up --build                    # Local multi-container deployment

# Frontend Development
cd frontend && npm install                   # Install frontend dependencies
cd frontend && npm run dev                   # Start frontend dev server
cd frontend && npm run build                 # Production build

# Manual Container Operations (if needed)
docker build -t azure-maintie-rag-backend .              # Build backend container
docker build -t azure-maintie-rag-frontend ./frontend    # Build frontend container
```

## Architecture Overview

This is a **production-ready multi-agent RAG system** built with PydanticAI framework and Azure services. The system implements **zero hardcoded domain bias** - all content characteristics are discovered dynamically rather than categorized into predetermined domains.

### Three Core PydanticAI Agents

1. **Domain Intelligence Agent** (`agents/domain_intelligence/agent.py`)
   - Discovers content characteristics without predetermined categories
   - Outputs: `UniversalDomainAnalysis` with vocabulary complexity, processing config
   - Key functions: `run_domain_analysis()`, `domain_intelligence_agent`
   
2. **Knowledge Extraction Agent** (`agents/knowledge_extraction/agent.py`)
   - Extracts entities and relationships using unified templates
   - Uses Domain Intelligence output to adapt extraction parameters
   - Key functions: `run_knowledge_extraction()`, `knowledge_extraction_agent`
   
3. **Universal Search Agent** (`agents/universal_search/agent.py`)
   - Orchestrates tri-modal search: Vector (Azure Cognitive Search) + Graph (Cosmos DB) + GNN (Azure ML)
   - Combines results based on domain characteristics
   - Key functions: `run_universal_search()`, `universal_search_agent`

### Critical Architecture Patterns

**Dependency Injection Pattern:**
```python
agents/core/universal_deps.py → UniversalDeps class
infrastructure/azure_auth/base_client.py → BaseAzureClient
# All Azure services accessed through UniversalDeps singleton
```

**PydanticAI Toolset Pattern (REQUIRED):**
```python
agents/core/agent_toolsets.py → get_*_toolset() functions
# MUST use FunctionToolset, NOT @agent.tool decorator
# Prevents Union type errors with PydanticAI 0.6.2
```

**Template Variable Flow:**
```python
agents/core/centralized_agent1_schema.py:Agent1TemplateMapping
→ infrastructure/prompt_workflows/templates/universal_knowledge_extraction.jinja2
# Domain Intelligence output provides template variables for extraction
```

**Universal Orchestrator Pattern (API Layer):**
```python
agents/orchestrator.py → UniversalOrchestrator class
api/endpoints/search.py → Uses orchestrator for agent coordination
# All API endpoints use orchestrator instead of calling agents directly
```

## Critical Development Rules

### 🚨 Zero Domain Bias (ENFORCED)

The system MUST discover content characteristics, not categorize into domains:

```python
# ❌ WRONG - Hardcoded domain categories
if domain in ["legal", "technical", "medical"]:
    complexity = 0.8

# ✅ CORRECT - Measure actual content properties
complexity = measure_vocabulary_complexity(content)
parameters = adapt_based_on_measured_properties(complexity)
```

**Pre-commit validation:** `scripts/hooks/pre-commit-domain-bias-check.sh` fails builds with domain bias

### 🏗️ PydanticAI Agent Pattern

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from agents.core.universal_deps import UniversalDeps

# 1. Create toolset FIRST
toolset = FunctionToolset()

# 2. Register tools on TOOLSET (not agent)
@toolset.tool  # ✅ CORRECT
async def analyze_content(ctx: RunContext[UniversalDeps], content: str):
    return await ctx.deps.azure_openai_client.analyze(content)

# 3. Create agent WITH toolset
agent = Agent[UniversalDeps, OutputModel](
    model=azure_openai_model,
    toolsets=[toolset],  # Pass toolset here
    deps_type=UniversalDeps
)
```

**⚠️ Using `@agent.tool` causes Union type errors with PydanticAI 0.6.2 + OpenAI 1.98.0**

## Environment Variables

Critical environment variables for development and testing:

```bash
# Required for all Python scripts
export PYTHONPATH=/workspace/azure-maintie-rag

# Azure authentication (local development)
export USE_MANAGED_IDENTITY=false  # Use DefaultAzureCredential locally

# Numerical stability for GNN/PyTorch operations  
export OPENBLAS_NUM_THREADS=1

# Combined for dataflow scripts (includes PyTorch stability)
OPENBLAS_NUM_THREADS=1 USE_MANAGED_IDENTITY=false PYTHONPATH=/workspace/azure-maintie-rag python <script>

# For GNN/PyTorch operations (essential for numerical stability)
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/workspace/azure-maintie-rag timeout 120 pytest -m performance
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| PydanticAI Union Type Error | `pip install 'openai==1.98.0'` and use FunctionToolset pattern |
| Azure Authentication Failed | `az login && ./scripts/deployment/sync-env.sh prod` |
| Import Errors | Always use `PYTHONPATH=/workspace/azure-maintie-rag` |
| Pre-commit Domain Bias Failed | Remove hardcoded domains, use measured properties |
| OpenBLAS Threading Issues | Use `OPENBLAS_NUM_THREADS=1` for numerical operations |
| Container Image Build Failures | Use `./scripts/acr-cloud-build.sh` instead of manual docker build |
| Bicep Template Compilation Errors | Ensure container image names match `azure.yaml` configuration |
| Frontend Docker Build Issues | Check frontend/.dockerignore and Dockerfile paths |
| Service Image Name Template Issues | Container images now use hardcoded names for reliability |
| Environment Sync Problems | Run `./scripts/deployment/sync-env.sh <env>` to resync configuration |
| PyTorch/GNN Threading Issues | Always use `OPENBLAS_NUM_THREADS=1` for numerical stability |
| Test Timeouts with Large Models | Use `timeout 120` or `timeout 180` for GNN operations |
| Session Management Issues | Use `make session-report` to view current session, `make clean` to reset |

## 6-Phase Data Pipeline

The system processes data through 6 phases (`scripts/dataflow/`):

| Phase | Purpose | Key Scripts |
|-------|---------|-------------|
| **0 - Cleanup** | Reset Azure services | `00_01_cleanup_all_services.py` |
| **1 - Validation** | Test all 3 agents | `01_0*_validate_*.py` |
| **2 - Ingestion** | Upload docs, create embeddings | `02_02_storage_upload.py`, `02_03_vector_embeddings.py` |
| **3 - Extraction** | Extract entities/relationships | `03_01_basic_entity_extraction.py`, `03_02_graph_storage.py` |
| **4 - Query** | Query analysis & search | `04_01_query_analysis.py` |
| **5 - Integration** | End-to-end pipeline | `05_01_full_pipeline_execution.py` |
| **6 - Advanced** | GNN training, monitoring | `06_01_gnn_training.py` |

**Quick execution:** `make dataflow-full` runs all phases sequentially

## Key Azure Services

The system integrates 9 Azure services:
- **Azure OpenAI**: GPT-4o for all agent intelligence
- **Cognitive Search**: Vector search with 1536D embeddings
- **Cosmos DB**: Graph database with Gremlin API
- **Blob Storage**: Document storage
- **Azure ML**: GNN model training and inference
- **Key Vault**: Secrets management
- **Container Apps**: Hosting backend and frontend
- **Application Insights**: Monitoring and telemetry
- **Managed Identity**: Passwordless authentication

## Technology Stack

- **Backend**: Python 3.11+, PydanticAI 0.6.2, FastAPI, OpenAI 1.98.0
- **Frontend**: React 19.1.0, TypeScript 5.8.3, Vite 7.0.4
- **Testing**: pytest 7.4.0+ with real Azure services (no mocks)
- **Code Quality**: Black (88 char), isort, ESLint 9.30.1, TypeScript strict
- **CI/CD**: GitHub Actions with OIDC, Azure Developer CLI (azd)

## Project Data

- **Real corpus**: 179 Azure AI Language Service documents in `data/raw/azure-ai-services-language-service_output/`
- **Production metrics**: 35-47 seconds per document, 88.4% confidence, 100% success rate

## Azure Deployment & Infrastructure

### Azure Developer CLI (azd) Configuration
- **Config file**: `azure.yaml` - Defines infrastructure, services, and deployment hooks
- **Infrastructure**: `infra/` directory with Bicep templates
  - `main.bicep` - Entry point for infrastructure
  - `modules/` - Modular Bicep templates for different service groups

### Container Image Management (Updated Workflow)
Container images are built and deployed via Azure Container Registry Cloud Build:

```bash
# Automated Cloud Build (Recommended)
./scripts/acr-cloud-build.sh                    # Build images via ACR Cloud Build
./scripts/complete-deployment.sh                # Complete deployment with containers

# Manual Container Registry Operations
az acr build --registry <registry-name> --image azure-maintie-rag/backend-prod:latest .
az acr build --registry <registry-name> --image azure-maintie-rag/frontend-prod:latest ./frontend

# Container Image Names (Current Configuration)
# Backend: azure-maintie-rag/backend-prod:latest
# Frontend: azure-maintie-rag/frontend-prod:latest
# Registry: Retrieved dynamically from azd environment
```

### Environment Synchronization
The system auto-syncs azd environment with backend configuration:
```bash
./scripts/deployment/sync-env.sh [environment]  # Switches and syncs environment
make sync-env                                    # Syncs current azd environment
```

### Deployment Troubleshooting
Based on recent fixes to container image management:
- **Image Name Conflicts**: Use `./scripts/acr-cloud-build.sh` for consistent naming
- **Bicep Template Issues**: Container image names are now hardcoded for reliability
- **Service Template Substitution**: Fixed SERVICE_*_IMAGE_NAME token replacement issues
- **azd URLs Not Displayed**: Run `./scripts/show-deployment-urls.sh` to see frontend/backend URLs
- **Docker Not Available**: azd may fail to package services if Docker isn't installed (containers already deployed via Bicep)

## Session Management & Current Status

### Enterprise Session Management
The Makefile implements sophisticated session tracking:
- Each command creates unique session ID (`YYYYMMDD_HHMMSS`)
- Session reports in `logs/dataflow_execution_*.md`
- Performance metrics in `logs/performance_*.log`
- Cumulative reports in `logs/cumulative_dataflow_report.md`
- Clean log replacement (no accumulation) for enterprise environments

### Current Development Status
Based on git status, active development files:
- `api/endpoints/search.py` - Universal Orchestrator API integration
- `frontend/src/services/api.ts` - Frontend API client updates
- `example_answer_generation.py` - Query generation examples

## Project Structure

```
/workspace/azure-maintie-rag/
├── agents/                      # Multi-agent system (PydanticAI)
│   ├── core/                    # Core infrastructure
│   │   ├── azure_pydantic_provider.py
│   │   ├── universal_models.py
│   │   ├── universal_deps.py
│   │   ├── agent_toolsets.py
│   │   └── centralized_agent1_schema.py
│   ├── domain_intelligence/
│   ├── knowledge_extraction/
│   └── universal_search/
├── infrastructure/              # Azure service clients
│   ├── azure_openai/
│   ├── azure_search/
│   ├── azure_cosmos/
│   ├── azure_storage/
│   ├── azure_ml/
│   └── prompt_workflows/templates/
├── api/                         # FastAPI endpoints
├── frontend/                    # React + TypeScript UI
├── config/                      # Environment configuration
├── scripts/
│   ├── dataflow/               # 6-phase pipeline scripts
│   ├── deployment/             # Deployment utilities
│   └── hooks/                  # Pre-commit hooks
├── infra/                      # Azure Bicep IaC
├── tests/                      # Real Azure service tests
├── data/raw/                   # 179 Azure AI docs
└── azure.yaml                  # Azure Developer CLI config
```

## Git Workflow & Development Process

### Pre-commit Hooks
- **Domain bias check**: `scripts/hooks/pre-commit-domain-bias-check.sh`
  - Enforces zero domain bias philosophy
  - Detects hardcoded domain categories
  - Fails builds with violations

### Code Quality Standards
- Python: Black (88 char line limit), isort
- TypeScript: ESLint 9.30.1, strict mode
- All tests must pass with real Azure services
- No mocks allowed in production code

### Recent Development Focus Areas
Based on recent commits, active development areas include:
- **Container Image Management**: Hardcoded image names for deployment reliability
- **Bicep Template Fixes**: SERVICE_*_IMAGE_NAME token resolution improvements
- **Frontend Docker Optimization**: Better build paths and .dockerignore configuration
- **Azure Container Registry Integration**: Cloud build scripts for consistent image creation
- **Universal Orchestrator API**: Modern agent delegation patterns in FastAPI endpoints
- **Session Management**: Enterprise session tracking with performance metrics

### Development Workflow
```bash
# 1. Feature development
git checkout -b feature/your-feature
# Make changes, ensure PYTHONPATH is set
PYTHONPATH=/workspace/azure-maintie-rag python <your-script>

# 2. Quality checks (must pass)
black . --check && isort . --check-only
./scripts/hooks/pre-commit-domain-bias-check.sh
pytest -m unit

# 3. Test with real Azure services
pytest -m azure_validation
make dataflow-validate

# 4. Container testing (if applicable)
./scripts/acr-cloud-build.sh  # Test image builds
docker-compose up --build     # Test local deployment

# 5. Commit and push
git add .
git commit -m "descriptive message"
git push origin feature/your-feature
```