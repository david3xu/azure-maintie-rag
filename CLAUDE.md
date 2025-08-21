# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Azure Universal RAG**: Production-ready multi-agent system with zero hardcoded domain bias for universal content processing.

- **Core Philosophy**: Discover characteristics from data, never assume domains
- **Framework**: PydanticAI 0.6.2 with 3 specialized agents  
- **Testing**: Real Azure services only (no mocks), fail-fast principles
- **Search**: Mandatory tri-modal (Vector + Graph + GNN) - no fallback

## Essential Commands

### Setup & Development
```bash
# Initial setup
cd /workspace/azure-maintie-rag
export PYTHONPATH=/workspace/azure-maintie-rag    # Required for imports
export OPENBLAS_NUM_THREADS=1                     # GNN stability
export USE_MANAGED_IDENTITY=false                 # Local auth

# Development
make health                    # System health check
make dev-backend              # Start API (port 8000)
make dev-frontend             # Start frontend (port 5174)
```

### Testing
```bash
# Run tests (uses real Azure services)
pytest                        # All tests with asyncio_mode=auto
pytest -m unit -x             # Unit tests, stop on first failure  
pytest -m integration         # Integration tests
pytest tests/test_agents.py::TestDomainIntelligenceAgent::test_agent_import  # Single test

# 6-Phase validation pipeline
make dataflow-validate        # Phase 1: Quick agent validation
make dataflow-full           # All phases: cleanup→validate→ingest→extract→query→gnn
```

### Deployment
```bash
# Authentication (required for enterprise)
az login && azd auth login

# Deploy with data
make deploy-with-data         # Full deployment + data pipeline

# Deploy infrastructure only
azd deploy                    # Fast container deployment
```

## Architecture

### Three PydanticAI Agents

1. **Domain Intelligence Agent** (`agents/domain_intelligence/agent.py`)
   - Discovers content characteristics without assumptions
   - Outputs: `UniversalDomainAnalysis` with vocabulary complexity, processing config
   - Key pattern: Measures properties → generates domain signature dynamically

2. **Knowledge Extraction Agent** (`agents/knowledge_extraction/agent.py`)
   - Extracts entities/relationships using Agent 1's analysis
   - Uses cached prompt library (~32% performance improvement)
   - Agent delegation: Calls Agent 1 first, uses output to guide extraction

3. **Universal Search Agent** (`agents/universal_search/agent.py`)
   - Orchestrates mandatory tri-modal search
   - No fallback: Vector + Graph + GNN all required
   - Optimization: Skips Agent 1 for queries (uses pre-analyzed data)

### Agent Communication Flow
```
Content → Domain Intelligence → Analysis + Entity Predictions
                ↓                           ↓
         Knowledge Extraction ←─────────────┘
                ↓
         Cosmos DB Graph
                ↓
Query → Universal Search (Vector + Graph + GNN) → Results
```

### Key Files & Patterns

**Core Models**: `agents/core/universal_models.py`
- Domain-agnostic Pydantic models
- No hardcoded categories or thresholds

**Dependency Injection**: `agents/core/universal_deps.py`
- Centralized Azure service initialization
- Lazy loading, environment detection

**Orchestrator**: `agents/orchestrator.py`
- `UniversalOrchestrator` coordinates all agents
- Implements caching to avoid redundant calls
- Manages shared dependencies

## Critical Development Rules

### 1. PydanticAI Agent Pattern (REQUIRED)
```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

# Create toolset FIRST
toolset = FunctionToolset()

# Register tools on TOOLSET (not agent)
@toolset.tool  # ✅ CORRECT
async def analyze_content(ctx: RunContext[UniversalDeps], content: str):
    return await ctx.deps.azure_openai_client.analyze(content)

# Create agent WITH toolset
agent = Agent[UniversalDeps, OutputModel](
    model=azure_openai_model,
    toolsets=[toolset],  # Pass toolset here
    deps_type=UniversalDeps
)

# ⚠️ NEVER use @agent.tool - causes Union type errors
```

### 2. Zero Domain Bias (Enforced by pre-commit)
```python
# ❌ WRONG - Hardcoded domains
if domain in ["legal", "technical", "medical"]:
    complexity = 0.8

# ✅ CORRECT - Discover from content
complexity = measure_vocabulary_complexity(content)
parameters = adapt_based_on_measured_properties(complexity)
```

Pre-commit check: `./scripts/hooks/pre-commit-domain-bias-check.sh`

### 3. Testing Philosophy
- **Real Azure services only** - no mocks
- **Fail-fast** - complete failure over degraded operation
- **Mandatory tri-modal** - all search modalities required

## 6-Phase Data Pipeline

Execute with: `make dataflow-full`

| Phase | Purpose      | Key Script                                           |
|-------|-------------|-----------------------------------------------------|
| 0     | Cleanup      | `phase0_cleanup/00_01_cleanup_azure_data.py`        |
| 1     | Validation   | `phase1_validation/01_00_basic_agent_connectivity.py`|
| 2     | Ingestion    | `phase2_ingestion/02_02_storage_upload_primary.py`  |
| 3     | Extraction   | `phase3_knowledge/03_01_basic_entity_extraction.py` |
| 4     | Query        | `phase4_query/04_01_query_analysis.py`              |
| 5     | Integration  | `phase5_integration/05_01_full_pipeline_execution.py`|
| 6     | Advanced     | `phase6_advanced/06_01_gnn_training.py`            |

## Azure Services

9 required services, all using RBAC authentication:
- Azure OpenAI (gpt-4o-mini)
- Cognitive Search (vector search)
- Cosmos DB (Gremlin graph)
- Blob Storage
- Azure ML (GNN)
- Key Vault, App Insights, Log Analytics, Container Apps

## Common Issues

### Import Errors
```bash
export PYTHONPATH=/workspace/azure-maintie-rag
python -c "from agents.core.universal_models import UniversalDomainAnalysis; print('✅')"
```

### PydanticAI Union Type Error
```bash
pip install 'openai==1.98.0'  # Required version
grep -r "@agent\.tool" agents/  # Should return nothing
```

### Azure Authentication Failed
```bash
az login && azd auth login
./scripts/deployment/sync-env.sh prod
```

### Domain Bias Detected
Remove hardcoded domains ("technical", "legal", etc.) and fixed thresholds.
Use discovery patterns instead of classification.

## Code Quality

```bash
# Format before commit
black . && isort .

# Check domain bias
./scripts/hooks/pre-commit-domain-bias-check.sh

# Full quality check
black . --check && isort . --check-only && pytest -m unit
```

## Frontend Development

```bash
cd frontend
npm install               # Install dependencies
npm run dev              # Start dev server (port 5173)
npm run build            # Production build
npm run lint             # Run ESLint
```

Tech stack: React 19.1.0, TypeScript 5.8.3, Vite 7.0.4

## Key Environment Variables

```bash
PYTHONPATH=/workspace/azure-maintie-rag         # Always required
OPENBLAS_NUM_THREADS=1                         # GNN/PyTorch stability
USE_MANAGED_IDENTITY=false                     # Local development
AZURE_OPENAI_ENDPOINT                          # From Azure portal
AZURE_SEARCH_ENDPOINT                          # From Azure portal
AZURE_COSMOS_ENDPOINT                          # From Azure portal
```

## Quick Validation

```bash
# Test agent imports
python -c "
from agents.domain_intelligence.agent import domain_intelligence_agent
from agents.knowledge_extraction.agent import knowledge_extraction_agent
from agents.universal_search.agent import universal_search_agent
print('✅ All agents import successfully')
"

# Test Azure connectivity
python -c "
import asyncio
from agents.core.universal_deps import get_universal_deps
asyncio.run(get_universal_deps())
print('✅ Azure services connected')
"

# Quick pipeline test
make dataflow-validate  # ~30 seconds
```

## Production URLs (after deployment)

- Frontend: `https://ca-frontend-maintie-rag-prod.<region>.azurecontainerapps.io`
- Backend API: `https://ca-backend-maintie-rag-prod.<region>.azurecontainerapps.io/health`
- API Docs: `https://ca-backend-maintie-rag-prod.<region>.azurecontainerapps.io/docs`