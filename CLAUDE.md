# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

```bash
# Always work from project root
cd /workspace/azure-maintie-rag

# Quick Start
make setup                    # Install all dependencies (backend + frontend)
make dev                      # Start API (port 8000) + Frontend (port 5174)
make health                   # System health check with Azure service status
make deploy-with-data         # Full deployment with automated data pipeline (RECOMMENDED)

# Testing (REAL Azure services, NO mocks)
pytest                        # All tests with asyncio_mode=auto
pytest -m unit                # Agent logic tests
pytest -m integration         # Multi-service integration
pytest -m azure_validation    # Azure service health
pytest -x -vvv                # Stop on first failure, verbose
pytest -k "specific_test"     # Run specific test by pattern

# Code Quality (REQUIRED before commits)
black . --check && isort . --check-only         # Python formatting
cd frontend && npm run lint && npx tsc --noEmit # Frontend checks
./scripts/hooks/pre-commit-domain-bias-check.sh # Domain bias validation (ENFORCED)

# 6-Phase Data Pipeline
make dataflow-full            # Execute all phases (complete pipeline)
make dataflow-validate        # Phase 1: Validate 3 PydanticAI agents (CRITICAL)
make dataflow-extract         # Phase 3: Knowledge extraction
make dataflow-query           # Phase 4: Query & search testing
make dataflow-graceful        # Execute with graceful GNN bootstrap handling

# Direct Script Execution (set environment variables)
OPENBLAS_NUM_THREADS=1 USE_MANAGED_IDENTITY=false PYTHONPATH=/workspace/azure-maintie-rag python <script>

# Azure Deployment
./scripts/deployment/sync-env.sh prod  # Switch environment & sync config
azd up                                  # Deploy Azure infrastructure
azd env set AUTO_POPULATE_DATA true    # Enable automated data pipeline
./scripts/show-deployment-urls.sh      # Get deployment URLs

# Container Building (when needed - usually automatic via azd)
az acr build --registry <acr-name> --image azure-maintie-rag/backend-prod:latest .
az acr build --registry <acr-name> --image azure-maintie-rag/frontend-prod:latest ./frontend
```

## Architecture Overview

**Production-ready multi-agent RAG system** with PydanticAI framework and **zero hardcoded domain bias**.

### Three Core PydanticAI Agents

1. **Domain Intelligence Agent** (`agents/domain_intelligence/agent.py:72-95`)
   - Discovers content characteristics dynamically (no predetermined categories)
   - Outputs: `UniversalDomainAnalysis` with vocabulary complexity, processing config
   - Entry: `run_domain_analysis()`, `domain_intelligence_agent`

2. **Knowledge Extraction Agent** (`agents/knowledge_extraction/agent.py:45-70`)
   - Extracts entities/relationships using Domain Intelligence output
   - Uses unified template: `infrastructure/prompt_workflows/templates/universal_knowledge_extraction.jinja2`
   - Entry: `run_knowledge_extraction()`, `knowledge_extraction_agent`

3. **Universal Search Agent** (`agents/universal_search/agent.py:50-75`)
   - Orchestrates mandatory tri-modal search: Vector + Graph + GNN
   - No fallback - all three modalities required (fail-fast design)
   - Entry: `run_universal_search()`, `universal_search_agent`

### Critical Architecture Files

```python
agents/core/universal_models.py         # Domain-agnostic data structures (1536 lines)
agents/core/agent_toolsets.py           # Centralized FunctionToolset management (REQUIRED pattern)
agents/core/universal_deps.py           # Azure service dependency injection
agents/core/azure_pydantic_provider.py  # Azure OpenAI provider with managed identity
agents/orchestrator.py                  # UniversalOrchestrator for agent coordination
infrastructure/azure_ml/gnn_inference_client.py  # GNN model inference with fail-fast
infrastructure/prompt_workflows/templates/universal_knowledge_extraction.jinja2  # Unified template
```

## Critical Development Rules

### 🚨 Zero Domain Bias (ENFORCED by pre-commit)

```python
# ❌ WRONG - Hardcoded domain categories
if domain in ["legal", "technical", "medical"]:
    complexity = 0.8

# ✅ CORRECT - Discover from content
complexity = measure_vocabulary_complexity(content)
parameters = adapt_based_on_measured_properties(complexity)
```

**Enforcement:** `scripts/hooks/pre-commit-domain-bias-check.sh` fails builds with violations

### 🏗️ PydanticAI Agent Pattern (REQUIRED)

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from agents.core.universal_deps import UniversalDeps

# 1. Create toolset FIRST
toolset = FunctionToolset()

# 2. Register tools on TOOLSET (not agent)
@toolset.tool  # ✅ CORRECT - use toolset.tool
async def analyze_content(ctx: RunContext[UniversalDeps], content: str):
    return await ctx.deps.azure_openai_client.analyze(content)

# 3. Create agent WITH toolset
agent = Agent[UniversalDeps, OutputModel](
    model=azure_openai_model,
    toolsets=[toolset],  # Pass toolset here
    deps_type=UniversalDeps,
    retries=3  # Fail-fast with limited retries
)
```

**⚠️ Using `@agent.tool` causes Union type errors with PydanticAI 0.6.2**

### 🚀 Fail-Fast Philosophy (STRICT ENFORCEMENT)

The system implements strict fail-fast principles:
- **NO fallback logic** when ANY service fails
- **NO fake success** responses ever
- **NO partial operation** - ALL THREE modalities required
- GNN endpoint returns **REAL failures** until model is ready
- System **FAILS COMPLETELY** if Vector, Graph, OR GNN unavailable
- Searches return **ERRORS** until ALL modalities operational
- This is **CORRECT behavior** - we fix issues, not bypass them

## Environment Variables

```bash
# Required for all Python scripts
export PYTHONPATH=/workspace/azure-maintie-rag

# Azure authentication (local development)
export USE_MANAGED_IDENTITY=false

# Numerical stability for GNN/PyTorch
export OPENBLAS_NUM_THREADS=1

# Combined for dataflow scripts
OPENBLAS_NUM_THREADS=1 USE_MANAGED_IDENTITY=false PYTHONPATH=/workspace/azure-maintie-rag python <script>
```

## 6-Phase Data Pipeline

| Phase | Purpose | Key Scripts | Success Criteria |
|-------|---------|-------------|------------------|
| **0 - Cleanup** | Clean Azure data (preserves infrastructure) | `00_01_cleanup_all_services.py` | 0 documents/entities |
| **1 - Validation** | Test 3 PydanticAI agents (CRITICAL) | `01_0*_validate_*.py` | All agents respond |
| **2 - Ingestion** | Upload all docs from data/raw | `02_02_storage_upload.py` | All docs indexed |
| **3 - Extraction** | Build knowledge graph | `03_01_basic_entity_extraction.py` | 13+ entities, 8+ relationships |
| **4 - Query** | Test tri-modal search | `04_01_query_analysis.py` | 3+ results, 80%+ confidence |
| **5 - Integration** | End-to-end validation | `05_01_full_pipeline_execution.py` | Complete pipeline works |
| **6 - Advanced** | GNN training + async bootstrap | `06_11_gnn_async_bootstrap.py` | Model deployed to Azure ML |

**Data source:** `data/raw/` (Universal system adapts to any documents placed here)
**Quick run:** `make dataflow-full` or `make dataflow-graceful` (with GNN bootstrap)

## Testing Strategy

**All tests use REAL Azure services (no mocks)**:

```bash
# Test execution with asyncio_mode=auto (pytest.ini:20)
pytest                        # All tests
pytest -m unit                # Agent logic
pytest -m integration         # Multi-service integration  
pytest -m azure_validation    # Service health
pytest -m performance         # Performance tests
pytest -x -vvv                # Debug mode (stop on first failure)

# Performance testing with timeout controls
OPENBLAS_NUM_THREADS=1 timeout 120 pytest -m performance -v

# Test specific dataflow phase
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_01_validate_domain_intelligence.py

# Run single test
pytest tests/test_agents.py::TestDomainIntelligenceAgent::test_agent_import -v
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| PydanticAI Union Type Error | `pip install 'openai==1.98.0'` + use FunctionToolset pattern |
| Azure Authentication Failed | `az login && ./scripts/deployment/sync-env.sh prod` |
| Import Errors | Always use `PYTHONPATH=/workspace/azure-maintie-rag` |
| Domain Bias Failed | Remove hardcoded domains, use measured properties |
| Agent Validation Failures | Check `agents/*/agent.py` imports individually |
| Knowledge Extraction Issues | Verify unified template in `infrastructure/prompt_workflows/templates/` |
| Search Returns 0 Results | Run `make check-data` to verify ingestion |
| Test Timeouts | Use `timeout 120` and `OPENBLAS_NUM_THREADS=1` |
| GNN Training Fails | Check `06_12_check_async_status.py` for async bootstrap status |

## Session Management

Enterprise session tracking with clean log replacement (Makefile:14-60):
- Unique session ID per command (`YYYYMMDD_HHMMSS`)
- Session reports: `logs/dataflow_execution_*.md`
- Performance metrics: `logs/performance_*.log`
- Cumulative report: `logs/cumulative_dataflow_report.md`

```bash
make session-report  # View current session
make clean          # Clean session (preserve cumulative)
make health         # Health check with session tracking
```

## Development Workflow

```bash
# 1. Setup environment
cd /workspace/azure-maintie-rag
./scripts/deployment/sync-env.sh prod
export PYTHONPATH=/workspace/azure-maintie-rag

# 2. Validate agents (CRITICAL first step)
make dataflow-validate

# 3. Make changes and test
black . && isort .  # Format code
./scripts/hooks/pre-commit-domain-bias-check.sh
pytest -m unit

# 4. Test with real data
make dataflow-extract  # Test knowledge extraction
make dataflow-query    # Test search

# 5. Full validation
make dataflow-full  # Or dataflow-graceful for GNN bootstrap

# 6. Check results
make session-report
make health
```

## Azure Deployment Details

### Automated Deployment with Data Pipeline
```bash
make deploy-with-data  # Recommended: Infrastructure + complete data pipeline
# OR
azd env set AUTO_POPULATE_DATA true && azd up
```

This automatically:
1. Deploys 9 Azure services with RBAC
2. Cleans existing Azure data
3. Validates all 3 PydanticAI agents
4. Uploads documents & creates embeddings
5. Runs Agent 1 to analyze all docs (one-time)
6. Creates GNN bootstrap endpoint (returns REAL failures initially)
7. Starts async GNN training in background
8. **IMPORTANT**: System will FAIL all searches until GNN ready (NO FALLBACK)

### Infrastructure Only
```bash
make deploy-infrastructure-only  # Deploy without data
# Then run: make dataflow-full
```

### Environment Sync
```bash
./scripts/deployment/sync-env.sh prod     # Switch to production
./scripts/deployment/sync-env.sh staging  # Switch to staging
make sync-env                              # Sync current environment
```

### Live System Access
```bash
# After deployment completes
./scripts/show-deployment-urls.sh

# Frontend Chat Interface
https://ca-frontend-maintie-rag-prod.<region>.azurecontainerapps.io

# Backend API
https://ca-backend-maintie-rag-prod.<region>.azurecontainerapps.io/health
https://ca-backend-maintie-rag-prod.<region>.azurecontainerapps.io/docs
https://ca-backend-maintie-rag-prod.<region>.azurecontainerapps.io/api/v1/rag
```

## Key Azure Services (9 integrated)

- **Azure OpenAI**: GPT-4o for all agent intelligence
- **Cognitive Search**: Vector search with 1536D embeddings
- **Cosmos DB**: Graph database with Gremlin API
- **Blob Storage**: Document storage
- **Azure ML**: GNN model training/inference (with async bootstrap)
- **Key Vault**: Secrets management
- **Container Apps**: Backend/frontend hosting
- **Application Insights**: Monitoring
- **Managed Identity**: Passwordless authentication

## Technology Stack

- **Backend**: Python 3.11+, PydanticAI 0.6.2, FastAPI, OpenAI 1.98.0 (critical version)
- **Frontend**: React 19.1.0, TypeScript 5.8.3, Vite 7.0.4
- **Testing**: pytest 7.4.0+ with real Azure services (no mocks), asyncio_mode=auto
- **Code Quality**: Black (88 char), isort, ESLint 9.30.1
- **CI/CD**: GitHub Actions with OIDC, Azure Developer CLI (azd), Bicep IaC

## Project Data & Performance

- **Universal corpus**: Adapts to any documents in `data/raw/` directory
- **Processing**: 35-47 seconds per document (varies by content complexity)
- **Confidence**: 88.4% average extraction confidence
- **Success rate**: 100% (all files processed successfully)
- **Production score**: 95/100

## Important Notes

- **NO MOCKS**: All testing uses real Azure services for validation
- **NO DOMAIN BIAS**: System discovers characteristics, doesn't categorize
- **MANDATORY TRI-MODAL**: Vector + Graph + GNN all required (no fallback)
- **FAIL-FAST**: No fake success, actual failures until services ready
- **UNIVERSAL DATA**: Processes any documents in data/raw/, no fake/sample data
- **AGENT PATTERN**: Must use FunctionToolset, not @agent.tool
- **CLEAN SESSIONS**: Makefile replaces logs, doesn't accumulate
- **ASYNC BOOTSTRAP**: GNN deployment uses async bootstrap for reproducibility