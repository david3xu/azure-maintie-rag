# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 Critical Quick Start

```bash
# Always work from project root
cd /workspace/azure-maintie-rag

# Required environment setup
export PYTHONPATH=/workspace/azure-maintie-rag    # Critical for all Python operations
export OPENBLAS_NUM_THREADS=1                     # Required for GNN/PyTorch stability

# First-run health check (always do this)
make health                                        # Verify system and Azure services
python -c "from agents.core.universal_models import UniversalDomainAnalysis; print('✅ Core imports working')"

# Most common development workflow
make dataflow-validate                             # Test all 3 PydanticAI agents (30s)
# Make your code changes, then:
black . && isort .                                 # Format code
./scripts/hooks/pre-commit-domain-bias-check.sh   # Check for domain bias
pytest -m unit -x                                  # Run unit tests
```

## Architecture Overview

**Azure Universal RAG**: Production-ready multi-agent system with PydanticAI framework and **zero hardcoded domain bias**.

### Three Core PydanticAI Agents

1. **Domain Intelligence Agent** (`agents/domain_intelligence/agent.py`)
   - Discovers content characteristics dynamically (no predetermined categories)
   - Entry point: `run_domain_analysis()`, `domain_intelligence_agent`

2. **Knowledge Extraction Agent** (`agents/knowledge_extraction/agent.py`)
   - Extracts entities/relationships using Domain Intelligence output
   - Entry point: `run_knowledge_extraction()`, `knowledge_extraction_agent`

3. **Universal Search Agent** (`agents/universal_search/agent.py`)
   - Orchestrates mandatory tri-modal search: Vector + Graph + GNN (no fallback)
   - Entry point: `run_universal_search()`, `universal_search_agent`

## Essential Commands

### Development & Testing
```bash
# Local development
make setup                    # Install all dependencies
make dev                      # Start API (8000) + Frontend (5174)
make health                   # System health check

# Testing (REAL Azure services, NO mocks)
pytest                        # All tests with asyncio_mode=auto
pytest -m unit                # Agent logic tests
pytest -m integration         # Multi-service integration
pytest -x -vvv                # Debug mode (stop on first failure)

# 6-Phase Data Pipeline
make dataflow-full            # Execute all phases (complete pipeline)
make dataflow-validate        # Phase 1: Validate agents (CRITICAL)
make dataflow-extract         # Phase 3: Knowledge extraction
make dataflow-query           # Phase 4: Query & search testing

# Code quality (REQUIRED before commits)
black . --check && isort . --check-only
./scripts/hooks/pre-commit-domain-bias-check.sh
```

### Deployment
```bash
# Authentication (Enterprise/University environments)
az login && azd auth login
./scripts/deployment/sync-auth.sh validate

# Production deployment (RECOMMENDED)
make deploy-with-data         # Full deployment with automated data pipeline

# Fast deployment (infrastructure only)
make deploy-fast              # 2-3 minutes, no data population

# Manual cleanup
./scripts/azd-down-fixed.sh --force --purge
```

## 🚨 Critical Development Rules

### Zero Domain Bias (ENFORCED by pre-commit)
```python
# ❌ WRONG - Hardcoded domain categories
if domain in ["legal", "technical", "medical"]:
    complexity = 0.8

# ✅ CORRECT - Discover from content
complexity = measure_vocabulary_complexity(content)
parameters = adapt_based_on_measured_properties(complexity)
```

### PydanticAI Agent Pattern (REQUIRED)
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
    retries=3
)
```

**⚠️ Using `@agent.tool` causes Union type errors with PydanticAI 0.6.2**

### Fail-Fast Philosophy
- **NO fallback logic** when ANY service fails
- **NO fake success** responses
- **NO partial operation** - ALL THREE modalities required for search
- System **FAILS COMPLETELY** if Vector, Graph, OR GNN unavailable

## 6-Phase Data Pipeline

| Phase | Purpose | Entry Script | Success Criteria |
|-------|---------|--------------|------------------|
| **0** | Clean Azure data | `scripts/dataflow/phase0_cleanup/00_01_cleanup_all_services.py` | 0 documents/entities |
| **1** | Validate agents | `scripts/dataflow/phase1_validation/01_00_basic_agent_connectivity.py` | All agents respond |
| **2** | Ingest data | `scripts/dataflow/phase2_ingestion/02_02_storage_upload_primary.py` | All docs indexed |
| **3** | Extract knowledge | `scripts/dataflow/phase3_knowledge/03_01_basic_entity_extraction.py` | 13+ entities, 8+ relationships |
| **4** | Query pipeline | `scripts/dataflow/phase4_query/04_01_query_analysis.py` | 3+ results, 80%+ confidence |
| **5** | Integration test | `scripts/dataflow/phase5_integration/05_01_full_pipeline_execution.py` | Complete pipeline works |
| **6** | Advanced (GNN) | `scripts/dataflow/phase6_advanced/06_11_gnn_async_bootstrap.py` | Model deployed |

## Common Issues & Solutions

### Import/Authentication Issues
```bash
# Import errors
export PYTHONPATH=/workspace/azure-maintie-rag
export OPENBLAS_NUM_THREADS=1

# Azure authentication failed
az login && azd auth login
./scripts/deployment/sync-env.sh prod

# Test individual agent imports
PYTHONPATH=/workspace/azure-maintie-rag python -c "from agents.domain_intelligence.agent import domain_intelligence_agent; print('✅')"
```

### PydanticAI Issues
```bash
# Union type error fix
pip install 'openai==1.98.0'  # Critical version

# Check for incorrect patterns
grep -r "@agent\.tool" agents/  # Should return NO results
grep -r "@.*_toolset\.tool" agents/  # Should find correct pattern
```

### Domain Bias Detection
```bash
# Run pre-commit check
./scripts/hooks/pre-commit-domain-bias-check.sh

# Manual scan
grep -r "legal\|technical\|medical\|financial" agents/ --include="*.py" | grep -v "# OK:"
```

## Key Architecture Files

### Core Agent Framework
- `agents/core/universal_models.py` - Domain-agnostic Pydantic models
- `agents/core/agent_toolsets.py` - Centralized FunctionToolset management
- `agents/core/universal_deps.py` - Azure service dependency injection
- `agents/core/azure_pydantic_provider.py` - Azure OpenAI provider
- `agents/orchestrator.py` - Multi-agent coordination

### Infrastructure
- `infrastructure/prompt_workflows/templates/` - Unified extraction templates
- `infrastructure/azure_ml/gnn_inference_client.py` - GNN model inference
- `config/azure_settings.py` - Centralized Azure configuration

### Testing & Quality
- `pytest.ini` - asyncio_mode=auto for PydanticAI agents
- `scripts/hooks/pre-commit-domain-bias-check.sh` - Domain bias enforcement

## Session Management

Enterprise session tracking with clean log replacement:
```bash
make session-report  # View current session
make clean          # Clean session, preserve cumulative
make health         # Complete health check with reporting
```

Session files:
- `logs/current_session` - Active session ID
- `logs/dataflow_execution_<SESSION_ID>.md` - Session report
- `logs/cumulative_dataflow_report.md` - Historical data

## Technology Stack

- **Backend**: Python 3.11+, PydanticAI 0.6.2, FastAPI, OpenAI 1.98.0 (critical)
- **Frontend**: React 19.1.0, TypeScript 5.8.3, Vite 7.0.4
- **Testing**: pytest 7.4.0+ with real Azure services (no mocks)
- **Azure Services**: 9 integrated services (OpenAI, Search, Cosmos DB, Storage, ML, etc.)
- **CI/CD**: GitHub Actions with OIDC, Azure Developer CLI (azd), Bicep IaC

## Infrastructure Details

### Azure Resources (Cost-Optimized)
- **Azure OpenAI**: gpt-4.1-mini (cost-optimized)
- **Cognitive Search**: BASIC tier (avoids quota conflicts)
- **Cosmos DB**: Serverless with Gremlin API
- **Container Apps**: Scale-to-zero capability
- **Azure ML**: CPU-only compute ($0.168/hour)

### Naming Pattern (Bicep)
```bicep
name: 'prefix-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id, resourcePrefix, environmentName)}'
```

## Project Philosophy

- **Zero Domain Bias**: System discovers characteristics, doesn't categorize
- **No Mocks**: All testing uses real Azure services
- **Fail-Fast**: Complete failure rather than fallback logic
- **Universal Design**: Adapts to any content in `data/raw/`
- **Tri-Modal Search**: Mandatory Vector + Graph + GNN (no fallback)

## Critical Versions

- **OpenAI**: Must be 1.98.0 (PydanticAI compatibility)
- **PydanticAI**: 0.6.2+ with Azure support
- **Pytest**: 7.4.0+ with asyncio_mode=auto
- **Python**: 3.11+ (Azure Container Apps requirement)