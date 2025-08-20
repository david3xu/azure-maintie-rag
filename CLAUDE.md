# CLAUDE.md - Development Guide

**Comprehensive development guide for Claude Code when working with the Azure Universal RAG system**

> **📝 Purpose**: This file provides complete development workflows, testing procedures, and architectural guidance for the production-grade multi-agent system.

## 🚀 **Quick Development Setup**

### **Environment Setup (CRITICAL)**

```bash
# Always work from project root
cd /workspace/azure-maintie-rag

# Required environment variables (MUST SET for all Python operations)
export PYTHONPATH=/workspace/azure-maintie-rag    # Required for agent imports
export OPENBLAS_NUM_THREADS=1                     # Required for GNN/PyTorch stability
export USE_MANAGED_IDENTITY=false                 # For local development

# Azure authentication
az login && azd auth login                         # Refresh enterprise tokens
./scripts/deployment/sync-env.sh prod             # Sync Azure environment
```

### **First-Run Validation**

```bash
# System health check
make health                                        # Check Azure services
python -c "from agents.core.universal_models import UniversalDomainAnalysis; print('✅ Core imports working')"

# Agent validation (CRITICAL)
make dataflow-validate                             # Test all 3 PydanticAI agents

# Code quality
black . && isort .                                 # Format code
./scripts/hooks/pre-commit-domain-bias-check.sh   # Check for domain bias
pytest -m unit -x                                  # Run unit tests
```

### **Most Common Development Workflows**

**1. Agent Development & Testing**:

```bash
# Test specific agents
PYTHONPATH=/workspace/azure-maintie-rag python agents/domain_intelligence/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/examples/demo_universal.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/examples/full_workflow_demo.py

# Full agent pipeline testing
make dataflow-full-progress                        # Complete 6-phase testing
```

**2. Azure Service Development**:

```bash
# Infrastructure changes
azd deploy                                         # Deploy infrastructure updates
make deploy-with-data                              # Deploy + populate data

# Service health monitoring
make azure-status                                  # Check service containers
make session-report                                # Performance metrics
```

## 🏗️ **System Architecture Overview**

**Azure Universal RAG**: Production-ready multi-agent system with **zero hardcoded domain bias** philosophy following strict fail-fast principles with real Azure services.

### **🤖 Three Core PydanticAI Agents**

1. **🌍 Domain Intelligence Agent** (`agents/domain_intelligence/agent.py`)

   - **Purpose**: Discovers content characteristics dynamically (no predetermined categories)
   - **Innovation**: LLM-based analysis generates entity types from content (not hardcoded)
   - **Output**: `UniversalDomainAnalysis` with vocabulary complexity, processing config
   - **Key Tools**: `analyze_content_characteristics()`, `predict_entity_types()`, `generate_processing_configuration()`

2. **📚 Knowledge Extraction Agent** (`agents/knowledge_extraction/agent.py`)

   - **Purpose**: Extracts entities/relationships using Agent 1's discoveries
   - **Innovation**: Cached prompt library (~32% performance improvement)
   - **Methods**: LLM + Pattern + Hybrid approaches with Agent 1 → Agent 2 delegation
   - **Storage**: Real Cosmos DB Gremlin integration
   - **Key Tools**: `extract_entities_and_relationships()`, `generate_extraction_prompts()`

3. **🔍 Universal Search Agent** (`agents/universal_search/agent.py`)
   - **Purpose**: Orchestrates **mandatory** tri-modal search: Vector + Graph + GNN
   - **Innovation**: No fallback logic - all three modalities required or system fails
   - **Optimization**: Skips Agent 1 for queries (uses pre-analyzed data)
   - **Key Tools**: `orchestrate_universal_search()`, `search_vector_index()`, `search_knowledge_graph()`

### **🔄 Agent Communication Flow**

```
User Content → Domain Intelligence → Content Analysis + Entity Predictions
                        ↓                               ↓
               Knowledge Extraction ←────────────────┘
                        ↓
               Cosmos DB Graph Storage
                        ↓
User Query → Universal Search (Vector + Graph + GNN) → Unified Results
```

**Key Patterns**:

- **Agent 1 → Agent 2**: Domain analysis guides extraction prompts
- **Agent 1 Caching**: Results cached to avoid duplicate calls
- **Agent 3 Optimization**: Skips Agent 1 for queries (production speed)
  ↓
  Universal Search (Vector + Graph + GNN)
  ↓
  Azure OpenAI Answer Generation

````

## Essential Commands

### Development & Testing
```bash
# Setup and run
make setup                    # Install all dependencies
make dev                      # Start API (8000) + Frontend (5174)
make health                   # Complete system health check

# Testing (REAL Azure services only - NO mocks)
pytest                        # All tests with asyncio_mode=auto
pytest -m unit -x             # Unit tests, stop on first failure
pytest -m integration -vvv    # Integration tests with verbose output
pytest tests/test_agents.py::TestDomainIntelligenceAgent  # Specific test

# 6-Phase Data Pipeline
make dataflow-full            # Execute complete pipeline (cleanup→validate→ingest→extract→query→gnn)
make dataflow-validate        # Phase 1: Validate all 3 agents (quick check)
make dataflow-extract         # Phase 3: Knowledge extraction
make dataflow-query           # Phase 4: Query & search testing
make dataflow-integrate       # Phase 5: Integration testing
make dataflow-advanced        # Phase 6: GNN & monitoring
````

## 🧪 **Comprehensive Testing Infrastructure**

### **6-Phase Testing Pipeline**

The system uses a comprehensive 6-phase testing pipeline as the primary validation method:

**Phase 0: Cleanup** - Reset Azure services to clean state

```bash
make dataflow-cleanup          # Clean all Azure data (preserves infrastructure)
```

**Phase 1: Agent Validation** - Test agent connectivity and imports

```bash
make dataflow-validate         # CRITICAL - Test all 3 agents
# Tests: Agent imports, initialization, basic functionality, Azure service connectivity
```

**Phase 2: Data Ingestion** - Upload real data to Azure services

```bash
make dataflow-ingest           # Upload data/raw/ to Storage, create embeddings, index in Search
```

**Phase 3: Knowledge Extraction** - Test Agent 1 → Agent 2 coordination

```bash
make dataflow-extract          # Test domain analysis → entity extraction → graph storage
```

**Phase 4: Query Pipeline** - Test Agent 3 tri-modal search

```bash
make dataflow-query            # Test Vector + Graph + GNN search (mandatory tri-modal)
```

**Phase 5: Integration Testing** - End-to-end workflow validation

```bash
make dataflow-integrate        # Test complete multi-agent workflows
```

**Phase 6: Advanced Features** - GNN training and monitoring

```bash
make dataflow-advanced         # Test GNN training, monitoring, advanced features
```

### **Testing Philosophy**

💪 **Real Azure Services Only**: No mocks, no fallbacks, no fake patterns
⚡ **Fail-Fast Behavior**: System fails completely on Azure service issues
🎯 **Zero Domain Bias**: Tests validate universal content processing
🔍 **Mandatory Tri-Modal**: Vector + Graph + GNN all required (no degradation)

# Code quality checks (REQUIRED before commits)

black . --check # Check formatting
isort . --check-only # Check import ordering
./scripts/hooks/pre-commit-domain-bias-check.sh # Domain bias detection

````

### Deployment
```bash
# Authentication setup (critical for enterprise environments)
az login && azd auth login
./scripts/deployment/sync-auth.sh validate

# Deployment options
make deploy-with-data         # Full deployment with data pipeline (10-15 min)
make deploy-fast              # Infrastructure only (2-3 min)
azd up                        # Standard Azure deployment

# Cleanup (azd down is broken - use fixed version)
./scripts/azd-down-fixed.sh --force --purge
````

## 🚨 Critical Development Rules

### 1. Zero Domain Bias (ENFORCED by pre-commit hook)

```python
# ❌ WRONG - Hardcoded domain categories
if domain in ["legal", "technical", "medical"]:
    complexity = 0.8

# ✅ CORRECT - Discover from content
complexity = measure_vocabulary_complexity(content)
parameters = adapt_based_on_measured_properties(complexity)
```

### 2. PydanticAI Agent Pattern (REQUIRED)

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from agents.core.universal_deps import UniversalDeps

# Create toolset FIRST (not agent)
toolset = FunctionToolset()

# Register tools on TOOLSET
@toolset.tool  # ✅ CORRECT - use toolset.tool
async def analyze_content(ctx: RunContext[UniversalDeps], content: str):
    return await ctx.deps.azure_openai_client.analyze(content)

# Create agent WITH toolset
agent = Agent[UniversalDeps, OutputModel](
    model=azure_openai_model,
    toolsets=[toolset],  # Pass toolset here
    deps_type=UniversalDeps,
    retries=3
)

# ⚠️ NEVER use @agent.tool - causes Union type errors
```

### 3. Fail-Fast Philosophy

- **NO fallback logic** when services fail
- **NO partial results** - all modalities required
- System **fails completely** rather than degrading gracefully
- This ensures data integrity and predictable behavior

## High-Level Architecture

### Multi-Agent Orchestration (`agents/orchestrator.py`)

- `UniversalOrchestrator` coordinates all three agents
- Implements caching to avoid redundant Domain Intelligence calls
- Manages shared `UniversalDeps` across agents
- Tracks costs and collects evidence for audit trails

### Dependency Injection (`agents/core/universal_deps.py`)

- Centralized Azure service initialization
- Auto-detects environment (Container Apps vs Local)
- Manages credentials via `DefaultAzureCredential` or `ManagedIdentityCredential`
- Lazy initialization of expensive resources

### Universal Models (`agents/core/universal_models.py`)

- Domain-agnostic Pydantic models
- No hardcoded domain types or categories
- Flexible configuration based on discovered characteristics
- Type-safe communication between agents

### Infrastructure Layer (`infrastructure/`)

- Azure service clients with retry logic
- Unified prompt templates in `prompt_workflows/templates/`
- GNN training and inference via Azure ML
- Cost tracking and monitoring utilities

## 6-Phase Data Pipeline

| Phase | Purpose   | Key Script                                            | Success Indicator       |
| ----- | --------- | ----------------------------------------------------- | ----------------------- |
| 0     | Cleanup   | `phase0_cleanup/00_01_cleanup_all_services.py`        | All services empty      |
| 1     | Validate  | `phase1_validation/01_00_basic_agent_connectivity.py` | Agents respond          |
| 2     | Ingest    | `phase2_ingestion/02_02_storage_upload_primary.py`    | Docs uploaded           |
| 3     | Extract   | `phase3_knowledge/03_01_basic_entity_extraction.py`   | Entities in Cosmos      |
| 4     | Query     | `phase4_query/04_01_query_analysis.py`                | Results with confidence |
| 5     | Integrate | `phase5_integration/05_01_full_pipeline_execution.py` | End-to-end works        |
| 6     | Advanced  | `phase6_advanced/06_01_gnn_training.py`               | GNN model deployed      |

## Common Issues & Solutions

### Import Errors

```bash
# Always set PYTHONPATH
export PYTHONPATH=/workspace/azure-maintie-rag

# Test specific imports
python -c "from agents.domain_intelligence.agent import domain_intelligence_agent; print('✅')"
```

### PydanticAI Union Type Error

```bash
# Must use specific OpenAI version
pip install 'openai==1.98.0'

# Check for incorrect patterns
grep -r "@agent\.tool" agents/  # Should return NOTHING
```

### Azure Authentication

```bash
# Full re-authentication
az login && azd auth login
./scripts/deployment/sync-env.sh prod

# Test connection
az account show
```

### Domain Bias Detection

```bash
# Run pre-commit check
./scripts/hooks/pre-commit-domain-bias-check.sh

# Manual scan for violations
grep -r '"technical"\|"legal"\|"medical"' agents/ --include="*.py"
```

## Testing Strategies

### Unit Tests

- Test agent logic in isolation
- Mock Azure services for speed
- Focus on business logic validation

### Integration Tests

- Use REAL Azure services (no mocks)
- Test multi-agent workflows
- Validate end-to-end processing

### Running Single Tests

```bash
# Run specific test class
pytest tests/test_agents.py::TestDomainIntelligenceAgent

# Run specific test method
pytest tests/test_agents.py::TestDomainIntelligenceAgent::test_agent_import

# Run with debugging output
pytest -xvvs tests/test_agents.py
```

## Key Configuration Files

- `pytest.ini` - Test configuration with `asyncio_mode=auto`
- `config/azure_settings.py` - Centralized Azure configuration
- `config/environments/*.env` - Environment-specific settings
- `.env` - Local development overrides
- `azure.yaml` - Azure Developer CLI configuration
- `infra/main.bicep` - Infrastructure as Code entry point

## Azure Services Integration

### Required Services (9 total)

1. **Azure OpenAI** - Text processing (gpt-4o-mini for cost optimization)
2. **Cognitive Search** - Vector search with 1536D embeddings
3. **Cosmos DB** - Graph database with Gremlin API
4. **Blob Storage** - Document storage
5. **Azure ML** - GNN training and inference
6. **Key Vault** - Secret management
7. **Application Insights** - Monitoring
8. **Log Analytics** - Centralized logging
9. **Container Apps** - Hosting (optional)

### Service Authentication

- Uses `DefaultAzureCredential` for local development
- Switches to `ManagedIdentityCredential` in Container Apps
- All secrets stored in Key Vault
- RBAC permissions for all services

## Development Workflow

### Making Changes

1. Create feature branch
2. Make code changes
3. Run formatters: `black . && isort .`
4. Check domain bias: `./scripts/hooks/pre-commit-domain-bias-check.sh`
5. Run unit tests: `pytest -m unit`
6. Test integration: `make dataflow-validate`
7. Commit with descriptive message

### Adding New Features

1. Follow existing patterns in `agents/core/`
2. Use universal models, avoid domain-specific types
3. Implement discovery over classification
4. Add tests for new functionality
5. Update relevant documentation

## Session Management

The system tracks development sessions with comprehensive logging:

```bash
make session-report   # View current session metrics
make clean           # Start fresh session
```

Session files in `logs/`:

- `current_session` - Active session ID
- `dataflow_execution_*.md` - Session reports
- `cumulative_dataflow_report.md` - Historical data

## Critical Versions

- **Python**: 3.11+ (Container Apps requirement)
- **OpenAI**: 1.98.0 (PydanticAI compatibility)
- **PydanticAI**: 0.6.2+ with Azure support
- **pytest**: 7.4.0+ with asyncio_mode=auto
- **React**: 19.1.0 (frontend)
- **TypeScript**: 5.8.3 (frontend)

## Philosophy & Principles

1. **Zero Domain Bias**: System discovers characteristics, never assumes
2. **No Mocks**: All testing uses real Azure services
3. **Fail-Fast**: Complete failure rather than degraded operation
4. **Universal Design**: Adapts to ANY content in `data/raw/`
5. **Mandatory Tri-Modal**: Vector + Graph + GNN always required

## Quick Debugging

```bash
# Check agent imports
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
async def test():
    deps = await get_universal_deps()
    print(f'Services: {list(deps.get_available_services())}')
asyncio.run(test())
"

# Quick agent validation
make dataflow-validate  # Should complete in ~30 seconds
```
