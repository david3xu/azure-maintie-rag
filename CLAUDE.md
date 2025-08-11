# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

```bash
# Always work from project root
cd /workspace/azure-maintie-rag

# Development (Complete workflow)
make setup                  # Full project setup (backend + frontend)  
make dev                    # Start API (8000) + Frontend (5174)
make health                 # Full system health check with Azure status
make clean                  # Clean sessions with log replacement

# Testing (Real Azure services - no mocks)
pytest                      # All tests with automatic asyncio handling
pytest -m unit             # Unit tests for agent logic
pytest -m integration      # Integration tests with Azure services
pytest -k "test_name"      # Run specific test pattern
pytest -x -vvv             # Stop on first failure with maximum verbosity

# Code Quality (MUST pass before commits)
black . --check && isort . --check-only              # Python formatting
cd frontend && npm run lint && npx tsc --noEmit      # Frontend checks
./scripts/hooks/pre-commit-domain-bias-check.sh      # Domain bias check (CRITICAL)

# 6-Phase Dataflow Pipeline (Production-ready with real data)
make dataflow-full          # Execute all 6 phases: cleanup → validate → ingest → extract → integrate → query → advanced
make dataflow-validate      # Phase 1: Validate all 3 PydanticAI agents
make dataflow-extract       # Phase 3: Knowledge extraction with unified templates

# Agent Testing (requires PYTHONPATH)
PYTHONPATH=/workspace/azure-maintie-rag python agents/domain_intelligence/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/knowledge_extraction/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/universal_search/agent.py

# Direct Dataflow Phase Execution
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_01_validate_domain_intelligence.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_02_validate_knowledge_extraction.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_03_validate_universal_search.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase5_integration/05_01_full_pipeline_execution.py

# Azure Deployment & Environment Management
./scripts/deployment/sync-env.sh prod      # Switch to production & sync backend
./scripts/deployment/sync-env.sh staging   # Switch to staging & sync backend  
./scripts/deployment/sync-env.sh development  # Switch to development & sync backend
azd up                                     # Deploy complete Azure infrastructure (9 services)
```

## Architecture Overview

**Azure Universal RAG System** - Production multi-agent platform using PydanticAI framework with **zero hardcoded domain bias**.

### Core Components

**Three PydanticAI Agents:**
1. **Domain Intelligence** (`agents/domain_intelligence/agent.py`) - Discovers content characteristics dynamically
2. **Knowledge Extraction** (`agents/knowledge_extraction/agent.py`) - Extracts entities and relationships
3. **Universal Search** (`agents/universal_search/agent.py`) - Tri-modal search (Vector + Graph + GNN)

**Key Architecture Files:**
- `agents/core/universal_models.py` - Domain-agnostic data structures (1,536 lines)
- `agents/core/agent_toolsets.py` - Centralized FunctionToolset management
- `agents/core/universal_deps.py` - Azure service dependency injection
- `agents/core/azure_pydantic_provider.py` - Azure OpenAI model provider
- `agents/core/centralized_agent1_schema.py` - Minimal essential fields with usage documentation (318 lines)
- `infrastructure/prompt_workflows/templates/universal_knowledge_extraction.jinja2` - Unified entity+relationship extraction template
- `infrastructure/prompt_workflows/prompt_workflow_orchestrator.py` - Template orchestration with Agent 1 variables
- `infrastructure/` - Azure service clients with unified prompt workflow system

## Critical Development Rules

### 🚨 Universal RAG Philosophy (ENFORCED BY PRE-COMMIT)

**NEVER hardcode domain assumptions:**
```python
# ❌ WRONG - Domain categories
if domain == "legal" or domain == "technical":
    complexity = 0.8

# ✅ CORRECT - Discover from content
complexity = measure_vocabulary_complexity(content)
parameters = adapt_based_on_measured_properties(complexity)
```

The pre-commit hook `scripts/hooks/pre-commit-domain-bias-check.sh` will fail builds with domain bias.

### 🏗️ PydanticAI Agent Pattern (REQUIRED)

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from agents.core.universal_deps import UniversalDeps

# Create toolset FIRST
toolset = FunctionToolset()

# Create agent WITH toolset
agent = Agent[UniversalDeps, OutputModel](
    model=azure_openai_model,
    toolsets=[toolset],  # Pass toolset here
    deps_type=UniversalDeps,
    retries=3
)

# Register tools on TOOLSET (not agent)
@toolset.tool  # ✅ Use toolset.tool
async def analyze_content(ctx: RunContext[UniversalDeps], content: str):
    return await ctx.deps.azure_openai_client.analyze(content)
```

**⚠️ Using `@agent.tool` causes Union type errors with PydanticAI 0.6.2**

### 🔒 Azure Integration

- **Authentication**: Use `DefaultAzureCredential` only (no API keys)
- **Testing**: Real Azure services only (no mocks)
- **Dependencies**: Access via `agents/core/universal_deps.py`
- **Environment**: Sync with `./scripts/deployment/sync-env.sh <env>` before deployment

## Testing Strategy

All tests use **real Azure services** (see `pytest.ini:22-28`). No mocks are used for comprehensive validation:

```bash
# Run all tests
pytest                      # Includes unit, integration, and azure_validation markers

# Specific test categories  
pytest -m unit              # Agent logic and configuration tests
pytest -m integration       # Multi-service integration tests
pytest -m azure_validation  # Azure service health and authentication
pytest -m performance       # Performance and SLA compliance tests

# Specific agent test
PYTHONPATH=/workspace/azure-maintie-rag pytest tests/test_agents.py::TestDomainIntelligenceAgent -v

# Single test with maximum detail
PYTHONPATH=/workspace/azure-maintie-rag pytest tests/test_agents.py::TestKnowledgeExtractionAgent::test_agent_import -v --tb=long

# Debug mode for agent development
export DEBUG=1
PYTHONPATH=/workspace/azure-maintie-rag python agents/domain_intelligence/agent.py

# Performance testing with timeout controls
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/workspace/azure-maintie-rag timeout 60 pytest -m performance -v

# Test specific dataflow scripts directly
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_00_basic_agent_connectivity.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase2_ingestion/02_00_validate_phase2_prerequisites.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_00_validate_phase3_prerequisites.py
```

## Common Issues & Solutions

### PydanticAI Union Type Error
```bash
# Fix: Downgrade OpenAI
pip install 'openai==1.98.0'
# Ensure using FunctionToolset pattern (see above)
```

### Azure Authentication Failed
```bash
az login
az account show
./scripts/deployment/sync-env.sh prod
```

### Import Errors
```bash
# Always use PYTHONPATH
PYTHONPATH=/workspace/azure-maintie-rag python <script>
```

### Pre-commit Domain Bias Failed
- Remove hardcoded domain categories ("technical", "legal", etc.)
- Replace fixed thresholds with measured properties
- Use discovery patterns, not classification

### Unified Template System (Current Architecture)
The system uses a unified template approach with centralized Agent 1 schema:
```bash
# Multi-step Phase 3 knowledge extraction (cleaned up architecture)
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_00_validate_phase3_prerequisites.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_01_basic_entity_extraction.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_02_graph_storage.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_03_verification.py
```

**Key Components:**
- `agents/core/centralized_agent1_schema.py:Agent1TemplateMapping` - Extracts template variables from Agent 1 output  
- `infrastructure/prompt_workflows/templates/universal_knowledge_extraction.jinja2` - Single template for both entities and relationships
- Template variables: `content_signature`, `discovered_entity_types`, `key_content_terms`, etc.

### Testing Directory Structure
The project uses validation scripts instead of traditional unit tests:
```bash
# Instead of pytest tests/test_agents.py, use:
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_01_validate_domain_intelligence.py

# Test markers are defined in pytest.ini but validation is script-based
pytest -m unit                # Uses pytest.ini markers for any future unit tests
pytest -m azure_validation    # Azure service health validation
pytest -m performance        # Performance and SLA compliance tests
```

## Session Management

The Makefile implements enterprise session tracking with clean log replacement:
- Each command creates unique session ID (`YYYYMMDD_HHMMSS`)
- Logs in `logs/session_report.md` (replaced each session, no accumulation)
- Performance metrics in `logs/performance.log`
- Azure status in `logs/azure_status.log`
- View with `make session-report`
- Clean with `make clean`

**Enterprise Session Features**:
- ✅ Clean log replacement prevents disk accumulation
- ✅ Comprehensive performance monitoring (memory, disk, processes)
- ✅ Azure service health checks integrated into every operation
- ✅ Session duration tracking and resource usage metrics

## Data Pipeline Scripts

Located in `scripts/dataflow/` with **6-phase execution structure**:

**Phase 0 - Cleanup**:
- `phase0_cleanup/00_01_cleanup_all_services.py` - Clean all Azure services
- `phase0_cleanup/00_03_verify_clean_state.py` - Verify clean state

**Phase 1 - Agent Validation** (Critical for Production):
- `phase1_validation/01_01_validate_domain_intelligence.py` - Test Domain Intelligence Agent
- `phase1_validation/01_02_validate_knowledge_extraction.py` - Test Knowledge Extraction Agent
- `phase1_validation/01_03_validate_universal_search.py` - Test Universal Search Agent

**Phase 2 - Data Ingestion**:
- `phase2_ingestion/02_02_storage_upload_primary.py` - Upload documents to Azure Storage
- `phase2_ingestion/02_03_vector_embeddings.py` - Create vector embeddings
- `phase2_ingestion/02_04_search_indexing.py` - Index in Azure Cognitive Search

**Phase 3 - Knowledge Extraction** (Streamlined with unified templates):
- `phase3_knowledge/03_02_knowledge_extraction.py` - Extract entities and relationships with unified template
- `phase3_knowledge/03_02_simple_extraction.py` - Simplified extraction for testing
- `phase3_knowledge/03_02_test_unified_template.py` - Test unified template system
- `phase3_knowledge/03_01_test_agent1_template_vars.py` - Test Agent 1 template variables
- `phase3_knowledge/03_03_simple_storage.py` - Simplified Cosmos DB storage
- `phase3_knowledge/03_04_simple_graph.py` - Simplified graph construction

**Phase 4 - Query Pipeline**:
- `phase4_query/04_01_query_analysis.py` - Query analysis and processing
- `phase4_query/04_02_universal_search_demo.py` - Universal search demonstration
- `phase4_query/04_06_complete_query_pipeline.py` - End-to-end query processing

**Phase 5 - Integration & Workflow**:
- `phase5_integration/05_01_full_pipeline_execution.py` - Complete pipeline test
- `phase5_integration/05_03_query_generation_showcase.py` - Query generation examples

**Phase 6 - Advanced Features**:
- `phase6_advanced/06_01_gnn_training.py` - GNN model training
- `phase6_advanced/06_02_streaming_monitor.py` - Real-time monitoring
- `phase6_advanced/06_03_config_system_demo.py` - Configuration system demo

**Key Execution Scripts**:
- `scripts/dataflow/phase0_cleanup/00_01_cleanup_all_services.py` - Clean all Azure services before testing
- `scripts/dataflow/phase5_integration/05_01_full_pipeline_execution.py` - Complete end-to-end pipeline execution
- `scripts/dataflow/phase5_integration/05_03_query_generation_showcase.py` - Query generation and processing examples

**Makefile Shortcuts**:
- `make data-prep-full` - Execute complete pipeline (alias for full pipeline script)
- `make knowledge-extract` - Run knowledge extraction phase
- `make query-demo` - Run query pipeline demonstration

## Configuration Files

- `config/environments/development.env` - Dev Azure settings
- `config/environments/staging.env` - Staging settings  
- `config/azure_settings.py` - Service configuration with validation
- `agents/core/simple_config_manager.py` - Runtime configuration
- `agents/core/constants.py` - Zero-hardcoded-values constants
- `agents/core/centralized_agent1_schema.py` - Minimal essential fields with complete usage mapping (318 lines)
- `infrastructure/prompt_workflows/templates/universal_knowledge_extraction.jinja2` - Unified entity+relationship extraction template (101 lines)

## Technology Stack

**Backend**: Python 3.11+, PydanticAI 0.6.2, FastAPI, OpenAI 1.98.0
**Frontend**: React 19.1.0, TypeScript 5.8.3, Vite 7.0.4
**Azure**: 9 services (OpenAI, Cognitive Search, Cosmos DB, ML, Storage, etc.)
**Testing**: pytest 7.4.0+ with asyncio, real Azure services
**Quality**: Black (88 chars), isort, ESLint, TypeScript strict mode

## Project Status

**Production Ready**: 95/100 score
- ✅ All 3 PydanticAI agents operational
- ✅ Zero domain bias enforced
- ✅ CI/CD with GitHub Actions
- ✅ Comprehensive test coverage
- ✅ Enterprise session management

**Real Data**: 179 Azure AI Language Service documents in `data/raw/azure-ai-services-language-service_output/`

## Quick Start for New Sessions

```bash
# 1. Navigate to project
cd /workspace/azure-maintie-rag

# 2. Check system health  
make health

# 3. Start development
make dev

# 4. Run quick validation
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/00_check_azure_state.py

# 5. Test all agents
pytest -m unit -v
```

## File Structure Key Locations

**Core Architecture**:
- `agents/core/universal_models.py:1-1536` - Domain-agnostic data structures
- `agents/core/agent_toolsets.py:1-200` - Centralized FunctionToolset management  
- `agents/core/universal_deps.py:1-150` - Azure service dependency injection
- `agents/core/azure_pydantic_provider.py:1-80` - Azure OpenAI model provider
- `agents/core/centralized_agent1_schema.py:1-50+` - Minimal essential fields for Domain Intelligence Agent

**Agent Implementations**:
- `agents/domain_intelligence/agent.py:72-95` - PydanticAI agent with toolsets
- `agents/knowledge_extraction/agent.py:45-70` - Knowledge extraction patterns
- `agents/universal_search/agent.py:50-75` - Tri-modal search coordination

**Infrastructure Layer** (`infrastructure/` - Multi-service Azure integration):
- `azure_openai/openai_client.py` - AsyncAzureOpenAI client with managed identity
- `azure_search/search_client.py` - Cognitive Search with 1536D vector operations
- `azure_cosmos/cosmos_gremlin_client.py` - Graph database with ThreadPoolExecutor (resolves async conflicts)
- `azure_storage/storage_client.py` - Document management with streaming upload
- `azure_ml/gnn_model.py` - Graph Neural Network training and inference
- `azure_auth/session_manager.py` - Centralized Azure authentication with DefaultAzureCredential
- `prompt_workflows/templates/universal_knowledge_extraction.jinja2` - Unified template for entity + relationship extraction
- `prompt_workflows/prompt_workflow_orchestrator.py` - Template orchestration and variable injection

**Critical Architecture Dependencies** (Must understand together):
- `agents/core/universal_deps.py` + `infrastructure/azure_auth/base_client.py` = Service injection pattern
- `agents/core/agent_toolsets.py` + `agents/*/agent.py` = PydanticAI toolset registration pattern
- `agents/core/centralized_agent1_schema.py:Agent1TemplateMapping` + `infrastructure/prompt_workflows/templates/universal_knowledge_extraction.jinja2` = Template variable extraction and unified extraction
- `agents/core/centralized_agent1_schema.py:Agent1UsageMapping` = Complete field usage documentation for downstream agent integration
- `config/azure_settings.py` + `config/environments/*.env` = Environment-specific Azure service configuration
- `scripts/deployment/sync-env.sh` + `azure.yaml` = Deployment environment synchronization

## Current Branch State (feature/universal-agents-clean)

**Status**: Agent architecture cleanup and unified template system implementation
**Key Changes**:
- **Modified**: Agent implementations (`agents/*/agent.py`) for PydanticAI 0.6.2 compatibility
- **Modified**: Centralized schema system (`agents/core/centralized_agent1_schema.py`) 
- **Modified**: Unified template system (`infrastructure/prompt_workflows/templates/universal_knowledge_extraction.jinja2`)
- **Added**: Phase 3 knowledge extraction validation scripts (`scripts/dataflow/phase3_knowledge/`)
- **Removed**: Individual entity/relationship templates (consolidated to unified approach)
- **Removed**: Legacy prompt workflow files and duplicate templates

**Current Issues**: Some dataflow execution scripts may reference removed files. Use validation scripts instead.

## Troubleshooting Current Branch Issues

### Missing Dataflow Files
Some Phase 3 scripts were removed during cleanup:
```bash
# Instead of missing scripts, use:
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_02_simple_extraction.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_03_simple_storage.py  
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_04_simple_graph.py
```

### Environment Variable Setup
Always use environment syncing before running agents:
```bash
./scripts/deployment/sync-env.sh prod
# or
USE_MANAGED_IDENTITY=false PYTHONPATH=/workspace/azure-maintie-rag python <script>
```

### OpenBLAS Threading Issues
For GNN and numerical operations, limit threads:
```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/workspace/azure-maintie-rag python <script>
```