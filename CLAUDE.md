# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference (Most Common Tasks)

```bash
# Essential Development (run from project root)
make dev                # Start API (8000) + Frontend (5174) with session tracking
make health             # Full system health check with enterprise reporting
make clean              # Clean sessions with log replacement
make session-report     # View current session status and metrics

# Testing (uses real Azure services - never mocks)
pytest                  # All tests with automatic asyncio handling
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests with real Azure services
pytest -k "test_name"   # Specific test pattern
pytest -x -vvv         # Stop on first failure with maximum verbosity

# Code Quality (must pass before commits)
black . --check && isort . --check-only  # Combined Python formatting
cd frontend && npm run lint               # Frontend linting
./scripts/hooks/pre-commit-domain-bias-check.sh  # Domain bias validation (CRITICAL)

# Agent Testing (direct execution - requires PYTHONPATH)
PYTHONPATH=/workspace/azure-maintie-rag python agents/domain_intelligence/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/knowledge_extraction/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/universal_search/agent.py

# Data Pipeline Testing (individual scripts)
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/00_check_azure_state.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/12_query_generation_showcase.py

# Quick Script Testing (with environment handling)
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/workspace/azure-maintie-rag python <script_name>  # For GNN-dependent scripts

# Project Structure Navigation
cd /workspace/azure-maintie-rag  # Always work from project root
```

## Architecture Overview

This is an **Azure Universal RAG system** - a production-grade multi-agent platform with PydanticAI framework, combining knowledge graphs, vector search, and Graph Neural Networks for intelligent document processing with **zero hardcoded domain bias**.

### Core Architecture: Multi-Agent System (PydanticAI)

**Three specialized agents with real Azure integration:**
- **Domain Intelligence Agent** (`agents/domain_intelligence/agent.py`): Analyzes document domains and generates dynamic configurations using Azure OpenAI
- **Knowledge Extraction Agent** (`agents/knowledge_extraction/agent.py`): Extracts entities and relationships with Azure Cosmos DB Gremlin integration  
- **Universal Search Agent** (`agents/universal_search/agent.py`): Orchestrates tri-modal search (Vector + Graph + GNN) across Azure services

### Critical Architecture Principles

**Universal RAG Philosophy (MOST IMPORTANT)**:
- **Zero Domain Assumptions**: Never use hardcoded domain categories (technical, legal, medical, etc.)
- **Content Discovery**: System discovers domain characteristics from content analysis
- **Adaptive Configuration**: Parameters adjust based on measured content properties, not domain labels
- **Pre-commit Enforcement**: `scripts/hooks/pre-commit-domain-bias-check.sh` detects domain bias violations

**Multi-Agent Coordination**:
- **PydanticAI Framework**: Type-safe agent communication with validation
- **Universal Models** (`agents/core/universal_models.py`): Domain-agnostic data structures
- **Azure Service Integration**: Real Azure OpenAI, Cosmos DB, Cognitive Search, and ML services
- **Azure Managed Identity**: Uses `agents/core/azure_pydantic_provider.py` for seamless authentication

## Critical Development Rules

### 🚨 Universal RAG Philosophy (MUST FOLLOW)

**NEVER use hardcoded domain assumptions.** The system must discover content characteristics dynamically:
- ❌ **WRONG**: `if domain == "legal"` or `technical_complexity = 0.8`
- ✅ **CORRECT**: Analyze content properties and adapt parameters based on discovered characteristics
- **Pre-commit Hook**: `./scripts/hooks/pre-commit-domain-bias-check.sh` will fail builds on domain bias violations

**Configuration Flow**: Content Analysis → Discovered Properties → Dynamic Configuration → Agent Processing

### 🏗️ PydanticAI Agent Patterns (REQUIRED)

All agents must follow these patterns:
```python
# Standard agent structure
from pydantic_ai import Agent, RunContext
from agents.core.universal_deps import UniversalDeps

agent = Agent[UniversalDeps, OutputModel](
    model=azure_openai_model,
    deps_type=UniversalDeps,
    retries=3
)

@agent.tool
async def analyze_content(ctx: RunContext[UniversalDeps], content: str):
    client = ctx.deps.azure_openai_client  # Access via dependency injection
    return await client.analyze(content)
```

### 🔒 Azure Integration Rules

- **Authentication**: Always use `DefaultAzureCredential` - never API keys
- **Testing**: Use real Azure services, never mocks
- **Dependencies**: Access Azure services via `agents/core/universal_deps.py`
- **Environment Sync**: Run `./scripts/deployment/sync-env.sh <environment>` before deployment

## Development Commands

### Essential Development Workflow
```bash
# Core development cycle (run from project root)
make setup              # Full project setup (API + frontend)  
make dev                # Start API (8000) and frontend UI (5174)
make health             # Comprehensive Azure service health check
make clean              # Clean sessions with log replacement

# Environment synchronization (critical for Azure services)
./scripts/deployment/sync-env.sh prod           # Switch to production + sync (default)
./scripts/deployment/sync-env.sh staging       # Switch to staging + sync  
make sync-env                                   # Sync with current azd environment

# Agent development workflow (use PYTHONPATH)
PYTHONPATH=/workspace/azure-maintie-rag python agents/domain_intelligence/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/knowledge_extraction/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/universal_search/agent.py

# Single test execution patterns
pytest tests/test_azure_services.py::test_azure_openai_connection -v  # Run specific test
pytest -k "knowledge_graph" -v                     # Run tests matching pattern
pytest -x -vvv tests/test_azure_services.py       # Stop on first failure, maximum verbosity
```

### Data Processing Pipeline
```bash
# Full pipeline commands (use these for end-to-end workflow)
make data-prep-full     # Complete data processing pipeline
make data-upload        # Upload documents & create chunks
make knowledge-extract  # Extract entities & relationships
make query-demo         # Query pipeline demonstration
make unified-search-demo # Tri-modal search demonstration
make full-workflow-demo # End-to-end agent orchestration demonstration

# Individual dataflow scripts for development (use PYTHONPATH)
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/00_check_azure_state.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/01_data_ingestion.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/01a_azure_storage.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/01b_azure_search.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/02_knowledge_extraction.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/03_cosmos_storage.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/07_unified_search.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/12_query_generation_showcase.py
```

### Testing Commands
```bash
# Testing strategy (tests use real Azure services - no mocks)
pytest                  # All tests with automatic asyncio handling
pytest -m unit          # Unit tests for agent logic and universal models
pytest -m integration   # Integration tests with real Azure services
pytest -m azure_validation # Azure service health validation
pytest -m performance   # SLA compliance and performance tests

# Specific test categories (actual test files in codebase)
pytest tests/test_azure_services.py                # Azure service integration tests
pytest tests/test_layer2_agents.py                 # PydanticAI agent tests
pytest tests/test_comprehensive_integration.py     # End-to-end integration tests
pytest tests/test_performance_benchmarking.py      # Performance and SLA tests

# Development testing patterns
pytest -v --tb=short    # Verbose output with short tracebacks
pytest -k "test_name"   # Run specific test pattern
pytest --collect-only   # Show available tests without running
pytest -x              # Stop on first failure (useful for debugging)
pytest -x -vvv         # Stop on first failure with maximum verbosity

# Running tests with proper environment setup
PYTHONPATH=/workspace/azure-maintie-rag pytest tests/test_agents.py -v
OPENBLAS_NUM_THREADS=1 pytest tests/test_gnn_comprehensive.py  # For GNN tests
```

### Individual Service Development
```bash
# API development (FastAPI with real Azure services)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000    # Development server
python -m api.main                                           # Alternative startup

# Frontend development (React 19.1.0 + TypeScript 5.8.3 + Vite 7.0.4)
cd frontend/
npm run dev             # Start Vite dev server (localhost:5174)
npm run build           # Production build with TypeScript compilation
npm run lint            # Run ESLint with React/TypeScript rules
npx tsc --noEmit        # TypeScript type checking without compilation
npm run preview         # Preview production build locally

# Code quality commands
black . --check                    # Python code formatting check
isort . --check-only               # Python import organization check
cd frontend && npm run lint        # Frontend linting
./scripts/hooks/pre-commit-domain-bias-check.sh  # Domain bias validation
```

### Azure Deployment
```bash
# Complete infrastructure deployment with Azure Developer CLI
azd up                  # Deploy complete Azure infrastructure (9 services)
azd env select production && azd up  # Deploy to production environment

# Environment management
azd env new prod && azd up            # Create and deploy production environment (default)
azd env new staging && azd up        # Create and deploy staging environment
azd env select <environment>         # Switch between environments

# Infrastructure automation
./scripts/deployment/setup-environments.sh     # Setup all environments
./scripts/deployment/azure_deployment_helper.sh # Deployment assistance
./scripts/deployment/test-infrastructure.sh    # Validate infrastructure

# CI/CD Pipeline setup
azd pipeline config     # Automatic GitHub Actions CI/CD setup
```

## Technology Stack

### Core Architecture
- **Python 3.11+** with async/await patterns throughout
- **FastAPI** with uvicorn for API endpoints and streaming
- **PydanticAI 0.1.0+** for multi-agent framework with type safety
- **Azure OpenAI** for LLM operations with AsyncAzureOpenAI client
- **Azure Cognitive Search** for vector search (1536D vectors)
- **Azure Cosmos DB** (Gremlin API) for knowledge graph storage
- **Azure ML** for GNN training and inference
- **Azure Blob Storage** for document management
- **Azure Key Vault** for secrets management
- **PyTorch + torch-geometric** for graph neural networks

### Frontend Architecture
- **React 19.1.0** with TypeScript 5.8.3
- **Vite 7.0.4** for build tooling and development
- **Axios 1.10.0** for HTTP requests
- **Server-Sent Events** for real-time progress updates

### Code Quality & Testing
- **Black** formatter (line length 88, configured in pyproject.toml)
- **isort** for import organization (black profile, first-party packages: agents, api, infrastructure, config)
- **ESLint 9.30.1** for TypeScript/React code quality with typescript-eslint 8.35.1
- **pytest 7.4.0+** with asyncio auto mode and comprehensive test discovery
- **Pre-commit hooks** for domain bias detection (no hardcoded domain assumptions)
- **MyPy** for type checking (optional dependency)
- **Real Azure Services Testing** - no mocks, comprehensive integration validation

### Build and Quality Commands
```bash
# Python code quality (must pass before commits)
black . --check && isort . --check-only  # Combined formatting check
black . && isort .                        # Apply all formatting

# Frontend quality
cd frontend && npm run lint && npm run build  # Lint + build
npx tsc --noEmit                              # Type check only

# Critical pre-commit validation
./scripts/hooks/pre-commit-domain-bias-check.sh  # Domain bias detection (REQUIRED)
```

## Configuration & Environment Management

### Environment Configuration
- `config/environments/development.env` - Development Azure settings  
- `config/environments/staging.env` - Staging Azure settings
- `config/azure_settings.py` - Azure service settings with validation
- `config/universal_config.py` - Universal RAG configuration
- `agents/core/simple_config_manager.py` - Simple configuration management
- Root level: `pyproject.toml`, `pytest.ini` for project tooling

**Environment Synchronization**:
- Configuration automatically syncs with `azd` environment selection
- Use `./scripts/deployment/sync-env.sh <environment>` to switch and sync
- Environment variables managed through `USE_MANAGED_IDENTITY` setting
- Multi-environment support: production (default), staging, with appropriate Azure SKUs

### Azure Service Authentication
- Uses **DefaultAzureCredential** for unified authentication across all services
- **Azure Managed Identity Provider** (`agents/core/azure_pydantic_provider.py`) for PydanticAI integration
- Supports managed identity (production) and CLI authentication (development)
- No API keys or connection strings in code - all through Azure authentication

## Testing & Validation Strategy

### Testing Architecture
- **Test Organization**: Tests organized by markers - unit, integration, azure_validation, performance  
- **Real Azure Services**: Tests use actual Azure services, not mocks (see `tests/conftest.py`)
- **Universal Testing**: Domain-agnostic test patterns using real test corpus
- **Agent Testing**: PydanticAI agent integration tests with real Azure OpenAI backends
- **Performance Testing**: SLA compliance testing with performance monitoring fixtures
- **Test Discovery**: Automatic async handling via `pytest.ini` with `asyncio_mode = auto`

### Performance Monitoring
- **Target Metrics**: Sub-3-second query processing (currently 0.8-1.8s uncached)
- **Cache Performance**: 60% cache hit rate (reduces to ~50ms response time)
- **Extraction Accuracy**: 85% relationship extraction accuracy
- **Concurrent Users**: 100+ concurrent users supported

### Session Management Pattern
- Clean session replacement (no log accumulation)
- Unique session timestamps for tracking
- Performance metrics capture in `logs/performance.log`
- Azure status monitoring in `logs/azure_status.log`

## Code Quality & Validation

### Pre-commit Validation Workflow (REQUIRED)
```bash
# Must pass before committing - run all together
black . --check && isort . --check-only && \
cd frontend && npm run lint && cd .. && \
./scripts/hooks/pre-commit-domain-bias-check.sh && \
pytest -m unit
```

### Service Validation Commands
```bash
make health                    # Full system health check
make azure-status             # Azure infrastructure status
make session-report           # Current session metrics
```

## Key File Structure

### Core Architecture Components
```
agents/                          # Multi-agent system (PydanticAI)
├── core/                        # Core infrastructure
│   ├── azure_pydantic_provider.py # Azure managed identity provider for PydanticAI
│   ├── universal_models.py      # Universal data models for domain-agnostic processing
│   ├── constants.py             # Zero-hardcoded-values constants
│   ├── simple_config_manager.py # Simple configuration management
│   └── universal_deps.py        # Universal dependencies
├── domain_intelligence/         # Domain analysis agent
│   └── agent.py                 # Domain Intelligence Agent with Azure OpenAI integration
├── knowledge_extraction/        # Entity/relationship extraction agent
│   └── agent.py                 # Knowledge Extraction Agent with Cosmos DB integration
├── universal_search/            # Tri-modal search agent
│   └── agent.py                 # Universal Search Agent with multi-modal search
├── shared/                      # Common agent utilities
│   └── query_tools.py           # Query processing tools
├── examples/                    # Agent workflow demonstrations
└── orchestrator.py             # Multi-agent orchestration

infrastructure/                  # Azure service clients (real implementation)
├── azure_openai/               # LLM operations with AsyncAzureOpenAI
├── azure_search/               # Vector search with Azure Cognitive Search
├── azure_cosmos/               # Graph database with Gremlin API
├── azure_storage/              # Blob storage for document management
├── azure_ml/                   # GNN training and inference
├── azure_auth/                 # Azure authentication utilities
├── azure_monitoring/           # Application Insights monitoring
├── utilities/                  # Common infrastructure utilities
└── prompt_workflows/           # Jinja2 templates for universal prompt engineering

config/                         # Environment-based configuration
├── universal_config.py         # Universal RAG configuration
├── azure_settings.py          # Azure service settings
├── timeouts.py                 # Service timeout configurations
├── settings.py                 # Application settings
└── environments/               # Environment-specific settings (.env files)

api/                            # FastAPI endpoints with streaming
├── main.py                     # FastAPI application entry point
├── endpoints/                  # Individual API endpoints
├── models/                     # API request/response models
└── streaming/                  # Server-sent events for real-time updates

tests/                          # Real Azure services testing (no mocks)
├── test_universal_content_processing.py    # Universal RAG processing tests
├── test_knowledge_graph_intelligence.py   # Knowledge extraction and graph tests
├── test_enterprise_deployment.py         # Enterprise deployment validation
├── test_advanced_search_discovery.py     # Multi-modal search testing
└── conftest.py                           # Real Azure services fixtures

scripts/                        # Development and deployment automation
├── dataflow/                   # Complete data processing pipeline
│   ├── 00_full_pipeline.py     # Complete pipeline orchestration
│   ├── 02_knowledge_extraction.py # Knowledge Extraction Agent processing
│   ├── 07_unified_search.py    # Universal Search Agent demonstration
│   └── 12_query_generation_showcase.py # Query generation demonstration
├── hooks/                      # Pre-commit validation hooks
│   └── pre-commit-domain-bias-check.sh # Universal RAG domain bias detection (899 lines)
└── deployment/                 # Azure deployment automation

frontend/                       # React TypeScript frontend with streaming
├── package.json                # Dependencies (React 19.1.0, TypeScript 5.8.3, Vite 7.0.4)
├── src/                        # Application source code
│   ├── components/             # React components (chat, domain, workflow, shared)
│   ├── hooks/                  # Custom React hooks (useUniversalRAG, useWorkflowStream)
│   ├── services/               # API communication and streaming
│   └── types/                  # TypeScript type definitions
└── public/                     # Static assets

data/                           # Test data and processing
├── raw/azure-ai-services-language-service_output/ # Real test corpus (179 files)
└── processed/                  # Processed data outputs

infra/                         # Azure Infrastructure as Code
├── main.bicep                  # Infrastructure entry point
├── modules/                    # Modular Bicep templates
└── main.parameters.json        # Environment parameters
```

## System Validation Status

### ✅ Comprehensive Lifecycle Validation Completed
**Date**: August 8, 2025 | **Score**: 95/100 | **Status**: Production Ready

The system has undergone comprehensive end-to-end lifecycle validation:
- **Multi-Agent Architecture**: All 3 PydanticAI agents validated and functional
- **Universal Design**: Zero hardcoded domain assumptions confirmed across all components
- **Data Pipeline**: Complete processing pipeline validated with 179 real Azure AI files
- **Service Integration**: All Azure service clients properly implemented with DefaultAzureCredential
- **Code Quality**: 20/20 core components successfully validated

**Validation Report**: See `COMPREHENSIVE_LIFECYCLE_VALIDATION_REPORT.md` for detailed results.

**Next Step**: Deploy Azure infrastructure with `azd up` to enable live Azure services.

## Current Development Context

### Active Branch: `feature/universal-agents-clean`
Focus on Universal RAG philosophy with zero domain bias and PydanticAI integration.

### Development Priorities
1. **Universal RAG Philosophy**: No hardcoded domain categories - discover content characteristics dynamically
2. **PydanticAI Integration**: Type-safe multi-agent communication with real Azure OpenAI backends
3. **Real Azure Services**: Comprehensive integration testing with actual Azure infrastructure
4. **Domain-Agnostic Design**: All patterns must work universally across any domain
5. **Azure Managed Identity**: Seamless authentication via azure_pydantic_provider.py

### Key Constraints
- **Universal Design**: Never assume domain types - analyze content properties instead
- **Security**: Never commit secrets - use Azure Key Vault and DefaultAzureCredential exclusively  
- **Real Services**: Test with actual Azure services, never mocks
- **Type Safety**: Use Pydantic models for all agent interfaces and data structures
- **Domain Bias Detection**: Pre-commit hooks enforce zero domain assumptions

### System Status: Production Ready (95/100)
- ✅ All 3 PydanticAI agents operational with real Azure integration
- ✅ Complete CI/CD pipeline with GitHub Actions and OIDC authentication  
- ✅ Universal RAG philosophy enforced across all components
- ✅ Comprehensive test suite with 179-file real Azure AI corpus
- ✅ Enterprise session management with clean log replacement

## Critical Development Patterns

### Session Management Pattern
- **Clean Session Replacement**: All make commands create unique session IDs and replace previous logs
- **No Log Accumulation**: System automatically cleans and replaces session data
- **Enterprise Reporting**: Each session generates comprehensive reports in `logs/session_report.md`
- **Performance Tracking**: Metrics captured in `logs/performance.log` and `logs/azure_status.log`

### Enterprise Session Management (Unique Feature)
The Makefile implements enterprise session tracking with clean output replacement - this is a key differentiator:

**Session Architecture**:
- Each `make` command creates a unique session ID (`YYYYMMDD_HHMMSS`)
- Previous session logs are **completely replaced** (no accumulation)
- Session reports capture Azure status, performance metrics, and detailed outputs
- Enterprise reporting in `logs/session_report.md` with timestamps and duration

**Session Commands**:
```bash
make session-report     # View current session with full metrics
make clean             # Clean current session and start fresh
make health            # Health check with comprehensive session report
```

**Session Files**:
- `logs/current_session` - Current session ID
- `logs/session_report.md` - Enterprise session report (replaced each command)
- `logs/azure_status.log` - Azure service health (replaced each command) 
- `logs/performance.log` - System performance metrics (replaced each command)

### Multi-Agent Development Architecture
The system uses PydanticAI with specific patterns that must be followed:
```python
# Agent structure pattern - all agents follow this
from pydantic_ai import Agent, RunContext
from agents.core.universal_deps import UniversalDeps

# Agent definition with proper typing
agent = Agent[UniversalDeps, OutputModel](
    model=azure_openai_model,
    deps_type=UniversalDeps,
    retries=3,
    system_prompt="..."
)

# Tool usage with RunContext dependency injection
@agent.tool
async def analyze_content(ctx: RunContext[UniversalDeps], content: str) -> Dict[str, Any]:
    # Access centralized services via ctx.deps
    client = ctx.deps.azure_openai_client
    return await client.analyze(content)
```

### Configuration Hierarchy (Critical)
1. **Universal Models** (`agents/core/universal_models.py`) - Domain-agnostic data structures
2. **Simple Config Manager** (`agents/core/simple_config_manager.py`) - Runtime configuration
3. **Universal Dependencies** (`agents/core/universal_deps.py`) - Centralized service access
4. **Azure Settings** (`config/azure_settings.py`) - Environment-specific Azure configuration

### Environment Synchronization Critical Workflow
```bash
# CRITICAL: Always sync environment before deployment
./scripts/deployment/sync-env.sh <environment>  # Switches azd environment AND syncs configuration
make sync-env                                   # Syncs configuration with current azd environment

# This ensures configuration matches azd environment selection
# Supports development, staging, and production environments with appropriate Azure SKUs
```

### Domain-Agnostic Development Pattern (Critical)
```bash
# Universal RAG development workflow - ZERO domain assumptions
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/12_query_generation_showcase.py
./scripts/hooks/pre-commit-domain-bias-check.sh         # REQUIRED: Validate no domain bias
pytest tests/test_azure_services.py                     # Universal processing tests

# Content analysis approach (NEVER domain classification)
# ✅ Analyze: vocabulary_complexity, concept_density, relationship_patterns  
# ✅ Adapt: parameters based on measured properties
# ✅ Use: universal models that work for ANY domain
# ✅ Let: Domain Intelligence Agent DISCOVER characteristics
# ❌ NEVER: hardcode "technical", "legal", "medical" categories
```

### Critical Pre-commit Domain Bias Enforcement
The `scripts/hooks/pre-commit-domain-bias-check.sh` (899 lines) enforces the Universal RAG philosophy:
- **Detects**: Hardcoded domain categories, assumptions, and classifications
- **Prevents**: Domain-specific logic that should be dynamic discovery
- **Enforces**: Universal patterns that adapt to ANY content type
- **Validates**: Zero hardcoded domain knowledge across all components

**Common Violations Caught**:
```bash
# ❌ These patterns will fail pre-commit
if domain == "technical"           # Hardcoded domain classification
technical_complexity = 0.8        # Assumed complexity value  
legal_pattern_weights = {...}     # Domain-specific hardcoding

# ✅ These patterns are allowed
complexity = analyze_vocabulary(content)     # Dynamic discovery
pattern_weights = discover_patterns(data)    # Content-driven
```

## Debugging and Troubleshooting

### Troubleshooting Quick Reference

**Azure Authentication Issues:**
```bash
az login && az account show && azd env select <environment> && make sync-env
```

**Agent Debugging:**
```bash
export DEBUG=1
PYTHONPATH=/workspace/azure-maintie-rag python agents/domain_intelligence/agent.py
```

**Service Health Check:**
```bash
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/00_check_azure_state.py
curl http://localhost:8000/api/v1/health         # API health
curl http://localhost:5174                        # Frontend health
make health                                       # Comprehensive health with session report
```

**Session Management Issues:**
```bash
make session-report                               # View current session
make clean && make dev                            # Clean start with new session
```

### Common Issues & Solutions

**Agent Failures**: 
- Check `UniversalDeps` configuration in `agents/core/universal_deps.py`
- Validate Pydantic models match expected structure
- Test agent tools independently with proper RunContext
- Review session logs in `logs/session_report.md`

**Pre-commit Hook Failures**: The domain bias detection is strict:
- ❌ Never use "technical", "legal", "medical" as hardcoded categories
- ❌ No mock/hardcoded values that should come from configuration
- ❌ Domain assumptions in prompts or logic
- ✅ Let Domain Intelligence Agent discover characteristics dynamically
- ✅ Use universal models that adapt to discovered content properties

**Python Path Issues**: Always use the full PYTHONPATH when running scripts:
```bash
PYTHONPATH=/workspace/azure-maintie-rag python <script_name>
```

**Azure Authentication Issues**: Ensure proper authentication chain:
```bash
az login
az account show  
./scripts/deployment/sync-env.sh prod  # or staging
# Verify: DefaultAzureCredential should work across all services
```

**Testing Issues**: 
- Tests use REAL Azure services, never mocks
- Ensure Azure services are deployed with `azd up`
- Use `pytest -x -vvv` for detailed failure debugging
- Check `tests/conftest.py` for real Azure fixtures

## Development Notes for Claude Code

### Important File Paths
- Main project directory: `/workspace/azure-maintie-rag/`
- Agent files: `agents/{domain_intelligence,knowledge_extraction,universal_search}/agent.py`
- Configuration: `config/azure_settings.py`, `config/universal_config.py`
- Test files: `tests/test_*.py` (all use real Azure services)

### Key Commands for Troubleshooting
1. **Check Azure services**: `PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/00_check_azure_state.py`
2. **Validate agents**: Run individual agent files with PYTHONPATH
3. **Check dependencies**: Look at `pyproject.toml` for exact versions
4. **Environment sync**: Use `./scripts/deployment/sync-env.sh <env>` before any Azure operations

### Makefile Commands Reference
The project uses a sophisticated Makefile with enterprise session tracking:
- `make dev` - Start API + Frontend with session tracking
- `make health` - Comprehensive health check with enterprise reporting  
- `make clean` - Clean session with log replacement (no accumulation)
- `make data-prep-full` - Complete data processing pipeline
- `make session-report` - View current session metrics
- `make azure-deploy` - Deploy Azure infrastructure

All Makefile commands create unique session IDs and replace previous logs (no accumulation).