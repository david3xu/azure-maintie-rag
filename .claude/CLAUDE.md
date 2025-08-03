# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Continuity Information

**Current Project State**: Advanced multi-agent RAG system in active development across multiple sessions
**Branch**: `fix/design-overlap-consolidation` (ongoing architectural improvements)
**Last Updated**: August 3, 2025
**Implementation Phase**: Phase 0 Complete - Agent 1 Learning Methods with Azure OpenAI ✅

### Key Session Context
- **Azure Infrastructure**: ✅ **5/6 Azure services connected and operational** (ConsolidatedAzureServices ready)
- **Agent Architecture**: ✅ **Enhanced 3-agent system with Phase 0 implementation** (Domain Intelligence, Knowledge Extraction, Universal Search)
- **Agent 1 Enhancement**: ✅ **Learning methods operational with Azure OpenAI** - statistical-only fallback disabled
- **PydanticAI Integration**: ✅ **95% compliance** with framework best practices and proper tool calling syntax
- **Error Handling**: ✅ **Statistical-only fallback disabled** - system now fails fast with clear error messages
- **Current Focus**: Phase 1 implementation - expanding learning capabilities and tool optimization

### Phase 0 Completion Summary
1. ✅ **Azure OpenAI Integration**: Agent 1 properly connected to Azure OpenAI (no statistical-only fallback)
2. ✅ **Learning Methods**: `create_fully_learned_extraction_config` tool working with proper PydanticAI syntax
3. ✅ **Error Handling**: Statistical-only fallback mode disabled - system fails fast with clear error messages
4. ✅ **Environment Configuration**: Fixed `AZURE_ENV_NAME=prod` environment mismatch issue

### Current Work Streams (Phase 1)
1. **Agent Learning Enhancement**: Expanding the 4 learning methods (_learn_entity_threshold, _learn_optimal_chunk_size, _learn_classification_rules, _estimate_response_sla)
2. **Tool Co-Location**: Migrating tools from separate directory to agent-specific Toolset classes
3. **Data-Driven Configuration**: Fully implement Agent 1's 100% learned configurations from corpus analysis
4. **Performance Monitoring**: Establishing baseline measurements for sub-3-second SLA validation

### Critical Files for Context
- `docs/implementation/AGENT_BOUNDARY_FIXES_IMPLEMENTATION.md`: Complete architectural analysis and improvement plan
- `docs/implementation/AGENT_1_COMPLETE_ANALYSIS.md`: Agent 1 data-driven configuration implementation
- `.claude/coding-rules.md`: Updated architectural patterns and coding standards
- `agents/domain_intelligence/agent.py`: Core Agent 1 implementation (Phase 0 complete - Azure OpenAI integration working)

### Implementation Status
- **Protected Competitive Advantages**: ✅ Tri-modal search unity, hybrid domain intelligence, config-extraction pipeline
- **Azure Integration**: ✅ Production-ready managed identity authentication + API key fallback for development
- **Phase 0 Complete**: ✅ Agent 1 learning methods operational, statistical-only fallback disabled, proper error handling
- **Next Phase**: Phase 1 implementation - enhanced learning capabilities and expanded tool coverage

---

# Azure Universal RAG with Intelligent Agents

## Project Overview
Enterprise-grade Universal RAG system with **4 competitive advantages**: (1) Tri-modal search orchestration (Vector + Graph + GNN), (2) Zero-config domain adaptation, (3) Sub-3-second response guarantee, (4) 100% data-driven intelligence. Features sophisticated **3-agent architecture** with PydanticAI framework integration and graph-based workflow orchestration.

## Key Architectural Principles
- **Multi-Agent Intelligence**: 3 specialized agents with clear boundaries (Domain Intelligence, Knowledge Extraction, Universal Search)
- **PydanticAI Graph Workflows**: State-persistent workflows with fault recovery and visual debugging
- **100% Data-Driven Configuration**: Agent 1 learns all critical parameters from corpus analysis (zero hardcoded domain assumptions)
- **Tool Co-Location**: PydanticAI Toolset classes instead of separate tools directories
- **Tri-Modal Unity**: Vector + Knowledge Graph + GNN working together with intelligent synthesis
- **Azure-Native Integration**: 5/6 Azure services connected with managed identity authentication

## Directory Structure (Root-Level Organization)
```
agents/                         # Multi-agent system (3 specialized agents)
├── core/                      # Shared Azure services and infrastructure
├── domain_intelligence/       # Agent 1: Zero-config pattern discovery
│   ├── agent.py              # Main agent with @agent.tool decorators
│   ├── toolsets.py           # ✅ PydanticAI Toolset classes (target)
│   └── tools/ (legacy)       # ⚠️ To be migrated to toolsets.py
├── knowledge_extraction/      # Agent 2: Multi-strategy extraction
│   ├── agent.py              # Main agent with validation framework
│   ├── toolsets.py           # ✅ PydanticAI Toolset classes (target)
│   └── processors/           # Entity/relationship processing logic
├── universal_search/          # Agent 3: Tri-modal search orchestration
│   ├── agent.py              # Vector + Graph + GNN coordination
│   ├── toolsets.py           # ✅ PydanticAI Toolset classes (target)
│   └── orchestrator.py       # ⚠️ To be replaced with graph workflow
├── orchestration/ (legacy)    # ⚠️ Multiple orchestrators to be consolidated
│   └── *.py                  # → Moving to single graph-based workflow
└── workflows/ (target)        # ✅ PydanticAI Graph workflows (target)
    ├── config_extraction_graph.py  # Unified Config-Extraction workflow
    └── search_workflow_graph.py    # Unified search workflow

api/                           # FastAPI application
├── endpoints/                # REST API endpoints
├── models/                   # Pydantic models
└── streaming/                # Real-time streaming

services/                     # Business logic layer (6 consolidated services)
infrastructure/               # Azure service integrations (5/6 connected)
config/                       # Configuration management
├── generated/domains/        # Agent 1 learned configurations
└── settings.py              # Azure environment settings
tests/                        # Test suites
frontend/                     # React TypeScript UI
```

## Common Development Commands

### Setup and Development
```bash
# Full setup (backend + frontend)
make setup

# Start development servers
make dev                    # Backend: :8000, Frontend: :5174

# Individual services
make backend               # Backend only
make frontend              # Frontend only
```

### Testing
```bash
# Run all tests
make test

# Quick architecture validation
python validate_architecture.py

# Test core features
python test_core_features.py

# Test Azure connectivity
python test_azure_services_direct.py

# Test data pipeline
python test_data_pipeline_simple.py

# Run specific test files
pytest tests/unit/test_api.py
pytest tests/integration/test_azure_integration.py
pytest tests/validation/validate_layer_boundaries.py

# Run tests with coverage
pytest --cov=agents --cov=api --cov=services --cov=infrastructure --cov=config
```

### Code Quality
```bash
# Pre-commit checks (black, isort, flake8, mypy, eslint, prettier)
make pre-commit

# Manual formatting (Python)
black agents/ api/ services/ infrastructure/ config/
isort agents/ api/ services/ infrastructure/ config/

# Manual formatting (Frontend)
cd frontend && npm run lint

# Type checking
mypy agents/ api/ services/ infrastructure/ config/

# Check for large files and artifacts
pre-commit run prevent-large-artifacts --all-files
pre-commit run check-venv-not-committed --all-files
```

### Deployment
```bash
# Azure deployment (full infrastructure)
make deploy

# Alternative deployment with fixes
./deploy_with_fixes.sh

# Health check
make health

# Clean artifacts and caches
make clean

# Setup local environment from scratch
python scripts/setup_local_environment.py

# Prepare for cloud deployment
python scripts/prepare_cloud_deployment.py
```

## Architecture Patterns

### PydanticAI Tool Co-Location (CRITICAL)
```python
# ✅ CORRECT - Tools co-located with agents using Toolset classes
from agents.domain_intelligence.toolsets import DomainIntelligenceToolset
from agents.knowledge_extraction.toolsets import ExtractionToolset
from agents.universal_search.toolsets import SearchToolset

# ✅ CORRECT - Agent with co-located tools
domain_agent = Agent(
    'azure-openai:gpt-4',
    deps_type=DomainDeps,
    toolsets=[DomainIntelligenceToolset()]
)

# ❌ FORBIDDEN - Separate tools directory
from agents.tools.search_tools import vector_search  # Violates co-location
```

### Graph-Based Workflow Control
```python
# ✅ CORRECT - PydanticAI Graph workflow
from pydantic_graph import Graph, BaseNode, GraphRunContext

@dataclass
class ConfigExtractionState:
    raw_data: str
    config: ExtractionConfig | None = None
    results: SearchResults | None = None

class AnalyzeDomainNode(BaseNode[ConfigExtractionState]):
    async def run(self, ctx: GraphRunContext[ConfigExtractionState]) -> ExtractKnowledgeNode:
        result = await domain_agent.run(ctx.state.raw_data, deps=ctx.deps)
        ctx.state.config = result.output
        return ExtractKnowledgeNode()

# ❌ FORBIDDEN - Multiple separate orchestrators
from agents.orchestration.config_extraction_orchestrator import ConfigOrchestrator  # Old pattern
```

### Service Dependencies
```python
# ✅ CORRECT - Dependency injection
@router.post("/api/v1/query")
async def universal_query(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
    agent_service: AgentService = Depends(get_agent_service)
):
    return await query_service.process_universal_query(request)
```

### Async Patterns
```python
# ✅ CORRECT - Parallel execution
async def process_tri_modal_search(query: str):
    tasks = [
        vector_search.search(query),
        graph_search.traverse(query),
        gnn_search.predict(query)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return synthesize_results(results)
```

## Data-Driven Design

### Agent 1: Complete Data-Driven Configuration Generation
```python
# ✅ CORRECT - Agent 1 generates 100% learned configurations
@domain_agent.tool
async def create_fully_learned_extraction_config(
    ctx: RunContext[DomainDeps],
    corpus_path: str  # e.g., "data/raw/Programming-Language"
) -> ExtractionConfiguration:
    """Generate configuration from subdirectory analysis with zero hardcoded values"""

    # Learn from actual corpus content
    stats = await analyze_corpus_statistics(ctx, corpus_path)
    patterns = await generate_semantic_patterns(ctx, sample_content)

    # Learn critical parameters from data
    entity_threshold = await _learn_entity_threshold(stats, patterns)  # From complexity
    chunk_size = await _learn_optimal_chunk_size(stats)                # From doc characteristics
    classification_rules = await _learn_classification_rules(stats)    # From token analysis
    response_sla = await _estimate_response_sla(stats)                 # From complexity

    return ExtractionConfiguration(
        domain_name=Path(corpus_path).name.lower().replace('-', '_'),
        entity_confidence_threshold=entity_threshold,  # ✅ LEARNED
        chunk_size=chunk_size,                         # ✅ LEARNED
        entity_classification_rules=classification_rules,  # ✅ LEARNED
        response_time_sla=response_sla,                # ✅ LEARNED
        # Only 10 acceptable hardcoded defaults for non-critical params
    )

# ❌ FORBIDDEN - Hardcoded domain assumptions
MEDICAL_ENTITIES = ["patient", "diagnosis", "treatment"]  # Never do this
```

### Universal Domain Discovery
- **Subdirectory-based discovery**: `data/raw/Programming-Language/` → `programming_language` domain
- **Statistical complexity assessment**: Vocabulary diversity within corpus determines processing parameters
- **Mathematical threshold optimization**: F1-score optimization for precision/recall balance
- **Performance prediction**: SLA estimation from content complexity analysis

## Performance Requirements
- **Simple queries**: <1 second
- **Complex agent reasoning**: <3 seconds
- **Domain discovery**: <30 seconds
- **Tool generation**: <5 seconds

## Layer Boundaries (Strict Enforcement)
```
API Layer → Services Layer → Infrastructure Layer → Azure Services
```

### API Layer (api/*)
- HTTP validation, authentication, dependency injection only
- Never instantiate services directly
- Never import from infrastructure layer

### Services Layer (services/*)
- Business logic, Azure service coordination
- Cache management, error handling
- Performance optimization

### Infrastructure Layer (infrastructure/*)
- Azure service clients only
- No business logic

## Agent Intelligence System

### Multi-Agent Architecture (3 Specialized Agents)
```python
# Agent 1: Domain Intelligence Agent
from agents.domain_intelligence.agent import domain_agent

# Agent 2: Knowledge Extraction Agent
from agents.knowledge_extraction.agent import extraction_agent

# Agent 3: Universal Search Agent
from agents.universal_search.agent import search_agent

# Unified workflow orchestration
result = await config_extraction_graph.run(
    query="How do I troubleshoot network issues?",
    state=WorkflowState(raw_data=query)
)
```

### PydanticAI Graph-Based Workflows
- **Config-Extraction Workflow**: Domain Analysis → Knowledge Extraction → Search
- **Graph state persistence** for fault recovery and long-running operations
- **Type-safe transitions** between workflow stages
- **Visual debugging** with automatic mermaid diagram generation

### Agent Responsibilities
- **Domain Intelligence Agent**: Zero-config pattern discovery and extraction configuration
- **Knowledge Extraction Agent**: Multi-strategy entity/relationship extraction with validation
- **Universal Search Agent**: Tri-modal search orchestration (Vector + Graph + GNN)

## Configuration Management

### Environment Variables
```bash
# Azure services
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_COSMOS_ENDPOINT=https://your-cosmos.documents.azure.com:443/
AZURE_CLIENT_ID=your-client-id
AZURE_LOCATION=westus2

# Authentication
USE_MANAGED_IDENTITY=false  # Set to true in production
OPENAI_API_TYPE=azure

# Development settings (CRITICAL: Must match azd environment)
AZURE_ENV_NAME=prod                   # ⚠️ Must match `azd env list` output
ENVIRONMENT=development
PYTHONPATH=/workspace/azure-maintie-rag
```

### Settings Files
- `config/settings.py`: Main configuration with Azure settings
- `azure.yaml`: Azure deployment configuration
- `.pre-commit-config.yaml`: Code quality checks
- `pyproject.toml`: Python project configuration (black, isort)
- `requirements.txt`: Python dependencies including PydanticAI

## Azure Services Integration

### Core Services (9 Azure services)
- **Azure OpenAI**: Entity extraction, reasoning
- **Azure Cognitive Search**: Vector search
- **Azure Cosmos DB**: Knowledge graph (Gremlin)
- **Azure ML**: GNN training and inference
- **Azure Blob Storage**: Document processing
- **Azure Key Vault**: Secrets management
- **Azure Monitor**: Observability
- **Azure Functions**: Serverless processing
- **Azure Container Registry**: Docker images

### Authentication Patterns
```python
# Development: Azure CLI authentication
az login

# Production: Managed Identity
USE_MANAGED_IDENTITY=true
```

## Common Workflows

### Adding New Domain Support
1. Create subdirectory in `data/raw/` (e.g., `data/raw/Medical-Documentation/`)
2. Add raw text documents to subdirectory (minimum 5 files)
3. Agent 1 automatically discovers domain and generates configuration
4. Zero manual configuration required - all learned from data

### Config-Extraction Workflow (Phase 0 Complete)
```bash
# Agent 1: Domain pattern discovery and config generation (Working with Azure OpenAI)
python -c "
import asyncio
from agents.domain_intelligence.agent import domain_agent
from agents.domain_intelligence.detailed_models import DomainDeps

async def test_learning():
    deps = DomainDeps()
    message = '''Use the create_fully_learned_extraction_config tool to analyze
                 the Programming-Language corpus and generate learned configuration.
                 Corpus path: data/raw/Programming-Language'''
    result = await domain_agent.run(message, deps=deps)
    print(f'Generated learned config: {result.output[:200]}...')

asyncio.run(test_learning())
"

# Agent 2: Knowledge extraction using learned config
python scripts/dataflow/02_knowledge_extraction.py

# Agent 3: Tri-modal search orchestration
python scripts/dataflow/10_query_pipeline.py
```

### Debugging Multi-Agent Workflows
```bash
# Check Azure services connectivity (5/6 services connected)
python test_azure_services_direct.py

# Validate agent boundary compliance
python -c "
from agents.validation.architecture_compliance_validator import validate_boundaries
result = await validate_boundaries()
print(f'Agent boundaries: {result.compliance_status}')
"

# Test complete Config-Extraction workflow
python test_config_extraction_workflow.py

# Monitor graph-based workflow execution
curl -X POST http://localhost:8000/api/v1/agent/workflow \
  -H "Content-Type: application/json" \
  -d '{"workflow": "config_extraction", "domain": "programming_language"}'
```

## Critical Development Guidelines

### Never Do ❌
- Hardcode critical parameters (entity thresholds, chunk sizes, classification rules, SLA targets)
- Create separate `tools/` directories (violates PydanticAI co-location)
- Use multiple orchestrators (violates graph-based workflow pattern)
- Return fake/placeholder data
- Import from old `backend.*` paths (use flat structure: `agents.*`, `services.*`, etc.)
- Bypass service layer abstractions
- Create competing search mechanisms
- **Allow statistical-only fallback mode** (Phase 0 requirement: Azure OpenAI required)
- Set incorrect `AZURE_ENV_NAME` in .env (must match `azd env list` output)
- Commit large ML artifacts (.pth, .pkl, .h5, .onnx files)
- Include virtual environments in git
- Use blocking synchronous operations for I/O

### Always Do ✅
- Learn critical parameters from corpus analysis (Agent 1 responsibility)
- Use PydanticAI Toolset classes for tool organization
- Implement graph-based workflows for complex orchestration
- Use async/await for all I/O operations with `asyncio.gather()` for parallelism
- Follow flat directory import structure (`agents.*`, `services.*`, `infrastructure.*`, `config.*`)
- Co-locate tools with agents using `@agent.tool` decorators
- **Ensure Azure OpenAI connectivity** (Phase 0 requirement: no statistical-only fallback)
- **Verify `AZURE_ENV_NAME` matches azd environment** before development
- Use proper PydanticAI tool calling syntax: `await agent.run(message, deps=deps)`
- Validate zero hardcoded values for critical parameters
- Maintain sub-3-second response times
- Use dependency injection for services via FastAPI `Depends()`
- Run pre-commit hooks before committing
- Use structured logging with operation context
- Implement comprehensive error handling with specific exceptions

### Code Quality Validation
```bash
# Before every commit, ensure these pass:
python -c "from services.query_service import QueryService; print('✅ Services import OK')"
python -c "from agents.domain_intelligence.agent import domain_agent; print('✅ Agent 1 Azure OpenAI OK')"
python -c "from infrastructure.azure_openai.openai_client import AzureOpenAIClient; print('✅ Infrastructure import OK')"

# Phase 0 Validation - Ensure no statistical-only fallback
python -c "
import os
from pathlib import Path
env_file = Path('.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

# Test Agent 1 Azure OpenAI connection
from agents.domain_intelligence.agent import domain_agent
print(f'✅ Agent 1 Model: {type(domain_agent.model).__name__ if domain_agent.model else \"❌ NO MODEL\"}')
"

make pre-commit
python validate_architecture.py
```
