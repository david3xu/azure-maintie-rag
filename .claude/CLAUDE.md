# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Continuity Information

**Current Project State**: Advanced multi-agent RAG system in active development across multiple sessions
**Branch**: `fix/design-overlap-consolidation` (ongoing architectural improvements)
**Last Updated**: August 3, 2025
**Implementation Phase**: Target Architecture ACHIEVED ✅ - PydanticAI Compliance Complete with 3 Agents

### Key Session Context
- **Azure Infrastructure**: ✅ **5/6 Azure services connected and operational** (ConsolidatedAzureServices ready)
- **Agent Architecture**: ✅ **PydanticAI-compliant 3-agent system with target architecture achieved** (Domain Intelligence, Knowledge Extraction, Universal Search)
- **Agent 1 ENHANCED**: ✅ **14 tools properly registered and working** - complete toolset restoration from legacy version
- **Domain Discovery**: ✅ **Subdirectory-based discovery operational** - `data/raw/Programming-Language` → `programming_language`
- **Learning Methods**: ✅ **All 4 learning methods working** - entity_threshold, chunk_size, classification_rules, response_sla
- **PydanticAI Integration**: ✅ **100% PydanticAI compliance achieved** - FunctionToolset pattern implemented across all agents
- **Tool Co-Location**: ✅ **Target architecture complete** - all tools moved from separate directories to agent-specific toolsets.py
- **Current Focus**: Production deployment ready - all architectural violations fixed

### Target PydanticAI Architecture ACHIEVED (100% Complete)
1. ✅ **Azure OpenAI Integration**: All 3 agents properly connected with OpenAIModel + AzureProvider
2. ✅ **Tool Co-Location Complete**: All tools migrated to agent-specific toolsets.py following PydanticAI FunctionToolset pattern
3. ✅ **Domain Intelligence Enhanced**: 14 tools properly registered and working (restored from legacy version)
4. ✅ **Knowledge Extraction Agent**: Converted to target architecture with toolsets.py
5. ✅ **Universal Search Agent**: Lazy initialization implemented with proper toolset pattern
6. ✅ **Architecture Compliance**: All violations fixed - no separate tools/ directories, proper import patterns
7. ✅ **File Organization**: Target directory structure achieved per AGENT_BOUNDARY_FIXES_IMPLEMENTATION.md

### Production Deployment Ready
1. **All Agent Tools Working**: 14 Domain Intelligence + Knowledge Extraction + Universal Search tools operational
2. **PydanticAI Compliance**: 100% adherence to FunctionToolset pattern and lazy initialization
3. **Import Architecture**: No import-time side effects, all agents use proper dependency injection
4. **File Structure**: Target architecture fully implemented with proper tool co-location

### Critical Files for Context
- `docs/implementation/AGENT_BOUNDARY_FIXES_IMPLEMENTATION.md`: Complete architectural analysis and improvement plan
- `docs/implementation/AGENT_1_COMPLETE_ANALYSIS.md`: Agent 1 data-driven configuration implementation
- `.claude/coding-rules.md`: Updated architectural patterns and coding standards
- `agents/domain_intelligence/agent.py`: Core Agent 1 implementation (Phase 0 complete - Azure OpenAI integration working)

### Implementation Status
- **Protected Competitive Advantages**: ✅ Tri-modal search unity, hybrid domain intelligence, config-extraction pipeline
- **Azure Integration**: ✅ Production-ready managed identity authentication + API key fallback for development
- **Target Architecture Complete**: ✅ PydanticAI compliance achieved, all architectural violations fixed
- **Ready for Production**: All 3 agents operational with proper toolset patterns and lazy initialization

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
├── domain_intelligence/       # Agent 1: Zero-config pattern discovery (14 tools)
│   ├── agent.py              # Main agent with lazy initialization
│   └── toolsets.py           # ✅ PydanticAI FunctionToolset with 14 tools
├── knowledge_extraction/      # Agent 2: Multi-strategy extraction
│   ├── agent.py              # Main agent with lazy initialization
│   ├── toolsets.py           # ✅ PydanticAI FunctionToolset classes
│   └── processors/           # Entity/relationship processing logic
├── universal_search/          # Agent 3: Tri-modal search orchestration
│   ├── agent.py              # Main agent with lazy initialization
│   ├── toolsets.py           # ✅ PydanticAI FunctionToolset classes
│   ├── vector_search.py      # Vector search engine
│   ├── graph_search.py       # Graph search engine
│   └── gnn_search.py         # GNN search engine
├── models/                    # Shared Pydantic models
│   └── domain_models.py      # Domain and extraction models
└── workflows/                 # ✅ PydanticAI Graph workflows
    └── tri_modal_orchestrator.py  # Unified tri-modal search orchestration

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
from agents.knowledge_extraction.toolsets import KnowledgeExtractionToolset
from agents.universal_search.toolsets import UniversalSearchToolset

# ✅ CORRECT - Agent with co-located tools (achieved)
domain_agent = Agent(
    'azure-openai:gpt-4',
    deps_type=DomainDeps,
    toolsets=[DomainIntelligenceToolset()]  # 14 tools co-located
)

# ❌ FORBIDDEN - Separate tools directory (eliminated)
from agents.tools.search_tools import vector_search  # Architecture violation - now fixed
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
# ✅ CORRECT - Agent 1 generates 100% learned configurations (14 tools working)
@domain_agent.tool
async def create_fully_learned_extraction_config(
    ctx: RunContext[DomainDeps],
    corpus_path: str  # e.g., "data/raw/Programming-Language"
) -> ExtractionConfiguration:
    """Generate configuration from subdirectory analysis with zero hardcoded values"""

    # Learn from actual corpus content using 14 specialized tools
    stats = await analyze_corpus_statistics(ctx, corpus_path)
    patterns = await generate_semantic_patterns(ctx, sample_content)

    # Learn critical parameters from data (4 core learning methods + 10 supporting tools)
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
# Agent 1: Domain Intelligence Agent (14 tools, lazy initialization)
from agents.domain_intelligence.agent import get_domain_intelligence_agent

# Agent 2: Knowledge Extraction Agent (lazy initialization)
from agents.knowledge_extraction.agent import get_knowledge_extraction_agent

# Agent 3: Universal Search Agent (lazy initialization)
from agents.universal_search.agent import get_universal_agent

# Unified tri-modal search orchestration
from agents.workflows.tri_modal_orchestrator import TriModalOrchestrator
orchestrator = TriModalOrchestrator()
result = await orchestrator.search(
    query="How do I troubleshoot network issues?",
    search_types=["vector", "graph", "gnn"]
)
```

### Tri-Modal Search Orchestration
- **Unified Search Workflow**: Vector + Graph + GNN executed simultaneously with result synthesis
- **Performance tracking** with modality-specific statistics and overall health monitoring
- **Fault-tolerant execution** with graceful degradation and timeout protection
- **Confidence-weighted synthesis** with tri-modal bonus for comprehensive results

### Agent Responsibilities (All Target Architecture Compliant)
- **Domain Intelligence Agent**: Zero-config pattern discovery with 14 specialized tools for corpus analysis
- **Knowledge Extraction Agent**: Multi-strategy entity/relationship extraction with PydanticAI toolsets
- **Universal Search Agent**: Tri-modal search orchestration with lazy initialization and proper tool co-location

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
# Agent 1: Domain pattern discovery and config generation (14 tools working with Azure OpenAI)
python -c "
import asyncio
from agents.domain_intelligence.agent import get_domain_intelligence_agent
from agents.models.domain_models import DomainDeps

async def test_learning():
    domain_agent = await get_domain_intelligence_agent()
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

# Test all agent imports with lazy initialization (no side effects)
python -c "
from agents.domain_intelligence.agent import get_domain_intelligence_agent
from agents.knowledge_extraction.agent import get_knowledge_extraction_agent 
from agents.universal_search.agent import get_universal_agent
print('✅ All agents import successfully with lazy initialization')
"

# Test tri-modal search orchestration
python -c "
import asyncio
from agents.workflows.tri_modal_orchestrator import TriModalOrchestrator

async def test_search():
    orchestrator = TriModalOrchestrator()
    result = await orchestrator.search('test query', search_types=['vector'])
    print(f'Search result confidence: {result.confidence}')

asyncio.run(test_search())
"
```

## Critical Development Guidelines

### Never Do ❌
- Hardcode critical parameters (entity thresholds, chunk sizes, classification rules, SLA targets)
- Create separate `tools/` directories (PydanticAI co-location violation - FIXED)
- Use import-time agent creation (causes Azure credential requirements at import - FIXED)
- Return fake/placeholder data
- Import from old `backend.*` paths (use flat structure: `agents.*`, `services.*`, etc.)
- Bypass service layer abstractions
- Create competing search mechanisms
- **Allow statistical-only fallback mode** (Target Architecture requirement: Azure OpenAI required)
- Set incorrect `AZURE_ENV_NAME` in .env (must match `azd env list` output)
- Commit large ML artifacts (.pth, .pkl, .h5, .onnx files)
- Include virtual environments in git
- Use blocking synchronous operations for I/O

### Always Do ✅
- Learn critical parameters from corpus analysis (Agent 1 with 14 tools)
- Use PydanticAI FunctionToolset classes for tool organization (ACHIEVED)
- Implement lazy initialization for all agents (ACHIEVED)
- Use async/await for all I/O operations with `asyncio.gather()` for parallelism
- Follow flat directory import structure (`agents.*`, `services.*`, `infrastructure.*`, `config.*`)
- Co-locate tools with agents in toolsets.py files (TARGET ARCHITECTURE ACHIEVED)
- **Ensure Azure OpenAI connectivity** (Target Architecture requirement: no statistical-only fallback)
- **Verify `AZURE_ENV_NAME` matches azd environment** before development
- Use proper PydanticAI agent creation: `await get_agent_function()` with lazy initialization
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
python -c "from agents.domain_intelligence.agent import get_domain_intelligence_agent; print('✅ Agent 1 lazy initialization OK')"
python -c "from infrastructure.azure_openai.openai_client import AzureOpenAIClient; print('✅ Infrastructure import OK')"

# Target Architecture Validation - Ensure lazy initialization and tool co-location
python -c "
# Test all agent imports without side effects (lazy initialization achieved)
from agents.domain_intelligence.agent import get_domain_intelligence_agent
from agents.knowledge_extraction.agent import get_knowledge_extraction_agent  
from agents.universal_search.agent import get_universal_agent
print('✅ All agents import without side effects - lazy initialization working')

# Test toolset imports (PydanticAI compliance achieved)
from agents.domain_intelligence.toolsets import DomainIntelligenceToolset
from agents.knowledge_extraction.toolsets import KnowledgeExtractionToolset
from agents.universal_search.toolsets import UniversalSearchToolset
print('✅ All toolsets import successfully - FunctionToolset pattern working')
"

make pre-commit
python validate_architecture.py
```
