# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is an **Azure Universal RAG system** - a production-grade platform that combines knowledge graphs, vector search, and Graph Neural Networks (GNNs) for intelligent document processing and retrieval.

### Core Components

- **Agents** (`agents/`): Multi-agent system with domain intelligence, knowledge extraction, and universal search capabilities
- **API** (`api/`): FastAPI service with streaming endpoints and real-time progress tracking  
- **Infrastructure** (`infrastructure/`): Azure service integrations (OpenAI, Search, Cosmos, Storage, ML)
- **Frontend** (`frontend/`): React + TypeScript UI with progressive disclosure workflow visualization
- **Config** (`config/`): Environment-based configuration management with Azure settings

### Agent Architecture

The system implements a **multi-agent architecture** using Pydantic AI:

- **Domain Intelligence Agent** (`agents/domain_intelligence/`): Analyzes document domains and generates configurations
- **Knowledge Extraction Agent** (`agents/knowledge_extraction/`): Extracts entities and relationships from documents
- **Universal Search Agent** (`agents/universal_search/`): Provides unified search across vector, graph, and GNN modalities

### Data Flow

```
Raw Documents → Azure Blob Storage
    ├─ Vector Embeddings → Azure Cognitive Search (1536D vectors)
    ├─ Knowledge Extraction → Azure Cosmos DB (Gremlin graph)
    └─ Graph Neural Networks → Azure ML (relationship learning)

User Query → Tri-modal Search (Vector + Graph + GNN) → Response Generation
```

## Common Development Commands

### Environment Management
```bash
# Switch environments and sync configuration
./scripts/sync-env.sh development    # Switch to development + sync backend
./scripts/sync-env.sh staging       # Switch to staging + sync backend
make sync-env                       # Sync backend with current azd environment
```

### Development Workflow
```bash
make setup              # Full project setup (backend + frontend)
make dev                # Start both backend API and frontend UI
make health             # Check service health across all components
make clean              # Clean sessions and logs
```

### Data Processing Pipeline
```bash
make data-prep-full     # Complete data processing pipeline
make data-upload        # Upload docs & create chunks  
make knowledge-extract  # Extract entities & relations
```

### Testing
```bash
pytest                  # Run all tests with automatic asyncio handling (project root)
pytest tests/agents/    # Run agent-specific tests  
pytest tests/agents/run_comprehensive_agent_tests.py  # Comprehensive multi-agent tests
pytest -v --tb=short    # Verbose output with short tracebacks
pytest tests/unit/      # Unit tests
pytest tests/integration/ # Integration tests
```

### Backend Development
```bash
# Backend runs on localhost:8000
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Backend testing happens at project root level (see Testing section above)
```

### Frontend Development  
```bash
cd frontend
npm run dev             # Start Vite dev server (localhost:5174)
npm run build           # Production build
npm run lint            # Run ESLint with TypeScript rules
```

### Azure Deployment
```bash
azd up                  # Deploy complete Azure infrastructure
azd env select production && azd up  # Deploy to production environment
```

## Key Architecture Patterns

### Multi-Agent Coordination
- **Agent Interfaces** (`agents/interfaces/agent_contracts.py`): Data-driven Pydantic contracts for agent communication
- **Workflow Orchestration** (`agents/workflows/`): Graph-based workflow coordination with tri-modal orchestrator
- **State Persistence** (`agents/workflows/state_persistence.py`): Maintains workflow state across agent interactions
- **Centralized Configuration** (`agents/core/centralized_config.py`): Dynamic configuration management

### Azure Service Integration
- **Consolidated Services** (`agents/core/azure_services.py`): Unified Azure service container with proper error handling
- **Authentication**: Uses DefaultAzureCredential for unified Azure authentication
- **Configuration**: Environment-based settings with automatic Azure environment synchronization

### Clean Architecture Principles
- **Layer Separation**: Agents depend on infrastructure, never the reverse
- **Dependency Injection**: Services are injected into agents for testability
- **Error Handling**: Comprehensive Azure service retry logic and graceful degradation

## Technology Stack

### Backend
- **Python 3.11+** with async/await patterns
- **FastAPI** with uvicorn for API endpoints  
- **Pydantic AI** for agent framework
- **Azure OpenAI** for LLM operations
- **Azure Cognitive Search** for vector search
- **Azure Cosmos DB** (Gremlin API) for knowledge graphs
- **Azure ML** for GNN training and inference
- **PyTorch + torch-geometric** for graph neural networks

### Frontend
- **React 19.1.0** with TypeScript 5.8.3
- **Vite 7.0.4** for build tooling
- **Axios 1.10.0** for HTTP requests
- **Server-Sent Events** for real-time progress updates

### Code Quality Tools
- **Black** formatter (line length 88, configured in pyproject.toml)
- **isort** for import organization (black profile, first-party packages configured)
- **ESLint** for TypeScript/React code
- **pytest** with asyncio auto mode and comprehensive test discovery

## Configuration Management

### Environment Files
- `config/environments/development.env` - Development Azure configuration
- `config/environments/staging.env` - Staging Azure configuration  
- `config/azure_settings.py` - Azure service settings with validation
- Root level project configuration in `pyproject.toml` and `pytest.ini`

### Azure Service Authentication
- Uses **DefaultAzureCredential** for unified authentication
- Supports both managed identity (production) and CLI authentication (development)
- Configuration automatically syncs with `azd` environment selection
- Environment variables managed through `USE_MANAGED_IDENTITY` setting

## Performance & Monitoring

### Key Metrics
- Sub-3-second query processing
- 85% relationship extraction accuracy  
- 60% cache hit rate with 99% reduction in repeat processing
- Multi-hop reasoning with semantic path discovery

### Session Management
- Clean session replacement pattern (no log accumulation)
- Performance metrics captured in `logs/performance.log`
- Azure status monitoring in `logs/azure_status.log`

## Development Guidelines

### Code Patterns
- **Async First**: All Azure operations use async/await
- **Error Handling**: Azure service retry logic with graceful degradation
- **Type Safety**: Full TypeScript on frontend, Pydantic models on backend
- **Testing**: Comprehensive test coverage with agent-specific test suites

### Agent Development
- Agents are stateless and communicate through well-defined interfaces
- Use dependency injection for Azure service access via `agents/core/azure_services.py`
- Follow the existing patterns in `agents/core/` and `agents/shared/`
- Implement proper error handling and logging
- All agent interfaces use data-driven Pydantic models to eliminate hardcoded values
- Use centralized configuration via `agents/core/centralized_config.py` for dynamic settings

### Security
- Never commit secrets or API keys
- Use Azure Key Vault for secret management  
- Follow RBAC patterns for service authentication
- Validate all inputs and sanitize outputs

### Testing Architecture
- **Comprehensive Agent Tests**: `tests/agents/run_comprehensive_agent_tests.py` runs all agent tests with JSON output
- **Individual Agent Tests**: Each agent has dedicated test files in `tests/agents/`
- **Test Results Storage**: Results stored in `test_results/agents/` with timestamped JSON files
- **Validation Tests**: Architecture and deployment validation in `tests/validation/`
- **Fixture Data**: Real data fixtures in `tests/fixtures/real_data.py`

## File Structure Guide

```
agents/                          # Multi-agent system (Pydantic AI)
├── core/                        # Core infrastructure (azure_services.py, centralized_config.py)
├── domain_intelligence/         # Domain analysis agent with statistical processing
├── knowledge_extraction/        # Entity/relationship extraction with validation processors
├── universal_search/            # Tri-modal search (vector + graph + GNN)
├── workflows/                   # Agent coordination (tri_modal_orchestrator.py, state_persistence.py)
├── interfaces/                  # Data-driven Pydantic contracts (agent_contracts.py)
├── models/                      # Domain models and workflow states
└── shared/                      # Common agent utilities and capability patterns

infrastructure/                  # Azure service clients
├── azure_openai/               # LLM operations and knowledge extraction
├── azure_search/               # Vector search integration
├── azure_cosmos/               # Graph database (Gremlin API)
├── azure_storage/              # Blob storage for documents
├── azure_ml/                   # ML models and GNN training
└── prompt_workflows/           # Jinja2 templates for prompt engineering

api/                            # FastAPI service
├── endpoints/                  # REST API route handlers
└── streaming/                  # Server-sent events for real-time updates

frontend/                       # React 19 + TypeScript UI
├── src/components/             # Modular React components (chat/, domain/, workflow/)
├── src/hooks/                  # Custom hooks (useUniversalRAG, useWorkflowStream)
├── src/services/               # API communication and streaming services
└── src/types/                  # TypeScript type definitions

tests/                          # Comprehensive testing
├── agents/                     # Agent-specific tests with JSON result output
├── validation/                 # Architecture validation tests
├── fixtures/                   # Real data fixtures
└── test_results/               # Timestamped test result storage

config/                         # Environment-based configuration
└── environments/               # Environment-specific Azure settings (.env files)
```

## Current Development Context

### Recent Architecture Improvements
- **Design Overlap Consolidation**: Ongoing consolidation of agent boundaries and responsibilities
- **Hardcoded Values Elimination**: Implementation of data-driven configuration patterns  
- **Centralized Configuration**: Migration to `agents/core/centralized_config.py` for dynamic settings
- **Performance Enhancement**: Cache optimization and learned pattern integration

### Active Branch: `fix/design-overlap-consolidation`
- Focus on agent boundary refinement and configuration consolidation
- Enhanced Pydantic AI integration patterns
- Elimination of legacy infrastructure workflows in favor of prompt workflows

This system represents a production-ready Azure Universal RAG platform with enterprise-grade architecture, comprehensive testing, and real-time workflow visualization.