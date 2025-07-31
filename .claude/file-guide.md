# Important File Locations

## Critical Architecture Violations (Fix First) ⚠️

### Global DI Anti-Pattern
- **File**: `backend/api/dependencies.py:18-23`
- **Issue**: Uses global variables instead of proper DI container
- **Impact**: Breaks testability, creates hidden dependencies
- **Priority**: CRITICAL - Must fix before other work

### Direct Service Instantiation
- **File**: `backend/api/endpoints/unified_search_endpoint.py:76`
- **Code**: `query_service = QueryService()`
- **Issue**: Bypasses dependency injection
- **Impact**: Tight coupling, untestable components

### Global State Anti-Pattern
- **File**: `backend/api/endpoints/gnn_endpoint.py:19`
- **Code**: `gnn_service = None`
- **Issue**: Global service variable
- **Impact**: Race conditions, testing difficulties

## Key Architecture Files 📋

### Master Documentation
- **`PROJECT_ARCHITECTURE.md`** - Complete system design and 6 architectural rules
- **`CODING_STANDARDS.md`** - Mandatory development rules (data-driven, no fake data)
- **`IMPLEMENTATION_ROADMAP.md`** - 12-week implementation plan with phases

### API Layer Structure
```
backend/api/
├── endpoints/
│   ├── query_endpoint.py              # ✅ Uses proper DI
│   ├── unified_search_endpoint.py     # ❌ Direct instantiation (line 76)
│   ├── gnn_endpoint.py               # ❌ Global state (line 19)
│   ├── demo_endpoint.py              # Demo functionality
│   ├── graph_endpoint.py             # Graph stats demo
│   ├── gremlin_endpoint.py           # Gremlin query demo
│   ├── workflow_endpoint.py          # ❌ Direct instantiation (line 11)
│   └── health_endpoint.py            # ✅ Health checks
├── models/
│   ├── query_models.py               # Query request/response models
│   ├── response_models.py            # API response models
│   └── azure_models.py               # Azure-specific models
└── dependencies.py                   # ❌ CRITICAL - Global DI pattern
```

## Service Layer (Current Architecture) 🔧

### Core Services
- **`backend/services/query_service.py`** - Main query processing service
- **`backend/services/infrastructure_service.py`** - Azure service coordination
- **`backend/services/performance_service.py`** - Performance monitoring
- **`backend/services/cache_service.py`** - Multi-level caching

### Domain and Configuration
- **`backend/config/domain_patterns.py`** - ❌ TO BE REPLACED by dynamic discovery
- **`backend/config/settings.py`** - Environment and Azure configuration
- **`backend/config/azure_config.py`** - Azure service settings

## Core Integration Layer ⚙️

### Search and Knowledge
```
backend/core/
├── search/
│   ├── cognitive_search_client.py    # Azure Cognitive Search integration
│   ├── vector_search_service.py      # Vector search coordination
│   └── search_result_processor.py    # Result processing
├── graph/
│   ├── cosmos_gremlin_client.py      # Cosmos DB Gremlin integration
│   ├── graph_traversal_service.py    # Knowledge graph operations
│   └── relationship_extractor.py     # Relationship discovery
├── gnn/
│   ├── azure_ml_client.py            # Azure ML integration
│   ├── gnn_model_service.py          # GNN model management
│   ├── gnn_inference_service.py      # Real-time inference
│   └── model_training_service.py     # Training pipeline
└── processing/
    ├── document_processor.py         # Document ingestion
    ├── embedding_service.py          # Vector embeddings
    └── data_pipeline.py              # Data processing workflows
```

## Infrastructure Layer ☁️

### Azure Service Clients
```
backend/infrastructure/
├── azure_openai_client.py           # OpenAI integration
├── azure_search_client.py           # Cognitive Search client
├── azure_cosmos_client.py           # Cosmos DB client
├── azure_ml_client.py               # Machine Learning client
├── azure_storage_client.py          # Blob Storage client
└── azure_monitor_client.py          # Monitoring and logging
```

### Configuration and Deployment
- **`infra/`** - Bicep infrastructure as code
- **`scripts/`** - Deployment and utility scripts
- **`.github/workflows/`** - CI/CD pipeline definitions

## Future Agent System (To Be Built) 🤖

### Planned Directory Structure (Phase 2-3)
```
backend/agents/                       # 🆕 NEW - Intelligent Agent System
├── base/                            # Agent foundation
│   ├── agent_interface.py           # Abstract agent interface
│   ├── reasoning_engine.py          # Core reasoning patterns
│   └── context_manager.py           # Context and memory management
├── discovery/                       # Data-driven discovery
│   ├── domain_discoverer.py         # Domain pattern discovery
│   ├── entity_extractor.py          # Dynamic entity extraction
│   ├── relationship_learner.py      # Relationship pattern learning
│   └── action_pattern_miner.py      # Action workflow discovery
├── reasoning/                       # Agent reasoning capabilities
│   ├── tri_modal_orchestrator.py    # Orchestrate tri-modal search
│   ├── reasoning_chain_builder.py   # Multi-step reasoning
│   ├── solution_synthesizer.py      # Solution generation
│   └── confidence_calculator.py     # Confidence scoring
├── learning/                        # Continuous learning
│   ├── pattern_extractor.py         # Success pattern extraction
│   ├── agent_evolution_manager.py   # Agent improvement
│   ├── cross_domain_learner.py      # Universal pattern discovery
│   └── feedback_processor.py        # User feedback integration
└── universal_agent.py               # Main agent implementation
```

### Dynamic Tool System (Phase 3)
```
backend/tools/                       # 🆕 NEW - Dynamic Tool System
├── base/                            # Tool foundation
├── discovery/                       # Dynamic tool discovery
├── registry/                        # Tool management
├── execution/                       # Tool execution
└── dynamic_tool.py                  # Main dynamic tool class
```

## Test Structure 🧪

### Current Test Organization
```
backend/tests/
├── unit/
│   ├── test_api.py                  # API endpoint tests
│   ├── test_core.py                 # Core functionality tests
│   ├── test_services.py             # Service layer tests
│   └── test_infrastructure.py       # Infrastructure tests
├── integration/
│   ├── test_azure_services.py       # Azure integration tests
│   ├── test_search_pipeline.py      # Search workflow tests
│   └── test_gnn_integration.py      # GNN integration tests
└── performance/
    ├── test_response_times.py       # Performance validation
    └── test_concurrent_users.py     # Load testing
```

## Frontend Structure 🌐

### React Application
```
frontend/
├── src/
│   ├── components/
│   │   ├── SearchInterface.tsx      # Main search interface
│   │   ├── ResultsDisplay.tsx       # Search results display
│   │   ├── TriModalView.tsx         # Tri-modal search visualization
│   │   └── AgentInteraction.tsx     # 🆕 Future agent interface
│   ├── services/
│   │   ├── apiClient.ts             # API communication
│   │   ├── streamingService.ts      # Server-sent events
│   │   └── agentService.ts          # 🆕 Future agent communication
│   └── types/
│       ├── searchTypes.ts           # Search-related types
│       └── agentTypes.ts            # 🆕 Future agent types
```

## Configuration Files 📝

### Environment and Settings
- **`.env`** - Local environment variables
- **`backend/config/settings.py`** - Application settings
- **`backend/config/azure_config.py`** - Azure service configuration
- **`pyproject.toml`** - Python project configuration
- **`requirements.txt`** - Python dependencies

### CI/CD and Infrastructure
- **`.github/workflows/`** - GitHub Actions workflows
- **`infra/main.bicep`** - Azure infrastructure definition
- **`azure.yaml`** - Azure Developer CLI configuration
- **`docker-compose.yml`** - Local development setup

## Navigation Tips 🧭

### Finding Specific Functionality
- **Query Processing**: Start with `backend/services/query_service.py`
- **Azure Integration**: Look in `backend/infrastructure/`
- **Search Logic**: Check `backend/core/search/`
- **GNN Features**: Explore `backend/core/gnn/`
- **API Endpoints**: Browse `backend/api/endpoints/`

### Understanding Data Flow
1. **Request**: `frontend/` → `backend/api/endpoints/`
2. **Processing**: `backend/services/` → `backend/core/`
3. **Azure Integration**: `backend/infrastructure/`
4. **Response**: Back through the same chain

### Debugging Common Issues
- **DI Problems**: Check `backend/api/dependencies.py`
- **Azure Connectivity**: Look in `backend/infrastructure/`
- **Performance Issues**: Check `backend/services/performance_service.py`
- **Search Problems**: Investigate `backend/core/search/`

## Quick Reference Commands 🚀

### Development
```bash
# Start backend
cd backend && python -m uvicorn main:app --reload

# Start frontend  
cd frontend && npm start

# Run tests
cd backend && python -m pytest

# Azure deployment
azd up
```

### File Patterns to Look For
- **`*_service.py`** - Service layer components
- **`*_client.py`** - Azure service integration
- **`*_endpoint.py`** - API endpoints
- **`test_*.py`** - Test files
- **`*_models.py`** - Data models and schemas