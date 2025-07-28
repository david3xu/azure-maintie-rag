# Backend Refactoring Plan - REVISED

**Azure Universal RAG System - Backend Architecture Refactoring**  
**UPDATED:** Based on comprehensive code exploration and dependency analysis

## 📝 Infrastructure Context Update

**Note:** This refactoring plan was created before the infrastructure deployment using Azure Developer CLI (azd). Some service files that appeared during the refactoring process (backup_service.py, deployment_service.py, monitoring_service.py, security_service.py, infrastructure_service_full.py) were related to infrastructure exploration but are not part of the final architecture. The core Azure services managed by InfrastructureService are:
- Azure OpenAI (text processing, embeddings)
- Azure Cognitive Search (vector search, indexing)
- Azure Blob Storage (data persistence)
- Azure Cosmos DB (knowledge graphs with Gremlin API)
- Azure ML Workspace (GNN training)
- Azure Application Insights (monitoring, telemetry)

## 🎯 Executive Summary

This document outlines a **corrected** refactoring plan based on deep analysis of the actual codebase. The original assumptions were wrong - this revision addresses the **real architectural issues** discovered through comprehensive code exploration.

## 📊 Current State Analysis

### Directory Structure (15 directories, 42+ Python files)
```
backend/
├── api/              # FastAPI endpoints (7 endpoints)
├── services/         # Business logic (4 services) ✅ CLEAN
├── integrations/     # External service wrappers (2 files) ⚠️ REDUNDANT
├── core/             # Azure clients & models (42 files) ✅ CONSOLIDATED 
├── config/           # Settings & environment ✅ CLEAN
├── utilities/        # Shared utilities (5 modules) ⚠️ SCATTERED
├── data/             # Raw/processed data
├── scripts/          # Processing scripts ⚠️ MIXED DEPENDENCIES
├── prompt_flows/     # Azure Prompt Flow configs
├── tests/            # Test suite
├── logs/             # Runtime logs
├── outputs/          # Generated models
└── docs/             # Documentation
```

### 🔍 Dependency Analysis

**Current Import Patterns:**
- ✅ `services/` → `core/` (4 files) - Clean business logic separation
- ❌ `api/` → `integrations/` → `core/` - Unnecessary intermediate layer
- ❌ `scripts/` → `integrations/` (24 files) - Wrong abstraction level
- ❌ `utilities/` scattered across multiple directories

## ⚠️ REAL Issues Discovered Through Code Analysis

### 1. **Dual Orchestration Problem** - CRITICAL

**Problem**: TWO massive orchestrators competing for same responsibility
- **`integrations/azure_services.py`** (921 lines) - Service management + data migration + health checks
- **`core/orchestration/rag_orchestration_service.py`** (754 lines) - Universal RAG orchestrator  
- **Both handle orchestration but with different approaches**

**Dependencies Found:**
- **20+ files import from integrations/** (api/, scripts/, workflows/)
- **Only 1 file imports from core/orchestration/** (api/endpoints/health.py)
- **integrations/ imports 5 core/* modules** (storage, search, cosmos, ml, monitoring)

**Impact**: 
- **Architectural confusion** - which orchestrator to use?
- **Code duplication** - overlapping responsibilities  
- **Maintenance nightmare** - changes need coordination between two systems

### 2. **Core Module Architecture Mixing** - MAJOR

**Problem**: `core/` mixes pure infrastructure with business logic
- **Pure Infrastructure**: `azure_*/` clients (auth, openai, search, cosmos, storage, ml, monitoring)
- **Business Logic**: `orchestration/`, `workflow/`, `prompt_*/` (should be in services)
- **Data Models**: `models/` (neutral, can stay)
- **Utilities**: Mixed utility functions

**Impact**:
- **Blurred boundaries** between infrastructure and business concerns
- **Testing complexity** - hard to unit test business logic separately
- **Deployment issues** - infrastructure changes affect business logic

### 3. **Scripts Mixed Dependencies**

**Problem**: Data processing scripts use application-layer integrations
- 24 scripts import `integrations.`
- Processing scripts should use infrastructure (`core/`) or business logic (`services/`)
- Not presentation layer (`integrations/`)

**Impact**:
- Scripts coupled to web application concerns
- Harder to run scripts independently

### 4. **Scattered Utilities**

**Problem**: Utilities spread across directories
- `utilities/` (root level)
- `core/utilities/` 
- No clear ownership or consolidation

## 🎯 Target Architecture with Detailed File Structure

### **Complete Target Directory Structure**
```
backend/
├── 📚 Configuration & Setup
│   ├── pyproject.toml                    ✅ KEEP - Python project config
│   ├── pytest.ini                        ✅ KEEP - Testing config
│   ├── requirements.txt                   ✅ KEEP - Dependencies
│   ├── Dockerfile                         ✅ KEEP - Container config
│   ├── Makefile                          ✅ KEEP - Build commands
│   └── README.md                         ✅ KEEP - Backend overview
│
├── 🚀 **Presentation Layer**
│   └── api/                              # FastAPI application
│       ├── __init__.py                   ✅ KEEP
│       ├── main.py                       ✅ KEEP - FastAPI app entry
│       ├── dependencies.py               ✅ KEEP - DI container
│       ├── middleware.py                 🆕 NEW - Extract from main.py
│       ├── models/
│       │   ├── __init__.py               ✅ KEEP
│       │   ├── query_models.py           ✅ KEEP - Request/response models
│       │   ├── response_models.py        🆕 NEW - Split from query_models
│       │   └── stream_models.py          🆕 NEW - Streaming models
│       ├── endpoints/
│       │   ├── __init__.py               ✅ KEEP
│       │   ├── health_endpoint.py        🔄 RENAME health.py → health_endpoint.py
│       │   ├── query_endpoint.py         🔄 RENAME azure-query-endpoint.py → query_endpoint.py
│       │   ├── graph_endpoint.py         🔄 RENAME knowledge_graph_demo.py → graph_endpoint.py
│       │   ├── gremlin_endpoint.py       🔄 RENAME gremlin_demo_api.py → gremlin_endpoint.py
│       │   ├── gnn_endpoint.py           🔄 RENAME gnn_enhanced_query.py → gnn_endpoint.py
│       │   ├── workflow_endpoint.py      🔄 RENAME workflow_evidence.py → workflow_endpoint.py
│       │   └── demo_endpoint.py          🔄 RENAME demo_simple.py → demo_endpoint.py
│       └── streaming/
│           ├── __init__.py               🆕 NEW
│           ├── workflow_stream.py        ✅ MOVE from root api/
│           └── progress_stream.py        🆕 NEW - Extract from workflow_stream
│
├── 🎯 **Business Logic Layer**
│   └── services/                         # High-level business services
│       ├── __init__.py                   ✅ KEEP
│       ├── query_service.py              ✅ KEEP - Query orchestration
│       ├── knowledge_service.py          ✅ KEEP - Knowledge extraction
│       ├── graph_service.py              ✅ KEEP - Graph operations
│       ├── ml_service.py                 ✅ KEEP - ML operations
│       ├── infrastructure_service.py     🆕 NEW - From integrations/azure_services.py (service mgmt)
│       ├── data_service.py               🆕 NEW - From integrations/azure_services.py (data ops)
│       ├── cleanup_service.py            🆕 NEW - From integrations/azure_services.py (cleanup)
│       ├── pipeline_service.py           🆕 NEW - From core/orchestration/enhanced_pipeline.py
│       ├── workflow_service.py           🆕 NEW - From core/workflow/* (4 files merged)
│       ├── prompt_service.py             🆕 NEW - From core/prompt_generation/*
│       └── flow_service.py               🆕 NEW - From core/prompt_flow/*
│
├── 🧠 **Infrastructure Layer**
│   ├── core/                             # Pure Azure clients & models
│   │   ├── __init__.py                   ✅ KEEP
│   │   ├── azure_auth/
│   │   │   ├── __init__.py               ✅ KEEP
│   │   │   ├── base_client.py            ✅ KEEP - Auth base class
│   │   │   └── session_manager.py        ✅ KEEP - Session management
│   │   ├── azure_openai/
│   │   │   ├── __init__.py               ✅ KEEP - Import aliases
│   │   │   └── openai_client.py          ✅ KEEP - Unified OpenAI client
│   │   ├── azure_search/
│   │   │   ├── __init__.py               ✅ KEEP - Import aliases
│   │   │   └── search_client.py          ✅ KEEP - Unified Search client
│   │   ├── azure_storage/
│   │   │   ├── __init__.py               ✅ KEEP - Import aliases
│   │   │   └── storage_client.py         ✅ KEEP - Unified Storage client
│   │   ├── azure_cosmos/
│   │   │   ├── __init__.py               ✅ KEEP - Import aliases
│   │   │   ├── cosmos_client.py          ✅ KEEP - Unified Cosmos client
│   │   │   └── cosmos_gremlin_client.py  ✅ KEEP - Gremlin implementation
│   │   ├── azure_ml/
│   │   │   ├── __init__.py               ✅ KEEP
│   │   │   ├── ml_client.py              ✅ KEEP - Core ML client
│   │   │   ├── classification_service.py ✅ KEEP - ML classification
│   │   │   ├── gnn_orchestrator.py       ✅ KEEP - GNN orchestration
│   │   │   ├── gnn_processor.py          ✅ KEEP - GNN processing
│   │   │   ├── gnn_training_evidence_orchestrator.py ✅ KEEP
│   │   │   └── gnn/                      # GNN implementation
│   │   │       ├── data_bridge.py        ✅ KEEP
│   │   │       ├── data_loader.py        ✅ KEEP
│   │   │       ├── feature_engineering.py ✅ KEEP
│   │   │       ├── model.py              ✅ KEEP
│   │   │       ├── model_quality_assessor.py ✅ KEEP
│   │   │       ├── trainer.py            ✅ KEEP
│   │   │       ├── train_gnn_workflow.py ✅ KEEP
│   │   │       └── unified_training_pipeline.py ✅ KEEP
│   │   ├── azure_monitoring/
│   │   │   ├── __init__.py               🆕 NEW
│   │   │   └── app_insights_client.py    ✅ KEEP - Application insights
│   │   ├── models/
│   │   │   ├── __init__.py               ✅ KEEP
│   │   │   ├── universal_rag_models.py   ✅ KEEP - Universal data models
│   │   │   └── gnn_data_models.py        ✅ KEEP - GNN-specific models
│   │   └── utilities/
│   │       ├── __init__.py               ✅ KEEP
│   │       ├── intelligent_document_processor.py ✅ KEEP
│   │       ├── config_loader.py          🔄 MOVE from utilities/
│   │       ├── file_utils.py             🔄 MOVE from utilities/
│   │       ├── logging_utils.py          🔄 RENAME from utilities/logging.py
│   │       └── validation_utils.py       🆕 NEW - Extract validation logic
│   ├── config/                           # Configuration management
│   │   ├── __init__.py                   ✅ KEEP
│   │   ├── settings.py                   ✅ KEEP - Main settings
│   │   ├── azure_config_validator.py     ✅ KEEP - Azure validation
│   │   ├── environments/
│   │   │   ├── dev.env                   ✅ KEEP
│   │   │   ├── staging.env               ✅ KEEP
│   │   │   └── prod.env                  ✅ KEEP
│   │   └── templates/                    ✅ KEEP - Config templates
│   └── integrations/                     # External service coordination
│       ├── __init__.py                   ✅ KEEP
│       ├── azure_manager.py              🆕 NEW - Thin coordinator only
│       └── azure_openai_wrapper.py       🔄 RENAME azure_openai.py → azure_openai_wrapper.py
│
├── 📊 **Data & Processing Layer**
│   ├── data/                             # Data storage (keep existing structure)
│   ├── scripts/                          # Consolidated operational tools
│   │   ├── __init__.py                   🆕 NEW
│   │   ├── rag_cli.py                    🆕 NEW - Unified CLI entry point
│   │   ├── data_pipeline.py              🆕 NEW - Consolidate data_processing/ (9 files)
│   │   ├── azure_setup.py                🆕 NEW - Consolidate azure_services/ (4 files)
│   │   ├── gnn_trainer.py                🆕 NEW - Consolidate gnn_training/ (8 files)
│   │   ├── test_validator.py             🆕 NEW - Consolidate testing/ (15 files)
│   │   ├── workflow_analyzer.py          🆕 NEW - Consolidate workflows/ (5 files)
│   │   └── demo_runner.py                🆕 NEW - Consolidate demos/ (3 files)
│   └── prompt_flows/                     ✅ KEEP - All existing files
│
└── 🔧 **Operations & Development**
    ├── tests/
    │   ├── __init__.py                   🆕 NEW
    │   ├── test_consolidated_codebase.py ✅ KEEP
    │   ├── unit/                         🆕 NEW - Unit tests by layer
    │   ├── integration/                  🆕 NEW - Integration tests
    │   └── fixtures/                     🆕 NEW - Test fixtures
    ├── logs/                             ✅ KEEP - Runtime logs
    ├── outputs/                          ✅ KEEP - Generated models & results
    └── docs/                             ✅ KEEP - All existing documentation
```

### **Critical File Naming Fixes**
- **Remove hyphens**: `azure-query-endpoint.py` → `query_endpoint.py` (fixes import issues)
- **Consistent suffixes**: `_endpoint.py`, `_service.py`, `_client.py`
- **Descriptive names**: `demo_simple.py` → `demo_endpoint.py`
- **Layer indicators**: Clear architectural layer identification

### **Dependency Rules**
1. **Downward Only**: api/ → services/ → core/ (never upward)
2. **Single Path**: One clear path to each functionality  
3. **Layer Skipping Forbidden**: API cannot directly use core/
4. **Focused Files**: No 900+ line files, split by responsibility

## 🔄 REVISED Refactoring Implementation Plan

### **Phase 1: File Renaming & Organization** - IMMEDIATE FIXES

**Critical Naming Issues:**
```bash
# Fix hyphen import issues (breaks Python imports)
api/endpoints/azure-query-endpoint.py → api/endpoints/query_endpoint.py

# Standardize endpoint naming
api/endpoints/health.py → api/endpoints/health_endpoint.py
api/endpoints/knowledge_graph_demo.py → api/endpoints/graph_endpoint.py
api/endpoints/gremlin_demo_api.py → api/endpoints/gremlin_endpoint.py
api/endpoints/gnn_enhanced_query.py → api/endpoints/gnn_endpoint.py
api/endpoints/workflow_evidence.py → api/endpoints/workflow_endpoint.py
api/endpoints/demo_simple.py → api/endpoints/demo_endpoint.py

# Clean utility naming
utilities/logging.py → core/utilities/logging_utils.py
```

**Actions:**
1. **Rename all endpoint files** with consistent `_endpoint.py` suffix
2. **Fix hyphen issue** in azure-query-endpoint.py
3. **Move utilities/** to **core/utilities/** for consolidation
4. **Update all import statements** to use new file names

**Files Affected:** 10 renames + all files importing these modules

### **Phase 2: Resolve Dual Orchestration & Split Massive Files**

**Problem 1**: TWO competing orchestrators
- **`integrations/azure_services.py`** (921 lines, 20+ dependencies)
- **`core/orchestration/rag_orchestration_service.py`** (754 lines, 1 dependency)

**Problem 2**: Massive files doing too much

**Solution**: 
1. **Keep integrations/azure_services.py** (higher adoption)
2. **Merge core/orchestration/ functionality** into integrations
3. **Split result into focused services**

**Actions:**
```bash
# Step 1: Merge orchestrators
core/orchestration/rag_orchestration_service.py → Merge into integrations/azure_services.py
core/orchestration/enhanced_pipeline.py → Extract to services/pipeline_service.py

# Step 2: Split massive file
integrations/azure_services.py (921+ lines) → Split into:
├── services/infrastructure_service.py   # Service management, health checks
├── services/data_service.py            # Data migration, storage operations  
├── services/cleanup_service.py         # Cleanup and maintenance
└── integrations/azure_manager.py       # Thin coordination layer only

# Step 3: Update 20+ dependent imports
- from integrations.azure_services import AzureServicesManager
+ from services.infrastructure_service import InfrastructureService
+ from services.data_service import DataService
```

**Files Affected:** 21+ files (20 import integrations + 1 imports core/orchestration)

### **Phase 3: Move Business Logic from Core to Services**

**Problem**: `core/` contains business logic that should be in `services/`

**Actions:**
1. **Move business logic modules:**
   ```bash
   core/orchestration/enhanced_pipeline.py → services/pipeline_service.py
   core/workflow/ → services/workflow_service.py (merge 4 files)
   core/prompt_generation/ → services/prompt_service.py  
   core/prompt_flow/ → services/flow_service.py
   ```

2. **Keep pure infrastructure in core:**
   ```bash
   core/azure_*/ → KEEP (pure Azure clients)
   core/models/ → KEEP (data models)
   core/utilities/ → KEEP (shared utilities)  
   ```

**Result**: Clean separation between infrastructure (core) and business logic (services)

### **Phase 4: Consolidate 44 Operational Scripts**

**Problem**: Too many scripts (44 files) scattered across categories

**Actions:**
1. **Consolidate by function:**
   ```bash
   scripts/organized/data_processing/ (9 files) → scripts/data_pipeline.py
   scripts/organized/azure_services/ (4 files) → scripts/azure_setup.py
   scripts/organized/gnn_training/ (8 files) → scripts/gnn_trainer.py
   scripts/organized/testing/ (15 files) → scripts/validate.py + scripts/test_runner.py
   ```

2. **Create unified CLI:**
   ```bash
   scripts/rag_cli.py  # Single entry point for all operations
   ```

**Files Affected:** 44 scripts → 6 consolidated tools

## ✅ REVISED Success Criteria

### **Architectural Goals**
1. **Single orchestrator** (eliminate dual orchestration confusion)
2. **Clean core/services separation** (infrastructure vs business logic)
3. **Manageable file sizes** (no 900+ line files)
4. **Consolidated operations** (44 scripts → 6 tools)

### **Technical Metrics**
- **Remove core/orchestration/** (eliminate competing orchestrator)
- **Split 921-line file** into 4 focused services  
- **Move 4 business modules** from core/ to services/
- **Consolidate 44 scripts** into 6 tools
- **Maintain all 20+ dependencies** with correct new imports

### **Validation Commands**
```bash
# Check clean dependencies
grep -r "from integrations\." backend/  # Should return empty
grep -r "from core\." backend/api/      # Should return empty  

# Verify layer structure
find backend/ -name "*.py" -exec grep -l "from services\." {} \;  # Only api/
find backend/ -name "*.py" -exec grep -l "from core\." {} \;      # services/ and scripts/

# Test functionality
python -m pytest tests/ -v  # All tests pass
```

## 🚨 Migration Risks & Mitigation

### **Risk 1: Breaking Changes**
- **Mitigation**: Update all imports atomically
- **Rollback**: Git branch for easy revert

### **Risk 2: Test Failures**  
- **Mitigation**: Run tests after each phase
- **Fix**: Update test imports to match new structure

### **Risk 3: Production Issues**
- **Mitigation**: Validate in development environment first
- **Monitoring**: Check all endpoints work after changes

## 📅 REVISED Implementation Timeline

1. **Phase 1** (2 days): Resolve dual orchestration (merge complex logic)
2. **Phase 2** (3 days): Split 921-line file into 4 services + update 20+ imports
3. **Phase 3** (1 day): Move business logic from core/ to services/  
4. **Phase 4** (2 days): Consolidate 44 scripts into 6 tools
5. **Testing** (1 day): Comprehensive validation of all changes

**Total Effort: 9 days** (much more complex than originally estimated)

## 🎉 Expected Benefits

### **Developer Experience**
- **Clear mental model**: Know exactly which layer to use
- **Faster development**: No confusion about import paths
- **Easier debugging**: Clean dependency chains

### **Code Quality**
- **Reduced duplication**: Single implementation per feature
- **Better testability**: Isolated layer testing
- **Maintainability**: Changes isolated to appropriate layers

### **Production Readiness**
- **Scalability**: Clean architecture supports growth
- **Reliability**: Fewer dependencies = fewer failure points
- **Monitoring**: Clear boundaries enable better observability

---

## 🎯 **TARGET BACKEND DIRECTORY STRUCTURE**

### **Complete File-Level Target Architecture**

```
backend/
├── 📚 Configuration & Setup
│   ├── pyproject.toml                    ✅ KEEP - Python project config
│   ├── pytest.ini                        ✅ KEEP - Testing config  
│   ├── requirements.txt                   ✅ KEEP - Dependencies
│   ├── Dockerfile                         ✅ KEEP - Container config
│   ├── Makefile                          ✅ KEEP - Build commands
│   └── README.md                         ✅ KEEP - Backend overview
│
├── 🚀 **PRESENTATION LAYER**
│   └── api/                              # FastAPI application
│       ├── __init__.py                   ✅ KEEP
│       ├── main.py                       ✅ KEEP - FastAPI app entry
│       ├── dependencies.py               ✅ KEEP - DI container
│       ├── middleware.py                 🆕 NEW - Extract from main.py
│       │
│       ├── models/                       # API data models
│       │   ├── __init__.py               ✅ KEEP
│       │   ├── query_models.py           ✅ KEEP - Request/response models
│       │   ├── response_models.py        🆕 NEW - Split from query_models
│       │   └── stream_models.py          🆕 NEW - Streaming models
│       │
│       ├── endpoints/                    # API endpoints
│       │   ├── __init__.py               ✅ KEEP
│       │   ├── health_endpoint.py        🔄 RENAME health.py → health_endpoint.py
│       │   ├── query_endpoint.py         🔄 RENAME azure-query-endpoint.py → query_endpoint.py
│       │   ├── graph_endpoint.py         🔄 RENAME knowledge_graph_demo.py → graph_endpoint.py
│       │   ├── gremlin_endpoint.py       🔄 RENAME gremlin_demo_api.py → gremlin_endpoint.py
│       │   ├── gnn_endpoint.py           🔄 RENAME gnn_enhanced_query.py → gnn_endpoint.py
│       │   ├── workflow_endpoint.py      🔄 RENAME workflow_evidence.py → workflow_endpoint.py
│       │   └── demo_endpoint.py          🔄 RENAME demo_simple.py → demo_endpoint.py
│       │
│       └── streaming/                    # Real-time streaming
│           ├── __init__.py               🆕 NEW
│           ├── workflow_stream.py        ✅ MOVE from root api/
│           └── progress_stream.py        🆕 NEW - Extract from workflow_stream
│
├── 🎯 **BUSINESS LOGIC LAYER**
│   └── services/                         # High-level business services
│       ├── __init__.py                   ✅ KEEP
│       │
│       ├── **Existing Services**
│       ├── query_service.py              ✅ KEEP - Query orchestration
│       ├── knowledge_service.py          ✅ KEEP - Knowledge extraction
│       ├── graph_service.py              ✅ KEEP - Graph operations
│       ├── ml_service.py                 ✅ KEEP - ML operations
│       │
│       ├── **New Services (from splits)**
│       ├── infrastructure_service.py     🆕 NEW - From integrations/azure_services.py (service mgmt)
│       ├── data_service.py               🆕 NEW - From integrations/azure_services.py (data ops)
│       ├── cleanup_service.py            🆕 NEW - From integrations/azure_services.py (cleanup)
│       ├── pipeline_service.py           🆕 NEW - From core/orchestration/enhanced_pipeline.py
│       ├── workflow_service.py           🆕 NEW - From core/workflow/* (4 files merged)
│       ├── prompt_service.py             🆕 NEW - From core/prompt_generation/*
│       └── flow_service.py               🆕 NEW - From core/prompt_flow/*
│
├── 🧠 **INFRASTRUCTURE LAYER**
│   ├── core/                             # Pure Azure clients & models
│   │   ├── __init__.py                   ✅ KEEP
│   │   │
│   │   ├── **Azure Service Clients**
│   │   ├── azure_auth/
│   │   │   ├── __init__.py               ✅ KEEP
│   │   │   ├── base_client.py            ✅ KEEP - Auth base class
│   │   │   └── session_manager.py        ✅ KEEP - Session management
│   │   ├── azure_openai/
│   │   │   ├── __init__.py               ✅ KEEP - Import aliases
│   │   │   └── openai_client.py          ✅ KEEP - Unified OpenAI client
│   │   ├── azure_search/
│   │   │   ├── __init__.py               ✅ KEEP - Import aliases
│   │   │   └── search_client.py          ✅ KEEP - Unified Search client
│   │   ├── azure_storage/
│   │   │   ├── __init__.py               ✅ KEEP - Import aliases
│   │   │   └── storage_client.py         ✅ KEEP - Unified Storage client
│   │   ├── azure_cosmos/
│   │   │   ├── __init__.py               ✅ KEEP - Import aliases
│   │   │   ├── cosmos_client.py          ✅ KEEP - Unified Cosmos client
│   │   │   └── cosmos_gremlin_client.py  ✅ KEEP - Gremlin implementation
│   │   ├── azure_ml/
│   │   │   ├── __init__.py               ✅ KEEP
│   │   │   ├── ml_client.py              ✅ KEEP - Core ML client
│   │   │   ├── classification_service.py ✅ KEEP - ML classification
│   │   │   ├── gnn_orchestrator.py       ✅ KEEP - GNN orchestration
│   │   │   ├── gnn_processor.py          ✅ KEEP - GNN processing
│   │   │   ├── gnn_training_evidence_orchestrator.py ✅ KEEP
│   │   │   └── gnn/                      # GNN implementation details
│   │   │       ├── data_bridge.py        ✅ KEEP
│   │   │       ├── data_loader.py        ✅ KEEP
│   │   │       ├── feature_engineering.py ✅ KEEP
│   │   │       ├── model.py              ✅ KEEP
│   │   │       ├── model_quality_assessor.py ✅ KEEP
│   │   │       ├── trainer.py            ✅ KEEP
│   │   │       ├── train_gnn_workflow.py ✅ KEEP
│   │   │       └── unified_training_pipeline.py ✅ KEEP
│   │   ├── azure_monitoring/
│   │   │   ├── __init__.py               🆕 NEW
│   │   │   └── app_insights_client.py    ✅ KEEP - Application insights
│   │   │
│   │   ├── **Data Models & Utilities**
│   │   ├── models/
│   │   │   ├── __init__.py               ✅ KEEP
│   │   │   ├── universal_rag_models.py   ✅ KEEP - Universal data models
│   │   │   └── gnn_data_models.py        ✅ KEEP - GNN-specific models
│   │   └── utilities/
│   │       ├── __init__.py               ✅ KEEP
│   │       ├── intelligent_document_processor.py ✅ KEEP
│   │       ├── config_loader.py          🔄 MOVE from utilities/
│   │       ├── file_utils.py             🔄 MOVE from utilities/
│   │       ├── logging_utils.py          🔄 RENAME from utilities/logging.py
│   │       └── validation_utils.py       🆕 NEW - Extract validation logic
│   │
│   ├── config/                           # Configuration management
│   │   ├── __init__.py                   ✅ KEEP
│   │   ├── settings.py                   ✅ KEEP - Main settings
│   │   ├── azure_config_validator.py     ✅ KEEP - Azure validation
│   │   ├── environments/
│   │   │   ├── dev.env                   ✅ KEEP
│   │   │   ├── staging.env               ✅ KEEP
│   │   │   └── prod.env                  ✅ KEEP
│   │   └── templates/                    ✅ KEEP - Config templates
│   │
│   └── integrations/                     # External service coordination
│       ├── __init__.py                   ✅ KEEP
│       ├── azure_manager.py              🆕 NEW - Thin coordinator only
│       └── azure_openai_wrapper.py       🔄 RENAME azure_openai.py → azure_openai_wrapper.py
│
├── 📊 **DATA & PROCESSING LAYER**
│   ├── data/                             # Data storage (existing structure)
│   │   ├── raw/                          ✅ KEEP - Raw input data
│   │   ├── processed/                    ✅ KEEP - Processed outputs
│   │   ├── cache/                        ✅ KEEP - Caching
│   │   ├── demo/                         ✅ KEEP - Demo datasets
│   │   ├── gnn_models/                   ✅ KEEP - Trained GNN models
│   │   ├── extraction_outputs/           ✅ KEEP - Knowledge extraction results
│   │   ├── loading_results/              ✅ KEEP - Data loading outputs
│   │   └── [other existing subdirs]      ✅ KEEP - All current data structure
│   │
│   ├── scripts/                          # Consolidated operational tools
│   │   ├── __init__.py                   🆕 NEW
│   │   ├── rag_cli.py                    🆕 NEW - Unified CLI entry point
│   │   ├── data_pipeline.py              🆕 NEW - Consolidate data_processing/ (9 files)
│   │   ├── azure_setup.py                🆕 NEW - Consolidate azure_services/ (4 files)
│   │   ├── gnn_trainer.py                🆕 NEW - Consolidate gnn_training/ (8 files)
│   │   ├── test_validator.py             🆕 NEW - Consolidate testing/ (15 files)
│   │   ├── workflow_analyzer.py          🆕 NEW - Consolidate workflows/ (5 files)
│   │   └── demo_runner.py                🆕 NEW - Consolidate demos/ (3 files)
│   │
│   └── prompt_flows/                     # Azure Prompt Flow configurations
│       └── universal_knowledge_extraction/ ✅ KEEP - All existing files
│           ├── flow.dag.yaml             ✅ KEEP
│           ├── azure_storage_writer.py   ✅ KEEP
│           ├── knowledge_graph_builder.py ✅ KEEP
│           ├── quality_assessor.py       ✅ KEEP
│           └── [other prompt flow files] ✅ KEEP
│
└── 🔧 **OPERATIONS & DEVELOPMENT**
    ├── tests/                            # Organized test suite
    │   ├── __init__.py                   🆕 NEW
    │   ├── test_consolidated_codebase.py ✅ KEEP
    │   ├── unit/                         🆕 NEW - Unit tests by layer
    │   │   ├── __init__.py               🆕 NEW
    │   │   ├── test_core.py              🆕 NEW - Infrastructure layer tests
    │   │   ├── test_services.py          🆕 NEW - Business logic tests
    │   │   └── test_api.py               🆕 NEW - Presentation layer tests
    │   ├── integration/                  🆕 NEW - Integration tests
    │   │   ├── __init__.py               🆕 NEW
    │   │   ├── test_azure_integration.py 🆕 NEW - Azure services integration
    │   │   └── test_workflow_integration.py 🆕 NEW - End-to-end workflows
    │   └── fixtures/                     🆕 NEW - Test fixtures and data
    │       ├── __init__.py               🆕 NEW
    │       ├── mock_data.py              🆕 NEW
    │       └── azure_mocks.py            🆕 NEW
    │
    ├── logs/                             ✅ KEEP - Runtime logs
    ├── outputs/                          ✅ KEEP - Generated models & results  
    └── docs/                             ✅ KEEP - All existing documentation
        ├── architecture/                 ✅ KEEP
        ├── demo/                         ✅ KEEP
        ├── execution/                    ✅ KEEP
        └── core/                         ✅ KEEP
```

### **File Migration Summary**

#### **🗑️ Files to Remove/Consolidate:**
- **`utilities/`** directory (3 files) → Move to `core/utilities/`
- **`integrations/azure_services.py`** (921 lines) → Split into 4 services  
- **`core/orchestration/`** directory (2 files) → Move to services/
- **`core/workflow/`** directory (4 files) → Merge into services/workflow_service.py
- **`core/prompt_generation/`** → Move to services/prompt_service.py
- **`core/prompt_flow/`** → Move to services/flow_service.py
- **`scripts/organized/`** (44 files) → Consolidate into 6 tools

#### **🔄 Files to Rename:**
- **7 API endpoints** - Add `_endpoint.py` suffix, fix hyphens
- **3 utility files** - Add `_utils` suffix for clarity
- **2 integration files** - Add descriptive suffixes

#### **🆕 Files to Create:**
- **9 new service files** from business logic extraction
- **6 new consolidated script tools** 
- **8 new API organization files** (middleware, models, streaming)
- **5 new test organization files** (unit, integration, fixtures)

#### **✅ Files to Keep (No Changes):**
- **All Azure client files** in core/azure_*/
- **All GNN implementation files** in core/azure_ml/gnn/
- **All data files and structure** 
- **All documentation files**
- **All configuration files**
- **All prompt flow files**

---

## 🏗️ **AZURE INFRASTRUCTURE FOUNDATION**

### **Azure Developer CLI (azd) Integration**

Based on the [Azure Search OpenAI Demo](https://github.com/Azure-Samples/azure-search-openai-demo) pattern, our backend should be built on top of automated Azure infrastructure provisioning using `azd up`.

#### **Current Infrastructure Gap**
**Problem**: Manual Azure service setup creates inconsistency and deployment friction
- ✅ **Have**: Bicep templates in `/infrastructure/`
- ❌ **Missing**: `azure.yaml` configuration for azd
- ❌ **Missing**: Automated service provisioning workflow
- ❌ **Missing**: Environment-specific deployment targets

#### **Target Infrastructure Architecture**
```yaml
# azure.yaml (NEW FILE NEEDED)
name: azure-maintie-rag
metadata:
  template: azure-search-openai-demo@main

services:
  backend:
    language: py
    host: containerapp
    path: ./backend

infra:
  provider: bicep
  path: ./infrastructure

hooks:
  preprovision:
    shell: sh
    run: echo "Preparing Azure infrastructure..."
  postprovision:
    shell: sh
    run: echo "Configuring deployed services..."
```

#### **Required Azure Services (via azd)**
**Core RAG Infrastructure:**
- ✅ **Azure OpenAI Service** - Text processing and embeddings
- ✅ **Azure Cognitive Search** - Vector search and indexing
- ✅ **Azure Cosmos DB** (Gremlin API) - Knowledge graphs
- ✅ **Azure Blob Storage** - Data persistence
- ✅ **Azure ML Workspace** - GNN training
- ✅ **Azure Application Insights** - Monitoring

**Container & Hosting:**
- 🆕 **Azure Container Apps** - Backend hosting
- 🆕 **Azure Container Registry** - Image storage
- 🆕 **Azure Key Vault** - Secrets management

#### **Infrastructure-as-Code Enhancement**
**Current Bicep Files (enhance existing):**
```
infrastructure/
├── main.bicep                     🆕 NEW - Main entry point for azd
├── main.parameters.json           🆕 NEW - Environment parameters
├── azure-resources-core.bicep     ✅ ENHANCE - Core services (OpenAI, Search, Storage)
├── azure-resources-cosmos.bicep   ✅ ENHANCE - Cosmos DB with Gremlin
├── azure-resources-ml-simple.bicep ✅ ENHANCE - ML workspace
└── modules/                       🆕 NEW - Modular components
    ├── openai.bicep              🆕 NEW - Azure OpenAI configuration
    ├── search.bicep              🆕 NEW - Cognitive Search setup
    ├── cosmos.bicep              🆕 NEW - Cosmos DB Gremlin API
    ├── storage.bicep             🆕 NEW - Multi-account blob storage
    ├── ml.bicep                  🆕 NEW - ML workspace with compute
    ├── monitoring.bicep          🆕 NEW - Application Insights
    ├── keyvault.bicep            🆕 NEW - Secrets management
    └── containerapp.bicep        🆕 NEW - Backend hosting
```

#### **Environment Configuration**
**Multi-Environment Support:**
```bash
# Development environment
azd env new development
azd env set AZURE_LOCATION eastus
azd env set AZURE_OPENAI_RESOURCE_GROUP rg-maintie-rag-dev

# Staging environment  
azd env new staging
azd env set AZURE_LOCATION westus2
azd env set AZURE_OPENAI_RESOURCE_GROUP rg-maintie-rag-staging

# Production environment
azd env new production
azd env set AZURE_LOCATION centralus
azd env set AZURE_OPENAI_RESOURCE_GROUP rg-maintie-rag-prod
```

#### **Backend Service Dependencies**
**Configuration Integration:**
```python
# config/azure_settings.py (ENHANCED)
class AzureSettings:
    # Automatically populated by azd deployment
    openai_endpoint: str = os.environ["AZURE_OPENAI_ENDPOINT"]
    search_endpoint: str = os.environ["AZURE_SEARCH_ENDPOINT"] 
    cosmos_endpoint: str = os.environ["AZURE_COSMOS_ENDPOINT"]
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"]
    ml_workspace_name: str = os.environ["AZURE_ML_WORKSPACE_NAME"]
    app_insights_key: str = os.environ["AZURE_APP_INSIGHTS_KEY"]
    
    # Managed Identity for security
    azure_client_id: str = os.environ.get("AZURE_CLIENT_ID", "")
```

#### **Deployment Workflow**
**One-Command Deployment:**
```bash
# Initial setup
azd auth login
azd init --template azure-search-openai-demo

# Infrastructure + Application deployment
azd up  # Provisions all Azure services + deploys backend

# Backend-only updates
azd deploy backend

# Infrastructure-only updates  
azd provision
```

#### **Integration with Backend Refactoring**
**Service Dependencies on Infrastructure:**
```python
# services/infrastructure_service.py (UPDATED)
class InfrastructureService:
    def __init__(self):
        # All Azure clients initialized from azd-provisioned services
        self.openai_client = UnifiedAzureOpenAIClient(
            endpoint=azure_settings.openai_endpoint,
            managed_identity=True  # Uses azd-configured identity
        )
        self.search_service = UnifiedSearchClient(
            endpoint=azure_settings.search_endpoint,
            managed_identity=True
        )
        # ... other services from azd-provisioned infrastructure
```

#### **Development to Production Pipeline**
**Consistent Environments:**
```bash
# Local development (uses real Azure services)
azd env select development
azd up
make dev  # Backend connects to dev Azure services

# Staging deployment
azd env select staging  
azd up
# Automated testing against staging services

# Production deployment
azd env select production
azd up
# Blue-green deployment with health checks
```

### **Infrastructure Prerequisites for Backend Refactoring**

#### **Phase 0: Infrastructure Foundation** (Before Phase 1)
1. **Create azure.yaml** - Configure azd for our project
2. **Enhance Bicep templates** - Add missing services and modules
3. **Setup environments** - Dev, staging, production configurations
4. **Test azd workflow** - Verify `azd up` provisions all services
5. **Update backend configuration** - Connect to azd-managed services

#### **Success Criteria**
- ✅ `azd up` provisions all 8 Azure services automatically
- ✅ Backend connects to azd-managed services (no manual configuration)
- ✅ Multiple environments (dev/staging/prod) work independently
- ✅ Secrets managed through Azure Key Vault (not .env files)
- ✅ Infrastructure changes deploy through azd (not manual portal changes)

### **Backend Architecture Dependencies**
**Critical Integration Points:**
1. **All services/** depend on azd-provisioned infrastructure
2. **Configuration management** driven by azd environment variables
3. **Deployment pipeline** uses azd for both infrastructure and application
4. **Local development** connects to real Azure services (not mocks)
5. **Testing environments** use separate azd-managed resource groups

**This infrastructure foundation is ESSENTIAL for the backend refactoring success** - without azd automation, the backend will continue to have configuration drift and deployment inconsistencies.

---

**Updated Next Steps**: 
1. **First**: Implement azd infrastructure foundation (azure.yaml + enhanced Bicep)
2. **Then**: Execute backend refactoring Phases 1-4 on azd-managed infrastructure
3. **Finally**: Validate end-to-end deployment through azd workflow