# Target Backend Directory Structure

**Azure Universal RAG System - Complete Backend Refactoring Plan**

## 🎯 Target Directory Structure with File Names

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
│       │   ├── health_endpoint.py        🔄 RENAME azure-query-endpoint.py → query_endpoint.py
│       │   ├── query_endpoint.py         🔄 RENAME health.py → health_endpoint.py
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
│   │   ├── settings.py                   ✅ KEEP - Main settings (split later)
│   │   ├── azure_config_validator.py     ✅ KEEP - Azure validation
│   │   ├── environments/
│   │   │   ├── dev.env                   ✅ KEEP
│   │   │   ├── staging.env               ✅ KEEP
│   │   │   └── prod.env                  ✅ KEEP
│   │   └── templates/                    ✅ KEEP - Config templates
│   └── integrations/                     # External service coordination
│       ├── __init__.py                   ✅ KEEP
│       ├── azure_manager.py              🆕 NEW - Thin coordinator only (from azure_services.py)
│       └── azure_openai_wrapper.py       🔄 RENAME azure_openai.py → azure_openai_wrapper.py
│
├── 📊 **Data & Processing Layer**
│   ├── data/                             # Data storage (existing structure)
│   │   ├── raw/                          ✅ KEEP - Raw input data
│   │   ├── processed/                    ✅ KEEP - Processed outputs
│   │   ├── cache/                        ✅ KEEP - Caching
│   │   ├── demo/                         ✅ KEEP - Demo data
│   │   ├── gnn_models/                   ✅ KEEP - Trained models
│   │   └── [other existing subdirs]      ✅ KEEP - All current data structure
│   ├── scripts/                          # Consolidated operational tools
│   │   ├── __init__.py                   🆕 NEW
│   │   ├── rag_cli.py                    🆕 NEW - Unified CLI entry point
│   │   ├── data_pipeline.py              🆕 NEW - Consolidate data_processing/ (9 files)
│   │   ├── azure_setup.py                🆕 NEW - Consolidate azure_services/ (4 files)
│   │   ├── gnn_trainer.py                🆕 NEW - Consolidate gnn_training/ (8 files)
│   │   ├── test_validator.py             🆕 NEW - Consolidate testing/ (15 files)
│   │   ├── workflow_analyzer.py          🆕 NEW - Consolidate workflows/ (5 files)
│   │   └── demo_runner.py                🆕 NEW - Consolidate demos/ (3 files)
│   └── prompt_flows/                     # Azure Prompt Flow configs
│       └── universal_knowledge_extraction/ ✅ KEEP - All existing files
│
├── 🔧 **Operations & Development**
│   ├── tests/
│   │   ├── __init__.py                   🆕 NEW
│   │   ├── test_consolidated_codebase.py ✅ KEEP
│   │   ├── unit/                         🆕 NEW - Unit tests by layer
│   │   │   ├── test_core.py              🆕 NEW
│   │   │   ├── test_services.py          🆕 NEW
│   │   │   └── test_api.py               🆕 NEW
│   │   ├── integration/                  🆕 NEW - Integration tests
│   │   │   ├── test_azure_integration.py 🆕 NEW
│   │   │   └── test_workflow_integration.py 🆕 NEW
│   │   └── fixtures/                     🆕 NEW - Test fixtures
│   ├── logs/                             ✅ KEEP - Runtime logs
│   ├── outputs/                          ✅ KEEP - Generated models & results
│   └── docs/                             ✅ KEEP - All existing documentation
│
└── 🗑️ **Removed/Consolidated**
    ├── utilities/                        ❌ DELETE - Moved to core/utilities/
    ├── integrations/azure_services.py    ❌ DELETE - Split into 4 services
    ├── core/orchestration/               ❌ DELETE - Moved to services/
    ├── core/workflow/                    ❌ DELETE - Merged into services/workflow_service.py
    ├── core/prompt_generation/           ❌ DELETE - Moved to services/prompt_service.py
    ├── core/prompt_flow/                 ❌ DELETE - Moved to services/flow_service.py
    └── scripts/organized/                ❌ DELETE - Consolidated into 6 scripts
```

## 🔥 File Naming Issues & Corrections

### **Current Naming Problems:**
1. **Hyphens in Python files**: `azure-query-endpoint.py` (requires special imports)
2. **Inconsistent conventions**: `azure_services.py` vs `azure-query-endpoint.py`
3. **Generic names**: `demo_simple.py`, `health.py` (not descriptive)
4. **Long descriptive names**: `gnn_training_evidence_orchestrator.py`
5. **Mixed conventions**: `workflow_stream.py` vs `azure-workflow-manager.py`

### **Naming Conventions Applied:**
- **Snake_case only** (no hyphens)
- **Descriptive suffixes**: `_endpoint.py`, `_service.py`, `_client.py`
- **Consistent prefixes**: `azure_*` for Azure-specific, none for generic
- **Layer indicators**: Clear indication of architectural layer

## 📊 File Migration Summary

### **Eliminations:**
- **4 directories removed**: utilities/, core/orchestration/, core/workflow/, core/prompt_*
- **1 massive file split**: integrations/azure_services.py (921 lines) → 4 services
- **44 scripts consolidated** → 6 focused tools
- **Core/orchestration eliminated** (754 lines merged)

### **Creations:**
- **9 new service files** from split/moved business logic
- **6 new script tools** from 44 consolidated scripts  
- **8 new API organization files** (middleware, models, streaming)
- **5 new test organization files** (unit, integration, fixtures)

### **Renames:**
- **7 API endpoints** cleaned up (remove hyphens, add suffixes)
- **3 utility files** moved and renamed for consistency
- **2 integration files** renamed for clarity

## ✅ Benefits of Target Structure

### **Architectural Benefits:**
1. **Clear Layer Separation**: Presentation → Business → Infrastructure → Data
2. **Single Responsibility**: Each file has focused purpose
3. **Clean Dependencies**: Unidirectional downward flow
4. **Testable Design**: Each layer independently testable

### **Developer Experience:**
1. **Intuitive Navigation**: Find files by layer and purpose
2. **Consistent Naming**: Predictable file locations and names  
3. **Manageable Size**: No 900+ line files
4. **Clear Boundaries**: Know which layer to modify for changes

### **Operations Benefits:**
1. **Simplified Scripts**: 6 tools instead of 44 files
2. **Unified CLI**: Single entry point for all operations
3. **Organized Testing**: Tests grouped by layer and type
4. **Clean Deployment**: Clear separation of concerns

---

**Next Step**: Create detailed file-by-file migration plan with specific commands and dependency updates.