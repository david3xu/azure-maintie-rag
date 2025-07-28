# Target Core Directory Structure

This document defines the **target structure** for the core/ directory after consolidation.

## 🎯 Target Structure Diagram

```
core/
├── __init__.py                     ✅ KEEP - Core module entry point
│
├── azure_unified/                  ✅ NEW - Our consolidated Azure clients
│   ├── __init__.py                 ✅ Entry point for unified clients
│   ├── base_client.py             ✅ Common patterns and base class
│   ├── openai_client.py           ✅ All OpenAI functionality (replaces 10 files)
│   ├── cosmos_client.py           ✅ All Cosmos DB functionality (replaces 2 files)
│   ├── search_client.py           ✅ All Search functionality (replaces 3 files)
│   └── storage_client.py          ✅ All Storage functionality (replaces 4 files)
│
├── services/                       ✅ NEW - High-level business logic services
│   ├── __init__.py                 ✅ Entry point for services
│   ├── knowledge_service.py       ✅ Knowledge extraction workflows
│   ├── graph_service.py           ✅ Graph operations and analysis
│   ├── ml_service.py              ✅ ML training and model management
│   └── query_service.py           ✅ Query processing and RAG workflows
│
├── azure_openai/                   🔄 CLEANED - Import redirection only
│   └── __init__.py                 🔄 Redirects to azure_unified/openai_client.py
│
├── azure_search/                   🔄 CLEANED - Import redirection only
│   └── __init__.py                 🔄 Redirects to azure_unified/search_client.py
│
├── azure_storage/                  🔄 CLEANED - Import redirection only
│   └── __init__.py                 🔄 Redirects to azure_unified/storage_client.py
│
├── azure_cosmos/                   🔄 SIMPLIFIED - Keep main client only
│   └── cosmos_gremlin_client.py    ✅ KEEP - Main implementation
│
├── azure_auth/                     ✅ KEEP - Authentication components
│   └── session_manager.py          ✅ KEEP - Session management
│
├── azure_monitoring/               ✅ KEEP - Monitoring components
│   └── app_insights_client.py      ✅ KEEP - Application insights
│
├── azure_ml/                       ✅ KEEP - ML specialized components
│   ├── __init__.py                 ✅ KEEP
│   ├── classification_service.py   ✅ KEEP - Specialized classification
│   ├── gnn_orchestrator.py        ✅ KEEP - GNN orchestration
│   ├── gnn_processor.py           ✅ KEEP - GNN processing
│   ├── gnn_training_evidence_orchestrator.py ✅ KEEP
│   ├── ml_client.py               ✅ KEEP - Core ML client
│   └── gnn/                       ✅ KEEP - GNN implementation details
│       ├── data_bridge.py         ✅ KEEP
│       ├── data_loader.py         ✅ KEEP
│       ├── feature_engineering.py ✅ KEEP
│       ├── model.py               ✅ KEEP
│       ├── model_quality_assessor.py ✅ KEEP
│       ├── train_gnn_workflow.py  ✅ KEEP
│       ├── trainer.py             ✅ KEEP
│       └── unified_training_pipeline.py ✅ KEEP
│
├── models/                         ✅ KEEP - Data models
│   ├── __init__.py                 ✅ KEEP
│   ├── gnn_data_models.py         ✅ KEEP
│   └── universal_rag_models.py    ✅ KEEP
│
├── orchestration/                  ✅ KEEP - High-level orchestration
│   ├── __init__.py                 ✅ KEEP
│   ├── enhanced_pipeline.py       ✅ KEEP
│   └── rag_orchestration_service.py ✅ KEEP
│
├── prompt_flow/                    ✅ KEEP - Prompt flow integration
│   ├── prompt_flow_integration.py ✅ KEEP
│   └── prompt_flow_monitoring.py  ✅ KEEP
│
├── prompt_generation/              ✅ KEEP - Prompt generation
│   └── adaptive_context_generator.py ✅ KEEP
│
├── utilities/                      ✅ KEEP - Utility functions
│   ├── __init__.py                 ✅ KEEP
│   └── intelligent_document_processor.py ✅ KEEP
│
└── workflow/                       ✅ KEEP - Workflow management
    ├── azure-workflow-manager.py  ✅ KEEP
    ├── cost_tracker.py            ✅ KEEP
    ├── data_workflow_evidence.py  ✅ KEEP
    └── progress_tracker.py        ✅ KEEP
```

---

## 📊 File Count Goals

### Before Consolidation:
- **Azure OpenAI**: 11 files → **1 file** (10 files eliminated)
- **Azure Search**: 4 files → **1 file** (3 files eliminated)  
- **Azure Storage**: 4 files → **1 file** (4 files eliminated, directory recreated)
- **Azure Cosmos**: 2 files → **1 file** (1 file eliminated)
- **Total Reduction**: 18 files eliminated

### After Consolidation:
- **azure_unified/**: 6 files (new consolidated clients)
- **services/**: 5 files (new high-level services)
- **Azure modules**: 4 files (import redirections only)
- **Other modules**: Unchanged (essential functionality preserved)

---

## ✅ Success Criteria

1. **18 redundant files deleted** from Azure modules
2. **6 unified client files** created in azure_unified/
3. **5 service files** created in services/
4. **Backwards compatibility** maintained via import redirections
5. **All tests passing** (100% success rate)
6. **Clean directory structure** with logical organization

---

## 🧪 Verification Commands

```bash
# Check structure matches target
find core/ -type f -name "*.py" | sort

# Count files in each module
echo "azure_openai: $(find core/azure_openai/ -name "*.py" | wc -l) files"
echo "azure_search: $(find core/azure_search/ -name "*.py" | wc -l) files"  
echo "azure_storage: $(find core/azure_storage/ -name "*.py" | wc -l) files"
echo "azure_cosmos: $(find core/azure_cosmos/ -name "*.py" | wc -l) files"
echo "azure_unified: $(find core/azure_unified/ -name "*.py" | wc -l) files"
echo "services: $(find core/../services/ -name "*.py" 2>/dev/null | wc -l) files"

# Test functionality
python test_consolidated_codebase.py
```

---

## 📋 Progress Tracking

- [x] Create azure_unified/ with 6 files
- [x] Create services/ with 5 files  
- [x] Delete 18 redundant Azure client files
- [x] Update import redirections for backwards compatibility
- [x] Test consolidated codebase
- [x] **VERIFY: Current structure matches target diagram exactly**
- [x] **CLEANUP COMPLETE: 100% success rate achieved**

**Use this document to verify we've achieved the target structure!**