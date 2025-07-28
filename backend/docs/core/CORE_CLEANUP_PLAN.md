# Core Directory Cleanup Plan
**Simple focused cleanup of core/ directory only**

## 📋 Current State Analysis

### What We Have:
- ✅ **NEW**: `core/azure_unified/` (6 files) - Our consolidated clients
- 🗑️ **OLD**: Multiple scattered Azure client files that are now redundant

### What We Need To Do:
1. **DELETE** old redundant Azure client files
2. **KEEP** essential non-Azure files 
3. **MODIFY** a few __init__.py files for backwards compatibility

---

## 🗑️ FILES TO DELETE (Simple removals)

### Azure OpenAI - DELETE 10 files:
```bash
rm core/azure_openai/azure_ml_quality_service.py
rm core/azure_openai/azure_monitoring_service.py  
rm core/azure_openai/azure_rate_limiter.py
rm core/azure_openai/azure_text_analytics_service.py
rm core/azure_openai/completion_service.py
rm core/azure_openai/dual_storage_extractor.py
rm core/azure_openai/extraction_client.py
rm core/azure_openai/improved_extraction_client.py
rm core/azure_openai/knowledge_extractor.py
rm core/azure_openai/text_processor.py
```

### Azure Cosmos - DELETE 1 file:
```bash
rm core/azure_cosmos/enhanced_gremlin_client.py
```

### Azure Search - DELETE 3 files:
```bash
rm core/azure_search/query_analyzer.py
rm core/azure_search/search_client.py
rm core/azure_search/vector_service.py
```

### Azure Storage - DELETE 4 files:
```bash
rm core/azure_storage/mock_azure_services.py
rm core/azure_storage/real_azure_services.py
rm core/azure_storage/storage_client.py
rm core/azure_storage/storage_factory.py
```

**Total to delete: 18 files**

---

## ✅ FILES TO KEEP (Don't touch these)

### Our New Unified Structure:
```
core/azure_unified/          ✅ KEEP ALL
├── __init__.py
├── base_client.py  
├── openai_client.py
├── cosmos_client.py
├── search_client.py
└── storage_client.py
```

### Essential Core Files:
```
core/models/                 ✅ KEEP ALL - Data models
core/utilities/              ✅ KEEP ALL - Utility functions  
core/workflow/               ✅ KEEP ALL - Workflow management
core/azure_auth/             ✅ KEEP ALL - Authentication
core/azure_monitoring/       ✅ KEEP ALL - Monitoring
core/azure_ml/               ✅ KEEP ALL - ML components
core/orchestration/          ✅ KEEP ALL - Orchestration
core/prompt_flow/            ✅ KEEP ALL - Prompt flow
core/prompt_generation/      ✅ KEEP ALL - Prompt generation
```

### Special Cases:
```
core/azure_cosmos/cosmos_gremlin_client.py    ✅ KEEP - Main cosmos client
core/azure_openai/__init__.py                 ✅ KEEP - Will modify later
core/azure_search/__init__.py                 ✅ KEEP - Will modify later
```

---

## 🔄 SIMPLE MODIFICATIONS (Just 2 files)

### 1. Update core/azure_openai/__init__.py:
```python
# Redirect to unified client for backwards compatibility
from ..azure_unified.openai_client import UnifiedAzureOpenAIClient as AzureOpenAIClient

__all__ = ['AzureOpenAIClient']
```

### 2. Update core/azure_search/__init__.py:
```python  
# Redirect to unified client for backwards compatibility
from ..azure_unified.search_client import UnifiedSearchClient as SearchClient

__all__ = ['SearchClient']
```

---

## 📊 SIMPLE IMPACT

### Before:
- `core/azure_openai/`: 11 files
- `core/azure_cosmos/`: 2 files  
- `core/azure_search/`: 4 files
- `core/azure_storage/`: 4 files
- **Total**: 21 Azure client files

### After:
- `core/azure_unified/`: 6 files (our new structure)
- `core/azure_cosmos/`: 1 file (keep main client)
- `core/azure_openai/`: 1 file (__init__.py only)
- `core/azure_search/`: 1 file (__init__.py only)  
- **Total**: 9 Azure client files

### **Result: Delete 18 files, keep clean structure**

---

## 🚀 EXECUTION STEPS

### Step 1: Delete redundant files (18 deletions)
```bash
# Delete Azure OpenAI redundant files (10 files)
cd core/azure_openai/
rm azure_ml_quality_service.py azure_monitoring_service.py azure_rate_limiter.py
rm azure_text_analytics_service.py completion_service.py dual_storage_extractor.py  
rm extraction_client.py improved_extraction_client.py knowledge_extractor.py text_processor.py

# Delete Azure Search redundant files (3 files)
cd ../azure_search/
rm query_analyzer.py search_client.py vector_service.py

# Delete Azure Storage redundant files (4 files)  
cd ../azure_storage/
rm mock_azure_services.py real_azure_services.py storage_client.py storage_factory.py

# Delete Azure Cosmos redundant file (1 file)
cd ../azure_cosmos/
rm enhanced_gremlin_client.py
```

### Step 2: Update __init__.py files (2 modifications)
```bash
# Will update these files to redirect imports
# core/azure_openai/__init__.py
# core/azure_search/__init__.py  
```

### Step 3: Test
```bash
python test_consolidated_codebase.py
```

---

## ✅ SUCCESS CRITERIA

- ✅ 18 redundant files deleted
- ✅ New unified structure intact  
- ✅ Essential files preserved
- ✅ Simple backwards compatibility maintained

**That's it! Simple, focused, effective cleanup of just the core directory.**