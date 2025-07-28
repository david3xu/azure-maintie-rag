# Organized Core Directory Cleanup Plan
**Systematic and careful cleanup based on actual directory structure**

## 📋 Current Structure Analysis

### What We Have Now:
```
core/
├── azure_unified/          ✅ NEW - Our consolidated clients (6 files)
├── azure_openai/           🗑️ OLD - 10 redundant files + 1 __init__.py  
├── azure_cosmos/           🔄 MIXED - 1 redundant + 1 keep
├── azure_search/           🗑️ OLD - 3 redundant files + 1 __init__.py
├── azure_storage/          🗑️ OLD - 4 redundant files (no __init__.py)
├── azure_auth/             ✅ KEEP - Authentication (1 file)
├── azure_ml/               ✅ KEEP - ML components (9+ files)
├── azure_monitoring/       ✅ KEEP - Monitoring (1 file)
├── models/                 ✅ KEEP - Data models (3 files)
├── orchestration/          ✅ KEEP - Orchestration (3 files)
├── prompt_flow/            ✅ KEEP - Prompt flow (2 files)
├── prompt_generation/      ✅ KEEP - Prompt generation (1 file)
├── utilities/              ✅ KEEP - Utilities (2 files)
└── workflow/               ✅ KEEP - Workflow (4 files)
```

---

## 🎯 PHASE 1: Delete Redundant Files (Careful Removal)

### 🗑️ azure_openai/ - DELETE 10 files, KEEP 1:
```bash
# Files to DELETE (functionality moved to azure_unified/openai_client.py):
rm core/azure_openai/azure_ml_quality_service.py          # 11KB
rm core/azure_openai/azure_monitoring_service.py          # 6KB  
rm core/azure_openai/azure_rate_limiter.py               # 10KB
rm core/azure_openai/azure_text_analytics_service.py      # 7KB
rm core/azure_openai/completion_service.py               # 13KB
rm core/azure_openai/dual_storage_extractor.py           # 19KB
rm core/azure_openai/extraction_client.py                # 26KB
rm core/azure_openai/improved_extraction_client.py       # 19KB
rm core/azure_openai/knowledge_extractor.py              # 41KB
rm core/azure_openai/text_processor.py                   # 15KB

# KEEP this file - will modify it:
# core/azure_openai/__init__.py ✅ KEEP
```

### 🗑️ azure_search/ - DELETE 3 files, KEEP 1:
```bash
# Files to DELETE (functionality moved to azure_unified/search_client.py):
rm core/azure_search/query_analyzer.py                   # 16KB
rm core/azure_search/search_client.py                    # 25KB
rm core/azure_search/vector_service.py                   # 16KB

# KEEP this file - will modify it:
# core/azure_search/__init__.py ✅ KEEP
```

### 🗑️ azure_storage/ - DELETE ALL 4 files:
```bash
# Files to DELETE (functionality moved to azure_unified/storage_client.py):
rm core/azure_storage/mock_azure_services.py             # 2KB
rm core/azure_storage/real_azure_services.py             # 10KB
rm core/azure_storage/storage_client.py                  # 11KB
rm core/azure_storage/storage_factory.py                 # 5KB

# NOTE: No __init__.py in this directory, so we'll remove the entire directory
rmdir core/azure_storage/
```

### 🗑️ azure_cosmos/ - DELETE 1 file, KEEP 1:
```bash
# File to DELETE (functionality moved to azure_unified/cosmos_client.py):
rm core/azure_cosmos/enhanced_gremlin_client.py          # 10KB

# KEEP this file - it's the main implementation:
# core/azure_cosmos/cosmos_gremlin_client.py ✅ KEEP
```

**Total to delete: 18 files (~167KB saved)**

---

## 🔄 PHASE 2: Update Import Redirections (2 files)

### 1. Modify core/azure_openai/__init__.py:
```python
"""
Azure OpenAI Integration - Redirected to Unified Client
"""
# Redirect to unified client for backwards compatibility
from ..azure_unified.openai_client import UnifiedAzureOpenAIClient

# Maintain backwards compatibility
AzureOpenAIKnowledgeExtractor = UnifiedAzureOpenAIClient
AzureOpenAITextProcessor = UnifiedAzureOpenAIClient
AzureOpenAICompletionService = UnifiedAzureOpenAIClient

__all__ = [
    'UnifiedAzureOpenAIClient',
    'AzureOpenAIKnowledgeExtractor', 
    'AzureOpenAITextProcessor',
    'AzureOpenAICompletionService'
]
```

### 2. Modify core/azure_search/__init__.py:
```python
"""
Azure Search Integration - Redirected to Unified Client  
"""
# Redirect to unified client for backwards compatibility
from ..azure_unified.search_client import UnifiedSearchClient

# Maintain backwards compatibility
SearchClient = UnifiedSearchClient
VectorService = UnifiedSearchClient

__all__ = [
    'UnifiedSearchClient',
    'SearchClient',
    'VectorService'
]
```

### 3. Create core/azure_storage/__init__.py:
```python
"""
Azure Storage Integration - Redirected to Unified Client
"""
# Redirect to unified client  
from ..azure_unified.storage_client import UnifiedStorageClient

# Maintain backwards compatibility
StorageClient = UnifiedStorageClient

__all__ = [
    'UnifiedStorageClient',
    'StorageClient'
]
```

---

## 📊 IMPACT SUMMARY

### File Count Reduction:
```
BEFORE:
├── azure_openai/     11 files (167KB)
├── azure_cosmos/      2 files ( 34KB)  
├── azure_search/      4 files ( 57KB)
├── azure_storage/     4 files ( 28KB)
Total: 21 files (286KB)

AFTER:  
├── azure_unified/     6 files ( 40KB) ✅ NEW
├── azure_openai/      1 file  (  1KB) 🔄 MODIFIED
├── azure_cosmos/      1 file  ( 24KB) ✅ KEPT
├── azure_search/      1 file  (  1KB) 🔄 MODIFIED  
├── azure_storage/     1 file  (  1KB) 🔄 NEW
Total: 10 files (67KB)

REDUCTION: 11 files deleted, 219KB saved (76% reduction)
```

### Directory Structure After Cleanup:
```
core/
├── azure_unified/          ✅ 6 files - Consolidated clients
├── azure_openai/           🔄 1 file  - Import redirection
├── azure_cosmos/           ✅ 1 file  - Main client kept
├── azure_search/           🔄 1 file  - Import redirection
├── azure_storage/          🔄 1 file  - Import redirection  
├── azure_auth/             ✅ 1 file  - Unchanged
├── azure_ml/               ✅ 9 files - Unchanged
├── azure_monitoring/       ✅ 1 file  - Unchanged
├── models/                 ✅ 3 files - Unchanged
├── orchestration/          ✅ 3 files - Unchanged
├── prompt_flow/            ✅ 2 files - Unchanged
├── prompt_generation/      ✅ 1 file  - Unchanged
├── utilities/              ✅ 2 files - Unchanged
└── workflow/               ✅ 4 files - Unchanged
```

---

## 🚀 CAREFUL EXECUTION STEPS

### Step 1: Backup (Safety First)
```bash
cp -r core/ core_backup_$(date +%Y%m%d_%H%M%S)/
```

### Step 2: Delete Files (One by one for safety)
```bash
# Azure OpenAI cleanup
cd core/azure_openai/
rm azure_ml_quality_service.py
rm azure_monitoring_service.py  
rm azure_rate_limiter.py
rm azure_text_analytics_service.py
rm completion_service.py
rm dual_storage_extractor.py
rm extraction_client.py
rm improved_extraction_client.py
rm knowledge_extractor.py
rm text_processor.py

# Azure Search cleanup  
cd ../azure_search/
rm query_analyzer.py
rm search_client.py
rm vector_service.py

# Azure Storage cleanup (remove entire directory)
cd ..
rm -rf azure_storage/

# Azure Cosmos cleanup
cd azure_cosmos/
rm enhanced_gremlin_client.py
cd ..
```

### Step 3: Create Import Redirections
```bash
# Will create/modify the __init__.py files as shown above
```

### Step 4: Test Everything
```bash
cd ../../
python test_consolidated_codebase.py
```

---

## ✅ SUCCESS CRITERIA

- ✅ **18 redundant files deleted safely**
- ✅ **Backwards compatibility maintained** 
- ✅ **No import errors**
- ✅ **Clean directory structure**
- ✅ **76% reduction in Azure client code**

This is a **careful, organized approach** that removes redundancy while preserving functionality!