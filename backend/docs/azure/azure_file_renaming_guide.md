# Azure-Aligned File Renaming Guide

**Universal RAG → Azure Services Design Patterns**

---

## 📋 Current Naming Issues Analysis

### **Problem: Generic "Universal" Naming**
Current Universal RAG files use generic naming that doesn't align with Azure services architecture:

```
❌ Current Naming Pattern:
universal_text_processor.py     → Generic, unclear service responsibility
universal_vector_search.py      → Generic, doesn't indicate Azure Cognitive Search
universal_llm_interface.py      → Generic, doesn't indicate Azure OpenAI
enhanced_rag_universal.py       → Confusing, mixed naming convention
```

### **Problem: Inconsistent Service Mapping**
Current names don't clearly map to Azure services they integrate with:

```
❌ Service Mapping Issues:
universal_vector_search.py      → Should indicate Azure Cognitive Search integration
universal_knowledge_extractor.py → Should indicate Azure ML/OpenAI integration  
universal_gnn_processor.py      → Should indicate Azure ML integration
```

---

## 🎯 Azure-Aligned Naming Strategy

### **Azure Service-Oriented Naming Pattern**
```
✅ New Pattern: {azure-service}-{function}-{component}.py

Examples:
azure-openai-text-processor.py     → Clear Azure OpenAI integration
azure-search-vector-service.py     → Clear Cognitive Search integration
azure-ml-knowledge-extractor.py    → Clear ML service integration
```

### **Directory Structure Alignment**
```
✅ Azure Services Directory Mapping:
core/azure-openai/          → Azure OpenAI integrations
core/azure-search/           → Azure Cognitive Search integrations  
core/azure-ml/               → Azure ML integrations
core/azure-cosmos/           → Azure Cosmos DB integrations
core/orchestration/          → Cross-service orchestration
```

---

## 🔄 Comprehensive File Renaming Plan

### **Phase 1: Core Service Components**

#### **Azure OpenAI Service Components**
```bash
# Current → Azure-Aligned Rename
universal_llm_interface.py              → azure-openai-completion-service.py
universal_text_processor.py             → azure-openai-text-processor.py
universal_knowledge_extractor.py        → azure-openai-knowledge-extractor.py
optimized_llm_extractor.py              → azure-openai-extraction-client.py
```

#### **Azure Cognitive Search Components**
```bash
# Current → Azure-Aligned Rename  
universal_vector_search.py              → azure-search-vector-service.py
universal_query_analyzer.py             → azure-search-query-analyzer.py
```

#### **Azure ML Service Components**
```bash
# Current → Azure-Aligned Rename
universal_gnn_processor.py              → azure-ml-gnn-processor.py
universal_classifier.py                 → azure-ml-classification-service.py
```

#### **Azure Orchestration Components**
```bash
# Current → Azure-Aligned Rename
universal_rag_orchestrator_complete.py  → azure-rag-orchestration-service.py
enhanced_rag_universal.py               → azure-rag-enhanced-pipeline.py
```

### **Phase 2: Supporting Infrastructure**

#### **Workflow Management**
```bash
# Current → Azure-Aligned Rename
universal_workflow_manager.py           → azure-workflow-manager.py
workflow_analysis.py                    → azure-workflow-analyzer.py
```

#### **Data Models**
```bash
# Current → Azure-Aligned Rename
universal_models.py                     → azure-rag-data-models.py
```

#### **Scripts & Utilities**
```bash
# Current → Azure-Aligned Rename
universal_rag_workflow_demo.py          → azure-rag-demo-script.py
data_preparation_workflow.py            → azure-data-preparation-pipeline.py
query_processing_workflow.py            → azure-query-processing-pipeline.py
```

### **Phase 3: Directory Restructuring**

#### **Current Structure**
```
backend/core/
├── orchestration/
├── generation/
├── retrieval/
├── enhancement/
├── knowledge/
├── extraction/
├── gnn/
└── workflow/
```

#### **Azure-Aligned Structure**
```
backend/core/
├── azure-openai/           # Azure OpenAI service integrations
│   ├── completion-service.py
│   ├── text-processor.py
│   ├── knowledge-extractor.py
│   └── extraction-client.py
├── azure-search/           # Azure Cognitive Search integrations
│   ├── vector-service.py
│   ├── query-analyzer.py
│   └── index-manager.py
├── azure-ml/               # Azure ML service integrations
│   ├── gnn-processor.py
│   ├── classification-service.py
│   └── model-registry.py
├── azure-cosmos/           # Azure Cosmos DB integrations
│   ├── graph-service.py
│   └── document-store.py
├── orchestration/          # Cross-service orchestration
│   ├── rag-orchestration-service.py
│   ├── enhanced-pipeline.py
│   └── workflow-manager.py
└── models/                 # Shared data models
    └── rag-data-models.py
```

---

## 🛠️ Step-by-Step Renaming Instructions

### **Pre-Migration Validation**
```bash
# 1. Backup current state
cd backend
cp -r core core_backup_$(date +%Y%m%d)

# 2. Verify current imports work
python -c "
from core.orchestration.universal_rag_orchestrator_complete import UniversalRAGOrchestrator
from core.generation.universal_llm_interface import UniversalLLMInterface
from core.retrieval.universal_vector_search import UniversalVectorSearch
print('✅ Current imports working')
"

# 3. Update pyproject.toml to include azure-* packages
# Add to include: ["api*", "core*", "data*", "config*", "utilities*", "integrations*", "azure*"]
```

### **Phase 1: Core Files Renaming**

#### **Step 1: Azure OpenAI Components**
```bash
# Create new Azure OpenAI directory
mkdir -p backend/core/azure-openai

# Rename and move files
cd backend/core
mv generation/universal_llm_interface.py azure-openai/completion-service.py
mv knowledge/universal_text_processor.py azure-openai/text-processor.py
mv extraction/universal_knowledge_extractor.py azure-openai/knowledge-extractor.py
mv extraction/optimized_llm_extractor.py azure-openai/extraction-client.py

# Create __init__.py
cat > azure-openai/__init__.py << 'EOF'
"""Azure OpenAI service integrations for Universal RAG"""
from .completion-service import UniversalLLMInterface as AzureOpenAICompletionService
from .text-processor import UniversalTextProcessor as AzureOpenAITextProcessor
from .knowledge-extractor import UniversalKnowledgeExtractor as AzureOpenAIKnowledgeExtractor
from .extraction-client import OptimizedLLMExtractor as AzureOpenAIExtractionClient

__all__ = [
    'AzureOpenAICompletionService',
    'AzureOpenAITextProcessor', 
    'AzureOpenAIKnowledgeExtractor',
    'AzureOpenAIExtractionClient'
]
EOF
```

#### **Step 2: Azure Cognitive Search Components**
```bash
# Create new Azure Search directory
mkdir -p backend/core/azure-search

# Rename and move files
mv retrieval/universal_vector_search.py azure-search/vector-service.py
mv enhancement/universal_query_analyzer.py azure-search/query-analyzer.py

# Create __init__.py
cat > azure-search/__init__.py << 'EOF'
"""Azure Cognitive Search integrations for Universal RAG"""
from .vector-service import UniversalVectorSearch as AzureSearchVectorService
from .query-analyzer import UniversalQueryAnalyzer as AzureSearchQueryAnalyzer

__all__ = [
    'AzureSearchVectorService',
    'AzureSearchQueryAnalyzer'
]
EOF
```

#### **Step 3: Azure ML Components**
```bash
# Create new Azure ML directory
mkdir -p backend/core/azure-ml

# Rename and move files
mv gnn/universal_gnn_processor.py azure-ml/gnn-processor.py
mv classification/universal_classifier.py azure-ml/classification-service.py

# Create __init__.py
cat > azure-ml/__init__.py << 'EOF'
"""Azure ML service integrations for Universal RAG"""
from .gnn-processor import UniversalGNNDataProcessor as AzureMLGNNProcessor
from .classification-service import UniversalClassificationPipeline as AzureMLClassificationService

__all__ = [
    'AzureMLGNNProcessor',
    'AzureMLClassificationService'
]
EOF
```

#### **Step 4: Orchestration Components**
```bash
# Rename orchestration files in place
cd backend/core/orchestration
mv universal_rag_orchestrator_complete.py rag-orchestration-service.py
mv enhanced_rag_universal.py enhanced-pipeline.py

# Update orchestration __init__.py
cat > __init__.py << 'EOF'
"""Azure RAG orchestration services"""
from .rag-orchestration-service import UniversalRAGOrchestrator as AzureRAGOrchestrationService
from .enhanced-pipeline import EnhancedUniversalRAG as AzureRAGEnhancedPipeline

__all__ = [
    'AzureRAGOrchestrationService',
    'AzureRAGEnhancedPipeline'
]
EOF
```

### **Phase 2: Update Import References**

#### **Step 1: Create Migration Compatibility Layer**
```bash
# Create compatibility imports in core/__init__.py
cat > backend/core/__init__.py << 'EOF'
"""Universal RAG Core - Azure Services Architecture"""

# Azure OpenAI Services
from .azure-openai import (
    AzureOpenAICompletionService as UniversalLLMInterface,
    AzureOpenAITextProcessor as UniversalTextProcessor,
    AzureOpenAIKnowledgeExtractor as UniversalKnowledgeExtractor
)

# Azure Search Services  
from .azure-search import (
    AzureSearchVectorService as UniversalVectorSearch,
    AzureSearchQueryAnalyzer as UniversalQueryAnalyzer
)

# Azure ML Services
from .azure-ml import (
    AzureMLGNNProcessor as UniversalGNNDataProcessor,
    AzureMLClassificationService as UniversalClassificationPipeline
)

# Azure Orchestration Services
from .orchestration import (
    AzureRAGOrchestrationService as UniversalRAGOrchestrator,
    AzureRAGEnhancedPipeline as EnhancedUniversalRAG
)

# Legacy compatibility aliases
UniversalLLMInterface = AzureOpenAICompletionService
UniversalTextProcessor = AzureOpenAITextProcessor
UniversalVectorSearch = AzureSearchVectorService
UniversalRAGOrchestrator = AzureRAGOrchestrationService

__all__ = [
    # Azure service components
    'AzureOpenAICompletionService', 'AzureOpenAITextProcessor',
    'AzureSearchVectorService', 'AzureSearchQueryAnalyzer', 
    'AzureMLGNNProcessor', 'AzureMLClassificationService',
    'AzureRAGOrchestrationService', 'AzureRAGEnhancedPipeline',
    
    # Legacy compatibility
    'UniversalLLMInterface', 'UniversalTextProcessor',
    'UniversalVectorSearch', 'UniversalRAGOrchestrator'
]
EOF
```

#### **Step 2: Update Internal Imports**
```bash
# Update imports in renamed files (example for completion-service.py)
cd backend/core/azure-openai

# Update completion-service.py imports
sed -i 's/from core\.models\.universal_models/from core.models.rag-data-models/g' completion-service.py
sed -i 's/from config\.settings/from config.settings/g' completion-service.py

# Repeat for all renamed files in each Azure service directory
```

#### **Step 3: Update Script References**
```bash
# Update script imports
cd backend/scripts

# Rename scripts with Azure naming
mv universal_rag_workflow_demo.py azure-rag-demo-script.py
mv data_preparation_workflow.py azure-data-preparation-pipeline.py  
mv query_processing_workflow.py azure-query-processing-pipeline.py

# Update imports in renamed scripts
sed -i 's/from core\.orchestration\.universal_rag_orchestrator_complete/from core.orchestration.rag-orchestration-service/g' *.py
sed -i 's/from core\.generation\.universal_llm_interface/from core.azure-openai.completion-service/g' *.py
```

### **Phase 3: Validation & Testing**

#### **Step 1: Import Validation**
```bash
# Test new Azure-aligned imports
cd backend
python -c "
# Test Azure service imports
from core.azure-openai import AzureOpenAICompletionService
from core.azure-search import AzureSearchVectorService  
from core.azure-ml import AzureMLGNNProcessor
from core.orchestration import AzureRAGOrchestrationService

# Test legacy compatibility
from core import UniversalLLMInterface, UniversalTextProcessor

print('✅ All Azure-aligned imports working')
"
```

#### **Step 2: Functionality Testing**
```bash
# Run existing tests with new structure
make test-unit

# Test specific Azure service components
python -c "
from core.azure-openai import AzureOpenAICompletionService
service = AzureOpenAICompletionService()
print('✅ Azure OpenAI service initialized')
"

python -c "  
from core.azure-search import AzureSearchVectorService
service = AzureSearchVectorService()
print('✅ Azure Search service initialized')
"
```

#### **Step 3: Integration Testing**
```bash
# Test full Azure RAG pipeline
cd backend
python scripts/azure-rag-demo-script.py

# Verify API endpoints still work
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/docs  # Verify API documentation
```

---

## 📊 Renaming Benefits

### **Azure Services Alignment**
✅ **Clear Service Mapping**: Each file clearly indicates which Azure service it integrates with  
✅ **Consistent Naming**: Follows Azure SDK and services naming conventions  
✅ **Service Separation**: Clear boundaries between different Azure services  
✅ **Future-Proof**: Easy to add new Azure services with consistent patterns

### **Developer Experience**
✅ **Intuitive Navigation**: Developers immediately understand file purposes  
✅ **Service Discovery**: Easy to locate specific Azure service integrations  
✅ **Maintenance Efficiency**: Clear separation reduces cross-service dependencies  
✅ **Documentation Clarity**: File names self-document their Azure service roles

### **Production Readiness**
✅ **Enterprise Naming**: Professional naming conventions for enterprise deployment  
✅ **Service Monitoring**: Clear service boundaries for Azure monitoring and alerts  
✅ **Deployment Isolation**: Azure services can be deployed and scaled independently  
✅ **Compliance Alignment**: Naming conventions align with Azure governance standards

---

## 🚀 Quick-Start Migration

### **Immediate Actions (30 minutes)**
```bash
# 1. Backup current state
cd backend && cp -r core core_backup_$(date +%Y%m%d)

# 2. Create Azure service directories
mkdir -p core/{azure-openai,azure-search,azure-ml,azure-cosmos}

# 3. Move core service files
mv core/generation/universal_llm_interface.py core/azure-openai/completion-service.py
mv core/retrieval/universal_vector_search.py core/azure-search/vector-service.py

# 4. Test basic imports still work
python -c "from core.azure-openai.completion-service import UniversalLLMInterface; print('✅')"
```

### **Validation Commands**
```bash
# Verify renamed structure works
make test-unit                           # Run existing tests
python scripts/azure-rag-demo-script.py # Test renamed demo script
curl http://localhost:8000/api/v1/health # Verify API functionality
```

### **Rollback Plan (if needed)**
```bash
# Restore original structure
cd backend
rm -rf core
mv core_backup_$(date +%Y%m%d) core
```

---

## 📈 Implementation Timeline

| **Phase** | **Duration** | **Actions** | **Validation** |
|-----------|--------------|-------------|----------------|
| **Phase 1** | 2 hours | Rename core service files | Import testing |
| **Phase 2** | 1 hour | Update import references | Functionality testing |  
| **Phase 3** | 30 minutes | Integration testing | Full system validation |

**Total Duration**: 3.5 hours  
**Risk Level**: Low (backward compatibility maintained)  
**Benefits**: Immediate Azure services alignment and enterprise readiness

---

## 📞 Support & Next Steps

### **Post-Migration Actions**
1. **Update Documentation**: Refresh all documentation with new Azure-aligned names
2. **Team Communication**: Notify team of new naming conventions
3. **IDE Configuration**: Update IDE imports and navigation bookmarks
4. **Monitoring Updates**: Update any monitoring dashboards with new service names

### **Future Azure Service Additions**
With the new naming structure, adding Azure services follows a clear pattern:
```bash
# New Azure service integration template
mkdir backend/core/azure-{service-name}
touch backend/core/azure-{service-name}/{function}-service.py
# Follow established naming conventions
```

---

**🎯 Migration Status**: Ready for immediate implementation  
**🚀 Next Action**: Execute Phase 1 renaming with provided scripts

This Azure-aligned naming structure positions your Universal RAG system for enterprise production deployment with clear Azure services integration patterns.