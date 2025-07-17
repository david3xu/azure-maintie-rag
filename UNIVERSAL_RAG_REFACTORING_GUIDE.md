# Universal RAG Architectural Refactoring Guide

**Transforming MaintIE Domain-Specific RAG to True Universal RAG**

---

## 🎯 **Refactoring Scope**

Our current backend is fundamentally architected around **MaintIE domain-specific assumptions**. To achieve true Universal RAG, we need comprehensive architectural refactoring across multiple layers.

### **Current Problems:**

| **Component** | **Domain-Specific Issue** | **Universal RAG Solution** |
|---------------|---------------------------|----------------------------|
| 🏗️ **Data Processing** | `MaintIEDataTransformer` expects gold/silver JSON | `UniversalTextProcessor` for pure text |
| 📊 **Entity Types** | Hardcoded `EntityType` enum | Dynamic LLM-discovered types |
| 🔗 **Relation Types** | Hardcoded `RelationType` enum | Dynamic LLM-discovered types |
| 📋 **Schema Files** | Requires `scheme.json` | No schema files needed |
| 🥇 **Quality Classification** | Gold/silver distinction | All text treated equally |
| 🧠 **Knowledge Extraction** | MaintIE annotation structure | Universal text processing |

---

## 🚀 **Phase 1: Core Components Refactored ✅**

### **1. Universal Text Processor**
**File**: `backend/core/knowledge/universal_text_processor.py`

**Key Features:**
- ✅ Works with pure text files (`.txt`, `.md`)
- ✅ No domain assumptions or hardcoded schemas
- ✅ No gold/silver classification
- ✅ LLM-powered knowledge extraction
- ✅ Dynamic entity/relation type discovery

**Replaces:**
- `MaintIEDataTransformer` (removes scheme.json dependency)
- Gold/silver processing logic
- Domain-specific annotation parsing

### **2. Universal Data Models**
**File**: `backend/core/models/universal_models.py`

**Key Features:**
- ✅ `UniversalEntity` with dynamic entity types (string, not enum)
- ✅ `UniversalRelation` with dynamic relation types (string, not enum)
- ✅ `UniversalDocument` works with any text content
- ✅ `UniversalQueryAnalysis` - domain-agnostic query understanding
- ✅ Legacy aliases for backward compatibility

**Replaces:**
- Hardcoded `EntityType` enum
- Hardcoded `RelationType` enum
- `MaintenanceEntity`, `MaintenanceRelation`, `MaintenanceDocument`

---

## 🔄 **Phase 2: Components to Refactor**

### **3. Universal Query Analyzer**
**Target**: `backend/core/enhancement/query_analyzer.py`

**Current Issues:**
```python
# PROBLEM: Hardcoded maintenance patterns
self.equipment_patterns = self._build_equipment_patterns()
self.failure_patterns = self._build_failure_patterns()

# PROBLEM: Domain-specific vocabulary
self.safety_critical = set(["pump", "valve", "motor"])
```

**Universal Solution:**
```python
# SOLUTION: Dynamic LLM-discovered patterns
self.discovered_concepts = llm_extractor.discover_domain_concepts(texts)
self.query_patterns = self._build_universal_patterns(self.discovered_concepts)

# SOLUTION: Domain-agnostic analysis
def analyze_query_universal(self, query: str) -> UniversalQueryAnalysis:
    # LLM-powered query understanding without domain assumptions
```

### **4. Universal Vector Search**
**Target**: `backend/core/retrieval/vector_search.py`

**Current Issues:**
```python
# PROBLEM: Expects MaintIE document structure
def build_index_from_documents(self, documents: Dict[str, MaintenanceDocument])

# PROBLEM: Domain-specific metadata
metadata = {"confidence_base": gold_vs_silver, "maintie_source": ...}
```

**Universal Solution:**
```python
# SOLUTION: Works with any document type
def build_index_from_documents(self, documents: Dict[str, UniversalDocument])

# SOLUTION: Domain-agnostic metadata
metadata = {"text_source": filename, "domain": domain_name}
```

### **5. Universal GNN Processing**
**Target**: `backend/core/gnn/data_preparation.py`

**Current Issues:**
```python
class MaintIEGNNDataProcessor:
    # PROBLEM: Expects MaintIE annotation structure
    def __init__(self, data_transformer: MaintIEDataTransformer)
```

**Universal Solution:**
```python
class UniversalGNNDataProcessor:
    # SOLUTION: Works with universal text processor
    def __init__(self, text_processor: UniversalTextProcessor)
```

### **6. Universal LLM Interface**
**Target**: `backend/core/generation/llm_interface.py`

**Current Issues:**
```python
# PROBLEM: Maintenance-specific response templates
safety_warning = "⚠️ SAFETY: This involves maintenance procedures..."
```

**Universal Solution:**
```python
# SOLUTION: Dynamic domain-appropriate responses
domain_context = self.get_domain_context(query.domain)
response_template = self.generate_domain_template(domain_context)
```

---

## 📋 **Phase 3: Orchestration Refactoring**

### **7. Universal RAG Orchestrator**
**Target**: `backend/core/orchestration/enhanced_rag.py`

**Current Issues:**
```python
# PROBLEM: Hard-wired to MaintIE components
self.data_transformer = MaintIEDataTransformer()
self.query_analyzer = MaintenanceQueryAnalyzer(transformer)
```

**Universal Solution:**
```python
# SOLUTION: Dynamic component initialization
self.text_processor = UniversalTextProcessor(domain_name)
self.query_analyzer = UniversalQueryAnalyzer(self.text_processor)
```

### **8. Universal API Endpoints**
**Target**: `backend/api/routes/streaming_query.py`

**Current Issues:**
```python
# PROBLEM: Hardcoded maintenance workflow
data_transformer = MaintIEDataTransformer()
query_analyzer = MaintenanceQueryAnalyzer(data_transformer)
```

**Universal Solution:**
```python
# SOLUTION: Dynamic domain workflow
domain = request.get("domain", "general")
text_processor = UniversalTextProcessor(domain)
query_analyzer = UniversalQueryAnalyzer(text_processor)
```

---

## 🛠️ **Migration Strategy**

### **Backward Compatibility**
```python
# Legacy aliases in universal_models.py maintain compatibility
MaintenanceEntity = UniversalEntity
MaintenanceRelation = UniversalRelation
MaintenanceDocument = UniversalDocument

# Existing code continues working while we migrate
```

### **Gradual Migration Steps**

1. **✅ Phase 1 Complete**: Core universal components created
2. **🔄 Phase 2**: Refactor component by component
3. **🔧 Phase 3**: Update orchestration and APIs
4. **🧹 Phase 4**: Remove legacy domain-specific code

### **Testing Strategy**
```bash
# Test with sample text files
echo "Sample maintenance text" > backend/data/raw/sample.txt

# Run universal processing
python backend/core/knowledge/universal_text_processor.py

# Verify no domain assumptions
# ✅ No scheme.json required
# ✅ No gold/silver classification
# ✅ Works with any text content
```

---

## 📊 **Impact Analysis**

### **Files to Modify:**

| **Priority** | **File** | **Change Type** | **Impact** |
|--------------|----------|-----------------|------------|
| 🔴 **High** | `query_analyzer.py` | Major refactor | Remove domain patterns |
| 🔴 **High** | `vector_search.py` | Medium refactor | Accept universal documents |
| 🟡 **Medium** | `gnn/data_preparation.py` | Major refactor | Remove MaintIE assumptions |
| 🟡 **Medium** | `llm_interface.py` | Medium refactor | Dynamic response generation |
| 🟢 **Low** | `orchestration/*.py` | Minor refactor | Update component initialization |

### **Files to Eventually Remove:**

- `schema_processor.py` (scheme.json dependency)
- `metadata_manager.py` (domain-specific metadata)
- `simple_extraction.py` (MaintIE-specific patterns)
- Any `maintie_*` prefixed files

---

## 🎯 **Success Criteria**

### **Universal RAG System Should:**
✅ Work with **any text files** (`.txt`, `.md`, etc.)
✅ **No configuration files** required (`scheme.json`, domain configs)
✅ **Dynamic entity/relation discovery** via LLM
✅ **No hardcoded domain assumptions**
✅ **Single processing pipeline** for all domains
✅ **Real-time workflow progress** for any content

### **Example Usage:**
```bash
# Medical domain
echo "Patient shows symptoms of fever" > data/raw/medical.txt
make dev  # Universal RAG discovers medical entities

# Legal domain
echo "Contract clause requires compliance" > data/raw/legal.txt
make dev  # Universal RAG discovers legal entities

# Any domain works automatically! 🚀
```

---

## 🚀 **Next Steps**

1. **Complete Phase 2 refactoring** of remaining components
2. **Update API endpoints** to use universal components
3. **Add domain parameter** to all endpoints
4. **Create universal configuration system**
5. **Add domain switching capabilities** to frontend
6. **Remove legacy domain-specific code**

The Universal RAG architecture will finally achieve the vision of **"one system, any domain"** with pure text-based processing! 🌟