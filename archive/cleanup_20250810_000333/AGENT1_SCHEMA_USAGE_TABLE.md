# Agent 1 Schema Usage Table - MINIMAL ESSENTIAL FIELDS

**Date**: 2025-08-09  
**Status**: Refactored to essential fields only  
**Schema Location**: `/workspace/azure-maintie-rag/agents/core/centralized_agent1_schema.py`

## 🎯 **Essential Fields Only**

This table contains **ONLY** the fields that are actually used by downstream systems.

### 📊 **Core Schema Fields**

| Field | Type | Usage Location | Status | Purpose |
|-------|------|----------------|--------|---------|
| **domain_signature** | `str` | All agents, templates `{{ domain_signature }}` | ✅ Working | Content identification |
| **content_type_confidence** | `float` | Validation, debug output | ✅ Working | Analysis confidence |

### 🧠 **Characteristics Fields** (6 fields)

| Field | Type | Used By | Line | Status | Purpose |
|-------|------|---------|------|--------|---------|
| **vocabulary_complexity_ratio** | `float` | Agent 2 entity scaling | 282 | ❌ **WRONG NAME** | Entity confidence adjustment |
| **lexical_diversity** | `float` | Agent 2 relationship scaling | 283 | ✅ Working | Relationship density calculation |
| **structural_consistency** | `float` | Agent 2 confidence adjustment | 284 | ✅ Working | Processing confidence |
| **vocabulary_richness** | `float` | Templates `{{ vocabulary_richness }}` | template | ❌ Not implemented | Adaptive prompts |
| **avg_document_length** | `int` | Debug output | debug | ✅ Working | Content metrics |
| **document_count** | `int` | Debug output | debug | ✅ Working | Content metrics |

### ⚙️ **Processing Config Fields** (8 fields)

| Field | Type | Used By | Line | Status | Purpose |
|-------|------|---------|------|--------|---------|
| **optimal_chunk_size** | `int` | Agent 2 chunking | 279 | ✅ **WORKING** | Chunk size optimization |
| **entity_confidence_threshold** | `float` | Agent 2 entity extraction | 278 | ✅ **WORKING** | Entity extraction threshold |
| **vector_search_weight** | `float` | Agent 3 search strategy | TBD | ❌ **NEEDS FIX** | Vector search weight |
| **graph_search_weight** | `float` | Agent 3 search strategy | TBD | ❌ **NEEDS FIX** | Graph search weight |
| **chunk_overlap_ratio** | `float` | Agent 2 chunking | 142 | ✅ Has default (0.2) | Chunk overlap |
| **relationship_density** | `float` | Agent 2 relationships | 157 | ✅ Has default (0.7) | Relationship extraction |
| **expected_extraction_quality** | `float` | Validation | validation | ✅ Has default (0.75) | Quality threshold |
| **processing_complexity** | `str` | Resource allocation | allocation | ✅ Has default ("medium") | Resource planning |

## 🎯 **Priority Fix Order**

### **Priority 1: Field Name Fix** 
```python
# Agent 1 currently outputs (WRONG):
"vocabulary_complexity": 0.818

# Must change to (CORRECT):  
"vocabulary_complexity_ratio": 0.818
```

### **Priority 2: Agent 3 Integration**
Fields that Agent 3 should use but doesn't:
- `vector_search_weight` (from Agent 1: 0.28)
- `graph_search_weight` (from Agent 1: 0.72)

### **Priority 3: Template Integration**  
Fields that templates expect but don't get:
- `vocabulary_richness` (from Agent 1: 0.65)

## 📋 **Current vs Target State**

| Component | Current State | Target State | Fix Required |
|-----------|---------------|--------------|--------------|
| **Agent 1 Output** | ❌ Wrong field name: `vocabulary_complexity` | ✅ Correct name: `vocabulary_complexity_ratio` | Fix Agent 1 field naming |
| **Agent 2 Integration** | ✅ Uses Agent 1 values (chunk_size=1494) | ✅ Working perfectly | None - already fixed |
| **Agent 3 Integration** | ❌ Ignores Agent 1 search weights | ✅ Use Agent 1 vector/graph weights | Fix Agent 3 integration |
| **Template Integration** | ❌ Uses fallback vocabulary_richness | ✅ Use Agent 1 vocabulary_richness | Fix template integration |

## ✅ **Success Metrics**

| Metric | Current | Target | Validation |
|--------|---------|--------|------------|
| **Field Name Consistency** | `vocabulary_complexity` | `vocabulary_complexity_ratio` | Agent 2 no longer gets field errors |
| **Agent 3 Search Weights** | Uses defaults (0.4/0.6) | Uses Agent 1 values (0.28/0.72) | Agent 3 logs show Agent 1 values |
| **Template Variables** | Uses fallback richness | Uses Agent 1 richness (0.65) | Templates show real data |

## 🔧 **Implementation Files**

| Fix | File to Update | Change Required |
|-----|----------------|-----------------|
| **Agent 1 Field Names** | `agents/domain_intelligence/agent.py` | Use `vocabulary_complexity_ratio` |
| **Agent 3 Integration** | `agents/universal_search/agent.py` | Use Agent 1 search weights |
| **Template Integration** | `infrastructure/prompt_workflows/` | Pass Agent 1 characteristics |

---

**Total Essential Fields**: 16 fields (down from 25+ in original schema)  
**Currently Working**: 11 fields (69%)  
**Need Implementation**: 3 fields (Agent 3 + templates)  
**Need Field Name Fix**: 1 field (vocabulary_complexity_ratio)