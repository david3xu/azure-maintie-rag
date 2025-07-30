# Data Flow-Aligned Scripts Design

**Objective**: Create scripts that directly reflect the README data flow architecture

**Current Issue**: Scripts are organized by function (test, demo, setup) rather than data flow stages

**Target**: Scripts that mirror the actual Azure Universal RAG data flow pipeline

---

## 📊 **README Data Flow Analysis**

### **Processing Phase:**
```
Raw Text → Blob Storage → Knowledge Extraction → Vector/Graph Storage → GNN Training
```

### **Query Phase:**
```
User Query → Query Analysis → Unified Search → Context Retrieval → Response Generation
```

### **Real-time Features:**
```
Progress Events → Frontend Progressive UI
```

---

## 🎯 **Target Scripts Design**

### **Core Data Flow Scripts (Processing Phase):**

1. **`01_data_ingestion.py`**
   - **Purpose**: Raw text → Azure Blob Storage
   - **Reflects**: First stage of processing phase
   - **Operations**: Document upload, chunking, blob storage

2. **`02_knowledge_extraction.py`**
   - **Purpose**: Blob Storage → Knowledge extraction (Azure OpenAI)
   - **Reflects**: Knowledge extraction stage
   - **Operations**: Entity/relationship extraction via LLM

3. **`03_vector_indexing.py`**
   - **Purpose**: Text → Vector embeddings (1536D) → Azure Cognitive Search
   - **Reflects**: Parallel processing - vector branch
   - **Operations**: Embedding generation, search index creation

4. **`04_graph_construction.py`**
   - **Purpose**: Entities/Relations → Azure Cosmos DB Gremlin Graph
   - **Reflects**: Parallel processing - graph branch
   - **Operations**: Graph database operations, relationship mapping

5. **`05_gnn_training.py`**
   - **Purpose**: Graph data → GNN training → Trained model storage
   - **Reflects**: Final processing stage
   - **Operations**: Azure ML GNN training, model storage

### **Query Processing Scripts (Query Phase):**

6. **`06_query_analysis.py`**
   - **Purpose**: User query → Query analysis (Azure OpenAI)
   - **Reflects**: First stage of query phase
   - **Operations**: Query understanding, intent analysis

7. **`07_unified_search.py`**
   - **Purpose**: Query → Unified search (Vector + Graph + GNN)
   - **Reflects**: Core search stage
   - **Operations**: Multi-modal search coordination

8. **`08_context_retrieval.py`**
   - **Purpose**: Search results → Context retrieval
   - **Reflects**: Context preparation stage
   - **Operations**: Result ranking, context assembly

9. **`09_response_generation.py`**
   - **Purpose**: Context → Azure OpenAI Response → Final answer with citations
   - **Reflects**: Final query stage
   - **Operations**: Response generation, citation formatting

### **Orchestration & Monitoring Scripts:**

10. **`00_full_pipeline.py`**
    - **Purpose**: Execute complete processing pipeline (stages 1-5)
    - **Reflects**: End-to-end processing coordination
    - **Operations**: Orchestrates all processing stages

11. **`10_query_pipeline.py`**
    - **Purpose**: Execute complete query pipeline (stages 6-9)
    - **Reflects**: End-to-end query processing
    - **Operations**: Orchestrates all query stages

12. **`11_streaming_monitor.py`**
    - **Purpose**: Real-time streaming progress events → Frontend
    - **Reflects**: Real-time features
    - **Operations**: Progress tracking, event streaming

### **Support Scripts:**

13. **`setup_azure_services.py`**
    - **Purpose**: Initialize and validate all Azure services
    - **Reflects**: Infrastructure preparation
    - **Operations**: Service health checks, configuration validation

14. **`demo_full_workflow.py`**
    - **Purpose**: Demonstrate complete RAG workflow
    - **Reflects**: End-to-end demonstration
    - **Operations**: Sample data processing + query demonstration

---

## 🔄 **Script Naming Convention**

**Processing Phase**: `01_` through `05_` (sequential processing)
**Query Phase**: `06_` through `09_` (sequential query handling)  
**Orchestration**: `00_` (full processing), `10_` (full query), `11_` (monitoring)
**Support**: Descriptive names without numbers

**Benefits:**
- ✅ **Clear Sequence**: Numbers indicate data flow order
- ✅ **README Alignment**: Each script matches a data flow stage
- ✅ **Easy Understanding**: Names directly reflect README architecture
- ✅ **Logical Grouping**: Processing vs Query vs Support clearly separated

---

## 🚀 **Implementation Strategy**

### **Replace Current Scripts**
- **Remove**: Current function-based scripts (demo_runner, test_runner, etc.)
- **Create**: New data flow-aligned scripts
- **Benefit**: Scripts directly demonstrate README architecture

### **CLI Integration**
```bash
# Execute by stage
python scripts/01_data_ingestion.py
python scripts/02_knowledge_extraction.py
# ... etc

# Execute full pipelines  
python scripts/00_full_pipeline.py        # Complete processing
python scripts/10_query_pipeline.py       # Complete query workflow

# Demonstrate end-to-end
python scripts/demo_full_workflow.py
```

### **Makefile Integration**
```makefile
# Data flow stages
data-ingestion:      cd backend && python scripts/01_data_ingestion.py
knowledge-extract:   cd backend && python scripts/02_knowledge_extraction.py
vector-indexing:     cd backend && python scripts/03_vector_indexing.py
graph-construction:  cd backend && python scripts/04_graph_construction.py
gnn-training:        cd backend && python scripts/05_gnn_training.py

# Full pipelines
data-prep-full:      cd backend && python scripts/00_full_pipeline.py
query-demo:          cd backend && python scripts/10_query_pipeline.py "sample query"
full-demo:           cd backend && python scripts/demo_full_workflow.py
```

---

## ✅ **Expected Benefits**

1. **📊 README Alignment**: Scripts directly demonstrate data flow architecture
2. **🔄 Clear Sequence**: Numbered scripts show exact processing order  
3. **🎯 Focused Purpose**: Each script has single data flow responsibility
4. **📚 Educational**: New developers can understand system by running scripts in order
5. **🧪 Testing**: Each stage can be tested independently
6. **🚀 Demonstration**: Perfect for showing system capabilities to stakeholders

**Result**: Scripts become living documentation of the README data flow architecture!