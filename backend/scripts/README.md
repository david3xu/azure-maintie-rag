# Backend Scripts

**Architecture**: Organized by data flow stages that directly reflect README pipeline

**Current Status**: ✅ Data flow-aligned architecture implemented

---

## 📁 **Script Organization**

```
backend/scripts/
├── dataflow/                    # 🔄 CURRENT - Data flow stage scripts
│   ├── 01_data_ingestion.py     # Raw text → Blob Storage
│   ├── 02_knowledge_extraction.py # Blob → Knowledge extraction  
│   ├── 03_vector_indexing.py    # Text → Vector embeddings
│   ├── 04_graph_construction.py # Entities → Graph database
│   ├── 05_gnn_training.py       # Graph → GNN training
│   ├── 06_query_analysis.py     # Query → Analysis
│   ├── 07_unified_search.py     # Query → Multi-modal search
│   ├── 08_context_retrieval.py  # Results → Context prep
│   ├── 09_response_generation.py # Context → Final response
│   ├── 00_full_pipeline.py      # Complete processing orchestration
│   ├── 10_query_pipeline.py     # Complete query orchestration
│   ├── 11_streaming_monitor.py  # Real-time progress events
│   ├── setup_azure_services.py  # Azure service initialization
│   ├── demo_full_workflow.py    # End-to-end demonstration
│   └── README.md                # Data flow documentation
├── legacy/                      # 📚 REFERENCE - Function-based scripts  
│   ├── rag_cli.py               # Unified CLI (reference)
│   ├── data_pipeline.py         # Pipeline orchestrator (reference)
│   ├── [other legacy scripts...]
│   └── README.md                # Legacy documentation
└── README.md                    # This file
```

## 🎯 **Philosophy: Scripts as Living Documentation**

**Previous Approach**: Scripts organized by function (CLI, demo, test)
- Obscured the actual data flow
- Difficult to understand README pipeline
- Generic processing without clear stages

**New Approach**: Scripts directly mirror README data flow
- ✅ Each script = one README pipeline stage
- ✅ Sequential numbering shows dependencies  
- ✅ Perfect demonstration of architecture
- ✅ Educational for new developers

## 🚀 **Quick Start**

### **Execute Complete Processing Pipeline**:
```bash
cd backend/scripts/dataflow
python 00_full_pipeline.py --source data/raw --domain general
```

### **Execute Individual Stages** (for testing/debugging):
```bash
python 01_data_ingestion.py --source data/raw
python 02_knowledge_extraction.py --domain general  
python 03_vector_indexing.py --index-name rag-index
# ... continue through stages
```

### **Execute Query Pipeline**:
```bash  
python 10_query_pipeline.py --query "How does maintenance work?"
```

### **Full Demonstration**:
```bash
python demo_full_workflow.py
```

## 📋 **Makefile Integration**

The root Makefile routes to these data flow scripts:

```bash
make data-prep-full     # Executes dataflow/00_full_pipeline.py
make knowledge-extract  # Executes dataflow/02_knowledge_extraction.py  
make query-demo         # Executes dataflow/10_query_pipeline.py
```

## 📊 **Benefits**

1. **README Alignment**: Scripts directly demonstrate README architecture
2. **Educational Value**: Sequential execution teaches system flow
3. **Granular Control**: Test/debug individual pipeline stages
4. **Perfect Demos**: Show exact data flow to stakeholders
5. **Clear Dependencies**: Numbered stages show processing order
6. **Living Documentation**: Scripts serve as executable examples

**Result**: Backend scripts are now living, executable documentation of your Azure Universal RAG data flow! 🎉