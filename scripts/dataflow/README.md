# Azure Universal RAG - Dataflow Pipeline Execution Guide

Complete step-by-step guide for executing the 6-phase Azure Universal RAG dataflow pipeline with REAL Azure services and REAL data.

## 🎯 **Universal RAG Philosophy**

This pipeline operates on the principle of **zero hardcoded domain bias** - it adapts to ANY data domain by discovering content characteristics dynamically rather than using predefined categories.

## 🏗️ **Architecture Overview**

**6-Phase Pipeline Structure:**
- **Phase 0**: Azure Data Cleanup (Optional - preserves infrastructure)
- **Phase 1**: Agent Validation (Critical - validates 3 PydanticAI agents)
- **Phase 2**: Data Ingestion (Upload real data to Azure services)
- **Phase 3**: Knowledge Extraction (Extract entities and build knowledge graphs)
- **Phase 4**: Query Pipeline (Query analysis and universal search)
- **Phase 5**: Integration Testing (End-to-end workflow validation)
- **Phase 6**: Advanced Features (GNN training and monitoring)

**Key Components:**
- **3 PydanticAI Agents**: Domain Intelligence, Knowledge Extraction, Universal Search
- **9 Azure Services**: OpenAI, Cognitive Search, Cosmos DB, Storage, ML, etc.
- **Real Data**: 179 Azure AI Language Service documents in `data/raw/`

## 🚨 **Critical Safety Information**

### ✅ **SAFE - Data Operations:**
All dataflow scripts only manipulate **data within Azure services**:
- Delete/create documents in Cognitive Search
- Delete/create entities and relationships in Cosmos DB
- Delete/create blobs in Azure Storage
- **Azure service infrastructure remains PROTECTED**

### ⚠️ **INFRASTRUCTURE PROTECTION:**
- NO `azd down` commands in dataflow pipeline
- NO resource group deletion
- NO service destruction
- Infrastructure deletion tools are isolated in `scripts/deployment/` only

## 📋 **Prerequisites**

### Required Environment:
```bash
# Navigate to project root
cd /workspace/azure-maintie-rag

# Verify Azure authentication
az login
az account show

# Sync with Azure environment
./scripts/deployment/sync-env.sh prod

# Verify Python path
export PYTHONPATH=/workspace/azure-maintie-rag
```

### Required Data:
- Real data must exist in `data/raw/azure-ai-services-language-service_output/`
- 179 Azure AI Language Service markdown files should be present
- NO sample data, NO fake values, NO placeholder content

### Azure Services:
All 9 Azure services must be operational:
- Azure OpenAI Service
- Azure Cognitive Search
- Azure Cosmos DB (Gremlin API)
- Azure Storage Account
- Azure Machine Learning
- Supporting services (Key Vault, Monitor, etc.)

## 🚀 **Quick Start - Complete Pipeline**

### Option 1: Execute All Phases (Recommended)
```bash
make dataflow-full
```

### Option 2: Individual Phase Execution
```bash
# Phase 0: Optional data cleanup
make dataflow-cleanup

# Phase 1: CRITICAL - Validate all agents
make dataflow-validate

# Phase 2: Upload and index real data
make dataflow-ingest

# Phase 3: Extract knowledge and build graphs
make dataflow-extract

# Phase 4: Query pipeline and search
make dataflow-query

# Phase 5: Integration testing
make dataflow-integrate

# Phase 6: Advanced features
make dataflow-advanced
```

## 📖 **Detailed Phase-by-Phase Execution**

### **Phase 0: Azure Data Cleanup** (Optional)
**Purpose**: Clean existing data for fresh start (preserves infrastructure)

```bash
# Method 1: Using Makefile
make dataflow-cleanup

# Method 2: Direct execution
cd /workspace/azure-maintie-rag
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase0_cleanup/00_01_cleanup_azure_data.py
```

**What it does:**
- Deletes existing documents from Cognitive Search
- Clears Cosmos DB graph (vertices and edges)
- Removes blobs from Azure Storage containers
- **Preserves**: Azure service infrastructure, original data files

**Expected Results:**
```
✅ Azure services cleaned
📊 Search: 0 documents remaining
📊 Cosmos: 0 entities, 0 relationships remaining  
📊 Storage: 0 blobs remaining
```

### **Phase 1: Agent Validation** (CRITICAL)
**Purpose**: Validate all 3 PydanticAI agents with real Azure services

```bash
# Method 1: Using Makefile
make dataflow-validate

# Method 2: Individual validation
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_01_validate_domain_intelligence.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_02_validate_knowledge_extraction.py
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_03_validate_universal_search.py
```

**What it validates:**
- **Domain Intelligence Agent**: Content analysis and characteristic discovery
- **Knowledge Extraction Agent**: Entity and relationship extraction
- **Universal Search Agent**: Multi-modal search (vector + graph + GNN)

**Expected Results:**
```
✅ Domain Intelligence: Working (signature: technical_documentation_0.847)
✅ Knowledge Extraction: Working (entities: 15+, relationships: 8+)
✅ Universal Search: Working (results: 3+, confidence: 80%+)
```

**🚨 CRITICAL**: If Phase 1 fails, DO NOT proceed. Fix agent issues first.

### **Phase 2: Data Ingestion** 
**Purpose**: Upload real data to Azure services and create indexes

```bash
# Method 1: Using Makefile
make dataflow-ingest

# Method 2: Step-by-step execution
# 2.1: Upload documents to Azure Storage
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase2_ingestion/02_02_storage_upload_primary.py --container documents-prod

# 2.2: Create vector embeddings
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase2_ingestion/02_03_vector_embeddings.py

# 2.3: Index in Cognitive Search
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase2_ingestion/02_04_search_indexing.py
```

**What it does:**
- Uploads 179 Azure AI docs to Storage containers
- Generates 1536-dimensional vector embeddings
- Indexes documents in Cognitive Search with metadata

**Expected Results:**
```
✅ Storage: 179 documents uploaded to documents-prod
✅ Embeddings: 179 vectors created (1536 dimensions each)
✅ Search Index: 179 documents indexed with metadata
```

### **Phase 3: Knowledge Extraction**
**Purpose**: Extract entities and relationships, build knowledge graph

```bash
# Method 1: Using Makefile
make dataflow-extract

# Method 2: Step-by-step execution
# 3.1: Validate prerequisites
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_00_validate_phase3_prerequisites.py

# 3.2: Extract knowledge using unified template
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_02_simple_extraction.py

# 3.3: Store in Cosmos DB
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_03_simple_storage.py

# 3.4: Validate graph construction
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_04_simple_graph.py
```

**Key Features:**
- Uses **unified template system** for consistent extraction
- **Fixed entity ID mapping** ensures relationships work correctly
- **Real data only** - no fake entities or relationships

**Expected Results:**
```
✅ Knowledge Extraction: 13+ entities, 8+ relationships extracted
✅ Cosmos Storage: Graph populated with real entities
✅ Relationships: Properly mapped (e.g., "FAQ bot → multi-turn prompts")
📊 Quality: Relationships correctly stored and queryable
```

### **Phase 4: Query Pipeline**
**Purpose**: Test query analysis and universal search with real data

```bash
# Method 1: Using Makefile
make dataflow-query

# Method 2: Individual execution
# 4.1: Query analysis
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase4_query/04_01_query_analysis.py "How to train custom models with Azure AI?"

# 4.2: Universal search demo
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase4_query/04_02_universal_search_demo.py

# 4.3: Complete query pipeline
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase4_query/04_06_complete_query_pipeline.py
```

**What it tests:**
- Query intent analysis using Domain Intelligence Agent
- Multi-modal search across vector, graph, and GNN systems
- Result ranking and confidence scoring

**Expected Results:**
```
✅ Query Analysis: Intent understood, domain characteristics identified
✅ Universal Search: 3+ results found with 80%+ confidence  
✅ Search Strategy: Vector + Graph combination used effectively
```

### **Phase 5: Integration Testing**
**Purpose**: End-to-end workflow validation with real scenarios

```bash
# Method 1: Using Makefile
make dataflow-integrate

# Method 2: Direct execution
# 5.1: Full pipeline execution test
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase5_integration/05_01_full_pipeline_execution.py --verbose

# 5.2: Query generation showcase
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase5_integration/05_03_query_generation_showcase.py
```

**What it validates:**
- Complete workflow from query to response
- Multi-agent coordination and data flow
- Real-world scenario testing

**Expected Results:**
```
✅ End-to-End Pipeline: Complete workflow operational
✅ Multi-Agent Coordination: Agents work together seamlessly
✅ Query Showcase: Multiple query types handled correctly
📊 Overall System Status: 90%+ operational
```

### **Phase 6: Advanced Features**
**Purpose**: GNN training, real-time monitoring, and advanced configuration

```bash
# Method 1: Using Makefile
make dataflow-advanced

# Method 2: Individual features
# 6.1: GNN model training
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase6_advanced/06_01_gnn_training.py

# 6.2: Real-time monitoring
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase6_advanced/06_02_streaming_monitor.py

# 6.3: Configuration system demo
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase6_advanced/06_03_config_system_demo.py
```

**Advanced Capabilities:**
- Graph Neural Network training with real knowledge graphs
- Real-time system monitoring and alerting
- Dynamic configuration management

**Expected Results:**
```
✅ GNN Training: Model trained on real graph data
✅ Monitoring: Real-time metrics and alerts operational
✅ Configuration: Dynamic system configuration working
```

## 🔧 **Troubleshooting Guide**

### Common Issues and Solutions:

#### **Azure Authentication Failed**
```bash
# Re-authenticate with Azure
az login
az account show
./scripts/deployment/sync-env.sh prod
```

#### **Import Errors**
```bash
# Always set PYTHONPATH
export PYTHONPATH=/workspace/azure-maintie-rag
```

#### **Agent Validation Failures**
```bash
# Check agent imports individually
PYTHONPATH=/workspace/azure-maintie-rag python agents/domain_intelligence/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/knowledge_extraction/agent.py
PYTHONPATH=/workspace/azure-maintie-rag python agents/universal_search/agent.py
```

#### **Data Upload Issues**
```bash
# Verify data exists
ls -la data/raw/azure-ai-services-language-service_output/
# Should show 179 .md files

# Check Azure Storage connectivity
az storage account list
```

#### **Knowledge Extraction Problems**
```bash
# Test unified template system
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_02_test_unified_template.py

# Validate prerequisites
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_00_validate_phase3_prerequisites.py
```

#### **Search Returns 0 Results**
```bash
# Check if data was properly ingested
make check-data

# Verify search index
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase2_ingestion/02_04_search_indexing.py
```

### **Performance Optimization**

#### **For Large Datasets:**
```bash
# Use threading control for NumPy operations
export OPENBLAS_NUM_THREADS=1

# Use timeout controls for long operations
timeout 300 python <script>
```

#### **For Development Speed:**
```bash
# Skip cleanup phase for faster iteration
make dataflow-validate  # Start from Phase 1
# Continue with other phases...
```

## 📊 **Success Metrics**

### **Phase 1 (Agents): 100% Success Required**
- Domain Intelligence Agent: Working signature generation
- Knowledge Extraction Agent: Entities + relationships extracted  
- Universal Search Agent: Results returned with confidence

### **Phase 2 (Ingestion): Data Quality**
- Storage: All 179 documents uploaded successfully
- Embeddings: Vector count matches document count
- Search: All documents indexed with proper metadata

### **Phase 3 (Knowledge): Graph Quality**
- Entities: 13+ meaningful entities extracted
- Relationships: 8+ valid relationships created
- Quality: Relationships properly mapped (not just IDs)

### **Phase 4 (Query): Search Effectiveness**
- Results: 3+ relevant results per query
- Confidence: 80%+ confidence scores
- Strategy: Appropriate search strategy selection

### **Phase 5 (Integration): System Reliability**
- End-to-End: Complete pipeline without errors
- Multi-Agent: Agents coordinate effectively
- Scenarios: Multiple query types handled

### **Phase 6 (Advanced): Enhanced Capabilities**
- GNN: Model training on real graph data
- Monitoring: Real-time metrics collection
- Config: Dynamic system adaptation

## 🎯 **Production Readiness Checklist**

- [ ] All 6 phases execute successfully
- [ ] Real Azure services operational (no mocks)
- [ ] Real data processed (179 Azure AI documents)
- [ ] Knowledge graph populated with quality relationships
- [ ] Universal search returns relevant results
- [ ] Multi-agent coordination working
- [ ] Zero hardcoded domain bias maintained
- [ ] Performance within acceptable thresholds
- [ ] Error handling and recovery working
- [ ] Monitoring and alerting operational

## 📚 **Additional Resources**

### **Documentation:**
- `/workspace/azure-maintie-rag/CLAUDE.md` - Project-specific guidance
- `/workspace/CLAUDE.md` - Workspace overview
- `docs/` - Detailed analysis and results

### **Configuration:**
- `config/environments/` - Environment-specific settings
- `agents/core/constants.py` - Zero-hardcoded-values constants  
- `infrastructure/prompt_workflows/templates/` - Unified templates

### **Testing:**
- `pytest` - Run all tests with real Azure services
- `pytest -m unit` - Unit tests for agent logic
- `pytest -m integration` - Integration tests with Azure

### **Session Management:**
```bash
make session-report  # View current session metrics
make clean          # Clean session with log replacement  
make health         # Comprehensive system health check
```

## 🚀 **Getting Started**

1. **Quick validation**: `make dataflow-validate`
2. **Full pipeline**: `make dataflow-full`  
3. **Check results**: `make session-report`
4. **Monitor health**: `make health`

For any issues, start with Phase 1 agent validation - this is the foundation of the entire system.

---

**Azure Universal RAG System** - Production-ready multi-agent platform with zero domain bias, real Azure integration, and comprehensive dataflow pipeline.