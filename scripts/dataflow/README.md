# Azure Universal RAG - Dataflow Architecture

Complete dataflow implementation using Azure cloud services with zero hardcoded domain knowledge.

## 🚀 Architecture Overview

This dataflow architecture implements:
- **Zero-Hardcoded-Values**: All parameters come from dynamic configuration and Domain Intelligence Agent
- **PydanticAI Integration**: Real agent orchestration with Azure services
- **Production-Ready**: Enterprise session management and comprehensive error handling
- **Universal Domain Adaptation**: Works with ANY content type automatically

## 📁 Dataflow Scripts

### Core Pipeline Scripts

| Script | Purpose | Integration |
|--------|---------|-------------|
| `00_check_azure_state.py` | Azure services health validation | Comprehensive state checking with JSON output |
| `00_full_pipeline.py` | Complete end-to-end pipeline | Orchestrates all agents and stages |
| `demo_full_workflow.py` | Interactive demonstration | Shows complete dataflow with sample content |

### Data Processing Scripts

| Script | Purpose | Azure Service |
|--------|---------|---------------|
| `01_data_ingestion.py` | Document ingestion pipeline | Azure Blob Storage |
| `01a_azure_storage.py` | Storage operations | Azure Blob Storage |
| `01b_azure_search.py` | Vector indexing | Azure Cognitive Search |
| `01c_vector_embeddings.py` | Embedding generation | Azure OpenAI |

### Knowledge Processing Scripts

| Script | Purpose | Agent Integration |
|--------|---------|------------------|
| `02_knowledge_extraction.py` | Entity/relationship extraction | Knowledge Extraction Agent |
| `03_cosmos_storage.py` | Graph database operations | Azure Cosmos DB |
| `04_graph_construction.py` | Knowledge graph building | Graph utilities |
| `05_gnn_training.py` | Neural network training | Azure ML |

### Query Processing Scripts

| Script | Purpose | Agent Integration |
|--------|---------|------------------|
| `06_query_analysis.py` | Query preprocessing | Domain Intelligence Agent |
| `07_unified_search.py` | Tri-modal search | Universal Search Agent |
| `08_context_retrieval.py` | Context assembly | Search orchestration |
| `09_response_generation.py` | LLM response synthesis | Azure OpenAI |
| `10_query_pipeline.py` | Complete query workflow | All agents |

### Monitoring & Utilities

| Script | Purpose | Features |
|--------|---------|----------|
| `11_streaming_monitor.py` | Real-time monitoring | Enterprise session tracking |
| `setup_azure_services.py` | Infrastructure setup | Azure deployment helpers |
| `load_outputs.py` | Data loading utilities | Output management |

## 🎯 Quick Start

### 1. Check Azure Services Status
```bash
# Basic health check
python scripts/dataflow/00_check_azure_state.py

# Verbose with JSON output
python scripts/dataflow/00_check_azure_state.py --verbose --json
```

### 2. Run Complete Pipeline
```bash
# Full pipeline with all stages
python scripts/dataflow/00_full_pipeline.py --data-dir data/raw

# Skip prerequisites check
python scripts/dataflow/00_full_pipeline.py --skip-prerequisites --verbose
```

### 3. Interactive Demo
```bash
# Complete dataflow demonstration
python scripts/dataflow/demo_full_workflow.py

# Custom query and verbose output
python scripts/dataflow/demo_full_workflow.py \
  --query "How does the multi-agent system work?" \
  --verbose --json
```

## 🔧 Advanced Usage

### Pipeline with Custom Configuration
```bash
# Full pipeline with custom settings
python scripts/dataflow/00_full_pipeline.py \
  --data-dir /path/to/your/data \
  --verbose \
  --output results/pipeline_results.json
```

### Individual Stage Testing
```bash
# Test specific components
python scripts/dataflow/02_knowledge_extraction.py
python scripts/dataflow/07_unified_search.py
python scripts/dataflow/10_query_pipeline.py
```

### Monitoring and Analysis
```bash
# Stream monitoring with session tracking
python scripts/dataflow/11_streaming_monitor.py

# Load and analyze previous outputs
python scripts/dataflow/load_outputs.py
```

## 📊 Output Formats

All scripts support:
- **Console Output**: Human-readable progress and results
- **JSON Output**: Machine-readable results with `--json`
- **File Output**: Save results with `--output filename.json`
- **Session Tracking**: Enterprise session management with unique IDs

### Example JSON Output Structure
```json
{
  "session_id": "pipeline_1703847600",
  "overall_status": "completed",
  "total_duration": 45.7,
  "stages_completed": [
    {
      "stage": "domain_analysis",
      "duration": 8.2,
      "status": "completed",
      "domain_signature": "technical_documentation"
    }
  ],
  "domain_analysis": {
    "domain_signature": "technical_documentation",
    "content_confidence": 0.89,
    "vocabulary_richness": 0.76,
    "technical_density": 0.82
  }
}
```

## 🌍 Universal Domain Adaptation

The dataflow architecture automatically adapts to ANY content type:

- **Technical Documentation**: High precision entity extraction
- **Business Documents**: Focus on processes and relationships  
- **Academic Papers**: Emphasis on concepts and citations
- **Creative Content**: Narrative and thematic analysis
- **Legal Documents**: Structured clause and reference processing

No manual configuration required - the Domain Intelligence Agent analyzes content and configures all downstream processing automatically.

## 🔗 Integration with Make Commands

The dataflow scripts integrate with the existing Make commands:

```bash
# Use dataflow through Make commands
make data-prep-full          # Runs 00_full_pipeline.py
make unified-search-demo     # Uses 07_unified_search.py
make query-demo             # Executes 10_query_pipeline.py
```

## 🚨 Error Handling

All scripts include comprehensive error handling:
- **Graceful Degradation**: Continue with available services
- **Detailed Logging**: Session-based error tracking
- **Retry Logic**: Automatic retry for transient failures
- **Fallback Modes**: Alternative processing when services unavailable

## 📈 Performance Metrics

The dataflow architecture targets:
- **Sub-3-second**: Query processing (end-to-end)
- **85%+ Accuracy**: Entity and relationship extraction
- **60%+ Cache Hit**: Reducing repeat processing
- **100+ Users**: Concurrent user support

All metrics are automatically tracked and reported in session outputs.

## 🔧 Customization

### Adding New Stages
1. Create new script in `scripts/dataflow/`
2. Follow naming convention: `##_stage_name.py`
3. Import and integrate with agent orchestrator
4. Add to pipeline orchestration

### Custom Agents
1. Implement PydanticAI agent in `agents/`
2. Add agent integration in dataflow scripts
3. Update orchestration workflows
4. Test with demo workflow

The architecture is designed for easy extension while maintaining zero-hardcoded-values principles.