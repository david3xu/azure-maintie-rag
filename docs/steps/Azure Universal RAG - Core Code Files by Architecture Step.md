# Azure Universal RAG - Core Code Files by Architecture Step

## Architecture Steps to Code Files Mapping

Based on the README "Workflow Components (Azure Enhanced)" table and actual codebase structure:

| **Step** | **README Architecture Phase** | **Azure Service** | **Core Implementation Files** | **Supporting Files** | **Configuration** |
|----------|-------------------------------|-------------------|-------------------------------|---------------------|------------------|
| **1** | **Data Ingestion**<br/>Text Processor | Azure Blob Storage (RAG) | `backend/core/azure_openai/text_processor.py`<br/>`backend/core/azure_storage/storage_client.py`<br/>`backend/core/azure_storage/storage_factory.py` | `backend/core/models/universal_rag_models.py` | `backend/config/settings.py`<br/>(Azure Blob Storage settings) |
| **2** | **Knowledge Extraction**<br/>LLM Extractor | Azure OpenAI GPT-4 | `backend/core/azure_openai/knowledge_extractor.py`<br/>`backend/core/azure_openai/completion_service.py` | `backend/core/models/universal_rag_models.py`<br/>(UniversalEntity, UniversalRelation) | `backend/config/settings.py`<br/>(Azure OpenAI settings) |
| **3** | **Vector Indexing**<br/>Azure Cognitive Search | Embedding + Vector DB | `backend/core/azure_search/vector_service.py`<br/>`backend/core/azure_search/search_client.py` | `backend/core/models/universal_rag_models.py`<br/>(UniversalDocument) | `backend/config/settings.py`<br/>(Azure Cognitive Search settings) |
| **4** | **Graph Construction**<br/>Azure Cosmos DB Gremlin | Native graph algorithms | `backend/core/azure_cosmos/cosmos_gremlin_client.py` | `backend/core/models/universal_rag_models.py`<br/>(Graph models) | `backend/config/settings.py`<br/>(Azure Cosmos DB settings) |
| **5** | **Query Processing**<br/>Query Analyzer | Azure OpenAI + Azure Services | `backend/core/azure_search/query_analyzer.py`<br/>`backend/core/azure_openai/completion_service.py` | `backend/core/workflow/universal_workflow_manager.py` | `backend/config/settings.py`<br/>(Query processing configuration) |
| **6** | **GNN Training**<br/>GNN Trainer | Azure Machine Learning | `backend/core/azure_ml/gnn_processor.py`<br/>`backend/core/azure_ml/ml_client.py`<br/>`backend/core/azure_ml/gnn/model.py`<br/>`backend/core/azure_ml/gnn/trainer.py`<br/>`backend/core/azure_ml/gnn/data_loader.py` | `backend/scripts/train_comprehensive_gnn.py`<br/>`backend/core/azure_ml/classification_service.py` | `backend/config/settings.py`<br/>(Azure ML settings) |
| **7** | **Retrieval**<br/>Unified Search | Azure Cognitive Search +<br/>Cosmos DB Gremlin + GNN | `backend/core/azure_search/vector_service.py`<br/>`backend/core/azure_cosmos/cosmos_gremlin_client.py`<br/>`backend/core/azure_ml/gnn_processor.py` | `backend/core/orchestration/rag_orchestration_service.py`<br/>(Unified search orchestration) | `backend/config/settings.py`<br/>(Multi-service integration) |
| **8** | **Generation**<br/>LLM Interface | Azure OpenAI GPT-4 | `backend/core/azure_openai/completion_service.py` | `backend/core/workflow/universal_workflow_manager.py`<br/>(Response streaming) | `backend/config/settings.py`<br/>(Response generation settings) |

## Main Orchestration Layer

| **Component** | **File Path** | **Purpose** | **Dependencies** |
|---------------|---------------|-------------|------------------|
| **RAG Orchestration Service** | `backend/core/orchestration/rag_orchestration_service.py` | Coordinates all 8 architecture steps in sequence | All Azure service clients |
| **Universal Workflow Manager** | `backend/core/workflow/universal_workflow_manager.py` | Three-layer progress disclosure and streaming | Orchestration service |
| **API Endpoints** | `backend/api/endpoints/azure-query-endpoint.py` | FastAPI endpoints exposing the workflow | Orchestration + Workflow managers |
| **Main Application** | `backend/api/main.py` | FastAPI application entry point | All API endpoints |

## Configuration & Data Models

| **Category** | **File Path** | **Purpose** |
|--------------|---------------|-------------|
| **Environment Configuration** | `backend/config/settings.py` | All Azure service configurations and credentials |
| **Universal Data Models** | `backend/core/models/universal_rag_models.py` | Domain-agnostic data structures (UniversalEntity, UniversalRelation, UniversalDocument) |
| **API Models** | `backend/api/models/query_models.py` | Request/response Pydantic models for API endpoints |

## Demo & Testing Scripts

| **Script** | **File Path** | **Architecture Steps Tested** |
|------------|---------------|-------------------------------|
| **Complete Workflow Demo** | `backend/scripts/azure-rag-workflow-demo.py` | All 8 steps end-to-end |
| **Query Processing Demo** | `backend/scripts/query_processing_workflow.py` | Steps 5-8 (Query → Response) |
| **Azure RAG Demo** | `backend/scripts/azure-rag-demo-script.py` | Full Azure services integration |
| **GNN Training** | `backend/scripts/train_comprehensive_gnn.py` | Step 6 (GNN Training) specifically |

## Architecture Notes

- **Data-Driven Design**: All components use `backend/config/settings.py` for configuration - no hardcoded values
- **Domain-Agnostic**: Universal models in `backend/core/models/` work with any domain
- **Azure-Native**: Each step integrates with specific Azure services using dedicated client files
- **Streaming Support**: Workflow manager provides real-time progress updates
- **Enterprise Ready**: Comprehensive error handling, monitoring, and scalability considerations

## Migration Path from Universal to Azure

The codebase shows a clear separation between:
1. **Universal Components** (`backend/core/models/universal_rag_models.py`) - Domain-agnostic data structures
2. **Azure Implementations** (`backend/core/azure_*/`) - Azure service-specific implementations
3. **Orchestration Layer** (`backend/core/orchestration/`) - Coordinates Azure services

This architecture enables migrating from universal RAG to Azure-specific implementations while maintaining the same universal interfaces.