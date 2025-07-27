# Organized Scripts Directory

This directory contains a logically organized structure of all Azure Universal RAG scripts, categorized by functionality and use case.

## Directory Structure

```
organized/
├── data_processing/     # Data ingestion, preparation, and extraction
├── azure_services/      # Azure service configuration and validation
├── gnn_training/        # Graph Neural Network training and optimization
├── testing/            # Testing, validation, and verification scripts
├── workflows/          # End-to-end workflow orchestration
├── demos/              # Demo and presentation scripts
└── README.md           # This file
```

## Quick Start - 10% Lifecycle Test

To run a complete end-to-end test using 10% sample data:

```bash
# Navigate to the backend directory
cd /workspace/azure-maintie-rag/backend

# Run the lifecycle test
python scripts/organized/workflows/lifecycle_test_10percent.py
```

This will execute all 6 stages of the Azure Universal RAG pipeline:
1. **Data Upload** - Upload sample to Azure Blob Storage
2. **Knowledge Extraction** - Extract entities/relationships via Azure OpenAI
3. **Vector Indexing** - Create search index in Azure Cognitive Search
4. **Graph Construction** - Build knowledge graph in Azure Cosmos DB
5. **GNN Training** - Train neural network using Azure ML
6. **Query Testing** - Test universal RAG queries

## Scripts by Category

### Data Processing (`data_processing/`)
- `data_preparation_workflow.py` - Complete data preparation pipeline
- `data_upload_workflow.py` - Upload documents to Azure storage
- `prepare_raw_data.py` - Raw data preprocessing
- `clean_knowledge_extraction.py` - Clean knowledge extraction process
- `knowledge_extraction_workflow.py` - Full extraction workflow
- `full_dataset_extraction.py` - Large dataset extraction

### Azure Services (`azure_services/`)
- `azure_config_validator.py` - Validate Azure service configuration
- `azure_credentials_setup.sh` - Set up Azure authentication
- `azure_data_state.py` - Check Azure data state
- `azure_services_consolidation.py` - Consolidate service connections
- `load_env_and_setup_azure.py` - Environment setup

### GNN Training (`gnn_training/`)
- `train_comprehensive_gnn.py` - Comprehensive GNN training
- `real_azure_ml_gnn_training.py` - Azure ML GNN training
- `gnn_training_optimized.py` - Optimized training pipeline
- `demo_real_gnn_training.py` - Demo GNN training
- `simple_gnn_test.py` - Simple GNN functionality test

### Testing (`testing/`)
- `test_clean_extraction.py` - Test clean extraction
- `test_context_aware_extraction.py` - Test context-aware processing
- `test_real_azure_extraction.py` - Test Azure extraction
- `validate_azure_config.py` - Validate Azure configuration
- `validate_azure_knowledge_data.py` - Validate knowledge data
- `verify_accuracy_calculation.py` - Verify accuracy metrics

### Workflows (`workflows/`)
- `lifecycle_test_10percent.py` - **🎯 Main 10% lifecycle test**
- `query_processing_workflow.py` - Query processing pipeline
- `orchestrate_gnn_pipeline.py` - GNN pipeline orchestration
- `data_preparation_workflow.py` - Data preparation workflow

### Demos (`demos/`)
- `demo_quick_loader.py` - Quick demo data loader
- `azure-rag-demo-script.py` - RAG demo script
- `azure-rag-workflow-demo.py` - Workflow demo
- `concrete_gnn_benefits_demo.py` - GNN benefits demonstration

## Performance Expectations (10% Sample)

Based on the 10% sample containing ~525 maintenance texts:

| Stage | Expected Duration | Expected Output |
|-------|------------------|-----------------|
| Data Upload | 5-10 seconds | ~50KB uploaded |
| Knowledge Extraction | 30-60 seconds | ~50-100 entities, ~30-60 relationships |
| Vector Indexing | 10-20 seconds | ~50-100 documents indexed |
| Graph Construction | 15-30 seconds | ~50-100 vertices, ~30-60 edges |
| GNN Training | 60-120 seconds | Model with ~0.4-0.6 accuracy |
| Query Testing | 10-20 seconds | 3 test queries processed |

**Total Expected Duration: 2-4 minutes**

## Environment Requirements

Ensure these environment variables are set (see `backend/.env`):
- Azure OpenAI credentials
- Azure Cognitive Search endpoint/key
- Azure Cosmos DB connection string
- Azure Blob Storage connection strings
- Azure ML workspace configuration

## Monitoring and Logs

All lifecycle test results are saved to:
- `backend/data/demo_outputs/lifecycle_test_10pct_[timestamp].json`
- Session logs in `backend/logs/`

## Troubleshooting

1. **Azure Connection Issues**: Run `python scripts/organized/azure_services/azure_config_validator.py`
2. **Data State Issues**: Run `python scripts/organized/azure_services/azure_data_state.py`
3. **Environment Issues**: Check `backend/.env` file configuration

## Next Steps

After successful 10% lifecycle test:
1. Scale to full dataset using `full_dataset_extraction.py`
2. Optimize GNN training with `gnn_training_optimized.py`
3. Deploy production pipeline using workflow scripts

---

*This organized structure makes the Azure Universal RAG system more maintainable and easier to understand for development and demonstration purposes.*