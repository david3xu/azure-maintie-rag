# Legacy Scripts

**Purpose**: Historical scripts organized by function rather than data flow

**Status**: ⚠️ **LEGACY** - Preserved for reference, replaced by data flow architecture

**Replaced By**: `../dataflow/` - Scripts that directly reflect README data flow stages

---

## 📁 **Legacy Script Organization**

These scripts were organized by **function** (CLI, demo, test, etc.) rather than **data flow stages**:

```
legacy/
├── rag_cli.py                    # Unified CLI entry point
├── data_pipeline.py              # Data processing orchestrator
├── demo_runner.py                # Demo execution
├── azure_setup.py                # Azure configuration  
├── gnn_trainer.py                # GNN training
├── test_runner.py                # Test execution + validation
├── workflow_runner.py            # Workflow execution + lifecycle
├── azure_credentials_setup.sh    # Credential setup
├── azure_ml_conda_env.yml        # ML environment
└── SCRIPT_REFACTORING_PLAN.md    # Legacy architecture documentation
```

## 🔄 **Why Replaced?**

**Issue**: Scripts didn't reflect the README data flow architecture
- Users couldn't see the processing pipeline stages
- No clear demonstration of: Raw Text → Knowledge Extraction → Vector/Graph → GNN → Query Processing
- Function-based organization obscured the actual data flow

**Solution**: New data flow-aligned scripts in `../dataflow/`

## 🚀 **Migration Path**

**Old Approach**:
```bash
python scripts/rag_cli.py data --mode full  # Generic data processing
python scripts/demo_runner.py               # Generic demo
```

**New Approach**:
```bash
python scripts/dataflow/01_data_ingestion.py     # Specific data flow stage
python scripts/dataflow/02_knowledge_extraction.py
python scripts/dataflow/03_vector_indexing.py
# ... sequential processing that matches README
```

## 📋 **Preservation Reason**

These scripts contain working implementations that may be useful for reference:
- **Unified CLI pattern** from `rag_cli.py`
- **Service integration examples** from various runners
- **Azure setup patterns** from `azure_setup.py`
- **Working pipeline orchestration** from `data_pipeline.py`

**Recommendation**: Use `../dataflow/` scripts for development, reference these for implementation patterns.