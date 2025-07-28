# Azure RAG Lifecycle - Quick Start Commands

**Rapid execution guide for running the complete RAG lifecycle**

## 🚀 Prerequisites Check
```bash
cd /workspace/azure-maintie-rag/backend
source venv/bin/activate  # or .venv/bin/activate
make health
```

## 📊 Complete Lifecycle Execution

### Option 1: One-Command Full Pipeline
```bash
# Run complete lifecycle with demo data
make demo-lifecycle

# Or with custom data
make data-prep-full DOMAIN=maintenance INPUT_PATH=data/raw/demo_sample_10percent.md
```

### Option 2: Step-by-Step Execution

#### Step 1: Data Upload & Processing
```bash
# Upload raw data to Azure
python scripts/data_pipeline.py --process \
    --domain maintenance \
    --input data/raw/demo_sample_10percent.md
```

#### Step 2: Knowledge Extraction
```bash
# Extract entities and relationships
python scripts/workflow_analyzer.py --extract-knowledge \
    --domain maintenance \
    --mode full
```

#### Step 3: Index & Graph Building
```bash
# Build vector index and knowledge graph in parallel
python scripts/data_pipeline.py --build-all \
    --domain maintenance
```

#### Step 4: GNN Training (Optional)
```bash
# Train GNN model
python scripts/gnn_trainer.py --quick-train \
    --domain maintenance
```

#### Step 5: Test Query
```bash
# Test the complete system
curl -X POST http://localhost:8000/api/v1/query/universal \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to fix pump failure?",
    "domain": "maintenance"
  }'
```

## 🔍 Monitoring Commands

### Real-time Progress
```bash
# Watch workflow progress
python scripts/workflow_analyzer.py --monitor --workflow-id latest
```

### Check Results
```bash
# Verify data in Azure
python scripts/test_validator.py --verify-data --domain maintenance

# Check extraction results
ls -la data/extraction_outputs/
cat data/extraction_outputs/*_summary.json | jq .
```

### Performance Metrics
```bash
# Get performance report
python scripts/workflow_analyzer.py --performance-report
```

## ⚡ Common Issues & Quick Fixes

### Azure Connection Issues
```bash
# Re-validate Azure configuration
python scripts/azure_setup.py --fix-connections
```

### Low Extraction Quality
```bash
# Re-run with higher quality settings
python scripts/workflow_analyzer.py --extract-knowledge \
    --quality high \
    --retry-failed
```

### Slow Queries
```bash
# Warm up caches
python scripts/demo_runner.py --warm-caches
```

### Cosmos DB Bulk Loading (Known Limitation)
```bash
# Use the production bulk loader with progress monitoring
python scripts/data_pipeline.py --bulk-load \
    --batch-size 100 \
    --show-progress

# Alternative: Skip to direct GNN training
python scripts/gnn_trainer.py --use-local-data
```

## 🏆 Production Achievements

> **⚠️ UPDATE NEEDED**: These are HISTORICAL results from previous code (July 2025). 
> After running the refactored backend, UPDATE these with your new results!

**Historical Benchmarks** *(July 2025 - Update with new results)*:
- **9,100 entities** extracted from full maintenance dataset
- **5,848 relationships** with semantic connections  
- **3,271 vertices** in production Azure Cosmos DB
- **5.38M edges** in knowledge graph
- **7.4M parameter** GNN model trained
- **1646x connectivity ratio** (highly connected graph)

**Your New Results**:
```
Date: [Your Date]
Entities: [Your Result]
Relationships: [Your Result]
Processing Time: [Your Result]  
Performance Notes: [Your Notes]
```

## 📈 Expected Execution Times

- **Small dataset (500 docs)**: ~5-10 minutes
- **Medium dataset (5000 docs)**: ~30-45 minutes  
- **Large dataset (50000 docs)**: ~3-4 hours
- **GNN Training**: +20-30 minutes

## 🎯 Success Indicators

✅ All services health check passing  
✅ Data uploaded to Azure Blob Storage  
✅ Entities > 100, Relationships > 50  
✅ Search index created with documents  
✅ Knowledge graph vertices and edges created  
✅ Query returns relevant results < 3 seconds  

---

**Pro tip**: Use `make demo-lifecycle` for a complete end-to-end test with sample data!