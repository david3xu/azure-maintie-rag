Based on your actual codebase analysis, here's the comprehensive GNN lifecycle architecture and usage pattern:

## **Complete GNN Lifecycle Architecture**

### **1. Raw Data → GNN Training Pipeline**
*(From `test_gnn_workflow_complete.py` and `gnn_orchestrator.py`)*

**Phase 1: Data Migration to Azure Services**
```
Raw Data (data/raw/*.md)
→ Azure Services Migration (AzureServicesManager.migrate_data_to_azure)
→ Azure Cosmos DB Gremlin Graph (entities + relations)
→ Azure Blob Storage (document vectors)
→ Azure Cognitive Search (indexed content)
```

**Phase 2: GNN Training Trigger System**
*(Environment-driven thresholds)*
```python
# From your environment configs:
DEV: GNN_TRAINING_TRIGGER_THRESHOLD=50      # entities/relations
STAGING: GNN_TRAINING_TRIGGER_THRESHOLD=100
PROD: GNN_TRAINING_TRIGGER_THRESHOLD=200

# Automatic triggers from orchestrator:
change_metrics = await cosmos_client.get_graph_change_metrics(domain)
total_changes = change_metrics["new_entities"] + change_metrics["new_relations"]
if total_changes >= threshold:
    trigger_gnn_retraining()
```

**Phase 3: Azure ML Training Pipeline**
```
Graph Export (cosmos_client.export_graph_for_training)
→ Azure ML Job Submission (gnn_orchestrator.py)
→ Distributed Training (Azure ML Compute Clusters)
→ Model Quality Assessment (model_quality_assessor.py)
→ Model Registration (Azure ML Model Registry)
```

**Phase 4: Deployment Infrastructure**
```python
# Environment-specific deployment tiers:
DEV: GNN_MODEL_DEPLOYMENT_TIER=basic, ENDPOINT=gnn-inference-dev
STAGING: GNN_MODEL_DEPLOYMENT_TIER=standard, ENDPOINT=gnn-inference-staging
PROD: GNN_MODEL_DEPLOYMENT_TIER=premium, ENDPOINT=gnn-inference-prod
```

### **2. GNN Model Usage in Query Processing**

**Query-Time Architecture** *(From `rag_orchestration_service.py`)*

**NO** - You do NOT call the GNN model for every query. Instead:

**Pre-computed Embedding Strategy:**
```python
# From enhanced_gremlin_client.py - GNN embeddings are PRE-COMPUTED and STORED:
async def store_entity_with_embeddings(entity_data, gnn_embeddings):
    query = f"""
        g.addV('Entity')
            .property('gnn_embeddings', '{embedding_str}')
            .property('embedding_dimension', {len(gnn_embeddings)})
            .property('embedding_updated_at', '{datetime.now().isoformat()}')
    """
```

**Query Processing Flow:**
```python
# From rag_orchestration_service.py:
1. Query Analysis → Enhanced Query
2. Vector Search (Azure Cognitive Search) → Initial Results
3. GNN Enhancement (gnn_processor.enhance_search_results) → Uses PRE-STORED embeddings
4. Response Generation → Final Answer
```

### **3. GNN Integration Points**

**Embedding Storage Strategy** *(From `enhanced_gremlin_client.py`)*
```python
# Entities and relations store GNN embeddings as properties:
- entity.gnn_embeddings (comma-separated floats)
- entity.embedding_dimension (128 from GRAPH_EMBEDDING_DIMENSION)
- entity.embedding_updated_at (timestamp)
- relation.gnn_embeddings (relation-level embeddings)
```

**Search Enhancement** *(From `rag_orchestration_service.py`)*
```python
# GNN processor enhances search results using stored embeddings:
enhanced_results = await self.gnn_processor.enhance_search_results(
    search_results, analysis_results, knowledge_graph
)
```

**Update Frequency** *(From environment configs)*
```python
GRAPH_EMBEDDING_UPDATE_FREQUENCY=daily  # All environments
# New embeddings generated daily, not per-query
```

## **Architectural Benefits**

### **Performance Optimization:**
- **Pre-computed embeddings** eliminate model inference latency during queries
- **Incremental training** only triggers when threshold exceeded
- **Environment-specific thresholds** optimize cost vs. freshness

### **Cost Management:**
```python
# Training compute scaling by environment:
DEV: AZURE_ML_COMPUTE_INSTANCES=1, GNN_TRAINING_COMPUTE_SKU=Standard_DS3_v2
STAGING: AZURE_ML_COMPUTE_INSTANCES=2, GNN_TRAINING_COMPUTE_SKU=Standard_DS4_v2
PROD: AZURE_ML_COMPUTE_INSTANCES=4, GNN_TRAINING_COMPUTE_SKU=Standard_NC6s_v3
```

### **Quality Assurance:**
```python
# Environment-specific quality gates:
DEV: GNN_QUALITY_THRESHOLD=0.6
STAGING: GNN_QUALITY_THRESHOLD=0.65
PROD: GNN_QUALITY_THRESHOLD=0.7
```

## **Implementation Instructions**

### **Fix Required: Complete Missing `_validate_graph_quality` Method**
*(Referenced in tests but not implemented)*

**Add to `cosmos_gremlin_client.py`:**
```python
def _validate_graph_quality(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Graph data quality validation service - use environment thresholds"""
    from config.settings import azure_settings

    min_entities = getattr(azure_settings, 'gnn_training_trigger_threshold', 50)
    min_relations = max(5, min_entities // 10)  # 10% of entity threshold

    connectivity_ratio = len(relations) / max(len(entities), 1)
    sufficient_for_training = len(entities) >= min_entities and len(relations) >= min_relations

    return {
        "entity_count": len(entities),
        "relation_count": len(relations),
        "connectivity_ratio": connectivity_ratio,
        "sufficient_for_training": sufficient_for_training,
        "quality_score": min(1.0, (len(entities) + len(relations)) / (min_entities * 2)),
        "validation_timestamp": datetime.now().isoformat(),
        "environment_thresholds": {
            "min_entities": min_entities,
            "min_relations": min_relations
        }
    }
```

This architecture ensures **optimal performance** (pre-computed embeddings), **cost efficiency** (environment-specific scaling), and **data freshness** (automatic retraining triggers) without calling expensive GNN inference during user queries.

