# Azure Universal RAG - API Reference

**Complete API Documentation**

📖 **Related Documentation:**
- ⬅️ [Back to Main README](README.md)
- 🏗️ [System Architecture](ARCHITECTURE.md)
- ⚙️ [Setup Guide](SETUP.md)
- 🚀 [Deployment Guide](DEPLOYMENT.md)

---

## 🌐 API Overview

The Azure Universal RAG system provides a comprehensive RESTful API built with FastAPI, offering real-time query processing, data management, and system monitoring capabilities.

### **Base URL**
- **Development**: `http://localhost:8000`
- **Production**: `https://[your-deployment-url]/`

### **API Versioning**
All endpoints are versioned with the prefix `/api/v1/`

### **Authentication**
- **Development**: No authentication required
- **Production**: Managed Identity + RBAC (auto-configured)

---

## 🔍 Query Processing Endpoints

### **Universal Query Processing**

#### **POST /api/v1/query/universal**

Process a query through the complete Azure Universal RAG pipeline with multi-source search and real-time streaming.

**Request Body:**
```json
{
  "query": "What are common maintenance issues with air conditioners?",
  "domain": "maintenance",
  "max_results": 10,
  "include_sources": true,
  "enable_streaming": true,
  "search_mode": "hybrid"
}
```

**Request Schema:**
```python
class UniversalQueryRequest(BaseModel):
    query: str                                    # User query text
    domain: str = "maintenance"                   # Domain for scoped search
    max_results: int = 10                         # Maximum results to return
    include_sources: bool = True                  # Include source citations
    enable_streaming: bool = True                 # Enable real-time progress
    search_mode: str = "hybrid"                   # "vector", "graph", "hybrid"
    confidence_threshold: float = 0.7             # Minimum confidence for results
    enable_multihop: bool = True                  # Enable multi-hop reasoning
```

**Response:**
```json
{
  "query_id": "query-20250728-143022-abc123",
  "original_query": "What are common maintenance issues with air conditioners?",
  "processed_query": "air conditioner maintenance problems troubleshooting repair",
  "response": "Common air conditioner maintenance issues include: 1. Thermostat malfunctions...",
  "sources": [
    {
      "id": "doc-001",
      "title": "HVAC Maintenance Guide",
      "content": "Thermostat calibration is critical for proper AC operation...",
      "confidence": 0.92,
      "source_type": "document",
      "azure_service": "cognitive_search"
    }
  ],
  "confidence_score": 0.89,
  "processing_time": 2.34,
  "workflow_steps": [
    {
      "step": "query_analysis",
      "status": "completed",
      "duration": 0.45,
      "azure_service": "openai"
    }
  ],
  "metadata": {
    "entities_found": 15,
    "relationships_discovered": 8,
    "vector_results": 5,
    "graph_results": 3,
    "gnn_predictions": 2
  }
}
```

**Status Codes:**
- `200`: Query processed successfully
- `400`: Invalid request format or parameters
- `429`: Rate limit exceeded
- `500`: Internal server error during processing

#### **Example Usage:**
```bash
curl -X POST "http://localhost:8000/api/v1/query/universal" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "troubleshooting air conditioner thermostat problems",
    "domain": "maintenance",
    "max_results": 5,
    "search_mode": "hybrid"
  }'
```

---

### **Streaming Query Progress**

#### **GET /api/v1/query/stream/{query_id}**

Real-time Server-Sent Events (SSE) stream for query processing progress with three-layer transparency.

**Path Parameters:**
- `query_id`: Unique query identifier from universal query response

**Query Parameters:**
- `detail_level`: `user` | `technical` | `diagnostic` (default: `user`)
- `include_metrics`: `true` | `false` (default: `false`)

**Response (Server-Sent Events):**
```
event: step_start
data: {"step": "query_analysis", "message": "🔍 Understanding your question...", "progress": 0.1}

event: step_progress  
data: {"step": "vector_search", "message": "🔍 Searching Azure Cognitive Search...", "progress": 0.4, "details": {"documents_found": 15}}

event: step_complete
data: {"step": "response_generation", "message": "📝 Generating comprehensive answer...", "progress": 1.0, "duration": 2.34}

event: query_complete
data: {"query_id": "query-20250728-143022-abc123", "status": "completed", "total_duration": 2.34}
```

**Event Types:**
- `step_start`: Processing step initiated
- `step_progress`: Progress update within step
- `step_complete`: Processing step completed
- `query_complete`: Entire query processing finished
- `error`: Error occurred during processing

#### **Example Usage:**
```javascript
// JavaScript/TypeScript client
const eventSource = new EventSource('/api/v1/query/stream/query-20250728-143022-abc123?detail_level=technical');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log(`Step: ${data.step}, Progress: ${data.progress * 100}%`);
};
```

---

## 📊 Data Management Endpoints

### **Domain Data Processing**

#### **POST /api/v1/data/process**

Trigger complete data processing pipeline for a domain through Azure services (Storage → Search → Cosmos → GNN).

**Request Body:**
```json
{
  "domain": "maintenance",
  "source_path": "/path/to/raw/data",  
  "force_reprocess": false,
  "batch_size": 100,
  "enable_gnn_training": true
}
```

**Response:**
```json
{
  "success": true,
  "domain": "maintenance", 
  "processing_id": "proc-20250728-143022-def456",
  "migration_summary": {
    "total_migrations": 3,
    "successful_migrations": 3,
    "failed_migrations": 0
  },
  "details": {
    "status": "completed",
    "duration": "0:02:45.123456",
    "migrations": {
      "storage": {
        "success": true,
        "container": "rag-data-maintenance",
        "uploaded_files": 4,
        "failed_uploads": 0
      },
      "search": {
        "success": true,
        "index_name": "maintie-index-maintenance",
        "documents_indexed": 327,
        "failed_documents": 0
      },
      "cosmos": {
        "success": true,
        "database": "maintie-rag-development",
        "entities_created": 207,
        "relationships_created": 156
      }
    }
  }
}
```

#### **Example Usage:**
```bash
curl -X POST "http://localhost:8000/api/v1/data/process" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "maintenance",
    "force_reprocess": false,
    "enable_gnn_training": true
  }'
```

---

### **Domain Data State**

#### **GET /api/v1/data/state/{domain}**

Get current state of data processing and service readiness for a domain.

**Path Parameters:**
- `domain`: Domain name (e.g., "maintenance")

**Query Parameters:**
- `include_details`: `true` | `false` (default: `false`)
- `validate_health`: `true` | `false` (default: `true`)

**Response:**
```json
{
  "domain": "maintenance",
  "requires_processing": false,
  "data_sources_ready": 3,
  "total_data_sources": 3,
  "services": {
    "storage": {
      "ready": true,
      "container": "rag-data-maintenance",
      "blob_count": 4,
      "last_modified": "2025-07-28T14:30:22Z"
    },
    "search": {
      "ready": true,
      "index_name": "maintie-index-maintenance", 
      "document_count": 327,
      "last_updated": "2025-07-28T14:30:45Z"
    },
    "cosmos": {
      "ready": true,
      "database": "maintie-rag-development",
      "vertex_count": 207,
      "edge_count": 156,
      "last_modified": "2025-07-28T14:31:02Z"
    }
  },
  "health_status": {
    "overall": "healthy",
    "azure_services": {
      "openai": "connected",
      "search": "connected", 
      "storage": "connected",
      "cosmos": "connected"
    }
  }
}
```

#### **Example Usage:**
```bash
curl "http://localhost:8000/api/v1/data/state/maintenance?include_details=true"
```

---

## 🧠 Machine Learning Endpoints

### **GNN Training**

#### **POST /api/v1/ml/train-gnn**

Initiate Graph Neural Network training on domain knowledge graph.

**Request Body:**
```json
{
  "domain": "maintenance",
  "model_config": {
    "hidden_dim": 512,
    "num_layers": 3,
    "dropout": 0.3,
    "learning_rate": 0.001
  },
  "training_config": {
    "epochs": 200,
    "batch_size": 64,
    "early_stopping": true,
    "validation_split": 0.2
  },
  "use_azure_ml": true
}
```

**Response:**
```json
{
  "training_id": "gnn-train-20250728-143022-ghi789",
  "status": "started",
  "domain": "maintenance",
  "estimated_duration": "15-30 minutes",
  "azure_ml_job": {
    "job_id": "azure-ml-job-123456",
    "compute_target": "cpu-cluster",
    "experiment_name": "universal-rag-gnn"
  },
  "model_info": {
    "architecture": "GCN",
    "parameters": "7.4M",
    "input_features": 1540
  }
}
```

#### **GET /api/v1/ml/training-status/{training_id}**

Get training status and progress for GNN model.

**Response:**
```json
{
  "training_id": "gnn-train-20250728-143022-ghi789",
  "status": "completed",
  "progress": 1.0,
  "current_epoch": 150,
  "total_epochs": 200,
  "metrics": {
    "train_accuracy": 0.432,
    "val_accuracy": 0.389,
    "train_loss": 1.234,
    "val_loss": 1.456
  },
  "early_stopped": true,
  "model_path": "models/gnn_model_maintenance_20250728.pt",
  "training_duration": "0:18:34.567890"
}
```

---

## 🌐 Graph Operations Endpoints

### **Entity Management**

#### **GET /api/v1/graph/entities/{domain}**

Retrieve entities from knowledge graph for specified domain.

**Path Parameters:**
- `domain`: Domain name

**Query Parameters:**
- `entity_type`: Filter by entity type (optional)
- `limit`: Maximum entities to return (default: 100)
- `offset`: Pagination offset (default: 0)
- `search`: Search term for entity text (optional)

**Response:**
```json
{
  "domain": "maintenance",
  "total_count": 207,
  "returned_count": 50,
  "entities": [
    {
      "id": "entity-001",
      "text": "air conditioner thermostat",
      "entity_type": "component",
      "confidence": 0.92,
      "relationships_count": 8,
      "centrality_score": 0.156,
      "metadata": {
        "source_documents": ["doc-001", "doc-005"],
        "extraction_date": "2025-07-28T14:30:22Z"
      }
    }
  ]
}
```

#### **POST /api/v1/graph/entities**

Add new entity to knowledge graph.

**Request Body:**
```json
{
  "domain": "maintenance",
  "text": "hvac system controller",
  "entity_type": "component", 
  "confidence": 0.87,
  "metadata": {
    "source": "manual_entry",
    "category": "equipment"
  }
}
```

---

### **Relationship Operations**

#### **GET /api/v1/graph/relationships/{domain}**

Retrieve relationships from knowledge graph.

**Query Parameters:**
- `source_entity`: Filter by source entity ID
- `target_entity`: Filter by target entity ID
- `relationship_type`: Filter by relationship type
- `min_confidence`: Minimum confidence threshold

**Response:**
```json
{
  "domain": "maintenance",
  "total_count": 156,
  "relationships": [
    {
      "id": "rel-001",
      "source_entity": "air_conditioner",
      "target_entity": "thermostat", 
      "relationship_type": "has_component",
      "confidence": 0.89,
      "metadata": {
        "co_occurrence_count": 15,
        "semantic_similarity": 0.83
      }
    }
  ]
}
```

#### **GET /api/v1/graph/paths**

Find paths between entities using multi-hop reasoning.

**Query Parameters:**
- `domain`: Domain name
- `start_entity`: Starting entity text
- `end_entity`: Target entity text  
- `max_hops`: Maximum path length (default: 3)
- `min_confidence`: Minimum path confidence

**Response:**
```json
{
  "domain": "maintenance",
  "start_entity": "air_conditioner",
  "end_entity": "repair_manual",
  "paths": [
    {
      "path_id": "path-001",
      "entities": ["air_conditioner", "thermostat", "troubleshooting", "repair_manual"],
      "relationships": ["has_component", "requires", "documented_in"],
      "confidence": 0.78,
      "path_length": 3,
      "semantic_coherence": 0.82
    }
  ],
  "total_paths": 5
}
```

---

## ⚙️ System Management Endpoints

### **Health Monitoring**

#### **GET /health**

Basic health check endpoint for load balancers and monitoring.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-28T14:30:22Z",
  "version": "1.0.0",
  "uptime": "2 days, 14:30:22"
}
```

#### **GET /api/v1/system/health-detailed**

Comprehensive health check including all Azure services.

**Response:**
```json
{
  "overall_status": "healthy",
  "timestamp": "2025-07-28T14:30:22Z",
  "services": {
    "openai": {
      "status": "healthy",
      "endpoint": "https://oai-maintie-rag-dev-abc123.openai.azure.com/",
      "models": ["gpt-4", "text-embedding-ada-002"],
      "response_time": 245
    },
    "search": {
      "status": "healthy", 
      "endpoint": "https://srch-maintie-rag-dev-abc123.search.windows.net",
      "indices": ["maintie-index-maintenance"],
      "response_time": 123
    },
    "storage": {
      "status": "healthy",
      "account": "stmaintieabc123",
      "containers": ["rag-data-maintenance", "ml-models"],
      "response_time": 67
    },
    "cosmos": {
      "status": "healthy",
      "endpoint": "https://cosmos-maintie-rag-dev-abc123.documents.azure.com/",
      "databases": ["maintie-rag-development"],
      "response_time": 189
    }
  },
  "performance_metrics": {
    "avg_query_time": 2.34,
    "cache_hit_rate": 0.67,
    "active_connections": 12
  }
}
```

---

### **System Metrics**

#### **GET /api/v1/system/metrics**

Get system performance metrics and statistics.

**Query Parameters:**
- `time_range`: Time range for metrics (`1h`, `24h`, `7d`, `30d`)
- `metric_types`: Comma-separated list of metric types

**Response:**
```json
{
  "time_range": "24h",
  "timestamp": "2025-07-28T14:30:22Z",
  "metrics": {
    "queries": {
      "total_count": 1247,
      "avg_response_time": 2.34,
      "success_rate": 0.987,
      "error_rate": 0.013
    },
    "data_processing": {
      "documents_processed": 4,
      "entities_extracted": 207, 
      "relationships_created": 156,
      "avg_processing_time": 165.43
    },
    "azure_services": {
      "openai_requests": 2401,
      "search_queries": 1847,
      "storage_operations": 234,
      "cosmos_operations": 1456
    },
    "performance": {
      "cache_hit_rate": 0.67,
      "memory_usage": 0.45,
      "cpu_usage": 0.23,
      "active_connections": 12
    }
  }
}
```

---

## 🔧 Configuration Endpoints

### **System Configuration**

#### **GET /api/v1/config/settings**

Get current system configuration (non-sensitive values only).

**Response:**
```json
{
  "domain_settings": {
    "default_domain": "maintenance",
    "supported_domains": ["maintenance", "technical", "general"]
  },
  "query_settings": {
    "max_results": 10,
    "default_search_mode": "hybrid",
    "confidence_threshold": 0.7,
    "enable_multihop": true,
    "max_hops": 3
  },
  "processing_settings": {
    "batch_size": 100,
    "enable_caching": true,
    "cache_ttl": 3600,
    "enable_streaming": true
  },
  "azure_settings": {
    "use_managed_identity": true,
    "region": "eastus",
    "environment": "development"
  }
}
```

#### **PUT /api/v1/config/settings**

Update system configuration (requires admin privileges).

**Request Body:**
```json
{
  "query_settings": {
    "max_results": 15,
    "confidence_threshold": 0.75
  },
  "processing_settings": {
    "batch_size": 150,
    "cache_ttl": 7200
  }
}
```

---

## 📡 Webhook Endpoints

### **Processing Webhooks**

#### **POST /api/v1/webhooks/processing-complete**

Webhook endpoint for Azure services to notify when processing is complete.

**Request Body:**
```json
{
  "event_type": "processing_complete",
  "processing_id": "proc-20250728-143022-def456",
  "domain": "maintenance",
  "status": "success",
  "timestamp": "2025-07-28T14:30:22Z",
  "details": {
    "documents_processed": 4,
    "entities_created": 207,
    "relationships_created": 156
  }
}
```

---

## 🚨 Error Handling

### **Standard Error Response Format**

All API errors return a consistent format:

```json
{
  "error": {
    "code": "QUERY_PROCESSING_FAILED",
    "message": "Failed to process query due to Azure service timeout",
    "details": {
      "service": "azure_openai",
      "timeout": 30000,
      "retry_count": 3
    },
    "timestamp": "2025-07-28T14:30:22Z",
    "request_id": "req-20250728-143022-xyz789"
  }
}
```

### **Common Error Codes**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Request format or parameters invalid |
| `AUTHENTICATION_FAILED` | 401 | Authentication credentials invalid |
| `AUTHORIZATION_FAILED` | 403 | Insufficient permissions for operation |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource does not exist |
| `RATE_LIMIT_EXCEEDED` | 429 | Request rate limit exceeded |
| `AZURE_SERVICE_ERROR` | 502 | Azure service unavailable or error |
| `PROCESSING_TIMEOUT` | 504 | Operation timed out |
| `INTERNAL_ERROR` | 500 | Unexpected internal server error |

---

## 📊 Rate Limits

### **Default Rate Limits**

| Endpoint Category | Requests per Minute | Burst Limit |
|------------------|-------------------|-------------|
| Query Processing | 60 | 10 |
| Data Management | 30 | 5 |
| Graph Operations | 120 | 20 |
| System Management | 300 | 50 |

### **Rate Limit Headers**

All responses include rate limit information:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
X-RateLimit-Retry-After: 60
```

---

## 🔍 API Examples

### **Complete Query Processing Workflow**

```python
import requests
import json
from time import sleep

# 1. Submit query
query_response = requests.post('http://localhost:8000/api/v1/query/universal', 
    json={
        "query": "air conditioner thermostat troubleshooting",
        "domain": "maintenance",
        "enable_streaming": True,
        "search_mode": "hybrid"
    }
)

query_data = query_response.json()
query_id = query_data['query_id']

print(f"Query submitted: {query_id}")
print(f"Processing time: {query_data['processing_time']}s")
print(f"Confidence: {query_data['confidence_score']}")
print(f"Sources found: {len(query_data['sources'])}")

# 2. Monitor progress (if streaming enabled)
import sseclient

messages = sseclient.SSEClient(f'http://localhost:8000/api/v1/query/stream/{query_id}')
for msg in messages:
    if msg.data:
        event_data = json.loads(msg.data)
        print(f"Progress: {event_data.get('step')} - {event_data.get('progress', 0) * 100:.1f}%")
        
        if msg.event == 'query_complete':
            break

# 3. Get final results
final_response = query_data
print(f"\nFinal Answer: {final_response['response'][:200]}...")
```

### **Data Processing Pipeline**

```python
import requests
import json

# 1. Check current data state
state_response = requests.get('http://localhost:8000/api/v1/data/state/maintenance')
state_data = state_response.json()

print(f"Data sources ready: {state_data['data_sources_ready']}/{state_data['total_data_sources']}")

# 2. Process data if needed
if state_data['requires_processing']:
    process_response = requests.post('http://localhost:8000/api/v1/data/process',
        json={
            "domain": "maintenance",
            "force_reprocess": False,
            "enable_gnn_training": True
        }
    )
    
    process_data = process_response.json()
    print(f"Processing started: {process_data['processing_id']}")
    print(f"Migrations: {process_data['migration_summary']}")

# 3. Verify processing completion
final_state = requests.get('http://localhost:8000/api/v1/data/state/maintenance')
final_data = final_state.json()

print(f"Final state: {final_data['data_sources_ready']}/3 services ready")
print(f"Documents indexed: {final_data['services']['search']['document_count']}")
print(f"Entities created: {final_data['services']['cosmos']['vertex_count']}")
```

---

## 📚 SDK and Client Libraries

### **Python SDK**

```python
# Install: pip install azure-maintie-rag-sdk
from azure_maintie_rag import UniversalRAGClient

# Initialize client
client = UniversalRAGClient(base_url="http://localhost:8000")

# Query processing
result = await client.query("air conditioner problems", domain="maintenance")
print(f"Answer: {result.response}")
print(f"Sources: {len(result.sources)}")

# Stream processing
async for progress in client.query_stream("thermostat issues"):
    print(f"Step: {progress.step}, Progress: {progress.progress}")

# Data management
await client.process_domain_data("maintenance")
state = await client.get_domain_state("maintenance")
```

### **JavaScript/TypeScript SDK**

```typescript
// Install: npm install @azure-maintie-rag/sdk
import { UniversalRAGClient } from '@azure-maintie-rag/sdk';

const client = new UniversalRAGClient({ baseUrl: 'http://localhost:8000' });

// Query processing
const result = await client.query({
  query: 'air conditioner problems',
  domain: 'maintenance',
  searchMode: 'hybrid'
});

console.log(`Answer: ${result.response}`);
console.log(`Confidence: ${result.confidenceScore}`);

// Streaming
const stream = client.queryStream(result.queryId);
stream.on('progress', (event) => {
  console.log(`${event.step}: ${event.progress * 100}%`);
});
```

---

## 🔧 Development and Testing

### **API Development Environment**

```bash
# Start development server with auto-reload
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# API will be available at:
# - API endpoints: http://localhost:8000/api/v1/
# - Interactive docs: http://localhost:8000/docs
# - OpenAPI schema: http://localhost:8000/openapi.json
```

### **Testing Endpoints**

```bash
# Health check
curl http://localhost:8000/health

# Detailed health check  
curl http://localhost:8000/api/v1/system/health-detailed

# Test query processing
curl -X POST http://localhost:8000/api/v1/query/universal \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "domain": "maintenance"}'

# Check data state
curl http://localhost:8000/api/v1/data/state/maintenance
```

### **API Testing Suite**

```python
# tests/test_api.py
import pytest
import httpx

@pytest.fixture
async def client():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        yield client

async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

async def test_universal_query(client):
    response = await client.post("/api/v1/query/universal", 
        json={"query": "test query", "domain": "maintenance"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "query_id" in data
    assert "response" in data
    assert "confidence_score" in data
```

---

**📖 Navigation:**
- ⬅️ [Back to Main README](README.md)
- 🏗️ [System Architecture](ARCHITECTURE.md)
- ⚙️ [Setup Guide](SETUP.md)
- 🚀 [Deployment Guide](DEPLOYMENT.md)

---

**API Status**: ✅ **Production-Ready** | **Version**: v1.0.0 | **Last Updated**: July 28, 2025