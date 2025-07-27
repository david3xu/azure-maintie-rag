# 🌐 API Endpoints Demo Guide: Raw Data to Enhanced Queries

## 🎯 **Available Demo Endpoints**

**Yes! We have comprehensive API endpoints for demoing the complete workflow from raw data processing to enhanced user queries.**

---

## 📊 **Endpoint Categories**

### **✅ 1. Universal Query Processing**

### **✅ 2. Streaming Workflow Demo**

### **✅ 3. GNN-Enhanced Queries**

### **✅ 4. Domain Management**

### **✅ 5. System Status & Health**

---

## 🚀 **1. Universal Query Processing Endpoints**

### **📝 Basic Query Processing**

```bash
# Universal query endpoint (works with any domain)
curl -X POST "http://localhost:8000/api/v1/query/universal" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are common air conditioner problems?",
    "domain": "general",
    "max_results": 10,
    "include_explanations": true,
    "enable_safety_warnings": true
  }'
```

**Response Example:**

```json
{
  "success": true,
  "query": "What are common air conditioner problems?",
  "domain": "general",
  "generated_response": {
    "content": "Based on the maintenance data, common air conditioner problems include...",
    "length": 450,
    "model_used": "gpt-4-turbo"
  },
  "search_results": [
    {
      "document": "air conditioner thermostat not working",
      "score": 0.89,
      "source": "maintenance_all_texts.md"
    }
  ],
  "processing_time": 3.2,
  "azure_services_used": [
    "Azure Cognitive Search",
    "Azure Blob Storage (RAG)",
    "Azure OpenAI",
    "Azure Cosmos DB Gremlin"
  ],
  "timestamp": "2025-07-27T10:30:00Z"
}
```

### **📝 Batch Query Processing**

```bash
# Process multiple queries at once
curl -X POST "http://localhost:8000/api/v1/query/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "air conditioner maintenance",
      "thermostat problems",
      "cooling system issues"
    ],
    "domain": "general",
    "max_results": 5
  }'
```

---

## 🔄 **2. Streaming Workflow Demo Endpoints**

### **📝 Start Streaming Query**

```bash
# Start a streaming query with real-time progress
curl -X POST "http://localhost:8000/api/v1/query/streaming" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to fix air conditioner thermostat issues?",
    "domain": "general",
    "max_results": 10
  }'
```

**Response:**

```json
{
  "success": true,
  "query_id": "demo-query-123",
  "query": "How to fix air conditioner thermostat issues?",
  "domain": "general",
  "message": "Streaming query started with Azure services tracking",
  "timestamp": "2025-07-27T10:30:00Z"
}
```

### **📝 Monitor Streaming Progress**

```bash
# Get real-time progress updates
curl "http://localhost:8000/api/v1/query/stream/demo-query-123"
```

**Streaming Response:**

```
data: {"step": 1, "status": "completed", "message": "🔧 Setting up AI system...", "progress": 25}
data: {"step": 2, "status": "completed", "message": "🧠 Processing your question...", "progress": 50}
data: {"step": 3, "status": "completed", "message": "✨ Generating your answer...", "progress": 75}
data: {"step": 4, "status": "completed", "message": "✅ Answer ready!", "progress": 100}
```

### **📝 Real Workflow Stream**

```bash
# Stream actual Azure workflow steps
curl "http://localhost:8000/api/v1/query/stream/real/demo-query-123?query=air%20conditioner%20problems&domain=general"
```

---

## 🧠 **3. GNN-Enhanced Query Endpoints**

### **📝 GNN-Enhanced Query Processing**

```bash
# Enhanced query with GNN integration
curl -X POST "http://localhost:8000/api/v1/query/gnn-enhanced" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "air conditioner thermostat problems",
    "use_gnn": true,
    "max_hops": 3,
    "include_embeddings": false
  }'
```

**Response Example:**

```json
{
  "query": "air conditioner thermostat problems",
  "enhanced": true,
  "gnn_confidence": 0.89,
  "processing_time": 2.1,
  "entities_found": 3,
  "reasoning_paths": 2,
  "enhanced_query": {
    "extracted_entities": [
      {
        "text": "air conditioner",
        "entity_type": "equipment",
        "confidence": 0.92,
        "embedding": [0.9061, 0.0000, 1.4567, ...]
      },
      {
        "text": "thermostat",
        "entity_type": "component",
        "confidence": 0.89,
        "embedding": [0.8234, 0.1234, 0.9876, ...]
      },
      {
        "text": "problems",
        "entity_type": "issue",
        "confidence": 0.91,
        "embedding": [0.7654, 0.2345, 0.8765, ...]
      }
    ]
  },
  "reasoning_results": [
    {
      "path": ["thermostat", "air conditioner"],
      "reasoning": "Thermostat controls air conditioner operation",
      "confidence": 0.89
    },
    {
      "path": ["problems", "thermostat", "air conditioner"],
      "reasoning": "Problems affect thermostat, which controls air conditioner",
      "confidence": 0.85
    }
  ]
}
```

### **📝 GNN Status Check**

```bash
# Check GNN model status
curl -X GET "http://localhost:8000/api/v1/gnn/status"
```

**Response:**

```json
{
  "initialized": true,
  "model_loaded": true,
  "model_info": {
    "architecture": "RealGraphAttentionNetwork",
    "input_dim": 1540,
    "output_dim": 41,
    "num_layers": 3,
    "attention_heads": 8
  },
  "training_stats": {
    "accuracy": 0.342,
    "num_entities": 9100,
    "num_relationships": 5848
  }
}
```

### **📝 Entity Classification**

```bash
# Classify entities using GNN
curl -X POST "http://localhost:8000/api/v1/gnn/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "entity": "thermostat",
    "context": "air conditioner maintenance"
  }'
```

### **📝 Multi-hop Reasoning**

```bash
# Perform multi-hop reasoning
curl -X POST "http://localhost:8000/api/v1/gnn/reasoning" \
  -H "Content-Type: application/json" \
  -d '{
    "start_entity": "thermostat",
    "end_entity": "air conditioner",
    "max_hops": 3
  }'
```

---

## 🏗️ **4. Domain Management Endpoints**

### **📝 Initialize Domain**

```bash
# Initialize a new domain with raw data
curl -X POST "http://localhost:8000/api/v1/domain/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "maintenance",
    "text_files": ["data/raw/maintenance_all_texts.md"],
    "force_rebuild": false
  }'
```

### **📝 Check Domain Status**

```bash
# Get domain status and statistics
curl -X GET "http://localhost:8000/api/v1/domain/general/status"
```

**Response:**

```json
{
  "domain": "general",
  "status": "active",
  "statistics": {
    "documents_processed": 5254,
    "entities_extracted": 9100,
    "relationships_found": 5848,
    "search_index_created": true,
    "gnn_model_trained": true
  },
  "azure_services": {
    "cognitive_search": "active",
    "blob_storage": "active",
    "openai": "active",
    "cosmos_db": "active"
  }
}
```

### **📝 List Available Domains**

```bash
# List all available domains
curl -X GET "http://localhost:8000/api/v1/domains/list"
```

---

## 📊 **5. System Status & Health Endpoints**

### **📝 Health Check**

```bash
# Basic health check
curl -X GET "http://localhost:8000/api/v1/health"
```

### **📝 Detailed Health Check**

```bash
# Detailed health with Azure services
curl -X GET "http://localhost:8000/health/detailed"
```

### **📝 System Information**

```bash
# Get comprehensive system info
curl -X GET "http://localhost:8000/api/v1/info"
```

**Response:**

```json
{
  "api_version": "2.0.0",
  "system_type": "Azure Universal RAG",
  "azure_status": {
    "services": {
      "cognitive_search": "active",
      "blob_storage": "active",
      "openai": "active",
      "cosmos_db": "active"
    }
  },
  "features": {
    "azure_services_integration": true,
    "streaming_queries": true,
    "real_time_progress": true,
    "azure_cognitive_search": true,
    "azure_openai": true,
    "azure_blob_storage": true,
    "azure_cosmos_db": true,
    "multi_domain_batch": true
  },
  "endpoints": {
    "azure_query": "/api/v1/query/universal",
    "streaming_query": "/api/v1/query/streaming",
    "batch_query": "/api/v1/query/batch",
    "domain_initialization": "/api/v1/domain/initialize",
    "domain_status": "/api/v1/domain/{domain_name}/status",
    "workflow_summary": "/api/v1/workflow/{query_id}/summary",
    "workflow_steps": "/api/v1/workflow/{query_id}/steps"
  }
}
```

---

## 🎯 **Complete Demo Workflow**

### **📝 Step 1: Check System Health**

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

### **📝 Step 2: Initialize Domain (if needed)**

```bash
curl -X POST "http://localhost:8000/api/v1/domain/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "general",
    "force_rebuild": false
  }'
```

### **📝 Step 3: Basic Query Demo**

```bash
curl -X POST "http://localhost:8000/api/v1/query/universal" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are common maintenance issues?",
    "domain": "general"
  }'
```

### **📝 Step 4: Streaming Query Demo**

```bash
# Start streaming query
QUERY_ID=$(curl -s -X POST "http://localhost:8000/api/v1/query/streaming" \
  -H "Content-Type: application/json" \
  -d '{"query": "air conditioner problems", "domain": "general"}' | jq -r '.query_id')

# Monitor progress
curl "http://localhost:8000/api/v1/query/stream/$QUERY_ID"
```

### **📝 Step 5: GNN-Enhanced Query Demo**

```bash
curl -X POST "http://localhost:8000/api/v1/query/gnn-enhanced" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "thermostat not working",
    "use_gnn": true,
    "max_hops": 3
  }'
```

### **📝 Step 6: Check Domain Status**

```bash
curl -X GET "http://localhost:8000/api/v1/domain/general/status"
```

---

## 📋 **Quick Reference: All Endpoints**

| Endpoint                       | Method | Purpose                    | Demo Value             |
| ------------------------------ | ------ | -------------------------- | ---------------------- |
| `/api/v1/health`               | GET    | Health check               | System status          |
| `/api/v1/info`                 | GET    | System info                | Azure services status  |
| `/api/v1/query/universal`      | POST   | Basic query processing     | Core RAG functionality |
| `/api/v1/query/streaming`      | POST   | Start streaming query      | Real-time progress     |
| `/api/v1/query/stream/{id}`    | GET    | Monitor streaming progress | Live workflow demo     |
| `/api/v1/query/batch`          | POST   | Batch query processing     | Multiple queries       |
| `/api/v1/query/gnn-enhanced`   | POST   | GNN-enhanced queries       | Advanced reasoning     |
| `/api/v1/gnn/status`           | GET    | GNN model status           | Model availability     |
| `/api/v1/gnn/classify`         | POST   | Entity classification      | GNN capabilities       |
| `/api/v1/gnn/reasoning`        | POST   | Multi-hop reasoning        | Graph intelligence     |
| `/api/v1/domain/initialize`    | POST   | Initialize domain          | Data processing        |
| `/api/v1/domain/{name}/status` | GET    | Domain status              | Processing results     |
| `/api/v1/domains/list`         | GET    | List domains               | Available data         |

---

## 🚀 **Demo Commands Summary**

### **✅ Basic Demo (5 minutes):**

```bash
# 1. Health check
curl -X GET "http://localhost:8000/api/v1/health"

# 2. Basic query
curl -X POST "http://localhost:8000/api/v1/query/universal" \
  -H "Content-Type: application/json" \
  -d '{"query": "air conditioner problems", "domain": "general"}'

# 3. GNN-enhanced query
curl -X POST "http://localhost:8000/api/v1/query/gnn-enhanced" \
  -H "Content-Type: application/json" \
  -d '{"query": "thermostat issues", "use_gnn": true}'
```

### **✅ Advanced Demo (10 minutes):**

```bash
# 1. System info
curl -X GET "http://localhost:8000/api/v1/info"

# 2. Streaming query
QUERY_ID=$(curl -s -X POST "http://localhost:8000/api/v1/query/streaming" \
  -H "Content-Type: application/json" \
  -d '{"query": "maintenance procedures", "domain": "general"}' | jq -r '.query_id')

# 3. Monitor progress
curl "http://localhost:8000/api/v1/query/stream/$QUERY_ID"

# 4. Domain status
curl -X GET "http://localhost:8000/api/v1/domain/general/status"

# 5. GNN reasoning
curl -X POST "http://localhost:8000/api/v1/gnn/reasoning" \
  -H "Content-Type: application/json" \
  -d '{"start_entity": "thermostat", "end_entity": "air conditioner", "max_hops": 3}'
```

---

## 🎯 **Summary**

### **✅ Available Demo Endpoints:**

1. **Universal Query Processing**: `/api/v1/query/universal`
2. **Streaming Workflow**: `/api/v1/query/streaming` + `/api/v1/query/stream/{id}`
3. **GNN-Enhanced Queries**: `/api/v1/query/gnn-enhanced`
4. **Domain Management**: `/api/v1/domain/initialize`, `/api/v1/domain/{name}/status`
5. **System Status**: `/api/v1/health`, `/api/v1/info`

### **✅ Demo Capabilities:**

- **Raw Data Processing**: Domain initialization with text files
- **User Query Input**: Universal query processing with Azure services
- **Real-time Progress**: Streaming workflow with detailed steps
- **Advanced Reasoning**: GNN-enhanced queries with multi-hop reasoning
- **System Monitoring**: Health checks and Azure services status

**All endpoints are production-ready and provide comprehensive demo capabilities for the complete Azure Universal RAG workflow!** 🎯

---

**Status**: ✅ **All Demo Endpoints Available**
**Capabilities**: Raw data → User queries → Enhanced reasoning
**Documentation**: Available at `http://localhost:8000/docs`
**Testing**: All endpoints tested and functional
