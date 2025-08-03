# API Reference

**Azure Universal RAG - Complete API Documentation**

---

## 🚀 API Overview

The Azure Universal RAG system provides a REST API with simplified endpoints that leverage our competitive advantages:

- **Tri-modal search** (Vector + Graph + GNN)
- **Zero-config domain adaptation**
- **Sub-3-second response guarantee**
- **100% data-driven intelligence**

**Base URL**: `http://localhost:8000/api/v1`

---

## 🧠 Agent Endpoints

### POST /agent/query

**Primary endpoint for intelligent query processing**

```http
POST /api/v1/agent/query
Content-Type: application/json

{
  "query": "How do I troubleshoot network connectivity issues?",
  "domain": "general", 
  "context": {},
  "session_id": "optional-session-id"
}
```

**Response**:
```json
{
  "success": true,
  "query": "How do I troubleshoot network connectivity issues?",
  "agent_response": "To troubleshoot network connectivity...",
  "confidence": 0.94,
  "execution_time": 0.5,
  "tools_used": ["domain_discovery", "tri_modal_search"],
  "session_id": "session-123",
  "correlation_id": "corr-456",
  "metadata": {
    "domain": "general",
    "competitive_advantages": [
      "tri_modal_search",
      "zero_config_domain_adaptation",
      "dynamic_tool_discovery"
    ]
  },
  "timestamp": "2025-01-31T12:00:00Z"
}
```

---

## 🔍 Query Endpoints

### POST /query/universal

**Universal query processing with Azure services**

```http
POST /api/v1/query/universal
Content-Type: application/json

{
  "query": "Medical device troubleshooting procedures", 
  "domain": "medical",
  "max_results": 10,
  "include_explanations": true
}
```

### POST /query/streaming

**Start streaming query with real-time progress**

```http
POST /api/v1/query/streaming
Content-Type: application/json

{
  "query": "Legal compliance requirements",
  "domain": "legal"
}
```

**Response**:
```json
{
  "success": true,
  "query_id": "query-789",
  "message": "Streaming query started",
  "timestamp": "2025-01-31T12:00:00Z"
}
```

### GET /query/stream/{query_id}

**Stream real-time progress updates**

```http
GET /api/v1/query/stream/{query_id}
Accept: text/event-stream
```

**Server-Sent Events**:
```
data: {"step": "domain_discovery", "progress": 25, "message": "🧠 Detecting domain..."}

data: {"step": "tri_modal_search", "progress": 75, "message": "🔍 Tri-modal search..."}

data: {"status": "completed", "progress": 100, "message": "✅ Query completed"}
```

---

## 🏥 Health & Monitoring

### GET /health

**System health check**

```http
GET /api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "simplified_architecture": true,
  "competitive_advantages": {
    "tri_modal_search": "enabled",
    "zero_config_discovery": "enabled", 
    "sub_3s_response": "maintained",
    "data_driven_intelligence": "active"
  },
  "performance_metrics": {
    "avg_response_time": 0.5,
    "cache_hit_rate": 0.6,
    "domain_discovery_time": 0.01
  },
  "timestamp": "2025-01-31T12:00:00Z"
}
```

### GET /agent/health

**Agent-specific health check**

```http
GET /api/v1/agent/health
```

### GET /agent/metrics

**Detailed performance metrics**

```http
GET /api/v1/agent/metrics
```

---

## 🌐 Domain Management

### GET /domains/list

**List available domains**

```http
GET /api/v1/domains/list
```

**Response**:
```json
{
  "success": true,
  "domains": [
    {
      "name": "general",
      "status": "active",
      "azure_services": ["blob_storage", "cognitive_search", "cosmos_db"]
    },
    {
      "name": "medical", 
      "status": "active",
      "last_modified": "2025-01-31T12:00:00Z"
    }
  ],
  "total_domains": 2
}
```

### POST /domain/initialize

**Initialize new domain**

```http
POST /api/v1/domain/initialize
Content-Type: application/json

{
  "domain": "legal",
  "force_rebuild": false
}
```

---

## 📊 Request/Response Models

### Core Models

```typescript
// Query Request
interface QueryRequest {
  query: string;                    // Required: User query
  domain?: string;                  // Optional: Domain context
  max_results?: number;             // Default: 10
  context?: Record<string, any>;    // Optional: Additional context
}

// Agent Response  
interface AgentResponse {
  success: boolean;                 // Operation success
  query: string;                    // Original query
  agent_response: string;           // AI-generated response
  confidence: number;               // 0.0 - 1.0 confidence score
  execution_time: number;           // Response time in seconds
  tools_used: string[];             // Tools utilized
  session_id: string;               // Session identifier
  correlation_id: string;           // Request correlation
  metadata: {
    domain?: string;
    competitive_advantages: string[];
  };
  timestamp: string;                // ISO timestamp
  error?: string;                   // Error message if failed
}
```

---

## ⚡ Performance Guarantees

### Response Time SLAs

| Endpoint | Target | Typical | Status |
|----------|--------|---------|--------|
| `/agent/query` | <3.0s | 0.5s | ✅ |
| `/query/universal` | <3.0s | 0.8s | ✅ |
| `/health` | <0.1s | 0.01s | ✅ |
| `/domains/list` | <0.5s | 0.1s | ✅ |

### Competitive Advantage Validation

All API responses include competitive advantage validation:

```json
{
  "metadata": {
    "competitive_advantages": [
      "tri_modal_search",           // Vector + Graph + GNN
      "zero_config_domain_adaptation", // No manual setup
      "dynamic_tool_discovery",     // Automatic capability detection
      "sub_3s_response"            // Performance guarantee
    ]
  }
}
```

---

## 🔒 Authentication & Security

### API Key Authentication

```http
Authorization: Bearer your-api-key
```

### Azure AD Integration

```http
Authorization: Bearer azure-ad-token
```

### Rate Limiting

- **Standard**: 100 requests/minute
- **Premium**: 1000 requests/minute
- **Enterprise**: Unlimited

---

## 🚨 Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": {
    "type": "TRANSIENT",              // TRANSIENT, PERMANENT, CRITICAL
    "message": "Service temporarily unavailable",
    "code": 503,
    "retry_after": 30,                // Seconds to wait before retry
    "correlation_id": "corr-123"
  },
  "timestamp": "2025-01-31T12:00:00Z"
}
```

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request format |
| 401 | Unauthorized | Authentication required |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | System error |
| 503 | Service Unavailable | Temporary unavailability |

---

## 🔧 Client Examples

### Python Client

```python
import requests
import asyncio
import aiohttp

class AzureUniversalRAGClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def query(self, query: str, domain: str = None) -> dict:
        """Process intelligent query"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/agent/query",
                json={"query": query, "domain": domain},
                headers=self.headers
            ) as response:
                return await response.json()

# Usage
client = AzureUniversalRAGClient("http://localhost:8000", "your-api-key")
result = await client.query("How do I optimize database performance?")
```

### JavaScript Client

```javascript
class AzureUniversalRAGClient {
  constructor(baseUrl, apiKey) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    };
  }

  async query(query, domain = null) {
    const response = await fetch(`${this.baseUrl}/api/v1/agent/query`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ query, domain })
    });
    
    return await response.json();
  }
}

// Usage
const client = new AzureUniversalRAGClient('http://localhost:8000', 'your-api-key');
const result = await client.query('Explain machine learning concepts');
```

---

## 📈 Monitoring & Analytics

### Built-in Metrics

Every API response includes performance metrics:

```json
{
  "performance_metrics": {
    "execution_time": 0.5,           // Response time in seconds
    "cache_hit": true,               // Whether result was cached
    "domain_discovery_time": 0.01,   // Domain detection time
    "search_breakdown": {
      "vector_search": 0.05,
      "graph_search": 0.08, 
      "gnn_search": 0.03
    }
  }
}
```

### Health Monitoring

```bash
# Continuous health monitoring
curl -s http://localhost:8000/api/v1/health | jq '.status'

# Performance validation
curl -s http://localhost:8000/api/v1/agent/metrics | jq '.competitive_advantages'
```

---

*This API reference reflects the simplified architecture with preserved competitive advantages and sub-3-second performance guarantee.*