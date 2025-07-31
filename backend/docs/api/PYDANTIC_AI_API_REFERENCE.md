# PydanticAI Universal RAG API Reference

## 🎯 Quick Start

### **Basic Agent Usage**
```python
from agents import agent, create_azure_service_container

# Initialize
container = await create_azure_service_container()

# Simple query
result = await agent.run(
    "Find information about quantum computing applications in healthcare",
    deps=container
)

print(result.output)
```

### **Health Check**
```python
from agents import health_check

health = await health_check()
print(f"Status: {health['system_status']} ({health['overall_health_score']:.1f}%)")
```

---

## 🔍 Search Tools API

### **tri_modal_search**
Execute unified Vector + Graph + GNN search simultaneously.

```python
result = await tri_modal_search(
    ctx: RunContext[AzureServiceContainer],
    query: str,
    search_types: List[str] = ["vector", "graph", "gnn"],
    domain: Optional[str] = None,
    max_results: int = 10
) -> str
```

**Parameters:**
- `query`: Search query text
- `search_types`: Modalities to use (`["vector", "graph", "gnn"]`)
- `domain`: Optional domain context for optimization
- `max_results`: Maximum results per modality

**Example:**
```python
result = await tri_modal_search(
    container, 
    query="machine learning algorithms for medical diagnosis",
    search_types=["vector", "graph", "gnn"],
    domain="healthcare",
    max_results=15
)
```

**Response Format:**
```
Tri-modal search results (confidence: 0.87):

UNIFIED SEARCH RESULTS for: machine learning algorithms for medical diagnosis

🔍 SEMANTIC SIMILARITY (Vector):
- Deep learning models for medical imaging analysis
- Neural networks in diagnostic applications
- Similarity score: 0.89

🌐 RELATIONAL CONTEXT (Graph):
- Medical AI → Diagnostic Tools → Patient Outcomes
- Healthcare Systems → ML Integration → Clinical Workflows
- Connected entities: 12

🧠 PATTERN PREDICTIONS (GNN):
- Emerging trends in medical AI applications
- Predictive models for disease diagnosis
- Pattern confidence: 0.85

💡 SYNTHESIZED INSIGHTS:
Based on the tri-modal analysis, this query benefits from semantic similarity matching,
relational context understanding, and pattern-based predictions working together.

Modality Contributions:
{'vector_contribution': {'content_influence': 0.4, 'confidence': 0.89},
 'graph_contribution': {'content_influence': 0.3, 'confidence': 0.82},
 'gnn_contribution': {'content_influence': 0.3, 'confidence': 0.85}}

Execution Time: 1.23s
Performance Met: True
```

### **vector_search**
Execute semantic similarity search.

```python
result = await vector_search(
    ctx: RunContext[AzureServiceContainer],
    query: str,
    similarity_threshold: float = 0.7,
    max_results: int = 10
) -> str
```

**Example:**
```python
result = await vector_search(
    container,
    query="natural language processing techniques",
    similarity_threshold=0.75,
    max_results=20
)
```

### **graph_search**
Execute relationship discovery search.

```python
result = await graph_search(
    ctx: RunContext[AzureServiceContainer],
    query: str,
    max_depth: int = 3,
    relationship_types: List[str] = []
) -> str
```

**Example:**
```python
result = await graph_search(
    container,
    query="artificial intelligence applications",
    max_depth=4,
    relationship_types=["implements", "uses", "relates_to"]
)
```

---

## 🎯 Discovery Tools API

### **domain_detection**
Zero-configuration domain identification and analysis.

```python
result = await domain_detection(
    ctx: RunContext[AzureServiceContainer],
    query: str,
    additional_context: List[str] = [],
    adaptation_strategy: str = "balanced",
    enable_learning: bool = True
) -> str
```

**Parameters:**
- `query`: Text to analyze for domain detection
- `additional_context`: Additional context for analysis
- `adaptation_strategy`: Strategy (`"conservative"`, `"balanced"`, `"aggressive"`, `"learning"`)
- `enable_learning`: Enable continuous learning from query

**Example:**
```python
result = await domain_detection(
    container,
    query="Patient care protocols and treatment guidelines for diabetes management",
    additional_context=[
        "Electronic health records integration",
        "Clinical decision support systems"
    ],
    adaptation_strategy="aggressive",
    enable_learning=True
)
```

**Response Format:**
```
Domain Detection Results:

Detected Domain: healthcare
Confidence: 0.92 (high)
Detection Time: 45.2ms

Similar Domains:
- medical: 0.87
- clinical: 0.82
- pharmaceuticals: 0.71

Adaptation Recommendations:
- search_optimization: Enable medical terminology recognition
- performance_tuning: Optimize for healthcare data patterns
- accuracy_boost: Use domain-specific validation rules

Analysis Details:
Patterns Found: 8 healthcare-specific patterns detected
Correlation ID: 1a2b3c4d-5e6f-7890-abcd-ef1234567890
```

### **agent_adaptation**
Adapt agent configuration based on detected domain.

```python
result = await agent_adaptation(
    ctx: RunContext[AzureServiceContainer],
    detected_domain: str,
    domain_confidence: float,
    base_config: Dict[str, Any] = {},
    adaptation_goals: List[str] = ["optimize_performance", "improve_accuracy"]
) -> str
```

**Example:**
```python
result = await agent_adaptation(
    container,
    detected_domain="healthcare",
    domain_confidence=0.92,
    base_config={
        "max_response_time": 2.5,
        "min_confidence": 0.8,
        "search_types": ["vector", "graph", "gnn"]
    },
    adaptation_goals=["optimize_performance", "improve_accuracy", "enhance_domain_specificity"]
)
```

### **pattern_learning**
Learn patterns from text examples using advanced algorithms.

```python
result = await pattern_learning(
    ctx: RunContext[AzureServiceContainer],
    text_examples: List[str],
    learning_mode: str = "unsupervised",
    domain_context: Optional[str] = None
) -> str
```

**Learning Modes:**
- `"unsupervised"`: Discover patterns without labels
- `"supervised"`: Learn from labeled examples
- `"reinforcement"`: Learn from feedback and rewards

**Example:**
```python
result = await pattern_learning(
    container,
    text_examples=[
        "Patient presents with chest pain and shortness of breath",
        "Clinical examination reveals elevated cardiac markers",
        "Diagnostic imaging shows coronary artery stenosis",
        "Treatment plan includes cardiac catheterization"
    ],
    learning_mode="unsupervised",
    domain_context="healthcare"
)
```

---

## ⚡ Dynamic Tools API

### **dynamic_tool_generation**
Generate specialized tools based on query analysis and intent detection.

```python
result = await dynamic_tool_generation(
    ctx: RunContext[AzureServiceContainer],
    query: str,
    domain: Optional[str] = None,
    existing_tools: List[str] = [],
    performance_requirements: Dict[str, float] = {}
) -> str
```

**Example:**
```python
result = await dynamic_tool_generation(
    container,
    query="I need to analyze financial market trends and predict stock movements",
    domain="finance",
    existing_tools=["tri_modal_search", "pattern_learning"],
    performance_requirements={
        "max_response_time": 2.0,
        "min_accuracy": 0.85,
        "memory_limit": 256
    }
)
```

**Response Format:**
```
Dynamic Tool Generation Results:

Recommended Tools (3):
- financial_trend_analyzer
- stock_prediction_model
- market_sentiment_processor

Tool Configurations:
  ✓ financial_trend_analyzer: Generated for Financial market analysis
  ✓ stock_prediction_model: Generated for Stock movement prediction
  ○ market_sentiment_processor: General purpose

Intent Analysis:
- analyze (confidence: 0.91)
- predict (confidence: 0.87)
- financial_data_processing (confidence: 0.83)

Reasoning Analysis:
- Tools Needed: ['data_analysis', 'prediction', 'sentiment_analysis']
- Reasoning Depth: advanced

Generation Metrics:
- Confidence: 0.89
- Generation Time: 156.7ms
- Domain Optimized: True

Correlation ID: f8e7d6c5-4b3a-2910-8765-fedcba098765
```

### **tool_performance_analysis**
Analyze tool performance and provide optimization recommendations.

```python
result = await tool_performance_analysis(
    ctx: RunContext[AzureServiceContainer],
    tool_name: str,
    execution_history: List[Dict[str, Any]],
    optimization_goals: List[str] = ["improve_speed", "reduce_errors"]
) -> str
```

**Example:**
```python
execution_history = [
    {"execution_time": 1.23, "success": True, "confidence": 0.87},
    {"execution_time": 2.45, "success": True, "confidence": 0.91},
    {"execution_time": 0.89, "success": False, "error": "timeout"},
    {"execution_time": 1.67, "success": True, "confidence": 0.85}
]

result = await tool_performance_analysis(
    container,
    tool_name="tri_modal_search",
    execution_history=execution_history,
    optimization_goals=["improve_speed", "reduce_errors", "increase_reliability"]
)
```

---

## 📊 System Tools API

### **performance_metrics**
Get comprehensive system performance metrics.

```python
result = await performance_metrics(
    ctx: RunContext[AzureServiceContainer],
    include_cache_stats: bool = True,
    include_tool_performance: bool = True
) -> str
```

**Example:**
```python
result = await performance_metrics(container)
```

**Response Format:**
```
🚀 AGENT PERFORMANCE METRICS

⏱️ Generated at: 2024-01-15 14:30:25 UTC

📊 CACHE PERFORMANCE:
- Hit Rate: 78.5%
- Total Requests: 1,247
- Cache Hits: 979 (Hot: 456, Warm: 321, Cold: 202)
- Memory Usage: 67.3MB / 100.0MB (67.3%)
- Avg Response Time: 234.5ms
- Performance Score: 89.2/100

🔧 AZURE SERVICES:
- Overall Status: HEALTHY
- Ready for PydanticAI: True
- Services Healthy: 6
- Components Healthy: 4

🎯 PERFORMANCE SUMMARY:
- System Status: Optimal
- SLA Compliance: ✅ Met
- Memory Health: ✅ Good

💡 Use this data to monitor system performance and identify optimization opportunities.
```

### **error_monitoring**
Get comprehensive error monitoring and resilience statistics.

```python
result = await error_monitoring(
    ctx: RunContext[AzureServiceContainer],
    time_window_hours: int = 1,
    include_recovery_stats: bool = True
) -> str
```

**Example:**
```python
result = await error_monitoring(container, time_window_hours=24)
```

**Response Format:**
```
🛡️ ERROR MONITORING & RESILIENCE

⏱️ Analysis Window: Last 24 hour(s)
📊 Health Score: 94.3/100

📈 ERROR STATISTICS:
- Total Errors (All Time): 23
- Recent Errors: 3
- Error Rate: 3 errors/hour
- Recovery Success Rate: 91.3%

🏷️ ERROR CATEGORIES:
- Timeout: 2 (66.7%)
- Azure Service: 1 (33.3%)

⚠️ ERROR SEVERITIES:
- Medium: 2 (66.7%)
- Low: 1 (33.3%)

🔧 TOP ERROR OPERATIONS:
- tri_modal_search: 2 errors
- domain_detection: 1 errors

🔄 CIRCUIT BREAKER STATUS:
- tri_modal_search: 🟢 CLOSED
- azure_openai: 🟢 CLOSED

🎯 SYSTEM STATUS:
- Resilience: 🟢 Excellent
- Error Handling: ✅ Active
- Recovery Capability: ✅ Proven

💡 Monitor this dashboard regularly to maintain system health and reliability.
```

---

## 🔗 Tool Chaining API

### **execute_tool_chain**
Execute predefined tool chains for complex multi-step operations.

```python
result = await execute_tool_chain(
    ctx: RunContext[AzureServiceContainer],
    chain_id: str,
    query: str,
    custom_parameters: Dict[str, Any] = {}
) -> str
```

**Available Chains:**
- `"comprehensive_search"`: Domain detection → Agent adaptation → Tri-modal search
- `"performance_optimization"`: Performance metrics → Error monitoring → Optimization
- `"learning_workflow"`: Pattern learning → Dynamic tool generation → Performance analysis

**Example:**
```python
result = await execute_tool_chain(
    container,
    chain_id="comprehensive_search",
    query="Find latest research on renewable energy storage solutions",
    custom_parameters={
        "domain": "energy",
        "adaptation_strategy": "aggressive",
        "search_types": ["vector", "graph", "gnn"],
        "max_results": 20
    }
)
```

**Response Format:**
```
🔗 TOOL CHAIN EXECUTION RESULTS

Chain: Comprehensive Search
Query: "Find latest research on renewable energy storage solutions"

📊 EXECUTION SUMMARY:
- Status: ✅ SUCCESS
- Steps Executed: 3
- Steps Successful: 3
- Success Rate: 100.0%
- Total Time: 2.45s
- Efficiency Score: 92.3%

🔧 STEP RESULTS:
✅ domain_detection: Completed successfully
✅ agent_adaptation: Completed successfully
✅ tri_modal_search: Completed successfully

📈 PERFORMANCE METRICS:
- Average Step Time: 0.82s
- Error Count: 0
- Efficiency: 🟢 Excellent

✅ No errors encountered

💡 Tool chains enable complex multi-step operations with automatic error handling and performance optimization.
```

### **list_available_chains**
List all available tool chains with descriptions and statistics.

```python
result = await list_available_chains(
    ctx: RunContext[AzureServiceContainer],
    include_stats: bool = True
) -> str
```

---

## 🏥 Health Check API

### **health_check**
Comprehensive system health check.

```python
async def health_check() -> Dict[str, Any]
```

**Response Structure:**
```python
{
    "agent_status": "healthy",
    "agent_initialized": True,
    "pydantic_ai_version": "0.4.10",
    "azure_integration": {
        "overall_status": "healthy",
        "ready_for_pydantic_ai": True,
        "services": {...},
        "unique_components": {...}
    },
    "tool_availability": {
        "tri_modal_search": True,
        "vector_search": True,
        "graph_search": True,
        "domain_detection": True,
        "agent_adaptation": True,
        "pattern_learning": True,
        "dynamic_tool_generation": True,
        "tool_performance_analysis": True,
        "performance_metrics": True,
        "error_monitoring": True,
        "execute_tool_chain": True,
        "list_available_chains": True
    },
    "tools_ready": True,
    "performance_systems": {
        "caching": {
            "status": "healthy",
            "hit_rate": 78.5,
            "memory_usage": 67.3
        },
        "error_handling": {
            "status": "healthy",
            "total_errors": 23,
            "recovery_rate": 0.913
        },
        "tool_chaining": {
            "status": "healthy",
            "available_chains": 3,
            "total_executions": 45,
            "success_rate": 0.956
        }
    },
    "overall_health_score": 94.3,
    "system_status": "excellent",
    "timestamp": 1705334425.123
}
```

---

## 🔧 Configuration API

### **Azure Service Container**
Create and configure Azure services dependency injection.

```python
from agents import create_azure_service_container

container = await create_azure_service_container(config={
    "azure_openai_endpoint": "https://your-openai.openai.azure.com/",
    "azure_search_endpoint": "https://your-search.search.windows.net",
    "performance_targets": {
        "max_response_time": 3.0,
        "min_confidence": 0.7,
        "max_memory_usage": 500
    }
})
```

### **Performance Cache Configuration**
```python
from agents.base import get_performance_cache

cache = get_performance_cache()
cache.max_memory_mb = 200  # Increase cache size
cache.hot_ttl = 600        # Extend hot cache TTL to 10 minutes
```

### **Error Handler Configuration**  
```python
from agents.base import get_error_handler

error_handler = get_error_handler()
# Error handling is automatically configured but can be customized
```

---

## 📝 Request/Response Models

### **Common Response Patterns**

#### **Search Results**
All search tools return formatted text with:
- Query and parameters
- Results with confidence scores
- Execution time and performance metrics
- Modality contributions (for tri-modal search)

#### **Discovery Results**
Discovery tools return formatted text with:
- Detection/analysis results
- Confidence levels and scores
- Recommendations and insights
- Correlation IDs for tracking

#### **System Metrics**
Performance and monitoring tools return formatted text with:
- Current status and health scores
- Statistical data and trends
- Actionable recommendations
- Time-based analysis windows

### **Error Handling**
All tools implement comprehensive error handling:
- Automatic retries with exponential backoff
- Circuit breaker protection
- Graceful degradation with fallback options
- Detailed error reporting and recovery suggestions

---

## 🚀 Performance Optimization

### **Caching Strategy**
- **HOT Cache** (< 100ms): Frequent operations, 5-minute TTL
- **WARM Cache** (< 500ms): Domain-specific operations, 30-minute TTL
- **COLD Cache** (< 3s): Complex computations, 1-hour TTL

### **Batch Operations**
For multiple operations, use tool chains for optimal performance:

```python
# Instead of multiple individual calls
result1 = await domain_detection(container, query1)
result2 = await agent_adaptation(container, domain, confidence)  
result3 = await tri_modal_search(container, query1)

# Use tool chain for better performance
result = await execute_tool_chain(
    container,
    chain_id="comprehensive_search",
    query=query1,
    custom_parameters={"domain": domain}
)
```

### **Memory Management**
- Cache automatically manages memory usage
- Circuit breakers prevent resource exhaustion
- Configurable limits and thresholds

---

## 🐛 Error Codes & Troubleshooting

### **Common Error Patterns**

#### **Azure Service Errors**
```python
# Check service availability
health = await health_check()
if not health["azure_integration"]["ready_for_pydantic_ai"]:
    # Handle Azure service issues
    pass
```

#### **Timeout Errors**
```python
# Monitor performance metrics
metrics = await performance_metrics(container)
if "timeout" in metrics:
    # Optimize or increase timeout limits
    pass
```

#### **Cache Issues**
```python
# Clear cache if performance degrades
cache = get_performance_cache()
await cache.clear_expired()
```

### **Debug Mode**
Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger("agents").setLevel(logging.DEBUG)
```

---

*This API reference provides comprehensive documentation for all PydanticAI Universal RAG system endpoints. For additional examples and advanced usage patterns, refer to the System Guide and integration tests.*