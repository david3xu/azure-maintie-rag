# 📊 Pipeline Monitoring System

## Overview

The MaintIE Enhanced RAG pipeline now includes a comprehensive monitoring system that provides granular tracking of sub-steps, performance metrics, and detailed insights for production debugging and optimization.

## 🎯 Key Features

### Granular Sub-Step Tracking
- **Timing**: Precise duration tracking for each sub-step (millisecond precision)
- **Input/Output**: Track input sizes and output characteristics
- **Success/Failure**: Monitor success rates and error isolation
- **Custom Metrics**: Add domain-specific metrics per step

### API Call Monitoring
- **Call Counting**: Track total API calls per query
- **Duration Tracking**: Monitor individual API call performance
- **Success Rates**: Track API call success/failure rates
- **Cost Analysis**: Monitor API usage for cost optimization

### Cache Performance
- **Hit Rates**: Track cache effectiveness
- **Performance Impact**: Measure cache hit vs miss performance
- **Optimization Insights**: Identify caching opportunities

### File Persistence
- **JSON Metrics**: Save detailed metrics to JSON files
- **Timestamped**: Each query gets a unique timestamped file
- **Query ID**: Track individual queries with UUID
- **Historical Analysis**: Build performance trends over time

## 🏗️ Architecture

### Core Components

```python
# Main monitoring class
PipelineMonitor
├── start_query()           # Initialize monitoring for a query
├── track_sub_step()        # Context manager for sub-step tracking
├── add_custom_metric()     # Add domain-specific metrics
├── track_api_call()        # Track individual API calls
├── track_cache_hit()       # Mark cache hits
└── end_query()            # Finalize metrics and save

# Data structures
SubStepMetrics             # Individual step metrics
PipelineMetrics           # Complete query metrics
```

### Integration Points

The monitoring system is integrated into key pipeline components:

1. **Query Analyzer** (`src/enhancement/query_analyzer.py`)
   - Query normalization timing
   - Entity extraction metrics
   - Query classification performance
   - Intent detection accuracy

2. **Vector Search** (`src/retrieval/vector_search.py`)
   - Embedding generation timing
   - FAISS search performance
   - Result assembly metrics

3. **Pipeline Orchestration** (`src/pipeline/rag_structured.py`)
   - Overall query processing
   - Component initialization
   - Response generation
   - Caching performance

## 📊 Usage Examples

### Basic Monitoring

```python
from src.monitoring.pipeline_monitor import get_monitor

# Start monitoring a query
monitor = get_monitor()
query_id = monitor.start_query("pump bearing failure", "structured")

# Track sub-steps
with monitor.track_sub_step("Query Analysis", "MaintenanceQueryAnalyzer", query):
    analysis = analyzer.analyze_query(query)
    monitor.add_custom_metric("Query Analysis", "entities_count", len(analysis.entities))

# End monitoring
metrics = monitor.end_query(confidence_score=0.85, sources_count=5)
```

### Custom Metrics

```python
# Add domain-specific metrics
monitor.add_custom_metric("Vector Search", "embedding_dimensions", 1536)
monitor.add_custom_metric("Entity Extraction", "entities_list", ["pump", "bearing", "failure"])
monitor.track_api_call("Embedding Generation", "Azure OpenAI", 250.5, True)
```

### Performance Analysis

```python
# Get performance summary
summary = monitor.get_performance_summary()
print(f"Total steps: {summary['summary']['total_steps']}")
print(f"API calls: {summary['summary']['total_api_calls']}")
print(f"Cache hits: {summary['summary']['cache_hits']}")

# Step-wise performance
for step_name, metrics in summary['step_performance'].items():
    print(f"{step_name}: {metrics['avg_duration_ms']:.1f}ms avg")
```

## 📁 Output Files

### Metrics Directory Structure
```
data/metrics/
├── pipeline_metrics_20241201_143022_abc12345.json
├── pipeline_metrics_20241201_143045_def67890.json
└── ...
```

### Sample Metrics File
```json
{
  "query_id": "abc12345-def6-7890-ghij-klmnopqrstuv",
  "query": "pump bearing failure analysis",
  "pipeline_type": "structured",
  "total_duration_ms": 3450.2,
  "total_steps": 12,
  "successful_steps": 12,
  "failed_steps": 0,
  "total_api_calls": 3,
  "cache_hits": 0,
  "confidence_score": 0.85,
  "sources_count": 5,
  "safety_warnings_count": 2,
  "sub_steps": [
    {
      "step_name": "Query Analysis",
      "component": "MaintenanceQueryAnalyzer",
      "duration_ms": 45.2,
      "success": true,
      "api_calls": 0,
      "custom_metrics": {
        "entities_count": 3,
        "entities_list": ["pump", "bearing", "failure"]
      }
    }
  ]
}
```

## 🔍 Debugging with Monitoring

### Performance Bottlenecks

```bash
# Run monitoring debug script
make debug-monitoring

# Check metrics files
ls -la data/metrics/
cat data/metrics/pipeline_metrics_*.json | jq '.total_duration_ms' | sort -n
```

### Error Isolation

```python
# Check failed steps
metrics = monitor.get_performance_summary()
failed_steps = [step for step in metrics['sub_steps'] if not step['success']]
for step in failed_steps:
    print(f"Failed: {step['step_name']} - {step['error_message']}")
```

### API Usage Analysis

```python
# Analyze API call patterns
api_calls = []
for step in metrics['sub_steps']:
    if step['api_calls'] > 0:
        api_calls.append({
            'step': step['step_name'],
            'calls': step['api_calls'],
            'duration': step['duration_ms']
        })
```

## 🚀 Production Benefits

### 1. Performance Optimization
- **Bottleneck Identification**: Find slowest sub-steps
- **Resource Usage**: Monitor API calls and costs
- **Cache Effectiveness**: Optimize caching strategies

### 2. Error Debugging
- **Error Isolation**: Pinpoint exact failure points
- **Error Patterns**: Identify recurring issues
- **Recovery Strategies**: Implement targeted fixes

### 3. Quality Assurance
- **Confidence Tracking**: Monitor response quality
- **Source Validation**: Track source count and relevance
- **Safety Monitoring**: Ensure safety warnings are generated

### 4. Cost Management
- **API Usage**: Track and optimize API calls
- **Resource Efficiency**: Monitor processing times
- **Scaling Insights**: Plan capacity requirements

## 🔧 Configuration

### Enable/Disable Monitoring

```python
# In your pipeline initialization
monitor = PipelineMonitor(
    enable_detailed_logging=True,  # Console output
    save_metrics=True              # File persistence
)
```

### Custom Metrics Directory

```python
# Default: data/metrics/
# Custom: Set in monitor initialization
monitor.metrics_dir = Path("/custom/metrics/path")
```

### Logging Levels

```python
# Adjust logging verbosity
import logging
logging.getLogger('src.monitoring').setLevel(logging.INFO)
```

## 📈 Future Enhancements

### Planned Features
1. **Real-time Dashboard**: Web-based metrics visualization
2. **Alerting System**: Performance threshold alerts
3. **Trend Analysis**: Historical performance trends
4. **Cost Tracking**: Detailed API cost analysis
5. **Integration**: Prometheus/Grafana integration

### Extensibility
- Add custom metric types
- Implement custom exporters
- Create domain-specific dashboards
- Build automated optimization systems

## 🧪 Testing

### Run Monitoring Tests
```bash
# Test monitoring system
make debug-monitoring

# Test with real queries
python debug/debug_monitoring.py

# Check generated metrics
ls -la data/metrics/
```

### Validation
- Verify metrics file generation
- Check timing accuracy
- Validate custom metrics
- Test error handling

---

**Note**: The monitoring system is designed to be lightweight and non-intrusive. It adds minimal overhead while providing comprehensive insights for production debugging and optimization.