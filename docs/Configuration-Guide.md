# 🔧 Configuration Guide: MaintIE Enhanced RAG

## Overview

The MaintIE Enhanced RAG system has been designed with extensive configurability to support different deployment environments, use cases, and performance requirements. This guide explains all configurable parameters and provides recommendations for different scenarios.

---

## 📋 Configuration Categories

### **1. Data Processing Settings**

These settings control how MaintIE data is processed and loaded:

| Parameter                | Default               | Description                                    | Environment Variable     |
| ------------------------ | --------------------- | ---------------------------------------------- | ------------------------ |
| `gold_data_filename`     | `gold_release.json`   | Filename for high-confidence annotations       | `GOLD_DATA_FILENAME`     |
| `silver_data_filename`   | `silver_release.json` | Filename for lower-confidence annotations      | `SILVER_DATA_FILENAME`   |
| `gold_confidence_base`   | `0.9`                 | Confidence threshold for gold data (0.0-1.0)   | `GOLD_CONFIDENCE_BASE`   |
| `silver_confidence_base` | `0.7`                 | Confidence threshold for silver data (0.0-1.0) | `SILVER_CONFIDENCE_BASE` |

**Usage Examples:**

```bash
# Use different dataset files
GOLD_DATA_FILENAME=maintie_gold_v2.json
SILVER_DATA_FILENAME=maintie_silver_v2.json

# Adjust confidence thresholds
GOLD_CONFIDENCE_BASE=0.95
SILVER_CONFIDENCE_BASE=0.75
```

### **2. Query Analysis Settings**

Control how queries are analyzed and enhanced:

| Parameter                 | Default | Description                          | Environment Variable      |
| ------------------------- | ------- | ------------------------------------ | ------------------------- |
| `max_related_entities`    | `15`    | Maximum related entities to find     | `MAX_RELATED_ENTITIES`    |
| `max_neighbors`           | `5`     | Maximum neighbors in knowledge graph | `MAX_NEIGHBORS`           |
| `concept_expansion_limit` | `10`    | Maximum concepts to expand           | `CONCEPT_EXPANSION_LIMIT` |

**Usage Examples:**

```bash
# More aggressive query expansion
MAX_RELATED_ENTITIES=25
CONCEPT_EXPANSION_LIMIT=15

# Conservative expansion for performance
MAX_RELATED_ENTITIES=10
CONCEPT_EXPANSION_LIMIT=5
```

### **3. Retrieval Settings**

Control vector search and retrieval behavior:

| Parameter              | Default       | Description                            | Environment Variable   |
| ---------------------- | ------------- | -------------------------------------- | ---------------------- |
| `embedding_batch_size` | `32`          | Batch size for embedding generation    | `EMBEDDING_BATCH_SIZE` |
| `faiss_index_type`     | `IndexFlatIP` | FAISS index type for similarity search | `FAISS_INDEX_TYPE`     |
| `similarity_threshold` | `0.7`         | Minimum similarity score (0.0-1.0)     | `SIMILARITY_THRESHOLD` |

**Usage Examples:**

```bash
# High-performance settings
EMBEDDING_BATCH_SIZE=64
FAISS_INDEX_TYPE=IndexIVFFlat

# Memory-constrained settings
EMBEDDING_BATCH_SIZE=16
FAISS_INDEX_TYPE=IndexFlatIP
```

### **4. Generation Settings**

Control LLM response generation:

| Parameter               | Default | Description                          | Environment Variable    |
| ----------------------- | ------- | ------------------------------------ | ----------------------- |
| `llm_top_p`             | `0.9`   | Nucleus sampling parameter (0.0-1.0) | `LLM_TOP_P`             |
| `llm_frequency_penalty` | `0.1`   | Frequency penalty (-2.0 to 2.0)      | `LLM_FREQUENCY_PENALTY` |
| `llm_presence_penalty`  | `0.1`   | Presence penalty (-2.0 to 2.0)       | `LLM_PRESENCE_PENALTY`  |

**Usage Examples:**

```bash
# More creative responses
LLM_TOP_P=0.95
LLM_FREQUENCY_PENALTY=0.2

# More focused responses
LLM_TOP_P=0.8
LLM_FREQUENCY_PENALTY=0.0
```

### **5. API Validation Settings**

Control API input validation:

| Parameter           | Default | Description            | Environment Variable |
| ------------------- | ------- | ---------------------- | -------------------- |
| `query_min_length`  | `3`     | Minimum query length   | `QUERY_MIN_LENGTH`   |
| `query_max_length`  | `500`   | Maximum query length   | `QUERY_MAX_LENGTH`   |
| `max_results_limit` | `50`    | Maximum search results | `MAX_RESULTS_LIMIT`  |

**Usage Examples:**

```bash
# Strict validation
QUERY_MIN_LENGTH=5
QUERY_MAX_LENGTH=200
MAX_RESULTS_LIMIT=20

# Relaxed validation for testing
QUERY_MIN_LENGTH=1
QUERY_MAX_LENGTH=1000
MAX_RESULTS_LIMIT=100
```

---

## 🎯 Environment-Specific Configurations

### **Development Environment**

```bash
# Development settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Relaxed limits for testing
QUERY_MIN_LENGTH=1
QUERY_MAX_LENGTH=1000
MAX_RESULTS_LIMIT=100

# Smaller batch sizes for faster iteration
EMBEDDING_BATCH_SIZE=16
MAX_RELATED_ENTITIES=10

# More verbose LLM responses
LLM_TOP_P=0.9
LLM_TEMPERATURE=0.5
```

### **Production Environment**

```bash
# Production settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Strict validation
QUERY_MIN_LENGTH=3
QUERY_MAX_LENGTH=500
MAX_RESULTS_LIMIT=50

# Optimized performance
EMBEDDING_BATCH_SIZE=64
MAX_RELATED_ENTITIES=15

# Balanced LLM settings
LLM_TOP_P=0.85
LLM_TEMPERATURE=0.3
```

### **High-Performance Environment**

```bash
# High-performance settings
EMBEDDING_BATCH_SIZE=128
MAX_RELATED_ENTITIES=25
CONCEPT_EXPANSION_LIMIT=20

# Aggressive retrieval
SIMILARITY_THRESHOLD=0.6
VECTOR_SEARCH_TOP_K=20

# Quality-focused generation
LLM_TOP_P=0.9
LLM_FREQUENCY_PENALTY=0.1
```

---

## 🔧 Domain Knowledge Configuration

### **Keyword Lists**

The system uses configurable keyword lists for query classification:

```bash
# Troubleshooting keywords (JSON array)
TROUBLESHOOTING_KEYWORDS=["failure","problem","issue","broken","not working","troubleshoot","diagnose","fix","repair","malfunction"]

# Procedural keywords (JSON array)
PROCEDURAL_KEYWORDS=["how to","procedure","steps","process","method","instructions","guide","manual","protocol"]

# Preventive keywords (JSON array)
PREVENTIVE_KEYWORDS=["preventive","maintenance schedule","inspection","service","routine","periodic","scheduled"]

# Safety keywords (JSON array)
SAFETY_KEYWORDS=["safety","hazard","risk","dangerous","caution","warning","lockout","ppe","procedure"]
```

### **Equipment Categories**

Configure equipment recognition patterns:

```python
# In config/advanced_settings.py
equipment_categories: Dict[str, List[str]] = Field(
    default={
        'rotating_equipment': ['pump', 'motor', 'compressor', 'turbine', 'fan'],
        'static_equipment': ['tank', 'vessel', 'pipe', 'valve'],
        'electrical': ['motor', 'generator', 'transformer', 'panel'],
        'hvac': ['fan', 'damper', 'coil', 'duct', 'filter'],
        'instrumentation': ['sensor', 'transmitter', 'gauge', 'indicator']
    }
)
```

### **Technical Abbreviations**

Configure abbreviation expansions:

```python
technical_abbreviations: Dict[str, str] = Field(
    default={
        'pm': 'preventive maintenance',
        'cm': 'corrective maintenance',
        'hvac': 'heating ventilation air conditioning',
        'loto': 'lockout tagout',
        'sop': 'standard operating procedure',
        'rca': 'root cause analysis'
    }
)
```

---

## 📊 Performance Tuning

### **Memory Optimization**

For memory-constrained environments:

```bash
# Reduce batch sizes
EMBEDDING_BATCH_SIZE=16
MAX_RELATED_ENTITIES=10
CONCEPT_EXPANSION_LIMIT=5

# Use simpler index type
FAISS_INDEX_TYPE=IndexFlatIP

# Limit results
MAX_RESULTS_LIMIT=20
VECTOR_SEARCH_TOP_K=5
```

### **Speed Optimization**

For high-throughput requirements:

```bash
# Increase batch sizes
EMBEDDING_BATCH_SIZE=64
MAX_RELATED_ENTITIES=20

# Use advanced index
FAISS_INDEX_TYPE=IndexIVFFlat

# Relax similarity threshold
SIMILARITY_THRESHOLD=0.6
```

### **Quality Optimization**

For high-quality responses:

```bash
# More comprehensive expansion
MAX_RELATED_ENTITIES=25
CONCEPT_EXPANSION_LIMIT=15

# Higher similarity threshold
SIMILARITY_THRESHOLD=0.8

# Quality-focused LLM settings
LLM_TOP_P=0.9
LLM_FREQUENCY_PENALTY=0.1
```

---

## 🔍 Configuration Validation

Use the built-in validation system:

```python
from config.validation import validate_configuration

# Validate current configuration
results = validate_configuration()

# Validate for specific environment
results = validate_configuration("production")

print(f"Configuration valid: {results['valid']}")
print(f"Warnings: {results['warnings']}")
print(f"Errors: {results['errors']}")
```

### **Common Validation Issues**

1. **Missing API Key**: Ensure `OPENAI_API_KEY` is set
2. **Invalid Ranges**: Check that confidence values are 0.0-1.0
3. **Performance Issues**: Monitor batch sizes and limits
4. **Security Concerns**: Validate query length limits

---

## 🚀 Deployment Examples

### **Docker Deployment**

```yaml
# docker-compose.yml
version: "3.8"
services:
  maintie-rag:
    build: .
    environment:
      - ENVIRONMENT=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - EMBEDDING_BATCH_SIZE=64
      - MAX_RELATED_ENTITIES=15
      - QUERY_MAX_LENGTH=500
      - MAX_RESULTS_LIMIT=50
    env_file:
      - .env
```

### **Kubernetes Deployment**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: maintie-rag
spec:
  template:
    spec:
      containers:
        - name: maintie-rag
          image: maintie-rag:latest
          env:
            - name: ENVIRONMENT
              value: "production"
            - name: EMBEDDING_BATCH_SIZE
              value: "64"
            - name: MAX_RELATED_ENTITIES
              value: "15"
          envFrom:
            - secretRef:
                name: maintie-rag-secrets
```

---

## 📝 Best Practices

### **1. Environment-Specific Files**

Create separate `.env` files for different environments:

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
EMBEDDING_BATCH_SIZE=16

# .env.production
ENVIRONMENT=production
DEBUG=false
EMBEDDING_BATCH_SIZE=64
```

### **2. Configuration Validation**

Always validate configuration before deployment:

```python
# In your startup script
from config.validation import validate_configuration

results = validate_configuration(settings.environment)
if not results['valid']:
    print("Configuration errors found:")
    for error in results['errors']:
        print(f"  - {error}")
    exit(1)
```

### **3. Monitoring Configuration**

Monitor configuration impact on performance:

```python
# Track configuration changes
import json
from datetime import datetime

config_snapshot = {
    "timestamp": datetime.now().isoformat(),
    "environment": settings.environment,
    "embedding_batch_size": advanced_settings.embedding_batch_size,
    "max_related_entities": advanced_settings.max_related_entities,
    # ... other key parameters
}

with open("config_snapshot.json", "w") as f:
    json.dump(config_snapshot, f, indent=2)
```

### **4. Gradual Rollouts**

Test configuration changes gradually:

1. **Development**: Test with relaxed limits
2. **Staging**: Test with production-like settings
3. **Production**: Monitor performance impact
4. **Rollback**: Keep previous configuration as backup

---

## 🔧 Troubleshooting

### **Common Issues**

1. **Memory Errors**: Reduce `EMBEDDING_BATCH_SIZE`
2. **Slow Performance**: Increase batch sizes, adjust similarity threshold
3. **Poor Quality**: Increase expansion limits, adjust LLM parameters
4. **API Errors**: Check validation limits and API key

### **Debugging Configuration**

```python
# Print current configuration
from config.advanced_settings import advanced_settings
from config.settings import settings

print("Current Configuration:")
print(f"Environment: {settings.environment}")
print(f"Batch Size: {advanced_settings.embedding_batch_size}")
print(f"Max Entities: {advanced_settings.max_related_entities}")
print(f"LLM Top P: {advanced_settings.llm_top_p}")
```

This comprehensive configuration system allows the MaintIE Enhanced RAG to be adapted to any deployment scenario while maintaining optimal performance and quality.
