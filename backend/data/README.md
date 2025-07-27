# Raw Data Preparation Guide

## 📁 Supported File Formats

The system supports the following file formats in the `data/raw/` directory:

- **`.md`** - Markdown files
- **`.txt`** - Plain text files
- **`.json`** - JSON files (structured data)

## 📋 Data Format Requirements

### **Option 1: Simple Text Records (Recommended)**

```
<id> your text record here
<id> another text record
<id> third text record
```

### **Option 2: Structured JSON**

```json
{
  "records": [
    { "id": "1", "text": "your text record here" },
    { "id": "2", "text": "another text record" },
    { "id": "3", "text": "third text record" }
  ]
}
```

### **Option 3: Plain Text with Delimiters**

```
record1|your text record here
record2|another text record
record3|third text record
```

## 🚀 Data Preparation Commands

### **1. Load and Process Raw Data**

```bash
cd azure-maintie-rag/backend
make data-prep
```

### **2. Extract Knowledge from Raw Data**

```bash
cd azure-maintie-rag/backend
make knowledge-extraction
```

### **3. Upload Data to Azure Services**

```bash
cd azure-maintie-rag/backend
make data-upload
```

## 📊 Data Processing Pipeline

### **Stage 1: Raw Data Loading**

- Reads all supported files from `data/raw/`
- Validates file formats and content
- Creates document objects with metadata

### **Stage 2: Knowledge Extraction**

- Uses Azure OpenAI GPT-4 for entity/relation extraction
- Discovers entities and relationships dynamically
- No predetermined schemas or hardcoded types

### **Stage 3: Azure Services Storage**

- **Azure Blob Storage**: Stores original documents
- **Azure Cognitive Search**: Creates vector embeddings
- **Azure Cosmos DB**: Builds knowledge graph

### **Stage 4: GNN Training**

- Prepares graph data for machine learning
- Trains Graph Neural Networks for enhanced retrieval
- Deploys models for production use

## 🔧 Configuration Options

### **Environment Variables**

```bash
# Data processing settings
SKIP_PROCESSING_IF_DATA_EXISTS=false
FORCE_DATA_REPROCESSING=false
RAW_DATA_INCLUDE_PATTERNS="*.md,*.txt,*.json"

# Azure service settings
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment
AZURE_SEARCH_SERVICE=your-search-service
AZURE_COSMOS_DB_ENDPOINT=your-cosmos-endpoint
```

### **Processing Thresholds**

```bash
# Minimum data requirements
MIN_ENTITIES_FOR_TRAINING=10
MIN_RELATIONS_FOR_TRAINING=5
GNN_TRAINING_TRIGGER_THRESHOLD=50
```

## 📈 Quality Guidelines

### **Text Quality Requirements**

- **Minimum length**: 10 characters per record
- **Maximum length**: 10,000 characters per record
- **Encoding**: UTF-8
- **Language**: Any (system is language-agnostic)

### **Content Guidelines**

- Include specific, meaningful information
- Avoid generic or repetitive content
- Maintain original terminology and context
- Include relevant technical details

### **Processing Recommendations**

- **Small datasets** (< 1,000 records): Process all data
- **Medium datasets** (1,000-10,000 records): Use sampling for discovery
- **Large datasets** (> 10,000 records): Use batch processing

## 🎯 Expected Output

After processing, your raw data becomes:

### **1. Structured Knowledge**

```json
{
  "entities": [
    {
      "entity_id": "entity_0",
      "text": "air conditioner",
      "entity_type": "equipment",
      "confidence": 0.9
    }
  ],
  "relations": [
    {
      "relation_id": "relation_0",
      "source_entity": "air conditioner",
      "target_entity": "thermostat",
      "relation_type": "has_component"
    }
  ]
}
```

### **2. Searchable Knowledge Base**

- Vector embeddings for semantic search
- Knowledge graph for relationship queries
- GNN models for enhanced retrieval

### **3. Intelligent RAG System**

- Can answer complex questions
- Provides contextual responses
- Maintains source citations

## 🔍 Troubleshooting

### **Common Issues**

**1. No raw data found**

```bash
# Check if files exist in data/raw/
ls -la azure-maintie-rag/backend/data/raw/
```

**2. Processing fails**

```bash
# Check Azure service configuration
cd azure-maintie-rag/backend
python -c "from config.settings import azure_settings; print(azure_settings.validate())"
```

**3. Low quality extraction**

```bash
# Use improved extraction
cd azure-maintie-rag/backend
python scripts/clean_knowledge_extraction.py
```

## 📚 Examples

### **Example 1: Maintenance Records**

```text
<id> air conditioner thermostat not working
<id> air receiver safety valves to be replaced
<id> analyse failed driveline component
```

### **Example 2: Technical Documentation**

```text
<id> The hydraulic system consists of a pump, reservoir, and control valves
<id> Regular maintenance includes checking fluid levels and filter replacement
<id> Common issues include leaks, pressure drops, and valve malfunctions
```

### **Example 3: Customer Support Data**

```text
<id> Customer reported login issues with error code 404
<id> System upgrade completed successfully at 2:30 PM
<id> Database connection timeout resolved by increasing pool size
```

## 🎯 Best Practices

1. **Start with clean, well-formatted data**
2. **Include diverse examples for better extraction**
3. **Use consistent terminology within your domain**
4. **Monitor extraction quality and adjust as needed**
5. **Test with sample queries to validate results**

## 📊 Performance Metrics

After processing, you can expect:

- **Entity extraction**: 80-95% accuracy
- **Relation extraction**: 70-85% accuracy
- **Processing speed**: 100-500 records/minute
- **Storage efficiency**: 10-50x compression ratio
