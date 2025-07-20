# Azure Storage Accounts Guide

## 📋 Overview

Your Azure RAG system uses **two storage accounts** with different purposes for optimal performance and security:

### **🏗️ Storage Account Architecture**

| Storage Account | Purpose | Container | Data Type |
|----------------|---------|-----------|-----------|
| `maintiedevmlstor1cdd8e11` | **RAG Data & ML Models** | `universal-rag-data` | Documents, embeddings, search data |
| `maintiedevstor1cdd8e11` | **Application Data** | `app-data` | Logs, cache, runtime data |

---

## 🎯 **Storage Account Purposes**

### **1. RAG Data Storage (`maintiedevmlstor1cdd8e11`)**
**Purpose**: Store RAG system data and ML models

**Data Types**:
- 📄 **Documents**: Raw text files, PDFs, markdown files
- 🔍 **Embeddings**: Vector embeddings for semantic search
- 🗂️ **Search Indexes**: FAISS indexes and metadata
- 🤖 **ML Models**: Trained models, model artifacts
- 📊 **Knowledge Graphs**: Extracted entities and relations

**Container**: `universal-rag-data`

### **2. Application Storage (`maintiedevstor1cdd8e11`)**
**Purpose**: Store application runtime data

**Data Types**:
- 📝 **Application Logs**: System logs, error logs, audit trails
- 💾 **Cache Data**: Temporary data, session information
- 🔄 **Runtime Data**: Temporary files, processing artifacts
- 📈 **Metrics**: Performance metrics, usage statistics

**Container**: `app-data`

---

## ⚙️ **Configuration**

### **Environment Variables**

Update your `.env` file with the correct storage account names:

```bash
# RAG Data Storage (ML Storage Account)
AZURE_STORAGE_ACCOUNT=maintiedevmlstor1cdd8e11
AZURE_STORAGE_KEY=your-rag-storage-key
AZURE_BLOB_CONTAINER=universal-rag-data
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=maintiedevmlstor1cdd8e11;AccountKey=your-key;EndpointSuffix=core.windows.net

# ML Models Storage (Same account, different container)
AZURE_ML_STORAGE_ACCOUNT=maintiedevmlstor1cdd8e11
AZURE_ML_STORAGE_KEY=your-ml-storage-key
AZURE_ML_BLOB_CONTAINER=ml-models
AZURE_ML_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=maintiedevmlstor1cdd8e11;AccountKey=your-key;EndpointSuffix=core.windows.net

# Application Storage (General Storage Account)
AZURE_APP_STORAGE_ACCOUNT=maintiedevstor1cdd8e11
AZURE_APP_STORAGE_KEY=your-app-storage-key
AZURE_APP_BLOB_CONTAINER=app-data
AZURE_APP_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=maintiedevstor1cdd8e11;AccountKey=your-key;EndpointSuffix=core.windows.net
```

---

## 🚀 **Usage Examples**

### **Using Storage Factory**

```python
from backend.core.azure_storage.storage_factory import get_storage_factory

# Get storage factory
storage_factory = get_storage_factory()

# Upload RAG document
await storage_factory.upload_rag_document(
    local_path=Path("data/raw/document.pdf"),
    blob_name="documents/document.pdf"
)

# Upload ML model
await storage_factory.upload_ml_model(
    local_path=Path("models/gnn_model.pkl"),
    blob_name="models/gnn_model_v1.pkl"
)

# Upload application log
await storage_factory.upload_app_data(
    local_path=Path("logs/app.log"),
    blob_name="logs/2024/01/app_20240115.log"
)
```

### **Direct Client Access**

```python
from backend.core.azure_storage.storage_factory import (
    get_rag_storage_client,
    get_ml_storage_client,
    get_app_storage_client
)

# Get specific clients
rag_client = get_rag_storage_client()
ml_client = get_ml_storage_client()
app_client = get_app_storage_client()

# Use RAG client for documents
result = await rag_client.upload_file(
    local_path=Path("data/raw/sample.txt"),
    blob_name="documents/sample.txt"
)

# Use ML client for models
result = await ml_client.upload_file(
    local_path=Path("models/embedding_model.pkl"),
    blob_name="models/embedding_v1.pkl"
)

# Use app client for logs
result = await app_client.upload_file(
    local_path=Path("logs/error.log"),
    blob_name="logs/errors/error_20240115.log"
)
```

---

## 📊 **Storage Status Monitoring**

### **Check Storage Status**

```python
from backend.core.azure_storage.storage_factory import get_storage_factory

storage_factory = get_storage_factory()

# Get status of all storage clients
status = storage_factory.get_storage_status()
print("Storage Status:", status)

# List available clients
clients = storage_factory.list_available_clients()
print("Available Clients:", clients)
```

### **Expected Status Output**

```json
{
  "rag_data": {
    "initialized": true,
    "connection_status": "healthy",
    "account_name": "maintiedevmlstor1cdd8e11",
    "container_name": "universal-rag-data"
  },
  "ml_models": {
    "initialized": true,
    "connection_status": "healthy",
    "account_name": "maintiedevmlstor1cdd8e11",
    "container_name": "ml-models"
  },
  "app_data": {
    "initialized": true,
    "connection_status": "healthy",
    "account_name": "maintiedevstor1cdd8e11",
    "container_name": "app-data"
  }
}
```

---

## 🔧 **Best Practices**

### **1. Data Organization**

**RAG Data Storage**:
```
universal-rag-data/
├── documents/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── embeddings/
│   ├── vectors/
│   └── metadata/
├── search_indexes/
│   ├── faiss/
│   └── metadata/
└── knowledge_graphs/
    ├── entities/
    └── relations/
```

**ML Models Storage**:
```
ml-models/
├── gnn_models/
├── embedding_models/
├── classification_models/
└── model_metadata/
```

**Application Storage**:
```
app-data/
├── logs/
│   ├── application/
│   ├── errors/
│   └── audit/
├── cache/
│   ├── sessions/
│   └── temporary/
└── metrics/
    ├── performance/
    └── usage/
```

### **2. Security Considerations**

- **RAG Data**: Contains sensitive documents and embeddings
- **ML Models**: Contains trained models and artifacts
- **Application Data**: Contains logs and runtime data

### **3. Performance Optimization**

- **RAG Data**: High-performance storage for frequent access
- **ML Models**: Optimized for large file transfers
- **Application Data**: Standard storage for logs and cache

---

## 🚨 **Error Handling**

With silent fallbacks removed, you'll see real errors:

```python
# This will now show real errors instead of silent failures
try:
    await storage_factory.upload_rag_document(
        local_path=Path("nonexistent.pdf"),
        blob_name="documents/test.pdf"
    )
except RuntimeError as e:
    print(f"Real error: {e}")
    # Error: File upload failed for test.pdf: [Errno 2] No such file or directory
```

---

## 📈 **Monitoring and Alerts**

### **Storage Metrics to Monitor**

1. **RAG Data Storage**:
   - Document upload/download rates
   - Embedding storage usage
   - Search index performance

2. **ML Models Storage**:
   - Model upload/download times
   - Storage capacity usage
   - Model version management

3. **Application Storage**:
   - Log file sizes
   - Cache hit rates
   - Storage cleanup operations

### **Health Checks**

```python
# Check all storage clients
status = storage_factory.get_storage_status()

for client_type, client_status in status.items():
    if not client_status['initialized']:
        logger.error(f"Storage client {client_type} failed: {client_status['error']}")
    elif client_status['connection_status']['status'] != 'healthy':
        logger.warning(f"Storage client {client_type} unhealthy: {client_status['connection_status']}")
```

---

## 🔄 **Migration Guide**

If you need to migrate data between storage accounts:

```python
# Migrate from old single storage to new multi-storage setup
async def migrate_storage_data():
    # Migrate documents to RAG storage
    await storage_factory.upload_rag_document(
        local_path=Path("old_documents/"),
        blob_name="migrated/documents/"
    )

    # Migrate models to ML storage
    await storage_factory.upload_ml_model(
        local_path=Path("old_models/"),
        blob_name="migrated/models/"
    )

    # Migrate logs to app storage
    await storage_factory.upload_app_data(
        local_path=Path("old_logs/"),
        blob_name="migrated/logs/"
    )
```

---

## ✅ **Summary**

Your two storage accounts serve distinct purposes:

- **`maintiedevmlstor1cdd8e11`**: RAG data and ML models (high-performance)
- **`maintiedevstor1cdd8e11`**: Application data and logs (standard)

This separation provides:
- 🎯 **Better Performance**: Optimized storage for different data types
- 🔒 **Enhanced Security**: Isolated storage for sensitive vs. non-sensitive data
- 📊 **Improved Monitoring**: Separate metrics and alerts
- 🚀 **Scalability**: Independent scaling of different data types

The storage factory makes it easy to use the right storage account for each data type automatically!