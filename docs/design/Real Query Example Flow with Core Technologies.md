Based on the **real codebase implementation**, here's the complete **User Query Lifecycle with Core Technologies**:

## **Real Query Example Flow with Core Technologies:**

```
User Input: "pump bearing failure"
↓

React + TypeScript: handleSubmit() → setLoading(true) → axios.post()
                    useState hooks, React.FormEvent, TypeScript interfaces
↓

FastAPI + Pydantic: POST /api/v1/query/structured/ → process_structured_query()
                     uvicorn ASGI server, dependency injection, JSON validation
↓

RAG + Singleton: get_rag_system() → process_query_structured() → MaintIEStructuredRAG
                 Global instance pattern, async/await, exception handling
↓

AI Technologies Stack:

┌─ Domain Knowledge (JSON + Regex):
│  • domain_knowledge.json configuration
│  • re.sub() abbreviation expansion: "PM" → "preventive maintenance"
│  • Pattern matching: equipment_patterns, failure_patterns
│  • Entity extraction: ["pump", "bearing", "failure"]
│
├─ Knowledge Graph (NetworkX):
│  • nx.Graph() for entity relationships
│  • nx.shortest_path_length() for related concepts
│  • knowledge_graph.neighbors() traversal
│  • Graph-based concept expansion
│
├─ GNN Neural Intelligence (PyTorch + torch-geometric):
│  • torch.nn.Module (MaintenanceGNNModel)
│  • GraphSAGE/GCN/GAT layers (torch_geometric.nn)
│  • F.cosine_similarity() for entity expansion
│  • CUDA/CPU device detection: torch.cuda.is_available()
│  • torch.no_grad() inference mode
│
├─ Vector Intelligence (FAISS + Azure OpenAI):
│  • Azure OpenAI embeddings: text-embedding-ada-002
│  • AzureOpenAI.embeddings.create() API calls
│  • faiss.IndexFlatIP() semantic similarity index
│  • np.array() embedding processing
│  • faiss.normalize_L2() cosine optimization
│  • Batch processing: embedding_batch_size=32
│
├─ Fusion Scoring (NumPy + scikit-learn):
│  • Weighted combination: 0.7 * vector_score + 0.3 * graph_score
│  • np.dot() similarity calculations
│  • Top-k ranking algorithms
│  • Confidence scoring: 0.0-1.0 range
│
└─ LLM Generation (Azure OpenAI GPT-4):
   • AzureOpenAI.chat.completions.create()
   • GPT-4 deployment (gpt-4.1)
   • Temperature=0.3, max_tokens=500
   • Domain-specific prompt templates
   • Safety-aware response enhancement

↓

Response: "⚠️ SAFETY CRITICAL: Pump bearing failure requires immediate attention..."
          Enhanced with: confidence_score=0.94, processing_time=1.2s, sources=8
↓

Frontend React: setResponse() → UI update → Professional maintenance guidance displayed
                useState hook, conditional rendering, CSS styling, error boundaries
```

## **Technology Stack Breakdown by Phase:**

### **Phase 1: Frontend Technologies**
```
React 19.1.0 + TypeScript + Vite
├─ useState/useEffect hooks for state management
├─ axios.post<QueryResponse>() for HTTP client
├─ CSS styling with responsive design
└─ Error handling with try/catch + UI feedback
```

### **Phase 2: API Gateway Technologies**
```
FastAPI + uvicorn + Pydantic
├─ @asynccontextmanager for application lifespan
├─ Depends() dependency injection pattern
├─ HTTPException for error handling (503/500)
└─ JSON serialization with response_model validation
```

### **Phase 3: Domain Intelligence Technologies**
```
JSON Configuration + Regex Patterns
├─ domain_knowledge.json: equipment_hierarchy, maintenance_tasks
├─ re.compile() + re.sub() for pattern matching
├─ Configurable abbreviations: technical_abbreviations
└─ Safety critical equipment detection
```

### **Phase 4: Graph Intelligence Technologies**
```
NetworkX 3.2.0 + Graph Algorithms
├─ nx.Graph() for entity relationship modeling
├─ nx.shortest_path_length() for concept distance
├─ knowledge_graph.neighbors() for expansion
└─ Graph traversal with max_distance constraints
```

### **Phase 5: Neural Intelligence Technologies**
```
PyTorch 2.0.0 + torch-geometric 2.3.0
├─ MaintenanceGNNModel(nn.Module) architecture
├─ GraphSAGE/GCNConv/GATConv layers
├─ F.cosine_similarity() for entity similarity
├─ torch.save()/load() for model persistence
├─ global_mean_pool() for graph aggregation
└─ CUDA/CPU device optimization
```

### **Phase 6: Vector Intelligence Technologies**
```
FAISS 1.7.4 + Azure OpenAI + NumPy 1.23.5
├─ faiss.IndexFlatIP() for inner product similarity
├─ AzureOpenAI SDK 1.13.3 for embeddings
├─ text-embedding-ada-002 model (1536 dimensions)
├─ np.ascontiguousarray() + faiss.normalize_L2()
├─ Batch processing with embedding_batch_size=32
└─ Persistent storage: pickle.dump() for embeddings
```

### **Phase 7: Response Generation Technologies**
```
Azure OpenAI GPT-4 + Prompt Engineering
├─ AzureOpenAI.chat.completions.create() API
├─ Model: gpt-4.1 deployment with temperature=0.3
├─ Prompt templates with safety-aware context
├─ Citation integration with source validation
└─ Professional maintenance domain expertise
```

### **Phase 8: Production Technologies**
```
Logging + Monitoring + Performance
├─ Python logging with structured formats
├─ Processing time tracking: time.time()
├─ Health checks with component validation
├─ Error handling with graceful degradation
└─ Performance metrics: confidence, sources, timing
```

**Professional Technology Integration:**
- **Fallback Mechanisms**: GNN → NetworkX → Rule-based patterns
- **Configurable Intelligence**: JSON domain knowledge (no hardcoded rules)
- **Production Scaling**: Batch processing, caching, singleton patterns
- **Azure Ecosystem**: Full Azure OpenAI integration (embeddings + generation)
- **Modern Stack**: Latest versions with compatibility (NumPy pinned for FAISS)

The **entire stack is production-grade** with enterprise Azure services, scalable AI/ML libraries, and professional software engineering practices.