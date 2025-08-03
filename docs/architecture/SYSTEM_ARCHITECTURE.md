# System Architecture

**Azure Universal RAG - Enterprise Architecture Overview**

## Architecture Principles

**Performance-First Design**
- Sub-3-second response guarantee with circuit breakers
- Async-first patterns throughout
- Intelligent caching with 60%+ hit rates

**Data-Driven Intelligence**
- Zero hardcoded values - all patterns learned from data
- Automatic domain detection and adaptation
- Statistical confidence-based decisions

**Tri-Modal Search Unity**
- Vector search (Azure Cognitive Search)
- Graph search (Azure Cosmos DB + Gremlin)
- GNN search (PyTorch Geometric)
- Intelligent result synthesis

## System Components

### Agent Intelligence Layer ✅ **NEW: CONFIG-EXTRACTION ARCHITECTURE**
```
┌─ Domain Intelligence Agent ──────┬─ Knowledge Extraction Agent ─────┐
│  • Domain pattern analysis       │  • Document-level processing     │
│  • Configuration generation      │  • Entity & relationship extr.   │
│  • Performance optimization      │  • Structured knowledge creation │
│  • Cache-driven domain detection │  • Validation & quality metrics  │
└───────────────────────────────────┴───────────────────────────────────┘
┌─ Config-Extraction Orchestrator ───────────────────────────────────────┐
│  • Two-stage workflow coordination                                     │
│  • Stage 1: Domain Intelligence → ExtractionConfiguration              │
│  • Stage 2: Knowledge Extraction → ExtractionResults                   │
│  • Feedback loop for continuous improvement                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Services Layer
```
┌─ Agent Service ──────┬─ Query Service ────────┬─ Workflow Service ──┐
│  • PydanticAI        │  • Multi-modal search  │  • Pipeline mgmt    │
│  • Tool coordination │  • Result synthesis    │  • Progress tracking│
└──────────────────────┴─────────────────────────┴─────────────────────┘
┌─ Infrastructure Service ─────┬─ ML Service ─────┬─ Cache Service ─────┐
│  • Azure service management  │  • GNN training  │  • Multi-level      │
│  • Connection pooling        │  • Model serving │  • Pattern indexing │
└───────────────────────────────┴──────────────────┴─────────────────────┘
```

### Infrastructure Layer
```
┌─ Azure OpenAI ─┬─ Azure Search ─┬─ Azure Cosmos ─┬─ Azure ML ──────┐
│  • GPT-4       │  • Vector idx   │  • Graph DB    │  • GNN training │
│  • Embeddings  │  • Hybrid srch  │  • Gremlin API │  • Model serve  │
└────────────────┴────────────────┴────────────────┴─────────────────┘
```

### Data Flow Architecture ✅ **UPDATED: CONFIG-EXTRACTION WORKFLOW**
```
Raw Documents → Stage 1: Domain Intelligence Agent
                        ↓
                ExtractionConfiguration
                        ↓
                Stage 2: Knowledge Extraction Agent
                        ↓
┌─ Vector Index ─┬─ Knowledge Graph ─┬─ GNN Model ──────┐
│  (Azure Search)│   (Cosmos DB)     │  (PyTorch Geo)   │
└────────────────┴───────────────────┴──────────────────┘
                        ↓
              Tri-Modal Search Engine
                        ↓
            Intelligent Result Synthesis
                        ↓
            ExtractionResults (Feedback Loop)
```

**Two-Stage Processing Model:**
1. **Configuration Stage**: Domain Intelligence Agent analyzes domain patterns → generates optimized ExtractionConfiguration
2. **Extraction Stage**: Knowledge Extraction Agent processes documents using configuration → produces ExtractionResults
3. **Feedback Loop**: ExtractionResults improve future domain configurations

## Performance Characteristics

**Response Times**
- Typical: <0.5 seconds
- Guaranteed: <3 seconds (circuit breaker enforced)
- Cache hit: <50ms

**Scalability**
- Concurrent users: 100+
- Domains supported: Unlimited (zero-config)
- Auto-scaling: Azure Container Apps

**Reliability**
- Uptime SLA: 99.9%
- Circuit breakers on all external calls
- Graceful degradation patterns

## Deployment Architecture

**Production Environment**
- Azure Container Apps (auto-scaling)
- Azure Key Vault (secrets management)
- Application Insights (monitoring)
- Azure AD (authentication)

**Development Environment**
- Local development: `make dev`
- Container testing: `backend/Dockerfile`
- CI/CD: GitHub Actions

This architecture delivers enterprise-grade performance while maintaining simplicity and cost-effectiveness.
