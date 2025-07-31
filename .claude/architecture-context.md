# Architecture Context

## System Type
Universal RAG with Intelligent Agents - first system combining tri-modal search with data-driven agents

## Core Architecture Flow
```
Query Input → Intelligent Agent Reasoning → Tri-Modal Orchestration
├── Vector Search (Azure Cognitive Search) - Semantic similarity matching
├── Knowledge Graph Traversal (Cosmos DB Gremlin) - Multi-hop relationship reasoning  
├── Graph Neural Network Enhancement (Azure ML) - Predictive pattern analysis
└── Dynamic Tool Generation - Domain-specific tools discovered from data patterns
```

## Layer Boundaries (CRITICAL)
```
API Layer (FastAPI) → Services Layer → Core Integration → Infrastructure (Bicep)
```

### API Layer Responsibilities
- HTTP-specific concerns only
- Authentication/Authorization
- Input validation  
- Dependency injection (never create services)
- Error translation to HTTP responses

### Services Layer Responsibilities  
- Business logic and domain processing
- Azure service coordination
- Error handling and transformation
- Caching strategy implementation
- Performance optimization

### Core Layer Responsibilities
- Azure service integration only
- No business logic
- Infrastructure concerns
- Service client management

## Agent Intelligence Features

### Zero-Configuration Deployment
```python
# Medical Domain (from medical text data)
medical_agent = await UniversalAgent.create_from_domain("medical")
# Agent automatically learns: diagnose → test → prescribe → monitor workflows

# Engineering Domain (from engineering manuals)  
engineering_agent = await UniversalAgent.create_from_domain("engineering")
# Agent automatically learns: troubleshoot → inspect → repair → validate workflows
```

### Dynamic Tool Discovery
- Tools generated from action patterns found in domain text
- No hardcoded tool definitions
- Automatic effectiveness scoring and lifecycle management
- Example: Engineering domain discovers "troubleshoot_vibration", "inspect_bearing" from maintenance manuals

### Multi-Step Reasoning Chains
```python
# Example: Complex engineering problem solving
query = "Pump is making unusual noise and vibrating excessively"

agent_reasoning_chain = [
    "Vector search identifies similar cases: bearing failure, misalignment, cavitation",
    "Knowledge graph traversal: Pump → has_component → Bearing → exhibits → Vibration → indicates → Misalignment", 
    "Neural network predicts: 87% probability misalignment, 72% probability bearing wear",
    "Selected tools: inspect_alignment_tool, check_bearing_condition_tool, measure_vibration_tool",
    "Recommended workflow: 1) Check alignment, 2) Inspect bearings, 3) Measure vibration patterns"
]
```

### Continuous Learning
- Agents improve from successful problem resolutions
- Cross-domain pattern discovery
- Dynamic tool generation from learned patterns
- Performance metrics drive agent evolution

## Technology Stack

### Backend
- **Framework**: FastAPI + Python 3.11+ with async/await patterns
- **Processing**: Concurrent execution with asyncio.gather()
- **Services**: Clean architecture with dependency injection

### Frontend  
- **Framework**: React 19.1.0 + TypeScript
- **Streaming**: Server-Sent Events for real-time updates
- **UI Pattern**: Progressive disclosure with transparency layers

### Azure Services (9-service integration)
- **Search**: Azure Cognitive Search (vector indexing)
- **Graph**: Cosmos DB Gremlin (knowledge relationships) 
- **ML**: Azure ML (GNN training and inference)
- **Storage**: Blob Storage (document processing)
- **OpenAI**: Entity extraction and reasoning
- **Key Vault**: Secrets management
- **Monitor**: Observability and metrics
- **Functions**: Serverless processing
- **Container Registry**: Docker image management

## Data-Driven Architecture Flow
```
Raw Text Data → Dynamic Domain Discovery → Universal Knowledge Generation:
├── NLP Pipeline → Entity/Relationship Extraction → Learned Domain Patterns
├── Vector Pipeline → Semantic Embeddings → Cognitive Search Index  
├── Graph Pipeline → Knowledge Relationships → Cosmos DB Gremlin
└── GNN Pipeline → Pattern Learning → Azure ML Models
```

### Core Principle: Raw Data In → Intelligence Out
- **Input**: Only raw text documents for any domain
- **Process**: Automated entity extraction, relationship discovery, pattern learning, agent reasoning discovery
- **Output**: Domain-specific knowledge graphs, search indices, GNN models, and intelligent agents with dynamic tools
- **Result**: Zero-configuration universal RAG with intelligent agent reasoning for any knowledge domain

## Performance Architecture

### Response Time Targets
- **Simple Queries**: < 1 second (tri-modal search only)
- **Complex Agent Reasoning**: < 3 seconds (full reasoning chain)
- **Tool Discovery**: < 5 seconds (new tool generation)
- **Domain Discovery**: < 30 seconds (new domain learning)

### Scalability Targets
- **Concurrent Users**: 100+ with agent processing
- **Cache Hit Rate**: 60%+ with multi-level caching
- **System Availability**: 99.9% target
- **Domain Scalability**: Unlimited domains with zero configuration

## Critical Architectural Constraints

### Never Violate These Rules
- **Unified Search**: Don't create parallel search mechanisms - agents orchestrate, don't replace
- **Data-Driven Domains**: Don't hardcode any domain-specific logic, configurations, or agent behaviors  
- **Universal Architecture**: System must work with ANY raw text corpus without manual configuration
- **Dynamic Agent Intelligence**: Agent reasoning patterns and tools must be learned from data
- **Service Boundaries**: Don't bypass service layer abstractions
- **Performance Targets**: Don't compromise sub-3-second response times (including agent reasoning)
- **Azure Patterns**: Follow established authentication and connection patterns

## Request Flow Pattern (Correct)
```
Frontend → API Layer (HTTP validation) → Services Layer (business logic) → Core Layer (Azure integration) → Azure Services
```

### Anti-Pattern (Never Do)
```
Frontend → API Layer → Direct Azure Service Access  # Bypasses service layer
```

## Configuration Management
- **Environment Variables**: Infrastructure settings
- **Domain Patterns**: Learned from data (not hardcoded)
- **Azure Key Vault**: Sensitive configuration
- **Runtime Parameters**: Dynamic domain discovery settings

## Observability Requirements
- **Structured logging** for all operations with context
- **Performance metrics** for response times and throughput
- **Error tracking** with specific exception handling
- **Agent reasoning traces** for explainable AI
- **Tool effectiveness monitoring** for continuous improvement