# 🤖 PydanticAI Agent Implementation Plan

**Document Type**: Implementation Execution Plan  
**Priority**: HIGH - Core Intelligent Agent System  
**Created**: 2025-07-31  
**Status**: 🔄 READY TO IMPLEMENT

This document outlines the complete implementation plan for integrating PydanticAI with our consolidated service architecture to create the intelligent agent system.

---

## 📊 Implementation Overview

### **🎯 Objective**
Complete the PydanticAI agent implementation leveraging our clean consolidated service architecture to create a production-ready intelligent agent system.

### **📈 Success Metrics**
- ✅ **Code Reduction**: Target 70%+ reduction in agent complexity
- ✅ **Performance**: Maintain <3s response time  
- ✅ **Integration**: Seamless with consolidated services
- ✅ **Functionality**: Full tri-modal search + intelligent reasoning
- ✅ **Architecture**: Clean layer boundaries maintained

### **🏗️ Current Architecture Foundation**
```
✅ COMPLETED - Clean Foundation:
- 6 consolidated services (down from 11)
- Zero layer boundary violations
- Proper dependency injection
- Clean naming conventions
- Infrastructure workflows in correct location
```

---

## 🚀 Implementation Phases

### **Phase 1: PydanticAI Core Integration (Day 1-2)**

#### **1.1 Install PydanticAI Dependencies**
- [ ] Add PydanticAI to requirements.txt
- [ ] Update pyproject.toml with agent dependencies
- [ ] Verify compatibility with existing Azure SDK versions

#### **1.2 Create Universal PydanticAI Agent**
- [ ] Implement `agents/universal_agent.py` using PydanticAI framework
- [ ] Integrate with ConsolidatedQueryService for search capabilities
- [ ] Integrate with ConsolidatedAgentService for coordination
- [ ] Connect to Azure OpenAI through infrastructure layer

#### **1.3 Agent Service Integration**
- [ ] Update ConsolidatedAgentService to work with PydanticAI
- [ ] Implement proper error handling and retry logic
- [ ] Add performance monitoring and metrics collection

### **Phase 2: Intelligent Tool System (Day 2-3)**

#### **2.1 Tri-Modal Search Tools**
- [ ] Convert existing tri-modal orchestrator to PydanticAI tools
- [ ] Implement vector search tool
- [ ] Implement graph search tool  
- [ ] Implement GNN-enhanced search tool
- [ ] Create unified search coordination tool

#### **2.2 Dynamic Tool Discovery**
- [ ] Implement tool discovery based on query analysis
- [ ] Create tool selection and chaining logic
- [ ] Implement domain-adaptive tool selection
- [ ] Add tool effectiveness learning

#### **2.3 Azure Integration Tools**
- [ ] Azure Cognitive Search integration tool
- [ ] Azure Cosmos DB graph traversal tool
- [ ] Azure OpenAI completion tool
- [ ] Azure ML GNN inference tool

### **Phase 3: API Integration (Day 3-4)**

#### **3.1 Update API Endpoints**
- [ ] Update query endpoints to use PydanticAI agent
- [ ] Implement streaming responses with agent reasoning
- [ ] Add agent health and status endpoints
- [ ] Update request/response models for agent data

#### **3.2 Dependency Injection Updates**
- [ ] Update DI container for PydanticAI agent
- [ ] Ensure proper service integration with agent
- [ ] Add agent lifecycle management
- [ ] Configure agent settings and parameters

#### **3.3 Error Handling and Monitoring**
- [ ] Implement agent-specific error handling
- [ ] Add agent performance monitoring
- [ ] Create agent reasoning trace logging
- [ ] Implement agent health checks

### **Phase 4: Testing and Validation (Day 4-5)**

#### **4.1 Unit Testing**
- [ ] Test PydanticAI agent initialization
- [ ] Test individual tool functionality
- [ ] Test service integration
- [ ] Test error handling scenarios

#### **4.2 Integration Testing**  
- [ ] Test end-to-end query processing
- [ ] Test tool chaining and selection
- [ ] Test Azure service integration
- [ ] Test performance under load

#### **4.3 Validation Testing**
- [ ] Validate <3s response time requirement
- [ ] Validate tri-modal search functionality
- [ ] Validate domain discovery capabilities
- [ ] Validate backward compatibility

---

## 📋 Detailed Implementation Tasks

### **Task 1: PydanticAI Agent Core**

**File**: `backend/agents/universal_agent.py`

```python
# Implementation structure:
from pydantic_ai import Agent
from config.inter_layer_contracts import AgentRequest, AgentResponse

class UniversalRAGAgent:
    """PydanticAI-powered universal RAG agent with tri-modal search"""
    
    def __init__(self, services_container):
        # Initialize with consolidated services
        self.query_service = services_container.query_service
        self.agent_service = services_container.agent_service
        self.cache_service = services_container.cache_service
        
        # Create PydanticAI agent with tools
        self.agent = Agent(
            'openai:gpt-4',
            tools=[
                self.tri_modal_search_tool,
                self.domain_discovery_tool,
                self.knowledge_extraction_tool
            ]
        )
    
    async def process_query(self, request: AgentRequest) -> AgentResponse:
        # Main agent processing logic
        pass
```

### **Task 2: Tool Implementation**

**File**: `backend/agents/tools/search_tools.py`

```python
# Tri-modal search tools using existing orchestrator
from pydantic_ai import tool
from agents.search.tri_modal_orchestrator import TriModalSearchOrchestrator

@tool
async def tri_modal_search(query: str, domain: str = "general") -> dict:
    """Execute tri-modal search (Vector + Graph + GNN)"""
    orchestrator = TriModalSearchOrchestrator()
    results = await orchestrator.execute_unified_search(query, domain)
    return results.dict()

@tool  
async def vector_search(query: str, max_results: int = 10) -> list:
    """Execute vector-only search for speed"""
    # Implementation using ConsolidatedQueryService
    pass
```

### **Task 3: Service Integration**

**Update File**: `backend/services/agent_service.py`

```python
# Add PydanticAI integration to ConsolidatedAgentService
class ConsolidatedAgentService(ServicesToAgentsInterface):
    def __init__(self):
        # Initialize PydanticAI agent
        from agents.universal_agent import UniversalRAGAgent
        self.pydantic_agent = UniversalRAGAgent(self)
    
    async def coordinate_agent_analysis(self, request: AgentRequest) -> OperationResult:
        # Use PydanticAI agent for analysis
        result = await self.pydantic_agent.process_query(request)
        return self._format_operation_result(result)
```

### **Task 4: API Integration**

**Update File**: `backend/api/endpoints/queries.py`

```python
# Update query endpoint to use PydanticAI agent
@router.post("/query/universal", response_model=AzureQueryResponse)
async def process_azure_query(
    request: AzureQueryRequest,
    agent_service: ConsolidatedAgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    # Convert to agent request
    agent_request = AgentRequest(
        operation_type="universal_query",
        query=request.query,
        domain=request.domain
    )
    
    # Process with PydanticAI agent
    result = await agent_service.coordinate_agent_analysis(agent_request)
    
    # Format response
    return format_query_response(result)
```

---

## 🔧 Technical Requirements

### **Dependencies to Add**
```toml
# pyproject.toml additions
[tool.poetry.dependencies]
pydantic-ai = "^0.0.14"
pydantic-ai-slim = "^0.0.14"
```

### **Environment Variables**
```bash
# .env additions
PYDANTIC_AI_MODEL=openai:gpt-4
PYDANTIC_AI_API_KEY=  # Will use Azure OpenAI
AGENT_MAX_TOOLS=10
AGENT_TIMEOUT=30
AGENT_CACHE_TTL=300
```

### **Configuration Updates**
```python
# config/settings.py additions
class AgentSettings(BaseSettings):
    pydantic_ai_model: str = "openai:gpt-4"
    agent_max_tools: int = 10
    agent_timeout: int = 30
    agent_cache_ttl: int = 300
```

---

## 🧪 Testing Strategy

### **Unit Test Files to Create**
- `tests/unit/test_universal_agent.py`
- `tests/unit/test_agent_tools.py`
- `tests/unit/test_agent_service_integration.py`

### **Integration Test Files to Update**
- `tests/integration/test_query_endpoints.py`
- `tests/integration/test_agent_azure_integration.py`

### **Performance Tests**
```python
# Performance benchmarks
async def test_agent_response_time():
    # Ensure <3s response time
    start_time = time.time()
    result = await agent.process_query(test_request)
    duration = time.time() - start_time
    assert duration < 3.0
```

---

## 🎯 Success Validation

### **Functional Validation**
- [ ] Agent responds to queries with structured output
- [ ] Tri-modal search tools execute correctly
- [ ] Domain discovery works across different domains
- [ ] Error handling gracefully manages failures
- [ ] Streaming responses work with agent reasoning

### **Performance Validation**  
- [ ] Response time <3s for complex queries
- [ ] Memory usage within acceptable limits
- [ ] Concurrent request handling (10+ simultaneous)
- [ ] Tool execution time optimization

### **Integration Validation**
- [ ] Consolidated services work with PydanticAI
- [ ] Azure service integration maintained
- [ ] Architecture compliance still passes
- [ ] Backward compatibility with existing APIs

---

## 📊 Implementation Timeline

### **Day 1: Foundation**
- ✅ Morning: Dependencies and core agent setup
- ✅ Afternoon: Basic PydanticAI agent implementation

### **Day 2: Tools and Integration** 
- ✅ Morning: Tri-modal search tools implementation
- ✅ Afternoon: Service integration and DI updates

### **Day 3: API and Endpoints**
- ✅ Morning: API endpoint updates
- ✅ Afternoon: Error handling and monitoring

### **Day 4: Testing**
- ✅ Morning: Unit and integration tests
- ✅ Afternoon: Performance testing and optimization

### **Day 5: Validation and Documentation**
- ✅ Morning: End-to-end validation
- ✅ Afternoon: Documentation updates and cleanup

---

## 🚨 Risk Mitigation

### **Technical Risks**
- **PydanticAI Compatibility**: Test with current Azure OpenAI setup
- **Performance Impact**: Monitor response times closely
- **Service Integration**: Ensure DI container works with agents

### **Mitigation Strategies**
- **Incremental Implementation**: Build and test piece by piece
- **Fallback Plan**: Keep existing agent code until migration complete
- **Performance Monitoring**: Real-time metrics during development

---

**Next Action**: Begin Phase 1 implementation with PydanticAI core integration.

**Ready to Execute**: ✅ All dependencies, architecture, and requirements defined.