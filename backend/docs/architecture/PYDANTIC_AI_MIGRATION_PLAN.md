# PydanticAI Migration Plan for Existing Agent Code

## Executive Summary

**Purpose**: Migrate existing custom agent implementation to PydanticAI framework  
**Current State**: 23 files, ~4800 lines of custom agent code  
**Target State**: 5 files, ~1400 lines with PydanticAI foundation  
**Code Reduction**: 71% reduction while preserving all unique capabilities  
**Timeline**: 5 days (Phase 2 Week 5)

---

## 🔍 **Current Code Analysis**

### **Existing Agent Structure**
```
backend/agents/
├── base/ (10 files, ~2000 lines)
│   ├── agent_interface.py
│   ├── agent_service_interface.py  
│   ├── reasoning_engine.py + optimized_reasoning_engine.py (DUPLICATE)
│   ├── memory_manager.py + integrated_memory_manager.py (DUPLICATE)
│   ├── context_manager.py
│   ├── react_engine.py
│   ├── plan_execute_engine.py
│   └── temporal_pattern_tracker.py
├── discovery/ (7 files, ~1500 lines)  
│   ├── pattern_learning_system.py
│   ├── domain_pattern_engine.py
│   ├── dynamic_pattern_extractor.py
│   ├── zero_config_adapter.py
│   └── domain_context_enhancer.py
├── search/ (2 files, ~500 lines)
│   └── tri_modal_orchestrator.py
└── universal_agent_service.py (~800 lines)

Total: 23 files, ~4800 lines
```

### **Key Existing Capabilities to Preserve**
1. **✅ Tri-Modal Search Orchestration** - `tri_modal_orchestrator.py`
2. **✅ Domain Discovery System** - `pattern_learning_system.py`, `zero_config_adapter.py`  
3. **✅ Dynamic Pattern Extraction** - `dynamic_pattern_extractor.py`
4. **✅ Intelligence Analysis** - `universal_agent_service.py`
5. **✅ Memory Management** - `integrated_memory_manager.py`
6. **✅ Context Management** - `context_manager.py`

---

## 🎯 **Migration Strategy**

### **Phase 1: Core Framework Migration (Days 1-2)**

#### **Step 1: PydanticAI Agent Foundation**
```python
# NEW: /agents/universal_agent.py (200 lines)
from pydantic_ai import Agent, RunContext
from typing import List, Dict, Any, Optional
from ..core.azure_services import AzureServiceContainer

# Create main PydanticAI agent with Azure integration
agent = Agent(
    'azure:gpt-4o',  # Use existing Azure OpenAI endpoint
    deps_type=AzureServiceContainer,
    system_prompt="""
    You are an intelligent Universal RAG system with tri-modal search capabilities.
    You can search using Vector, Graph, and GNN modalities simultaneously.
    You can discover new domains automatically and generate tools dynamically.
    You maintain high performance with sub-3-second response times.
    """,
    retries=2,
    result_type=str  # Can be enhanced later with structured outputs
)
```

#### **Step 2: Tri-Modal Search Tool Migration**
```python
# NEW: /agents/tools/search_tools.py (300 lines)
from ..search.tri_modal_orchestrator import TriModalOrchestrator
from pydantic_ai import RunContext
from pydantic import BaseModel

class TriModalSearchRequest(BaseModel):
    query: str
    search_types: List[str] = ["vector", "graph", "gnn"]
    max_results: int = 10
    domain: Optional[str] = None

@agent.tool
async def execute_tri_modal_search(
    ctx: RunContext[AzureServiceContainer],
    request: TriModalSearchRequest
) -> Dict[str, Any]:
    """Execute our proprietary tri-modal search (Vector + Graph + GNN)"""
    
    # Use existing tri-modal orchestrator implementation
    orchestrator = ctx.deps.tri_modal_orchestrator
    
    search_result = await orchestrator.execute_unified_search(
        query=request.query,
        search_types=request.search_types,
        max_results=request.max_results,
        domain=request.domain
    )
    
    return {
        "results": search_result.content,
        "confidence": search_result.confidence,
        "modality_contributions": search_result.modality_contributions,
        "execution_time": search_result.execution_time,
        "metadata": search_result.metadata
    }
```

#### **Step 3: Domain Discovery Tool Migration**
```python
# NEW: /agents/tools/discovery_tools.py (400 lines)
from ..discovery.zero_config_adapter import ZeroConfigAdapter
from ..discovery.pattern_learning_system import PatternLearningSystem

class DomainDiscoveryRequest(BaseModel):
    text_corpus: List[str]
    domain_hint: Optional[str] = None
    adaptation_strategy: str = "balanced"
    learning_enabled: bool = True

@agent.tool
async def discover_domain_patterns(
    ctx: RunContext[AzureServiceContainer],
    request: DomainDiscoveryRequest
) -> Dict[str, Any]:
    """Discover domain patterns using our zero-config adaptation system"""
    
    # Use existing zero-config adapter
    adapter = ctx.deps.zero_config_adapter
    
    adaptation_result = await adapter.adapt_agent_to_domain(
        raw_text_data=request.text_corpus,
        domain_name=request.domain_hint,
        adaptation_strategy=request.adaptation_strategy,
        learning_enabled=request.learning_enabled
    )
    
    return {
        "discovered_domain": adaptation_result.get('discovered_domain'),
        "domain_patterns": adaptation_result.get('patterns', []),
        "confidence": adaptation_result.get('confidence', 0.8),
        "learned_features": adaptation_result.get('learned_features', {}),
        "adaptation_time": adaptation_result.get('execution_time', 0.0)
    }
```

### **Phase 2: Dynamic Tool Discovery (Days 3-4)**

#### **Step 4: PydanticAI Dynamic Tools Implementation**
```python
# NEW: /agents/tools/dynamic_tools.py (300 lines)
from pydantic_ai.tools import ToolDefinition
from typing import Union, List
import json

async def prepare_dynamic_tools(
    ctx: RunContext[AzureServiceContainer], 
    tool_defs: List[ToolDefinition]
) -> Union[List[ToolDefinition], None]:
    """Dynamically discover and add domain-specific tools"""
    
    # Extract current query context  
    current_query = getattr(ctx, 'current_query', '')
    
    if not current_query:
        return tool_defs
    
    # Use existing pattern extractor to analyze query for tool opportunities
    pattern_extractor = ctx.deps.dynamic_pattern_extractor
    
    tool_patterns = await pattern_extractor.extract_tool_patterns(
        query=current_query,
        context=getattr(ctx, 'conversation_context', {})
    )
    
    # Generate dynamic tools from discovered patterns
    new_tools = []
    for pattern in tool_patterns:
        if pattern.automation_potential > 0.7:
            dynamic_tool = await _generate_tool_from_pattern(pattern, ctx.deps)
            if dynamic_tool:
                new_tools.append(dynamic_tool)
    
    # Add discovered tools to existing tool list
    tool_defs.extend(new_tools)
    
    return tool_defs

# Register dynamic tool preparation with agent
agent = Agent(
    'azure:gpt-4o',
    deps_type=AzureServiceContainer,
    prepare_tools=prepare_dynamic_tools,
    system_prompt="..."  # Same as before
)
```

#### **Step 5: Tool Effectiveness Scoring**
```python
async def _generate_tool_from_pattern(pattern, azure_services) -> Optional[ToolDefinition]:
    """Generate executable tool from discovered pattern"""
    
    # Use existing pattern analysis to create tool specification
    tool_spec = {
        "name": f"dynamic_{pattern.action_type}_{pattern.target_domain}",
        "description": f"Execute {pattern.action_type} operation for {pattern.target_domain}",
        "parameters": pattern.extracted_parameters,
        "implementation": pattern.suggested_implementation
    }
    
    # Create PydanticAI tool definition
    async def dynamic_tool_execution(**kwargs):
        """Dynamically generated tool execution"""
        # Use existing Azure services for actual execution
        return await azure_services.execute_dynamic_operation(
            operation=pattern.action_type,
            parameters=kwargs,
            context=pattern.context
        )
    
    return ToolDefinition(
        name=tool_spec["name"],
        description=tool_spec["description"],
        function=dynamic_tool_execution,
        parameters_json_schema=pattern.parameter_schema
    )
```

### **Phase 3: Service Integration (Day 5)**

#### **Step 6: Simplified Agent Service**
```python
# NEW: /agents/services/agent_service.py (200 lines)
from ..universal_agent import agent
from ...core.azure_services import AzureServiceContainer

class PydanticAIAgentService:
    """Simplified agent service using PydanticAI foundation"""
    
    def __init__(self, azure_services: AzureServiceContainer):
        self.azure_services = azure_services
        self.agent = agent
    
    async def process_intelligent_query(self, request: QueryRequest) -> QueryResponse:
        """Main query processing using PydanticAI agent"""
        
        # Execute query through PydanticAI agent
        result = await self.agent.run_async(
            request.query,
            deps=self.azure_services,
            message_history=request.conversation_history
        )
        
        # Extract performance metrics and tool usage
        tools_used = self._extract_tools_used(result.all_messages())
        reasoning_chain = self._extract_reasoning_chain(result.all_messages())
        
        return QueryResponse(
            answer=result.output,
            sources=self._extract_sources(result),
            reasoning_chain=reasoning_chain,
            tools_used=tools_used,
            confidence=self._calculate_confidence(result),
            execution_time=self._calculate_execution_time(result),
            performance_metrics={
                "response_time": result.usage.total_tokens if result.usage else 0,
                "tools_invoked": len(tools_used),
                "reasoning_steps": len(reasoning_chain)
            }
        )
    
    async def adapt_to_domain(self, domain_request: DomainAdaptationRequest) -> DomainAdaptationResult:
        """Domain adaptation using PydanticAI tools"""
        
        # Use discover_domain_patterns tool through agent
        discovery_query = f"Analyze and adapt to domain from these documents: {domain_request.domain_name}"
        
        result = await self.agent.run_async(
            discovery_query,
            deps=self.azure_services
        )
        
        # Extract domain adaptation results from agent response
        return self._parse_domain_adaptation_result(result, domain_request)
```

---

## 📊 **Migration Benefits Analysis**

### **Code Complexity Reduction**

| Component | Before (Custom) | After (PydanticAI) | Reduction |
|-----------|-----------------|-------------------|-----------|
| **Agent Interface** | 10 files, ~2000 lines | 1 file, ~200 lines | 90% |
| **Tool System** | Built from scratch | PydanticAI built-in | 100% framework code eliminated |
| **Validation** | Custom parameter validation | Pydantic built-in | 100% |
| **Error Handling** | Custom retry logic | PydanticAI built-in | 80% |
| **Multi-modal Support** | Custom implementation | PydanticAI native | 70% |
| **Tool Discovery** | 7 files, ~1500 lines | 1 file, ~300 lines | 80% |

**Overall: 71% code reduction (4800 → 1400 lines)**

### **Capability Preservation**

| Unique Capability | Migration Strategy | Preservation Level |
|-------------------|-------------------|-------------------|
| **Tri-Modal Search** | Convert to PydanticAI tool | 100% - Full functionality preserved |
| **Domain Discovery** | Convert to PydanticAI tool | 100% - Zero-config adaptation maintained |
| **Dynamic Patterns** | Use PydanticAI dynamic tools | 95% - Enhanced with PydanticAI features |
| **Azure Integration** | Dependency injection through RunContext | 100% - All Azure services accessible |
| **Performance** | PydanticAI optimizations + caching | 105% - Likely performance improvement |

### **New Capabilities Gained**

| PydanticAI Feature | Benefit | Impact |
|-------------------|---------|---------|
| **Built-in Validation** | Automatic parameter validation | Improved reliability |
| **Structured Outputs** | Type-safe responses | Better integration |
| **Multi-modal Native** | Images, documents, audio support | Enhanced capabilities |
| **Tool Marketplace** | LangChain, MCP integration | Ecosystem access |
| **Testing Framework** | Built-in testing support | Better quality assurance |

---

## 🚧 **Migration Risks & Mitigation**

### **High Risk: Functionality Loss**
- **Risk**: Complex custom logic lost in migration
- **Mitigation**: Thorough mapping of all existing capabilities
- **Validation**: Comprehensive test suite execution before/after

### **Medium Risk: Performance Regression**  
- **Risk**: PydanticAI overhead impacts <3s requirement
- **Mitigation**: Performance benchmarking at each step
- **Fallback**: Optimize tool execution and add caching

### **Low Risk: Azure Integration Issues**
- **Risk**: PydanticAI conflicts with Azure services
- **Mitigation**: Use dependency injection pattern with RunContext
- **Validation**: Test all Azure service integrations

---

## 🧪 **Migration Validation Plan**

### **Functional Validation**
```python
# Test existing capabilities work with PydanticAI
async def test_tri_modal_search_migration():
    """Validate tri-modal search functionality preserved"""
    
    # Test with same query as before migration
    test_query = "Find information about machine learning algorithms"
    
    # PydanticAI version
    result_new = await pydantic_agent.run_async(test_query, deps=azure_services)
    
    # Validate results match previous implementation expectations
    assert result_new.output is not None
    assert len(result_new.output) > 0
    # Add specific tri-modal validation logic
```

### **Performance Validation**
```python
async def test_performance_requirements():
    """Validate <3s response time maintained"""
    
    start_time = time.time()
    result = await pydantic_agent.run_async("Complex query", deps=azure_services)
    execution_time = time.time() - start_time
    
    assert execution_time < 3.0, f"Response time {execution_time}s exceeds 3s requirement"
```

### **Integration Validation**
```python
async def test_azure_services_integration():
    """Validate all Azure services accessible through PydanticAI"""
    
    # Test each Azure service through tools
    services_to_test = [
        'azure_openai', 'cosmos_db', 'cognitive_search', 
        'blob_storage', 'application_insights'
    ]
    
    for service in services_to_test:
        result = await test_azure_service_access(service)
        assert result.success, f"Azure service {service} not accessible"
```

---

## 📋 **Implementation Checklist**

### **Day 1: Foundation Setup**
- [ ] Install PydanticAI and configure with Azure OpenAI
- [ ] Create basic agent structure (`universal_agent.py`)
- [ ] Set up dependency injection with `AzureServiceContainer`
- [ ] Test basic agent functionality

### **Day 2: Core Tool Migration**
- [ ] Migrate tri-modal search to PydanticAI tool (`search_tools.py`)
- [ ] Migrate domain discovery to PydanticAI tool (`discovery_tools.py`)
- [ ] Test tool execution and parameter validation
- [ ] Validate Azure service integration

### **Day 3: Dynamic Tool Discovery**
- [ ] Implement PydanticAI dynamic tools (`dynamic_tools.py`)
- [ ] Create tool generation from pattern analysis
- [ ] Add tool effectiveness scoring
- [ ] Test dynamic tool discovery workflow

### **Day 4: Advanced Features**
- [ ] Add caching and performance optimization
- [ ] Implement comprehensive error handling
- [ ] Add tool composition and chaining
- [ ] Create monitoring and metrics collection

### **Day 5: Integration & Validation**
- [ ] Create simplified agent service wrapper
- [ ] Run comprehensive test suite
- [ ] Performance benchmarking and optimization
- [ ] Documentation and deployment preparation

---

**Migration Success Criteria**: All existing capabilities preserved, 71% code reduction achieved, <3s response time maintained, production-ready PydanticAI foundation established.