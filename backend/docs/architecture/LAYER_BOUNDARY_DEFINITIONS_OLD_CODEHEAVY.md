# 🏗️ Layer Boundary Definitions & Interface Contracts

## Document Overview

**Document Type**: Architecture Boundary Specifications  
**Priority**: CRITICAL - Foundation Architecture  
**Created**: 2025-07-31  
**Status**: ✅ ACTIVE - Implementation Guidelines

This document defines the precise boundaries, responsibilities, and interface contracts between all architectural layers in the Universal RAG with Intelligent Agents system.

---

## 🎯 **Executive Summary**

### **Purpose**
Establish clear, enforceable boundaries between architectural layers to prevent:
- ❌ Responsibility overlap and confusion
- ❌ Tight coupling between layers  
- ❌ Business logic leaking into wrong layers
- ❌ Infrastructure concerns mixed with intelligence

### **Architectural Principle**
**"Each layer has ONE primary responsibility and communicates through well-defined interfaces"**

---

## 🏛️ **Layer Hierarchy & Dependencies**

### **Dependency Flow (One-Way Only)**
```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (HTTP Interface)              │
│                    ↓ depends on                            │
├─────────────────────────────────────────────────────────────┤
│                Services Layer (Business Orchestration)      │
│                    ↓ depends on                            │
├─────────────────────────────────────────────────────────────┤
│                 Agents Layer (Intelligent Reasoning)        │
│                    ↓ depends on                            │
├─────────────────────────────────────────────────────────────┤
│                 Tools Layer (Functional Capabilities)       │
│                    ↓ depends on                            │
├─────────────────────────────────────────────────────────────┤
│                 Core Layer (Infrastructure)                 │
│                    ↓ depends on                            │
└─────────────────────────────────────────────────────────────┘
                  External Services (Azure, etc.)
```

### **Dependency Rules**
- ✅ **Higher layers can depend on lower layers**
- ❌ **Lower layers CANNOT depend on higher layers**
- ✅ **Layers can skip levels for direct infrastructure access**
- ❌ **No circular dependencies allowed**

---

## 📋 **Layer Specifications**

### **1. API Layer** (`/api/`)

#### **Primary Responsibility**
HTTP request/response handling and API contract enforcement

#### **Detailed Responsibilities**
- ✅ Route definition and HTTP method handling
- ✅ Request validation using Pydantic models
- ✅ Response formatting and serialization
- ✅ Authentication and authorization middleware
- ✅ API documentation and OpenAPI specs
- ✅ Error handling and HTTP status codes

#### **Forbidden Responsibilities**
- ❌ Business logic implementation
- ❌ Direct database or Azure service calls
- ❌ Intelligent decision making
- ❌ Data processing or transformation
- ❌ Caching or performance optimization logic

#### **Interface Contract**
```python
class APILayerContract:
    """Contract defining what API layer can and cannot do"""
    
    # ✅ ALLOWED: HTTP concerns
    async def handle_request(self, request: HTTPRequest) -> HTTPResponse:
        """Process HTTP request and return response"""
        
    def validate_request(self, request_data: Dict[str, Any]) -> RequestModel:
        """Validate incoming request data"""
        
    def format_response(self, service_result: Any) -> ResponseModel:
        """Format service results for HTTP response"""
    
    # ❌ FORBIDDEN: Business logic
    # def process_query(self, query: str) -> ProcessedResult:  # NO!
    # def detect_domain(self, text: str) -> Domain:            # NO!
```

#### **Dependencies**
- ✅ **Can depend on**: Services Layer only
- ❌ **Cannot depend on**: Agents, Tools, Core directly

---

### **2. Services Layer** (`/services/`)

#### **Primary Responsibility**
Business workflow orchestration and resource management

#### **Detailed Responsibilities**
- ✅ High-level business workflow coordination
- ✅ Cross-cutting concerns (caching, performance, monitoring)
- ✅ Resource management and lifecycle
- ✅ Infrastructure service orchestration
- ✅ Service composition and integration
- ✅ Transaction and consistency management

#### **Forbidden Responsibilities**
- ❌ Intelligent reasoning or decision making
- ❌ Domain-specific knowledge processing
- ❌ Tool selection or execution logic
- ❌ Direct Azure service client operations
- ❌ Query intent analysis or pattern recognition

#### **Interface Contract**
```python
class ServiceLayerContract:
    """Contract defining service layer responsibilities"""
    
    # ✅ ALLOWED: Orchestration and resource management
    async def orchestrate_query_workflow(self, query: str) -> WorkflowResult:
        """Orchestrate complete query processing workflow"""
        # Delegates intelligence to agents
        agent_result = await self.agent_service.process_intelligent_query(query)
        # Manages resources and caching
        cached_result = await self.cache_service.get_or_set(key, agent_result)
        return self._build_workflow_result(cached_result)
    
    async def manage_system_resources(self) -> ResourceStatus:
        """Manage system resources and health"""
        
    # ❌ FORBIDDEN: Intelligence or direct infrastructure
    # def analyze_query_intent(self, query: str) -> Intent:     # NO! Agent responsibility
    # def execute_azure_search(self, query: str) -> Results:    # NO! Core responsibility
```

#### **Dependencies**
- ✅ **Can depend on**: Agents Layer, Core Layer
- ❌ **Cannot depend on**: API Layer, Tools Layer directly

---

### **3. Agents Layer** (`/agents/`)

#### **Primary Responsibility**
Intelligent reasoning, decision making, and cognitive processing

#### **Detailed Responsibilities**
- ✅ Query analysis and intent detection
- ✅ Multi-step reasoning and planning
- ✅ Context management and memory
- ✅ Tool selection and coordination
- ✅ Pattern recognition and learning
- ✅ Domain adaptation and discovery
- ✅ Confidence scoring and uncertainty handling

#### **Forbidden Responsibilities**
- ❌ Infrastructure management or resource orchestration
- ❌ HTTP request/response handling
- ❌ Direct tool implementation
- ❌ System-wide caching or performance monitoring
- ❌ Azure service client management

#### **Interface Contract**
```python
class AgentLayerContract:
    """Contract defining agent layer capabilities"""
    
    # ✅ ALLOWED: Intelligence and reasoning
    async def analyze_query_intelligence(self, query: str, context: AgentContext) -> IntelligenceResult:
        """Perform intelligent analysis of query"""
        intent = await self.reasoning_engine.analyze_intent(query, context)
        tools_needed = await self.tool_coordinator.select_tools(intent)
        reasoning_plan = await self.planning_engine.create_plan(intent, tools_needed)
        return IntelligenceResult(intent, tools_needed, reasoning_plan)
    
    async def execute_reasoning_workflow(self, plan: ReasoningPlan) -> ReasoningResult:
        """Execute multi-step reasoning workflow"""
    
    async def learn_from_interaction(self, interaction: Interaction) -> LearningResult:
        """Learn and adapt from successful interactions"""
    
    # ❌ FORBIDDEN: Infrastructure or orchestration
    # def manage_cache_lifecycle(self) -> CacheStatus:          # NO! Service responsibility
    # async def call_azure_openai(self, prompt: str) -> str:    # NO! Core responsibility
```

#### **Dependencies**
- ✅ **Can depend on**: Tools Layer, Core Layer
- ❌ **Cannot depend on**: API Layer, Services Layer

---

### **4. Tools Layer** (`/tools/`) - **TO BE IMPLEMENTED**

#### **Primary Responsibility**
Specific functional capabilities and domain-specific operations

#### **Detailed Responsibilities**
- ✅ Specific task execution (search, analysis, generation)
- ✅ Tool discovery from domain patterns
- ✅ Tool validation and effectiveness scoring
- ✅ Result processing and formatting
- ✅ Tool lifecycle management
- ✅ Domain-specific functionality

#### **Forbidden Responsibilities**
- ❌ Decision making about tool usage
- ❌ Multi-tool coordination or orchestration
- ❌ Infrastructure management
- ❌ Context or memory management
- ❌ Business workflow orchestration

#### **Interface Contract**
```python
class ToolLayerContract:
    """Contract defining tool layer capabilities"""
    
    # ✅ ALLOWED: Specific functional execution
    async def execute_tool(self, tool_request: ToolRequest) -> ToolResult:
        """Execute specific tool functionality"""
        
    async def validate_tool_effectiveness(self, tool: Tool, context: ToolContext) -> EffectivenessScore:
        """Validate and score tool effectiveness"""
        
    async def discover_tools_from_patterns(self, patterns: List[Pattern]) -> List[Tool]:
        """Discover and generate tools from domain patterns"""
    
    # ❌ FORBIDDEN: Coordination or decision making
    # def select_best_tool(self, options: List[Tool]) -> Tool:   # NO! Agent responsibility
    # def orchestrate_multi_tool_workflow(self) -> Result:      # NO! Agent responsibility
```

#### **Dependencies**
- ✅ **Can depend on**: Core Layer only
- ❌ **Cannot depend on**: API, Services, Agents layers

---

### **5. Core Layer** (`/core/`)

#### **Primary Responsibility**
Infrastructure services, Azure integrations, and technical utilities

#### **Detailed Responsibilities**
- ✅ Azure service client wrappers and integrations
- ✅ Data models, contracts, and shared types
- ✅ Technical utilities and helper functions
- ✅ System monitoring and health checks
- ✅ Configuration and settings management
- ✅ Cross-cutting technical infrastructure

#### **Forbidden Responsibilities**
- ❌ Business logic or workflow orchestration
- ❌ Intelligent decision making or reasoning
- ❌ HTTP request/response handling
- ❌ Tool coordination or selection
- ❌ Domain-specific processing logic

#### **Interface Contract**
```python
class CoreLayerContract:
    """Contract defining core infrastructure capabilities"""
    
    # ✅ ALLOWED: Infrastructure and technical services
    async def execute_azure_service_operation(self, operation: AzureOperation) -> AzureResult:
        """Execute Azure service operations with proper error handling"""
        
    def provide_data_model(self, model_type: str) -> DataModel:
        """Provide data models and contracts"""
        
    async def monitor_system_health(self) -> HealthStatus:
        """Monitor and report system health"""
    
    # ❌ FORBIDDEN: Business logic or intelligence
    # def process_business_workflow(self, data: Any) -> Result:  # NO! Service responsibility
    # def make_intelligent_decision(self, options: List) -> Any: # NO! Agent responsibility
```

#### **Dependencies**
- ✅ **Can depend on**: External services (Azure, databases) only
- ❌ **Cannot depend on**: Any internal layers

---

## 🔌 **Inter-Layer Interface Contracts**

### **API → Services Interface**
```python
class APIToServicesInterface:
    """Standardized interface between API and Services layers"""
    
    async def process_query_request(self, request: QueryRequest) -> QueryResponse:
        """Process query through business logic layer"""
        
    async def get_system_status(self) -> SystemStatus:
        """Get system status and health information"""
        
    async def manage_user_session(self, session: UserSession) -> SessionResult:
        """Manage user session lifecycle"""
```

### **Services → Agents Interface**
```python
class ServicesToAgentsInterface:
    """Standardized interface between Services and Agents layers"""
    
    async def request_intelligent_analysis(self, query: str, context: Dict[str, Any]) -> IntelligenceResult:
        """Request intelligent analysis from agents"""
        
    async def coordinate_reasoning_workflow(self, workflow: ReasoningWorkflow) -> WorkflowResult:
        """Coordinate multi-step reasoning workflow"""
        
    async def adapt_to_domain(self, domain_data: DomainData) -> AdaptationResult:
        """Request domain adaptation from discovery system"""
```

### **Agents → Tools Interface**
```python
class AgentsToToolsInterface:
    """Standardized interface between Agents and Tools layers"""
    
    async def execute_selected_tools(self, tool_selection: ToolSelection) -> ToolResults:
        """Execute tools selected by agent reasoning"""
        
    async def discover_domain_tools(self, domain_context: DomainContext) -> AvailableTools:
        """Discover available tools for domain context"""
        
    async def validate_tool_results(self, results: ToolResults, context: ValidationContext) -> ValidationResult:
        """Validate tool execution results"""
```

### **All Layers → Core Interface**
```python
class AllLayersToCoreInterface:
    """Standardized interface from all layers to Core infrastructure"""
    
    async def get_azure_client(self, service_type: AzureServiceType) -> AzureClient:
        """Get Azure service client instance"""
        
    def get_data_model(self, model_name: str) -> DataModel:
        """Get data model or contract definition"""
        
    async def log_operation(self, operation: Operation, metadata: Dict[str, Any]) -> None:
        """Log operation with proper correlation tracking"""
        
    async def get_configuration(self, config_key: str) -> ConfigValue:
        """Get configuration value"""
```

---

## 🚨 **Boundary Violation Detection**

### **Automated Detection Rules**
```python
class BoundaryViolationDetector:
    """Detect violations of layer boundaries"""
    
    VIOLATION_PATTERNS = {
        'api_business_logic': [
            'api/.*\\.py.*def.*process.*query',  # API doing query processing
            'api/.*\\.py.*detect.*domain',       # API doing domain detection
        ],
        'service_intelligence': [
            'services/.*\\.py.*analyze.*intent', # Service doing intelligent analysis
            'services/.*\\.py.*reasoning',       # Service doing reasoning
        ],
        'core_business_logic': [
            'core/.*\\.py.*workflow',            # Core containing workflows
            'core/.*\\.py.*orchestrat',          # Core doing orchestration
        ]
    }
    
    def detect_violations(self, codebase_path: str) -> List[BoundaryViolation]:
        """Scan codebase for boundary violations"""
```

### **Common Violation Patterns**

#### **❌ Service Layer Violations**
```python
# WRONG: Service doing intelligent analysis
class QueryService:
    async def process_query(self, query: str):
        # ❌ Intelligence belongs in Agents layer
        if "medical" in query.lower():
            domain = "medical"
        
        # ❌ Direct reasoning logic
        if query.endswith("?"):
            intent = "question"
```

#### **✅ Corrected Service Implementation**
```python
# RIGHT: Service orchestrates, Agent analyzes
class QueryService:
    async def process_query(self, query: str):
        # ✅ Delegate intelligence to agents
        intelligence_result = await self.agent_service.analyze_query_intelligence(query)
        
        # ✅ Focus on orchestration and resource management
        workflow_result = await self.orchestrate_processing_workflow(intelligence_result)
        return workflow_result
```

#### **❌ API Layer Violations**
```python
# WRONG: API doing business logic
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    # ❌ Business logic in API layer
    if request.domain is None:
        domain = detect_domain(request.query)
    
    # ❌ Direct service orchestration
    results = await search_multiple_sources(request.query)
```

#### **✅ Corrected API Implementation**
```python
# RIGHT: API delegates to services
@app.post("/query")
async def query_endpoint(
    request: QueryRequest, 
    query_service: QueryService = Depends(get_query_service)
):
    # ✅ Pure HTTP handling, delegate to services
    result = await query_service.process_universal_query(
        query=request.query,
        domain=request.domain,
        max_results=request.max_results
    )
    return QueryResponse.from_service_result(result)
```

---

## 📏 **Boundary Enforcement Mechanisms**

### **1. Interface Contracts (Runtime)**
```python
class LayerBoundaryContract:
    """Enforce boundary contracts at runtime"""
    
    def __init__(self, layer_name: str, allowed_dependencies: List[str]):
        self.layer_name = layer_name
        self.allowed_dependencies = allowed_dependencies
    
    def validate_dependency(self, target_layer: str) -> bool:
        """Validate if dependency is allowed"""
        if target_layer not in self.allowed_dependencies:
            raise BoundaryViolationError(
                f"{self.layer_name} cannot depend on {target_layer}"
            )
        return True
```

### **2. Import Guards (Development)**
```python
# In each layer's __init__.py
class ImportGuard:
    """Prevent invalid imports between layers"""
    
    FORBIDDEN_IMPORTS = {
        'api': ['agents.*', 'tools.*', 'core.*'],  # API can only import services
        'services': ['api.*'],                      # Services cannot import API
        'agents': ['api.*', 'services.*'],         # Agents cannot import upper layers
        'tools': ['api.*', 'services.*', 'agents.*'], # Tools only import core
    }
```

### **3. Testing Boundaries**
```python
class BoundaryComplianceTest:
    """Test boundary compliance"""
    
    def test_layer_dependencies(self):
        """Test that layers only depend on allowed layers"""
        
    def test_no_circular_dependencies(self):
        """Test no circular dependencies exist"""
        
    def test_interface_contracts(self):
        """Test that inter-layer interfaces are respected"""
```

---

## 📋 **Implementation Checklist**

### **Phase 1: Documentation & Contracts (Week 1)**
- [x] ✅ Layer boundary definitions documented
- [ ] 🔄 Interface contracts implemented
- [ ] 🔄 Boundary violation detection rules created
- [ ] 🔄 Development guidelines updated

### **Phase 2: Code Refactoring (Week 1-2)**
- [ ] 🔄 Service layer intelligence violations fixed
- [ ] 🔄 Agent layer infrastructure violations fixed
- [ ] 🔄 Core layer business logic violations fixed
- [ ] 🔄 API layer business logic violations fixed

### **Phase 3: Enforcement & Testing (Week 2)**
- [ ] 🔄 Import guards implemented
- [ ] 🔄 Runtime boundary validation added
- [ ] 🔄 Boundary compliance tests created
- [ ] 🔄 CI/CD boundary checks integrated

---

## 🎯 **Success Criteria**

### **Functional Success**
- [ ] All layers have single, clear responsibility
- [ ] No circular dependencies between layers
- [ ] All inter-layer communication through defined interfaces
- [ ] Boundary violations automatically detected and prevented

### **Quality Success**
- [ ] Code is more maintainable and testable
- [ ] Clear separation of concerns
- [ ] Easier to understand and modify individual layers
- [ ] Consistent architectural patterns across codebase

### **Performance Success**
- [ ] No performance degradation from boundary enforcement
- [ ] Efficient inter-layer communication
- [ ] Minimal overhead from interface contracts

---

**Document Status**: ✅ ACTIVE - Implementation Guidelines  
**Next Review**: Weekly during boundary refactoring  
**Success Metrics**: Zero boundary violations in automated scans