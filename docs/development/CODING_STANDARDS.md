# Coding Standards

**Azure Universal RAG - Essential Development Rules**

## Core Principles

1. **Data-Driven Everything** - Learn from actual data, never hardcode assumptions
2. **Zero Fake Data** - Real processing results, no placeholders or mock responses
3. **Universal Design** - Works with any domain without configuration
4. **Production-Ready** - Complete implementations with comprehensive error handling
5. **Performance-First** - Async operations, <3s response times, proper monitoring

### 1. Data-Driven Implementation
- **✅ DO**: Learn patterns from actual text corpus analysis
- **✅ DO**: Use statistical thresholds derived from real data
- **❌ DON'T**: Hardcode domain-specific entity lists or rules
- **❌ DON'T**: Use arbitrary confidence scores without calculation

### 2. No Fake Data
- **✅ DO**: Return real results from actual Azure services
- **✅ DO**: Throw explicit errors when operations fail
- **❌ DON'T**: Return placeholder data like `["entity1", "entity2"]`
- **❌ DON'T**: Use mock responses in production code

### 3. Universal Design
- **✅ DO**: Build domain-agnostic components that learn patterns
- **✅ DO**: Use configuration-driven behavior
- **❌ DON'T**: Create separate classes per domain (MedicalProcessor, LegalProcessor)
- **❌ DON'T**: Use if/else chains for domain-specific logic

### 4. Production-Ready Code
- **✅ DO**: Implement comprehensive error handling with context
- **✅ DO**: Validate inputs and outputs
- **❌ DON'T**: Leave TODO comments in production code
- **❌ DON'T**: Use silent try/catch blocks that hide errors

### 5. Performance Requirements
- **✅ DO**: Use async/await for all I/O operations
- **✅ DO**: Implement parallel processing where possible
- **❌ DON'T**: Use blocking synchronous operations
- **❌ DON'T**: Ignore response time requirements (<3s target)

## Code Examples

### ✅ Good Example - Data-Driven Entity Extraction
```python
class EntityExtractor:
    async def extract_entities(self, text_corpus: List[str]) -> List[Entity]:
        # Learn from actual content, no hardcoded entities
        patterns = await self.pattern_learner.discover_patterns(text_corpus)
        entities = await self.nlp_service.extract_using_patterns(text_corpus, patterns)
        return self._validate_with_learned_thresholds(entities, patterns)
```

### ❌ Bad Example - Hardcoded Domain Logic
```python
# FORBIDDEN - Domain-specific hardcoded processing
def process_query(self, query: str, domain: str):
    if domain == "medical":
        return self._process_medical_query(query)  # Hardcoded logic
    elif domain == "legal":
        return self._process_legal_query(query)    # Not universal
```

### ✅ Good Example - Comprehensive Error Handling
```python
async def process_document(self, path: str) -> ProcessingResult:
    try:
        content = await self.load_document(path)
        return await self.analyze_content(content)
    except FileNotFoundError:
        raise DocumentNotFoundError(f"Document not found: {path}")
    except Exception as e:
        logger.error(f"Processing failed for {path}: {e}", exc_info=True)
        raise ProcessingError(f"Document processing failed: {str(e)}") from e
```

### ❌ Bad Example - Silent Failures
```python
# FORBIDDEN - Silent failure with no error handling
async def process_data(self, data: Any):
    try:
        return await self.real_processing(data)
    except:
        return None  # Hides errors, returns fake success
```

## Review Checklist

Before merging any code, verify:

- [ ] **Data-driven**: No hardcoded values, learns from actual data
- [ ] **No fake data**: Real processing results, no placeholders or mock data
- [ ] **Production-ready**: Complete implementation, comprehensive error handling
- [ ] **Domain-agnostic**: Works universally without domain-specific assumptions
- [ ] **Performance**: Async operations, <3s response time, proper monitoring

## PydanticAI Implementation Rules

### 6. Agent Architecture Patterns

- **✅ DO**: Co-locate agent tools with agent definitions using `@agent.tool` decorator
- **✅ DO**: Use proper `RunContext[Dependencies]` type hints for dependency injection
- **❌ DON'T**: Create separate tool modules - tools belong with their agent
- **❌ DON'T**: Import tools across agent boundaries - use agent delegation instead

### 7. Agent Tool Co-location

```python
# ✅ CORRECT - Tools co-located with agent
@agent.tool
async def domain_specific_search(
    ctx: RunContext[AzureServiceContainer], query: str
) -> SearchResult:
    # Tool implementation here
    pass
```

```python
# ❌ FORBIDDEN - Separate tool modules
# tools/search_tools.py - DON'T DO THIS
def execute_search():  # Tools should not be separate
    pass
```

### 8. Agent Delegation Patterns

- **✅ DO**: Use agent delegation for cross-domain functionality
- **✅ DO**: Pass `RunContext` and track usage across agent calls
- **❌ DON'T**: Import and call other agents' methods directly
- **❌ DON'T**: Break agent boundaries with direct tool access

```python
# ✅ CORRECT - Agent delegation
result = await domain_agent.run(
    "detect_domain_from_query",
    message_history=[{"role": "user", "content": f"Detect domain: {query}"}],
    deps=ctx.deps
)
```

### 9. Graph-Based State Management

- **✅ DO**: Use `pydantic-graph` for workflow state machines
- **✅ DO**: Implement proper state transitions with error recovery
- **❌ DON'T**: Use simple linear workflows for complex operations
- **❌ DON'T**: Ignore state persistence for fault tolerance

```python
# ✅ CORRECT - Graph-based workflow with error recovery
@dataclass
class SearchNode(BaseNode[SearchState]):
    async def run(self, ctx: GraphRunContext[SearchState]) -> ResultNode | TimeoutNode:
        start_time = time.time()
        result = await execute_search(ctx.state.query)
        if time.time() - start_time > 3.0:
            return TimeoutNode("SLA violation detected")
        return ResultNode(result)
```

### 10. Azure Service Integration

- **✅ DO**: Use `DefaultAzureCredential` for unified authentication
- **✅ DO**: Initialize services with proper health checks and graceful degradation
- **❌ DON'T**: Hardcode service endpoints or authentication keys
- **❌ DON'T**: Create service instances without dependency injection

### 11. Performance Requirements (Sub-3-Second SLA)

- **✅ DO**: Implement background processing for startup optimization
- **✅ DO**: Use intelligent caching with sub-5ms lookup times
- **❌ DON'T**: Block initialization with synchronous operations
- **❌ DON'T**: Execute searches without timeout protection

```python
# ✅ CORRECT - Background processing optimization
async def _run_background_processing(self):
    """Pre-compute domain intelligence for runtime performance"""
    stats = await run_startup_background_processing()
    self.background_processed = True
    logger.info(f"⚡ Processing rate: {stats.files_per_second:.1f} files/sec")
```

### 12. State Persistence and Recovery

- **✅ DO**: Persist workflow state to Azure Cosmos DB for fault recovery
- **✅ DO**: Implement checkpointing for long-running operations
- **❌ DON'T**: Lose processing state on service restarts
- **❌ DON'T**: Re-process completed workflow steps after failures

## Critical Agent Boundary Rules

### 13. Strict Agent Responsibility Separation

**Prevent hardcoded values by enforcing clear agent boundaries:**

- **✅ DO**: Domain Intelligence Agent performs ONLY statistical analysis
- **✅ DO**: Knowledge Extraction Agent uses ONLY provided configurations
- **❌ DON'T**: Allow Domain Intelligence Agent to do extraction-like pattern matching
- **❌ DON'T**: Allow Knowledge Extraction Agent to generate configuration parameters

```python
# ✅ CORRECT - Domain Intelligence Agent: Pure Mathematical Analysis
class DomainIntelligenceAgent:
    async def analyze_domain_corpus(self, domain_path: Path) -> ExtractionConfiguration:
        # 1. ONLY statistical analysis - no hardcoded patterns
        corpus_stats = await self._calculate_frequency_distributions(domain_path)
        entity_clusters = await self._cluster_using_entropy(corpus_stats)
        optimal_threshold = self._calculate_confidence_percentiles(entity_clusters)

        # 2. Generate configuration from PURE DATA (no hardcoded values)
        return ExtractionConfiguration(
            entity_confidence_threshold=optimal_threshold,
            expected_entity_types=self._derive_from_clusters(entity_clusters),
            chunk_size=self._optimize_from_statistics(corpus_stats)
        )

# ✅ CORRECT - Knowledge Extraction Agent: Tool Delegation Only
class KnowledgeExtractionAgent:
    async def extract_from_document(self, document: str, config: ExtractionConfiguration) -> ExtractedKnowledge:
        # 1. NEVER generate config - only USE provided config
        entities = await self.extraction_tools.extract_entities(document, config)
        relationships = await self.extraction_tools.extract_relationships(document, config)

        # 2. Return results (NO configuration decisions)
        return ExtractedKnowledge(entities=entities, relationships=relationships)
```

**❌ FORBIDDEN Patterns That Enable Hardcoded Values:**

```python
# FORBIDDEN - Domain Intelligence doing extraction-like tasks
class DomainIntelligenceAgent:
    def analyze_domain(self, content: str):
        # ❌ This is extraction, not configuration
        if "programming" in content:
            return hardcoded_programming_config  # Hardcoded!

# FORBIDDEN - Knowledge Extraction generating config
class KnowledgeExtractionAgent:
    def extract_knowledge(self, document: str):
        # ❌ Should receive config, not generate it
        threshold = 0.7  # Hardcoded!
        entities = self.extract_with_threshold(document, threshold)
```

### 14. Mathematical Foundation Requirements

**Eliminate hardcoded patterns with pure statistical analysis:**

- **✅ DO**: Use entropy, frequency distributions, and clustering for pattern discovery
- **✅ DO**: Calculate confidence thresholds from percentile analysis
- **❌ DON'T**: Use regex patterns or hardcoded classification dictionaries
- **❌ DON'T**: Make domain assumptions without mathematical validation

```python
# ✅ CORRECT - Mathematical pattern discovery
def discover_entity_patterns(corpus: List[str]) -> EntityPatterns:
    # Pure mathematical analysis
    word_frequencies = calculate_frequency_distribution(corpus)
    entropy_scores = calculate_information_entropy(word_frequencies)
    entity_clusters = kmeans_clustering(entropy_scores, n_clusters="auto")

    # Data-driven thresholds
    confidence_threshold = np.percentile(entity_clusters.confidence_scores, 95)

    return EntityPatterns(
        clusters=entity_clusters,
        threshold=confidence_threshold,
        derived_from="statistical_analysis"
    )

# ❌ FORBIDDEN - Hardcoded pattern matching
def discover_entity_patterns(content: str) -> EntityPatterns:
    # ❌ Hardcoded regex patterns
    if re.match(r"programming|code|software", content):
        return HARDCODED_PROGRAMMING_PATTERNS  # Not data-driven!
```

### 15. Tool Architecture Enforcement

**Prevent logic duplication by enforcing tool delegation:**

- **✅ DO**: Agents coordinate tools using `@agent.tool` decorators
- **✅ DO**: Implement business logic in tools, coordination in agents
- **❌ DON'T**: Implement extraction logic directly in agents
- **❌ DON'T**: Create standalone tool modules separate from agents

```python
# ✅ CORRECT - Agent coordinates tools
class KnowledgeExtractionAgent:
    @agent.tool
    async def extract_entities_with_config(
        ctx: RunContext[AzureServiceContainer],
        document: str,
        config: ExtractionConfiguration
    ) -> List[Entity]:
        # Tool implementation co-located with agent
        return await ctx.deps.nlp_service.extract_entities(
            document,
            confidence_threshold=config.entity_confidence_threshold,
            expected_types=config.expected_entity_types
        )

    async def process_document(self, document: str, config: ExtractionConfiguration) -> ExtractedKnowledge:
        # Agent coordinates tools - no direct implementation
        entities = await self.extract_entities_with_config(document, config)
        return ExtractedKnowledge(entities=entities)
```

### 16. Config-Extraction Orchestration Pattern

**Ensure proper two-stage workflow integration:**

- **✅ DO**: Use `ConfigExtractionOrchestrator` for unified workflow
- **✅ DO**: Pass `ExtractionConfiguration` objects between stages
- **❌ DON'T**: Allow direct communication between Domain Intelligence and Knowledge Extraction agents
- **❌ DON'T**: Skip the two-stage architecture for "simpler" approaches

```python
# ✅ CORRECT - Orchestrated two-stage workflow
class UniversalAgent:
    async def process_query(self, query: str) -> QueryResult:
        # Stage 1: Configuration generation
        config = await self.config_extraction_orchestrator.generate_config(query)

        # Stage 2: Knowledge extraction using config
        knowledge = await self.config_extraction_orchestrator.extract_knowledge(query, config)

        return self._generate_response(knowledge)

# ❌ FORBIDDEN - Direct agent-to-agent communication
class UniversalAgent:
    async def process_query(self, query: str) -> QueryResult:
        # ❌ Bypasses orchestration, enables hardcoded fallbacks
        domain_config = await self.domain_agent.analyze(query)  # Direct call
        knowledge = await self.extraction_agent.extract(query)   # No config passed
```

### 17. Graph-Based Workflow State Management

**Implement fault-tolerant workflows with `pydantic-graph`:**

- **✅ DO**: Use graph-based state machines for complex multi-step operations
- **✅ DO**: Implement proper state transitions with error recovery paths
- **✅ DO**: Persist workflow state to Azure Cosmos DB for fault tolerance
- **❌ DON'T**: Use simple linear workflows for operations requiring SLA guarantees
- **❌ DON'T**: Lose processing state on service restarts

```python
# ✅ CORRECT - Graph-based workflow with fault recovery
from pydantic_graph import BaseNode, GraphRunContext
from dataclasses import dataclass
import time

@dataclass
class SearchState:
    query: str
    domain: str
    start_time: float
    max_duration: float = 3.0

@dataclass
class SearchNode(BaseNode[SearchState]):
    async def run(self, ctx: GraphRunContext[SearchState]) -> 'ResultNode | TimeoutNode | ErrorNode':
        start_time = time.time()

        try:
            # Execute search with SLA monitoring
            result = await self._execute_tri_modal_search(
                query=ctx.state.query,
                domain=ctx.state.domain
            )

            # Check SLA compliance
            elapsed = time.time() - start_time
            if elapsed > ctx.state.max_duration:
                return TimeoutNode(f"Search exceeded SLA: {elapsed:.2f}s > {ctx.state.max_duration}s")

            return ResultNode(result=result, elapsed_time=elapsed)

        except Exception as e:
            return ErrorNode(error=str(e), recovery_strategy="fallback_search")

@dataclass
class ResultNode(BaseNode[SearchState]):
    result: Any
    elapsed_time: float

    async def run(self, ctx: GraphRunContext[SearchState]) -> None:
        # Workflow completed successfully
        await self._persist_success_metrics(ctx.state, self.result, self.elapsed_time)

@dataclass
class TimeoutNode(BaseNode[SearchState]):
    reason: str

    async def run(self, ctx: GraphRunContext[SearchState]) -> 'FallbackSearchNode':
        # SLA violation - trigger fallback
        await self._log_sla_violation(ctx.state, self.reason)
        return FallbackSearchNode(fallback_strategy="cached_results")

@dataclass
class ErrorNode(BaseNode[SearchState]):
    error: str
    recovery_strategy: str

    async def run(self, ctx: GraphRunContext[SearchState]) -> 'RetryNode | FallbackSearchNode':
        # Determine recovery path based on error type
        if "timeout" in self.error.lower():
            return FallbackSearchNode(fallback_strategy="degraded_search")
        else:
            return RetryNode(max_retries=2, backoff_seconds=0.5)
```

### 18. Azure Service Integration Best Practices

**Ensure enterprise-grade Azure service integration:**

- **✅ DO**: Use `DefaultAzureCredential` for unified managed identity authentication
- **✅ DO**: Initialize services with health checks and graceful degradation
- **✅ DO**: Implement circuit breaker patterns for service failures
- **❌ DON'T**: Hardcode service endpoints or authentication keys
- **❌ DON'T**: Create service instances without proper dependency injection

```python
# ✅ CORRECT - Enterprise Azure service integration
from azure.identity import DefaultAzureCredential
from agents.core.azure_services import ConsolidatedAzureServices

class AzureServiceContainer:
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.azure_services = None
        self.health_status = {}

    async def initialize_services(self) -> Dict[str, bool]:
        """Initialize all Azure services with health monitoring"""
        self.azure_services = ConsolidatedAzureServices(credential=self.credential)

        # Initialize with comprehensive health checks
        service_status = await self.azure_services.initialize_all_services()

        # Validate critical services for SLA compliance
        critical_services = ["ai_foundry", "search", "cosmos"]
        critical_available = sum(
            service_status.get(service, False) for service in critical_services
        )

        if critical_available < 2:  # At least 2 of 3 critical services
            raise RuntimeError(f"Insufficient critical services available: {critical_available}/3")

        return service_status

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check with circuit breaker"""
        if not self.azure_services:
            return {"status": "not_initialized", "healthy_services": 0}

        health_result = await self.azure_services.health_check()

        # Update circuit breaker status based on health
        self._update_circuit_breakers(health_result)

        return health_result

    def _update_circuit_breakers(self, health_result: Dict[str, Any]):
        """Update circuit breaker state based on service health"""
        for service_name, status in health_result.get("service_health", {}).items():
            if status != "healthy":
                # Open circuit breaker for unhealthy services
                self.health_status[f"{service_name}_circuit_breaker"] = "open"
            else:
                # Close circuit breaker for healthy services
                self.health_status[f"{service_name}_circuit_breaker"] = "closed"

# ❌ FORBIDDEN - Hardcoded service initialization
class BadAzureIntegration:
    def __init__(self):
        # ❌ Hardcoded endpoint and credentials
        self.openai_client = AzureOpenAI(
            api_key="hardcoded-key",  # Security violation!
            azure_endpoint="https://hardcoded.openai.azure.com/"  # Not configurable!
        )
```

### 19. Performance Monitoring for Sub-3-Second SLA

**Implement comprehensive performance monitoring:**

- **✅ DO**: Monitor end-to-end response times with detailed breakdowns
- **✅ DO**: Implement performance budgets for each workflow stage
- **✅ DO**: Use background processing to optimize startup times
- **❌ DON'T**: Execute operations without timeout protection
- **❌ DON'T**: Ignore SLA violations or performance degradation

```python
# ✅ CORRECT - Comprehensive performance monitoring
import time
import asyncio
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class PerformanceMetrics:
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    stage_timings: Dict[str, float] = field(default_factory=dict)
    sla_target: float = 3.0

    @property
    def total_duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def sla_compliance(self) -> bool:
        return self.total_duration <= self.sla_target

    def record_stage(self, stage_name: str, duration: float):
        """Record timing for individual workflow stage"""
        self.stage_timings[stage_name] = duration

        # Alert if stage exceeds budget
        stage_budgets = {
            "domain_analysis": 0.5,    # 500ms budget
            "config_generation": 0.3,  # 300ms budget
            "knowledge_extraction": 1.5, # 1.5s budget
            "search_orchestration": 0.7  # 700ms budget
        }

        budget = stage_budgets.get(stage_name, 1.0)
        if duration > budget:
            logger.warning(
                f"⚠️ Stage '{stage_name}' exceeded budget: {duration:.3f}s > {budget:.3f}s"
            )

class PerformanceMonitor:
    def __init__(self):
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.sla_violations: List[Dict[str, Any]] = []

    async def monitor_operation(self, operation_name: str, sla_target: float = 3.0):
        """Context manager for operation performance monitoring"""

        class OperationMonitor:
            def __init__(self, parent: PerformanceMonitor, name: str, target: float):
                self.parent = parent
                self.metrics = PerformanceMetrics(operation_name=name,
                                                start_time=time.time(),
                                                sla_target=target)

            async def __aenter__(self):
                self.parent.active_operations[self.metrics.operation_name] = self.metrics
                return self.metrics

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.metrics.end_time = time.time()

                # Check SLA compliance
                if not self.metrics.sla_compliance:
                    self.parent._record_sla_violation(self.metrics)

                # Clean up active operation
                self.parent.active_operations.pop(self.metrics.operation_name, None)

                # Log performance summary
                logger.info(
                    f"🎯 Operation '{self.metrics.operation_name}' completed in "
                    f"{self.metrics.total_duration:.3f}s (SLA: {self.metrics.sla_target}s, "
                    f"Compliant: {self.metrics.sla_compliance})"
                )

        return OperationMonitor(self, operation_name, sla_target)

    def _record_sla_violation(self, metrics: PerformanceMetrics):
        """Record SLA violation for analysis"""
        violation = {
            "operation": metrics.operation_name,
            "duration": metrics.total_duration,
            "sla_target": metrics.sla_target,
            "violation_amount": metrics.total_duration - metrics.sla_target,
            "stage_timings": metrics.stage_timings,
            "timestamp": time.time()
        }

        self.sla_violations.append(violation)

        logger.error(
            f"🚨 SLA VIOLATION: {metrics.operation_name} took {metrics.total_duration:.3f}s "
            f"(target: {metrics.sla_target}s, violation: +{violation['violation_amount']:.3f}s)"
        )

# Usage example in agent
class UniversalAgent:
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()

    async def process_query(self, query: str) -> QueryResult:
        async with self.performance_monitor.monitor_operation("query_processing", sla_target=3.0) as metrics:
            # Stage 1: Domain analysis
            stage_start = time.time()
            config = await self._generate_config(query)
            metrics.record_stage("config_generation", time.time() - stage_start)

            # Stage 2: Knowledge extraction
            stage_start = time.time()
            knowledge = await self._extract_knowledge(query, config)
            metrics.record_stage("knowledge_extraction", time.time() - stage_start)

            # Stage 3: Search orchestration
            stage_start = time.time()
            result = await self._orchestrate_search(knowledge)
            metrics.record_stage("search_orchestration", time.time() - stage_start)

            return result
```

## Implementation Readiness Checklist

Before implementing any multi-agent feature, verify:

### Agent Architecture Compliance
- [ ] **Agent Boundaries**: Tools are co-located with their respective agents using `@agent.tool`
- [ ] **Mathematical Foundation**: No hardcoded patterns, pure statistical analysis only
- [ ] **Tool Delegation**: Agents coordinate tools, never implement extraction logic directly
- [ ] **Config-Extraction Separation**: Domain Intelligence generates config, Knowledge Extraction uses config
- [ ] **Orchestration Pattern**: ConfigExtractionOrchestrator manages two-stage workflow
- [ ] **Delegation Pattern**: Cross-agent communication uses proper delegation, not direct imports

### Workflow and State Management
- [ ] **Graph Workflows**: Complex operations use `pydantic-graph` state machines with error recovery
- [ ] **State Persistence**: Workflow state persists to Azure Cosmos DB for fault tolerance
- [ ] **Error Recovery**: Graceful degradation and retry mechanisms with circuit breakers
- [ ] **SLA Monitoring**: All operations have timeout protection and performance budgets

### Azure Service Integration
- [ ] **Azure Integration**: Services use `DefaultAzureCredential` with proper health checks
- [ ] **Service Health**: Circuit breaker patterns implemented for service failures
- [ ] **Dependency Injection**: All Azure services properly injected via container pattern
- [ ] **Configuration**: No hardcoded endpoints, all settings configurable

### Performance and Monitoring
- [ ] **Performance SLA**: Background processing and caching maintain sub-3-second responses
- [ ] **Stage Budgets**: Each workflow stage has defined performance budget
- [ ] **SLA Compliance**: End-to-end monitoring with violation tracking and alerting
- [ ] **Monitoring**: Comprehensive logging and metrics collection is in place

### Data-Driven Architecture
- [ ] **Zero Hardcoded Values**: All thresholds derived from statistical analysis
- [ ] **Universal Design**: Components work with any domain without hardcoded assumptions
- [ ] **Mathematical Validation**: Pattern discovery uses entropy, clustering, percentile analysis
- [ ] **Configuration Interface**: Clean contracts between configuration and extraction phases

## Architecture Violation Prevention

### Automated Enforcement
- **Pre-commit Hooks**: Validate agent boundary compliance, mathematical foundation requirements
- **Type Checking**: Ensure proper `RunContext[Dependencies]` usage and PydanticAI patterns
- **Performance Tests**: Verify sub-3-second SLA compliance and stage budget adherence
- **Architecture Validation**: Block hardcoded values, verify tool co-location, check orchestration patterns

### Code Review Requirements
- **Agent Boundary Review**: Verify strict separation between Domain Intelligence and Knowledge Extraction
- **Mathematical Foundation Review**: Confirm no hardcoded patterns, validate statistical analysis
- **Tool Architecture Review**: Ensure proper `@agent.tool` usage and delegation patterns
- **Performance Review**: Validate SLA monitoring, error recovery, and circuit breaker implementation
- **Azure Integration Review**: Confirm `DefaultAzureCredential` usage and health check patterns

### Blocking Conditions
- **❌ MERGE BLOCKED**: Any hardcoded domain assumptions or configuration values
- **❌ MERGE BLOCKED**: Agent boundary violations (extraction in Domain Intelligence, config generation in Knowledge Extraction)
- **❌ MERGE BLOCKED**: Missing mathematical foundation (regex patterns instead of statistical analysis)
- **❌ MERGE BLOCKED**: Tool architecture violations (standalone tool modules, direct logic implementation)
- **❌ MERGE BLOCKED**: Missing SLA monitoring or timeout protection
- **❌ MERGE BLOCKED**: Azure service integration without `DefaultAzureCredential`

### Architecture Gates
- **Implementation Gate**: All readiness checklist items must be completed before starting implementation
- **Performance Gate**: Sub-3-second SLA must be demonstrated with load testing
- **Fault Tolerance Gate**: Workflow state persistence and error recovery must be validated
- **Production Gate**: Complete health monitoring and circuit breaker patterns must be operational

---

*These standards eliminate the root causes of hardcoded values and ensure production-ready, PydanticAI-compliant multi-agent architecture with enterprise scalability and sub-3-second SLA compliance.*
