# Phase 1: Tool Co-Location Implementation Plan

**Date**: August 3, 2025
**Duration**: 1 week
**Priority**: High
**Status**: Ready for Implementation

## Overview

Phase 1 focuses on implementing PydanticAI best practices by co-locating tools with their respective agents using the official `Toolset` pattern. This organizational improvement will enhance maintainability while preserving all competitive advantages.

## Current State Analysis

### Tool Distribution Assessment

**Current Structure** (Anti-Pattern):
```
agents/tools/
├── config_tools.py         # 4 tools for domain intelligence
├── extraction_tools.py     # 6 tools for knowledge extraction
├── search_tools.py         # 5 tools for universal search
├── consolidated_tools.py   # 8 shared tools
├── discovery_tools.py      # 3 tools for domain discovery
└── dynamic_tools.py        # 2 utility tools
```

**Total**: 28 tools across 6 files

### Agent-Tool Mapping Analysis

| Agent | Current Tools | Location | Target Location |
|-------|---------------|----------|-----------------|
| **Domain Intelligence** | 7 tools | `tools/config_tools.py`, `tools/discovery_tools.py` | `domain_intelligence/toolsets.py` |
| **Knowledge Extraction** | 6 tools | `tools/extraction_tools.py` | `knowledge_extraction/toolsets.py` |
| **Universal Search** | 5 tools | `tools/search_tools.py` | `universal_search/toolsets.py` |
| **Shared Infrastructure** | 10 tools | `tools/consolidated_tools.py`, `tools/dynamic_tools.py` | `shared/toolsets.py` |

## Implementation Strategy

### Phase 1.1: Analysis & Preparation (Day 1)

#### Tool Dependency Mapping
1. **Analyze current tool usage patterns**
   ```bash
   # Find all tool imports across codebase
   grep -r "from agents.tools" . --include="*.py"
   grep -r "import.*tools" . --include="*.py"
   ```

2. **Document tool dependencies**
   - Map which agents use which tools
   - Identify shared tools used by multiple agents
   - Document Azure service dependencies for each tool

3. **Validate competitive advantage preservation**
   - Ensure tri-modal search tools remain functional
   - Verify hybrid domain intelligence tools are preserved
   - Confirm configuration-extraction tools maintain workflow

#### Preparation Tasks
- [ ] Create backup branch: `feature/phase1-tool-colocation`
- [ ] Document current tool performance baselines
- [ ] Set up testing framework for tool validation
- [ ] Create rollback strategy

### Phase 1.2: Toolset Class Implementation (Day 2-3)

#### Create Agent-Specific Toolsets

**1. Domain Intelligence Toolset**
```python
# agents/domain_intelligence/toolsets.py
from pydantic_ai import tool
from pydantic_ai.tools import Toolset

class DomainIntelligenceToolset(Toolset):
    """Tools for domain pattern analysis and configuration generation"""

    @tool
    async def analyze_corpus_statistics(
        self,
        ctx: RunContext[DomainDeps],
        corpus_path: str
    ) -> StatisticalAnalysis:
        """Statistical analysis of corpus content for pattern detection"""
        # Migrate from tools/config_tools.py

    @tool
    async def generate_semantic_patterns(
        self,
        ctx: RunContext[DomainDeps],
        content_sample: str
    ) -> SemanticPatterns:
        """LLM-powered semantic pattern discovery"""
        # Migrate from tools/config_tools.py

    @tool
    async def create_extraction_config(
        self,
        ctx: RunContext[DomainDeps],
        patterns: CombinedPatterns
    ) -> ExtractionConfiguration:
        """Generate extraction configuration from discovered patterns - FULLY DATA-DRIVEN"""
        # ✅ DATA-DRIVEN - Learn all parameters from actual corpus analysis
        entity_types = await self._discover_entity_types_from_patterns(patterns.statistical_patterns)
        relationship_types = await self._discover_relationship_types_from_patterns(patterns.semantic_patterns)

        # Calculate optimal thresholds from validation data
        entity_threshold = await self._calculate_confidence_threshold(patterns, "entity")
        relationship_threshold = await self._calculate_confidence_threshold(patterns, "relationship")

        return ExtractionConfiguration(
            entity_types=entity_types,
            relationship_types=relationship_types,
            entity_confidence_threshold=entity_threshold,
            relationship_confidence_threshold=relationship_threshold
        )

    @tool
    async def validate_pattern_quality(
        self,
        ctx: RunContext[DomainDeps],
        config: ExtractionConfiguration
    ) -> QualityMetrics:
        """Validate quality of discovered patterns and configuration"""
        # Migrate from tools/config_tools.py

    @tool
    async def discover_domain_patterns(
        self,
        ctx: RunContext[DomainDeps],
        domain_data: str
    ) -> DomainPatterns:
        """Discover domain-specific patterns from raw data"""
        # Migrate from tools/discovery_tools.py
```

**2. Knowledge Extraction Toolset**
```python
# agents/knowledge_extraction/toolsets.py
class KnowledgeExtractionToolset(Toolset):
    """Tools for multi-strategy entity and relationship extraction"""

    @tool
    async def extract_entities_multi_strategy(
        self,
        ctx: RunContext[ExtractionDeps],
        text: str,
        config: ExtractionConfig
    ) -> EntityResults:
        """Multi-strategy entity extraction with confidence scoring"""
        # Migrate from tools/extraction_tools.py

    @tool
    async def extract_relationships_contextual(
        self,
        ctx: RunContext[ExtractionDeps],
        text: str,
        entities: List[Entity]
    ) -> RelationshipResults:
        """Context-aware relationship extraction between entities"""
        # Migrate from tools/extraction_tools.py

    @tool
    async def validate_extraction_quality(
        self,
        ctx: RunContext[ExtractionDeps],
        results: ExtractionResults
    ) -> ValidationResults:
        """Comprehensive quality validation of extraction results"""
        # Migrate from tools/extraction_tools.py

    @tool
    async def store_knowledge_graph(
        self,
        ctx: RunContext[ExtractionDeps],
        validated_results: ValidatedResults
    ) -> StorageResults:
        """Store validated knowledge graph in Azure Cosmos DB"""
        # Migrate from tools/extraction_tools.py
```

**3. Universal Search Toolset**
```python
# agents/universal_search/toolsets.py
class UniversalSearchToolset(Toolset):
    """Tools for tri-modal search coordination and execution"""

    @tool
    async def execute_vector_search(
        self,
        ctx: RunContext[SearchDeps],
        query: str,
        filters: SearchFilters
    ) -> VectorResults:
        """Execute semantic vector search via Azure Cognitive Search"""
        # Migrate from tools/search_tools.py

    @tool
    async def execute_graph_search(
        self,
        ctx: RunContext[SearchDeps],
        query: str,
        graph_context: GraphContext
    ) -> GraphResults:
        """Execute graph traversal search via Azure Cosmos DB"""
        # Migrate from tools/search_tools.py

    @tool
    async def execute_gnn_search(
        self,
        ctx: RunContext[SearchDeps],
        query: str,
        pattern_context: PatternContext
    ) -> GNNResults:
        """Execute GNN pattern prediction via Azure ML"""
        # Migrate from tools/search_tools.py

    @tool
    async def synthesize_search_results(
        self,
        ctx: RunContext[SearchDeps],
        tri_modal_results: TriModalResults
    ) -> FinalResults:
        """Synthesize and rank tri-modal search results"""
        # Migrate from tools/search_tools.py
```

**4. Shared Infrastructure Toolset**
```python
# agents/shared/toolsets.py
class AzureServiceToolset(Toolset):
    """Common Azure service operations available to all agents"""

    @tool
    async def get_azure_credentials(self, ctx: RunContext[SharedDeps]) -> AzureCredentials:
        """Get managed identity credentials for Azure services"""

    @tool
    async def monitor_service_health(self, ctx: RunContext[SharedDeps]) -> ServiceHealth:
        """Monitor health status of Azure services"""

    @tool
    async def track_usage_metrics(self, ctx: RunContext[SharedDeps]) -> UsageMetrics:
        """Track usage metrics for Azure services"""

class PerformanceToolset(Toolset):
    """Performance monitoring and optimization tools"""

    @tool
    async def measure_response_time(self, ctx: RunContext[SharedDeps]) -> ResponseMetrics:
        """Measure and validate response time SLAs"""

    @tool
    async def optimize_cache_strategy(self, ctx: RunContext[SharedDeps]) -> CacheOptimization:
        """Optimize caching strategies for performance"""

    @tool
    async def validate_sla_compliance(self, ctx: RunContext[SharedDeps]) -> SLAStatus:
        """Validate compliance with sub-3-second SLA"""
```

### Phase 1.3: Agent Integration (Day 4-5)

#### Update Agent Implementations

**1. Domain Intelligence Agent Update**
```python
# agents/domain_intelligence/agent.py
from pydantic_ai import Agent
from .toolsets import DomainIntelligenceToolset
from ..shared.toolsets import AzureServiceToolset, PerformanceToolset

# Create agent with co-located toolsets
domain_agent = Agent(
    'azure-openai:gpt-4',
    deps_type=DomainDeps,
    toolsets=[
        DomainIntelligenceToolset(),
        AzureServiceToolset(),
        PerformanceToolset()
    ]
)

# Remove direct tool imports - now handled via toolsets
# OLD: from ..tools.config_tools import analyze_corpus_statistics
# NEW: Tools available via @domain_agent.tool decorators in toolsets
```

**2. Knowledge Extraction Agent Update**
```python
# agents/knowledge_extraction/agent.py
knowledge_agent = Agent(
    'azure-openai:gpt-4',
    deps_type=ExtractionDeps,
    toolsets=[
        KnowledgeExtractionToolset(),
        AzureServiceToolset(),
        PerformanceToolset()
    ]
)
```

**3. Universal Search Agent Update**
```python
# agents/universal_search/agent.py
search_agent = Agent(
    'azure-openai:gpt-4',
    deps_type=SearchDeps,
    toolsets=[
        UniversalSearchToolset(),
        AzureServiceToolset(),
        PerformanceToolset()
    ]
)
```

### Phase 1.4: Import Updates & Migration (Day 6)

#### Update All Import Statements

**1. API Layer Updates**
```python
# api/endpoints/queries.py
# OLD: from agents.tools.search_tools import execute_tri_modal_search
# NEW: Tools automatically available via agent.run() with toolsets

# api/endpoints/health.py
# OLD: from agents.tools.consolidated_tools import check_service_health
# NEW: Available via AzureServiceToolset on all agents
```

**2. Service Layer Updates**
```python
# services/agent_service.py
# OLD: from agents.tools.extraction_tools import validate_extraction_quality
# NEW: Called via knowledge_agent.run() with integrated toolsets

# services/query_service.py
# OLD: from agents.tools.search_tools import synthesize_search_results
# NEW: Called via search_agent.run() with integrated toolsets
```

**3. Orchestration Layer Updates**
```python
# agents/orchestration/config_extraction_orchestrator.py
# OLD: from ..tools.config_tools import create_extraction_config
# NEW: await domain_agent.run("create_extraction_config", deps=deps)

# agents/orchestration/unified_orchestrator.py
# OLD: from ..tools.search_tools import execute_vector_search
# NEW: await search_agent.run("execute_vector_search", deps=deps)
```

### Phase 1.5: Testing & Validation (Day 7)

#### Comprehensive Validation Suite

**1. Tool Functionality Tests**
```python
# tests/phase1/test_tool_colocation.py
class TestToolColocation:
    async def test_domain_intelligence_toolset(self):
        """Validate domain intelligence tools work via toolset"""
        agent = domain_agent
        result = await agent.run(
            "analyze_corpus_statistics",
            corpus_path="test_data/sample_corpus",
            deps=test_domain_deps
        )
        assert result.statistical_patterns is not None

    async def test_knowledge_extraction_toolset(self):
        """Validate knowledge extraction tools work via toolset"""
        agent = knowledge_agent
        result = await agent.run(
            "extract_entities_multi_strategy",
            text="sample text",
            config=test_extraction_config,
            deps=test_extraction_deps
        )
        assert len(result.entities) > 0

    async def test_universal_search_toolset(self):
        """Validate universal search tools work via toolset"""
        agent = search_agent
        result = await agent.run(
            "execute_vector_search",
            query="test query",
            filters=test_filters,
            deps=test_search_deps
        )
        assert result.vector_results is not None
```

**2. Competitive Advantage Preservation Tests**
```python
# tests/phase1/test_competitive_advantages.py
class TestCompetitiveAdvantagePreservation:
    async def test_tri_modal_search_unity(self):
        """Ensure tri-modal search coordination remains functional"""
        result = await search_agent.run(
            "synthesize_search_results",
            tri_modal_results=sample_tri_modal_results,
            deps=test_deps
        )
        assert result.vector_score > 0
        assert result.graph_score > 0
        assert result.gnn_score > 0

    async def test_hybrid_domain_intelligence(self):
        """Ensure hybrid LLM+Statistical analysis preserved"""
        result = await domain_agent.run(
            "generate_semantic_patterns",
            content_sample=sample_content,
            deps=test_deps
        )
        assert result.statistical_patterns is not None
        assert result.semantic_patterns is not None

    async def test_config_extraction_workflow(self):
        """Ensure configuration-extraction pipeline preserved"""
        config_result = await domain_agent.run(
            "create_extraction_config",
            patterns=sample_patterns,
            deps=test_deps
        )

        extraction_result = await knowledge_agent.run(
            "extract_entities_multi_strategy",
            text=sample_text,
            config=config_result.output,
            deps=test_deps
        )

        assert extraction_result.entities is not None
```

**3. Performance Validation**
```python
# tests/phase1/test_performance_preservation.py
class TestPerformancePreservation:
    async def test_sub_3_second_response(self):
        """Validate sub-3-second response time maintained"""
        start_time = time.time()

        result = await search_agent.run(
            "execute_vector_search",
            query="performance test query",
            filters=standard_filters,
            deps=production_deps
        )

        execution_time = time.time() - start_time
        assert execution_time < 3.0, f"Response time {execution_time}s exceeds 3s SLA"

    async def test_azure_service_integration(self):
        """Validate Azure service integration preserved"""
        health_result = await domain_agent.run(
            "monitor_service_health",
            deps=production_deps
        )

        assert health_result.azure_openai_status == "healthy"
        assert health_result.azure_search_status == "healthy"
        assert health_result.azure_cosmos_status == "healthy"
```

## Risk Mitigation

### Rollback Strategy
1. **Immediate Rollback**: Keep original `agents/tools/` directory until Phase 1 validation complete
2. **Import Aliases**: Maintain backward compatibility during transition period
3. **Feature Flags**: Use feature flags to toggle between old and new tool organization
4. **Performance Monitoring**: Continuous monitoring during migration

### Error Handling
1. **Tool Migration Errors**: Fallback to original tool imports if toolset calls fail
2. **Dependency Injection Issues**: Validate all deps_type definitions work with new toolsets
3. **Azure Service Integration**: Ensure Azure service dependencies transfer correctly

### Validation Checkpoints
- [ ] Day 3: Toolset classes implemented and unit tested
- [ ] Day 5: Agent integration complete with toolsets
- [ ] Day 6: All imports updated and migration complete
- [ ] Day 7: Full validation suite passing

## Data-Driven Implementation Methods

### Required Data-Driven Helper Functions

**1. Statistical Analysis Methods**
```python
class DataDrivenConfigurationManager:
    """Generate all configuration parameters from actual operational data"""

    async def _discover_entity_types_from_patterns(self, statistical_patterns: Dict) -> List[str]:
        """Discover entity types from statistical corpus analysis"""
        # Use TF-IDF and clustering to identify entity-like tokens
        token_frequencies = statistical_patterns["token_frequencies"]
        tfidf_scores = self._calculate_tfidf(token_frequencies)

        # Cluster high-value tokens to identify entity types
        entity_clusters = self._cluster_by_statistical_properties(tfidf_scores)
        return self._extract_types_from_clusters(entity_clusters)

    async def _discover_relationship_types_from_patterns(self, semantic_patterns: Dict) -> List[str]:
        """Discover relationship types from semantic pattern analysis"""
        # Use dependency parsing and co-occurrence patterns
        dependency_patterns = semantic_patterns["dependency_patterns"]
        cooccurrence_matrix = semantic_patterns["cooccurrence_matrix"]

        # Analyze syntactic patterns to identify relationship types
        relationship_types = self._extract_relationships_from_dependencies(dependency_patterns)
        return self._validate_relationships_with_cooccurrence(relationship_types, cooccurrence_matrix)

    async def _calculate_confidence_threshold(self, patterns: Dict, threshold_type: str) -> float:
        """Calculate optimal confidence thresholds from validation data"""
        validation_data = patterns.get("validation_metrics", {})

        if threshold_type == "entity":
            precision_scores = validation_data.get("entity_precision_by_threshold", [])
            recall_scores = validation_data.get("entity_recall_by_threshold", [])
        else:  # relationship
            precision_scores = validation_data.get("relationship_precision_by_threshold", [])
            recall_scores = validation_data.get("relationship_recall_by_threshold", [])

        # Calculate F1 scores and find optimal threshold
        f1_scores = [
            2 * (p * r) / (p + r) if (p + r) > 0 else 0
            for p, r in zip(precision_scores, recall_scores)
        ]

        optimal_idx = np.argmax(f1_scores)
        return validation_data["thresholds"][optimal_idx]
```

**2. Performance Analytics Methods**
```python
    async def _optimize_chunk_size_from_performance_data(self, corpus_statistics: Dict) -> int:
        """Learn optimal chunk size from performance analytics"""
        document_sizes = corpus_statistics["document_size_distribution"]
        processing_times = corpus_statistics["processing_time_by_chunk_size"]

        # Find chunk size that optimizes processing speed vs accuracy
        optimal_chunk_size = self._find_performance_optimal_chunk_size(document_sizes, processing_times)
        return max(1, min(optimal_chunk_size, 10))  # Reasonable bounds

    async def _get_learned_sla_target(self, domain: str) -> float:
        """Learn SLA targets from historical performance data"""
        historical_data = await self._load_performance_history(domain)

        if not historical_data:
            return 3.0  # Fallback to default if no data

        # Use 95th percentile of historical performance as SLA target
        response_times = historical_data["response_times"]
        return np.percentile(response_times, 95)
```

## Success Metrics

### Technical KPIs
- [ ] **100% tool migration** - All 28 tools successfully moved to agent-specific toolsets
- [ ] **Zero functionality regression** - All existing tool functionality preserved
- [ ] **Performance maintained** - Sub-3-second response time SLA preserved
- [ ] **Azure integration intact** - All Azure service tools functional via toolsets

### Architectural KPIs
- [ ] **PydanticAI compliance** - 100% compliance with official toolset patterns
- [ ] **Code organization** - Tools co-located with respective agents
- [ ] **Maintainability improvement** - Clearer agent boundaries and responsibilities
- [ ] **Import simplification** - Reduced import complexity across codebase

### Competitive Advantage KPIs
- [ ] **Tri-modal search unity** - Vector+Graph+GNN coordination preserved
- [ ] **Hybrid domain intelligence** - LLM+Statistical analysis functionality maintained
- [ ] **Configuration-extraction workflow** - Zero-config automation preserved
- [ ] **Enterprise features** - Cost tracking, evidence collection, streaming responses intact

### Data-Driven Compliance KPIs
- [ ] **Zero hardcoded entity types** - All entity types discovered from corpus analysis
- [ ] **Zero hardcoded confidence thresholds** - All thresholds calculated from validation data
- [ ] **Zero hardcoded relationship types** - All relationship types learned from semantic patterns
- [ ] **Mathematical foundation validation** - All parameters derived from statistical analysis
- [ ] **Universal design compliance** - No domain-specific hardcoded assumptions

## Next Steps

Upon successful completion of Phase 1:
1. **Phase 2 Preparation**: Begin Phase 2: Graph-Based Orchestration implementation
2. **Documentation Update**: Update all documentation to reflect new toolset organization
3. **Training Materials**: Create training materials for team on new toolset patterns
4. **Production Deployment**: Deploy Phase 1 changes to staging environment for validation

## Dependencies

### Prerequisites
- [ ] Azure services fully operational (✅ completed - 5/6 services connected)
- [ ] Test environment with sample data available
- [ ] Backup and rollback procedures established
- [ ] Team training on PydanticAI toolset patterns

### External Dependencies
- **Azure OpenAI**: Required for LLM-powered tools
- **Azure Cognitive Search**: Required for vector search tools
- **Azure Cosmos DB**: Required for graph storage tools
- **Azure ML**: Required for GNN prediction tools

## Communication Plan

### Stakeholder Updates
- **Daily**: Progress updates to development team
- **Mid-week**: Status report to project leadership
- **End of week**: Completion report with metrics and next steps

### Documentation
- **Implementation logs**: Daily logs of changes and decisions
- **Tool migration mapping**: Complete mapping of old → new tool locations
- **Performance benchmarks**: Before/after performance comparisons
- **Lessons learned**: Document insights for future phases
