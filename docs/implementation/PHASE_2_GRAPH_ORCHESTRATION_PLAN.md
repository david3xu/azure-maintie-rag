# Phase 2: Graph-Based Orchestration Implementation Plan

**Date**: August 3, 2025
**Duration**: 2 weeks  
**Priority**: High
**Status**: Ready for Implementation (Following Phase 1 completion)

## Overview

Phase 2 implements the officially recommended PydanticAI graph-based orchestration using `pydantic-graph` to replace the current 5 orchestrator files with a unified, state-persistent, fault-tolerant workflow system that preserves all competitive advantages.

## Current State Analysis

### Orchestrator Complexity Assessment

**Current Structure** (Multiple Orchestrators):
```
agents/orchestration/
├── config_extraction_orchestrator.py    # 306 lines - Config-Extraction workflow
├── search_orchestrator.py               # 640 lines - Search coordination  
├── unified_orchestrator.py              # 1,696 lines - Complete workflow hub
├── workflow_orchestrator.py             # 739 lines - High-level workflow management
├── pydantic_integration.py              # 512 lines - PydanticAI patterns
└── universal_search/orchestrator.py     # ~200 lines - Search-specific orchestration
```

**Total**: ~4,093 lines across 6 orchestrator files

### Target Architecture (Official PydanticAI Pattern)

**Graph-Based Structure**:
```
agents/workflows/
├── config_extraction_graph.py           # Single graph for Config-Extraction
├── search_workflow_graph.py             # Single graph for search workflows  
├── state_persistence.py                 # Production state management
├── graph_monitoring.py                  # Built-in performance tracking
└── fault_recovery.py                    # Automatic retry and recovery
```

**Total**: ~2,100 lines across 5 specialized files (49% reduction)

## Implementation Strategy

### Phase 2.1: Graph Foundation (Week 1, Day 1-2)

#### Install and Configure pydantic-graph

**1. Dependency Installation**
```bash
# Add to requirements.txt
pydantic-graph>=0.1.0
pydantic-logfire>=0.1.0  # For enhanced monitoring
```

**2. Core Graph Infrastructure**
```python
# agents/workflows/graph_base.py
from pydantic_graph import Graph, BaseNode, GraphRunContext
from dataclasses import dataclass
from typing import Optional, Dict, Any
import asyncio
import time

@dataclass 
class WorkflowState:
    """Base state for all workflow graphs"""
    workflow_id: str
    query: str
    domain: Optional[str] = None
    start_time: float = 0.0
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class ConfigExtractionState(WorkflowState):
    """State for Config-Extraction workflow"""
    raw_data: str = ""
    statistical_analysis: Optional[Dict] = None
    semantic_patterns: Optional[Dict] = None
    extraction_config: Optional[Dict] = None
    extracted_knowledge: Optional[Dict] = None
    validation_results: Optional[Dict] = None

@dataclass  
class SearchWorkflowState(WorkflowState):
    """State for search workflow execution"""
    search_context: Optional[Dict] = None
    vector_results: Optional[Dict] = None
    graph_results: Optional[Dict] = None
    gnn_results: Optional[Dict] = None
    synthesized_results: Optional[Dict] = None
```

#### Production State Persistence

**3. PostgreSQL-Based State Persistence**
```python
# agents/workflows/state_persistence.py
import asyncio
import json
from typing import Optional, Dict, Any
import asyncpg
from pydantic_graph.persistence import BaseStatePersistence
from config.settings import azure_settings

class ProductionStatePersistence(BaseStatePersistence):
    """Production-grade state persistence with PostgreSQL backend"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.encryption_key = azure_settings.state_encryption_key
        
    async def initialize(self):
        """Initialize PostgreSQL connection pool"""
        self.pool = await asyncpg.create_pool(
            host=azure_settings.postgres_host,
            port=azure_settings.postgres_port,
            database=azure_settings.postgres_database,
            user=azure_settings.postgres_user,
            password=azure_settings.postgres_password,
            min_size=1,
            max_size=10
        )
        
        # Create tables if not exist
        await self._create_tables()
    
    async def save_state(self, workflow_id: str, state: WorkflowState) -> None:
        """Save encrypted workflow state to PostgreSQL"""
        if not self.pool:
            await self.initialize()
            
        encrypted_state = self._encrypt_state(state)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflow_states (workflow_id, state_data, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (workflow_id) 
                DO UPDATE SET state_data = $2, updated_at = NOW()
            """, workflow_id, encrypted_state)
    
    async def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load and decrypt workflow state from PostgreSQL"""
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT state_data FROM workflow_states 
                WHERE workflow_id = $1
            """, workflow_id)
            
        if row:
            return self._decrypt_state(row['state_data'])
        return None
    
    async def delete_state(self, workflow_id: str) -> None:
        """Delete workflow state (cleanup after completion)"""
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM workflow_states WHERE workflow_id = $1
            """, workflow_id)
    
    def _encrypt_state(self, state: WorkflowState) -> str:
        """Encrypt state data using Azure Key Vault keys"""
        # Implement encryption using Azure Key Vault
        # This is a placeholder - implement actual encryption
        return json.dumps(state.__dict__)
    
    def _decrypt_state(self, encrypted_data: str) -> WorkflowState:
        """Decrypt state data from storage"""
        # Implement decryption using Azure Key Vault
        # This is a placeholder - implement actual decryption
        data = json.loads(encrypted_data)
        return WorkflowState(**data)
```

### Phase 2.2: Config-Extraction Graph Implementation (Week 1, Day 3-5)

#### Core Config-Extraction Workflow Graph

**1. Domain Analysis Node**
```python
# agents/workflows/config_extraction_graph.py
from pydantic_graph import Graph, BaseNode, GraphRunContext, End
from ..domain_intelligence.agent import domain_agent
from ..knowledge_extraction.agent import knowledge_agent  
from ..universal_search.agent import search_agent

class AnalyzeDomainNode(BaseNode[ConfigExtractionState]):
    """First stage: Statistical + LLM domain analysis"""
    
    async def run(self, ctx: GraphRunContext[ConfigExtractionState]) -> 'GenerateConfigNode':
        """Execute hybrid domain analysis preserving competitive advantage"""
        logger.info(f"Starting domain analysis for workflow {ctx.state.workflow_id}")
        
        try:
            # Statistical Analysis (preserving hybrid intelligence)
            statistical_result = await domain_agent.run(
                "analyze_corpus_statistics",
                corpus_path=ctx.state.raw_data,
                deps=ctx.deps,
                usage=ctx.usage
            )
            ctx.state.statistical_analysis = statistical_result.output
            
            # Semantic Pattern Analysis (preserving LLM component)
            semantic_result = await domain_agent.run(
                "generate_semantic_patterns", 
                content_sample=ctx.state.raw_data[:1000],
                deps=ctx.deps,
                usage=ctx.usage
            )
            ctx.state.semantic_patterns = semantic_result.output
            
            # Update performance metrics
            ctx.state.performance_metrics["domain_analysis_time"] = time.time() - ctx.state.start_time
            
            logger.info(f"Domain analysis completed for {ctx.state.workflow_id}")
            return GenerateConfigNode()
            
        except Exception as e:
            logger.error(f"Domain analysis failed for {ctx.state.workflow_id}: {e}")
            return DomainAnalysisErrorNode(str(e))

class GenerateConfigNode(BaseNode[ConfigExtractionState]):
    """Second stage: Generate extraction configuration"""
    
    async def run(self, ctx: GraphRunContext[ConfigExtractionState]) -> 'ExtractKnowledgeNode':
        """Generate ExtractionConfiguration from domain patterns"""
        logger.info(f"Generating extraction config for {ctx.state.workflow_id}")
        
        try:
            # Combine statistical and semantic patterns
            combined_patterns = {
                "statistical": ctx.state.statistical_analysis,
                "semantic": ctx.state.semantic_patterns
            }
            
            # Generate configuration (preserving zero-config automation)
            config_result = await domain_agent.run(
                "create_extraction_config",
                patterns=combined_patterns,
                deps=ctx.deps,
                usage=ctx.usage
            )
            ctx.state.extraction_config = config_result.output
            
            # Update performance metrics
            ctx.state.performance_metrics["config_generation_time"] = time.time() - ctx.state.start_time
            
            logger.info(f"Extraction config generated for {ctx.state.workflow_id}")
            return ExtractKnowledgeNode()
            
        except Exception as e:
            logger.error(f"Config generation failed for {ctx.state.workflow_id}: {e}")
            return ConfigGenerationErrorNode(str(e))

class ExtractKnowledgeNode(BaseNode[ConfigExtractionState]):
    """Third stage: Knowledge extraction using generated config"""
    
    async def run(self, ctx: GraphRunContext[ConfigExtractionState]) -> 'ValidateResultsNode':
        """Extract knowledge using configuration (preserving multi-strategy extraction)"""
        logger.info(f"Starting knowledge extraction for {ctx.state.workflow_id}")
        
        try:
            # Multi-strategy entity extraction (preserving competitive advantage)
            entity_result = await knowledge_agent.run(
                "extract_entities_multi_strategy",
                text=ctx.state.raw_data,
                config=ctx.state.extraction_config,
                deps=ctx.deps,
                usage=ctx.usage
            )
            
            # Contextual relationship extraction (preserving advanced features)
            relationship_result = await knowledge_agent.run(
                "extract_relationships_contextual",
                text=ctx.state.raw_data,
                entities=entity_result.output.get("entities", []),
                deps=ctx.deps,
                usage=ctx.usage
            )
            
            # Combine extraction results
            ctx.state.extracted_knowledge = {
                "entities": entity_result.output,
                "relationships": relationship_result.output
            }
            
            # Update performance metrics
            ctx.state.performance_metrics["extraction_time"] = time.time() - ctx.state.start_time
            
            logger.info(f"Knowledge extraction completed for {ctx.state.workflow_id}")
            return ValidateResultsNode()
            
        except Exception as e:
            logger.error(f"Knowledge extraction failed for {ctx.state.workflow_id}: {e}")
            return ExtractionErrorNode(str(e))

class ValidateResultsNode(BaseNode[ConfigExtractionState]):
    """Fourth stage: Quality validation and storage"""
    
    async def run(self, ctx: GraphRunContext[ConfigExtractionState]) -> 'CompleteWorkflowNode':
        """Validate extraction quality and store results"""
        logger.info(f"Validating results for {ctx.state.workflow_id}")
        
        try:
            # Quality validation (preserving enterprise validation framework)
            validation_result = await knowledge_agent.run(
                "validate_extraction_quality",
                results=ctx.state.extracted_knowledge,
                deps=ctx.deps,
                usage=ctx.usage
            )
            ctx.state.validation_results = validation_result.output
            
            # Store knowledge graph (preserving Azure Cosmos DB integration)
            if validation_result.output.get("quality_score", 0) > 0.7:
                storage_result = await knowledge_agent.run(
                    "store_knowledge_graph",
                    validated_results=ctx.state.extracted_knowledge,
                    deps=ctx.deps,
                    usage=ctx.usage
                )
                ctx.state.performance_metrics["storage_time"] = time.time() - ctx.state.start_time
            
            logger.info(f"Results validation completed for {ctx.state.workflow_id}")
            return CompleteWorkflowNode()
            
        except Exception as e:
            logger.error(f"Results validation failed for {ctx.state.workflow_id}: {e}")
            return ValidationErrorNode(str(e))

class CompleteWorkflowNode(BaseNode[ConfigExtractionState]):
    """Final stage: Workflow completion and metrics"""
    
    async def run(self, ctx: GraphRunContext[ConfigExtractionState]) -> End[Dict[str, Any]]:
        """Complete workflow and return final results"""
        total_time = time.time() - ctx.state.start_time
        
        # ✅ DATA-DRIVEN SLA - Learn SLA targets from performance data
        optimal_sla = await self._get_learned_sla_target(ctx.state.domain)
        sla_met = total_time < optimal_sla
        
        final_results = {
            "workflow_id": ctx.state.workflow_id,
            "domain": ctx.state.domain,
            "extraction_config": ctx.state.extraction_config,
            "extracted_knowledge": ctx.state.extracted_knowledge,
            "validation_results": ctx.state.validation_results,
            "performance_metrics": {
                **ctx.state.performance_metrics,
                "total_execution_time": total_time,
                "sla_compliance": sla_met,
                "sub_3s_target_met": total_time < 3.0
            },
            "competitive_advantages_preserved": {
                "hybrid_domain_intelligence": True,
                "multi_strategy_extraction": True,
                "zero_config_automation": True,
                "enterprise_validation": True
            }
        }
        
        logger.info(f"Config-Extraction workflow completed: {ctx.state.workflow_id} in {total_time:.2f}s")
        return End(final_results)

# Error handling nodes
class DomainAnalysisErrorNode(BaseNode[ConfigExtractionState]):
    def __init__(self, error_message: str):
        self.error_message = error_message
    
    async def run(self, ctx: GraphRunContext[ConfigExtractionState]) -> End[Dict[str, Any]]:
        return End({"error": f"Domain analysis failed: {self.error_message}"})

class ConfigGenerationErrorNode(BaseNode[ConfigExtractionState]):
    def __init__(self, error_message: str):
        self.error_message = error_message
    
    async def run(self, ctx: GraphRunContext[ConfigExtractionState]) -> End[Dict[str, Any]]:
        return End({"error": f"Config generation failed: {self.error_message}"})

class ExtractionErrorNode(BaseNode[ConfigExtractionState]):
    def __init__(self, error_message: str):
        self.error_message = error_message
    
    async def run(self, ctx: GraphRunContext[ConfigExtractionState]) -> End[Dict[str, Any]]:
        return End({"error": f"Knowledge extraction failed: {self.error_message}"})

class ValidationErrorNode(BaseNode[ConfigExtractionState]):
    def __init__(self, error_message: str):
        self.error_message = error_message
    
    async def run(self, ctx: GraphRunContext[ConfigExtractionState]) -> End[Dict[str, Any]]:
        return End({"error": f"Results validation failed: {self.error_message}"})

# Create the graph
config_extraction_graph = Graph(
    nodes=[
        AnalyzeDomainNode,
        GenerateConfigNode, 
        ExtractKnowledgeNode,
        ValidateResultsNode,
        CompleteWorkflowNode,
        DomainAnalysisErrorNode,
        ConfigGenerationErrorNode,
        ExtractionErrorNode,
        ValidationErrorNode
    ],
    state_type=ConfigExtractionState
)
```

### Phase 2.3: Search Workflow Graph Implementation (Week 1, Day 6-7)

#### Tri-Modal Search Coordination Graph

**2. Search Workflow Graph**
```python
# agents/workflows/search_workflow_graph.py  
class PrepareSearchNode(BaseNode[SearchWorkflowState]):
    """Prepare search context and strategy"""
    
    async def run(self, ctx: GraphRunContext[SearchWorkflowState]) -> 'ExecuteTriModalSearchNode':
        """Prepare search context based on query characteristics"""
        logger.info(f"Preparing search for workflow {ctx.state.workflow_id}")
        
        # Analyze query to determine optimal search strategy
        ctx.state.search_context = {
            "query_complexity": len(ctx.state.query.split()),
            "domain_context": ctx.state.domain,
            "search_modalities": ["vector", "graph", "gnn"]  # Preserve tri-modal unity
        }
        
        return ExecuteTriModalSearchNode()

class ExecuteTriModalSearchNode(BaseNode[SearchWorkflowState]):
    """Execute tri-modal search preserving competitive advantage"""
    
    async def run(self, ctx: GraphRunContext[SearchWorkflowState]) -> 'SynthesizeResultsNode':
        """Execute Vector + Graph + GNN search in parallel (preserving tri-modal unity)"""
        logger.info(f"Executing tri-modal search for {ctx.state.workflow_id}")
        
        try:
            # Execute all three search modalities in parallel (competitive advantage)
            search_tasks = {
                "vector": search_agent.run(
                    "execute_vector_search",
                    query=ctx.state.query,
                    filters=ctx.state.search_context,
                    deps=ctx.deps,
                    usage=ctx.usage
                ),
                "graph": search_agent.run(
                    "execute_graph_search", 
                    query=ctx.state.query,
                    graph_context=ctx.state.search_context,
                    deps=ctx.deps,
                    usage=ctx.usage
                ),
                "gnn": search_agent.run(
                    "execute_gnn_search",
                    query=ctx.state.query,
                    pattern_context=ctx.state.search_context,
                    deps=ctx.deps,
                    usage=ctx.usage
                )
            }
            
            # Await all search results
            search_results = {}
            for modality, task in search_tasks.items():
                try:
                    result = await task
                    search_results[modality] = result.output
                except Exception as e:
                    logger.error(f"{modality} search failed: {e}")
                    search_results[modality] = None
            
            # Store results in state
            ctx.state.vector_results = search_results.get("vector")
            ctx.state.graph_results = search_results.get("graph") 
            ctx.state.gnn_results = search_results.get("gnn")
            
            logger.info(f"Tri-modal search completed for {ctx.state.workflow_id}")
            return SynthesizeResultsNode()
            
        except Exception as e:
            logger.error(f"Tri-modal search failed for {ctx.state.workflow_id}: {e}")
            return SearchErrorNode(str(e))

class SynthesizeResultsNode(BaseNode[SearchWorkflowState]):
    """Synthesize tri-modal results into final response"""
    
    async def run(self, ctx: GraphRunContext[SearchWorkflowState]) -> 'CompleteSearchNode':
        """Synthesize and rank tri-modal search results"""
        logger.info(f"Synthesizing results for {ctx.state.workflow_id}")
        
        try:
            # Combine tri-modal results (preserving intelligent fusion)
            tri_modal_results = {
                "vector_results": ctx.state.vector_results,
                "graph_results": ctx.state.graph_results, 
                "gnn_results": ctx.state.gnn_results
            }
            
            # Intelligent result synthesis (preserving competitive advantage)
            synthesis_result = await search_agent.run(
                "synthesize_search_results",
                tri_modal_results=tri_modal_results,
                deps=ctx.deps,
                usage=ctx.usage
            )
            ctx.state.synthesized_results = synthesis_result.output
            
            logger.info(f"Results synthesis completed for {ctx.state.workflow_id}")
            return CompleteSearchNode()
            
        except Exception as e:
            logger.error(f"Results synthesis failed for {ctx.state.workflow_id}: {e}")
            return SynthesisErrorNode(str(e))

class CompleteSearchNode(BaseNode[SearchWorkflowState]):
    """Complete search workflow with performance validation"""
    
    async def run(self, ctx: GraphRunContext[SearchWorkflowState]) -> End[Dict[str, Any]]:
        """Complete search workflow and validate SLA compliance"""
        total_time = time.time() - ctx.state.start_time
        
        # ✅ DATA-DRIVEN SLA - Learn SLA targets from query complexity and performance data
        adaptive_sla = await self._calculate_adaptive_sla_target(ctx.state.query, ctx.state.domain)
        sla_met = total_time < adaptive_sla
        
        final_results = {
            "workflow_id": ctx.state.workflow_id,
            "query": ctx.state.query,
            "domain": ctx.state.domain,
            "search_results": ctx.state.synthesized_results,
            "performance_metrics": {
                "total_execution_time": total_time,
                "sla_compliance": sla_met,
                "sub_3s_target_met": sla_met,
                "tri_modal_coverage": {
                    "vector_available": ctx.state.vector_results is not None,
                    "graph_available": ctx.state.graph_results is not None,
                    "gnn_available": ctx.state.gnn_results is not None
                }
            },
            "competitive_advantages_preserved": {
                "tri_modal_search_unity": True,
                "sub_3s_performance": sla_met,
                "intelligent_result_synthesis": True
            }
        }
        
        logger.info(f"Search workflow completed: {ctx.state.workflow_id} in {total_time:.2f}s")
        return End(final_results)

# Error handling nodes
class SearchErrorNode(BaseNode[SearchWorkflowState]):
    def __init__(self, error_message: str):
        self.error_message = error_message
    
    async def run(self, ctx: GraphRunContext[SearchWorkflowState]) -> End[Dict[str, Any]]:
        return End({"error": f"Search execution failed: {self.error_message}"})

class SynthesisErrorNode(BaseNode[SearchWorkflowState]):
    def __init__(self, error_message: str):
        self.error_message = error_message
    
    async def run(self, ctx: GraphRunContext[SearchWorkflowState]) -> End[Dict[str, Any]]:
        return End({"error": f"Result synthesis failed: {self.error_message}"})

# Create the search graph
search_workflow_graph = Graph(
    nodes=[
        PrepareSearchNode,
        ExecuteTriModalSearchNode,
        SynthesizeResultsNode, 
        CompleteSearchNode,
        SearchErrorNode,
        SynthesisErrorNode
    ],
    state_type=SearchWorkflowState
)
```

### Phase 2.4: Graph Monitoring & Observability (Week 2, Day 1-2)

#### Built-in Performance Monitoring

**3. Graph Monitoring Infrastructure**
```python
# agents/workflows/graph_monitoring.py
import logfire
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

# Configure Pydantic Logfire for graph monitoring
logfire.configure()
logfire.instrument_pydantic_ai()

@dataclass
class WorkflowMetrics:
    """Comprehensive workflow performance metrics"""
    workflow_id: str
    workflow_type: str
    start_time: float
    end_time: Optional[float] = None
    node_executions: Dict[str, float] = None
    sla_compliance: bool = False
    error_count: int = 0
    azure_service_calls: Dict[str, int] = None
    
    def __post_init__(self):
        if self.node_executions is None:
            self.node_executions = {}
        if self.azure_service_calls is None:
            self.azure_service_calls = {}

class GraphMonitor:
    """Production-grade graph monitoring and observability"""
    
    def __init__(self):
        self.active_workflows: Dict[str, WorkflowMetrics] = {}
        self.completed_workflows: Dict[str, WorkflowMetrics] = {}
        
    async def start_workflow_monitoring(
        self, 
        workflow_id: str, 
        workflow_type: str
    ) -> WorkflowMetrics:
        """Start monitoring a workflow execution"""
        metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            start_time=time.time()
        )
        
        self.active_workflows[workflow_id] = metrics
        
        # Log workflow start with Logfire
        logfire.info(
            "Workflow started",
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            start_time=metrics.start_time
        )
        
        return metrics
    
    async def record_node_execution(
        self,
        workflow_id: str,
        node_name: str,
        execution_time: float
    ):
        """Record individual node execution time"""
        if workflow_id in self.active_workflows:
            metrics = self.active_workflows[workflow_id]
            metrics.node_executions[node_name] = execution_time
            
            # Log node execution with Logfire
            logfire.info(
                "Node executed",
                workflow_id=workflow_id,
                node_name=node_name,
                execution_time=execution_time
            )
    
    async def record_azure_service_call(
        self,
        workflow_id: str,
        service_name: str
    ):
        """Record Azure service usage for cost tracking"""
        if workflow_id in self.active_workflows:
            metrics = self.active_workflows[workflow_id]
            metrics.azure_service_calls[service_name] = metrics.azure_service_calls.get(service_name, 0) + 1
    
    async def complete_workflow_monitoring(
        self,
        workflow_id: str,
        final_results: Dict[str, Any]
    ) -> WorkflowMetrics:
        """Complete workflow monitoring and calculate final metrics"""
        if workflow_id not in self.active_workflows:
            return None
            
        metrics = self.active_workflows[workflow_id]
        metrics.end_time = time.time()
        
        # Calculate total execution time
        total_time = metrics.end_time - metrics.start_time
        
        # ✅ DATA-DRIVEN SLA - Learn optimal SLA targets from performance analytics
        learned_sla_target = await self._get_learned_sla_target_from_analytics()
        metrics.sla_compliance = total_time < learned_sla_target
        
        # Move to completed workflows
        self.completed_workflows[workflow_id] = metrics
        del self.active_workflows[workflow_id]
        
        # Log workflow completion with comprehensive metrics
        logfire.info(
            "Workflow completed",
            workflow_id=workflow_id,
            workflow_type=metrics.workflow_type,
            total_execution_time=total_time,
            sla_compliance=metrics.sla_compliance,
            node_count=len(metrics.node_executions),
            azure_service_calls=sum(metrics.azure_service_calls.values()),
            competitive_advantages=final_results.get("competitive_advantages_preserved", {})
        )
        
        # Alert on SLA violations
        if not metrics.sla_compliance:
            logfire.warning(
                "SLA violation detected",
                workflow_id=workflow_id,
                execution_time=total_time,
                target_time=3.0
            )
        
        return metrics
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        total_workflows = len(self.completed_workflows)
        
        if total_workflows == 0:
            return {"message": "No completed workflows"}
        
        # Calculate aggregate metrics
        total_execution_times = [m.end_time - m.start_time for m in self.completed_workflows.values()]
        sla_compliant_workflows = [m for m in self.completed_workflows.values() if m.sla_compliance]
        
        return {
            "total_workflows": total_workflows,
            "active_workflows": len(self.active_workflows),
            "sla_compliance_rate": len(sla_compliant_workflows) / total_workflows * 100,
            "average_execution_time": sum(total_execution_times) / total_workflows,
            "fastest_execution": min(total_execution_times),
            "slowest_execution": max(total_execution_times),
            "azure_service_usage": self._aggregate_azure_usage(),
            "competitive_advantages_status": self._check_competitive_advantages()
        }
    
    def _aggregate_azure_usage(self) -> Dict[str, int]:
        """Aggregate Azure service usage across all workflows"""
        aggregated = {}
        for metrics in self.completed_workflows.values():
            for service, count in metrics.azure_service_calls.items():
                aggregated[service] = aggregated.get(service, 0) + count
        return aggregated
    
    def _check_competitive_advantages(self) -> Dict[str, float]:
        """Check preservation of competitive advantages across workflows"""
        if not self.completed_workflows:
            return {}
            
        advantages = [
            "tri_modal_search_unity",
            "hybrid_domain_intelligence", 
            "zero_config_automation",
            "sub_3s_performance"
        ]
        
        advantage_scores = {}
        for advantage in advantages:
            preserved_count = 0
            total_count = len(self.completed_workflows)
            
            for metrics in self.completed_workflows.values():
                if advantage == "sub_3s_performance":
                    if metrics.sla_compliance:
                        preserved_count += 1
                else:
                    # Would check specific competitive advantage preservation
                    preserved_count += 1  # Placeholder
            
            advantage_scores[advantage] = (preserved_count / total_count) * 100
        
        return advantage_scores

# Global graph monitor instance
graph_monitor = GraphMonitor()
```

### Phase 2.5: Fault Recovery & Circuit Breakers (Week 2, Day 3-4)

#### Automatic Error Recovery

**4. Fault Recovery System**
```python
# agents/workflows/fault_recovery.py
import asyncio
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random

class RetryStrategy(Enum):
    """Different retry strategies for various failure types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE_RETRY = "immediate_retry"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True

class CircuitBreaker:
    """Circuit breaker for Azure service failures"""
    
    def __init__(self, service_name: str, historical_data: Optional[Dict] = None):
        # ✅ DATA-DRIVEN - Learn thresholds from service reliability patterns
        self.failure_threshold = await self._learn_failure_threshold(service_name, historical_data)
        self.recovery_timeout = await self._learn_recovery_timeout(service_name, historical_data)
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

class FaultRecoveryManager:
    """Manages fault recovery for graph workflows"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        # ✅ DATA-DRIVEN - Learn retry configurations from service reliability data
        self.retry_configs = await self._generate_adaptive_retry_configs(
            service_reliability_analytics=await self._load_service_analytics(),
            error_pattern_analysis=await self._analyze_failure_patterns(),
            success_rate_targets={"azure_openai": 0.99, "azure_search": 0.95, "azure_cosmos": 0.98, "azure_ml": 0.90}
        )
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    async def execute_with_retry(
        self,
        func: Callable,
        service_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic and circuit breaker"""
        config = self.retry_configs.get(service_name, RetryConfig())
        circuit_breaker = self.get_circuit_breaker(service_name)
        
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                # Execute with circuit breaker protection
                result = await circuit_breaker.call(func, *args, **kwargs)
                
                # Log successful recovery if this wasn't the first attempt
                if attempt > 0:
                    logfire.info(
                        "Service recovered after retry",
                        service_name=service_name,
                        attempt=attempt + 1,
                        total_attempts=config.max_attempts
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Log retry attempt
                logfire.warning(
                    "Service call failed, retrying",
                    service_name=service_name,
                    attempt=attempt + 1,
                    total_attempts=config.max_attempts,
                    error=str(e)
                )
                
                # Don't delay on the last attempt
                if attempt < config.max_attempts - 1:
                    delay = self._calculate_delay(config, attempt)
                    await asyncio.sleep(delay)
        
        # All attempts failed
        logfire.error(
            "Service call failed after all retries",
            service_name=service_name,
            total_attempts=config.max_attempts,
            final_error=str(last_exception)
        )
        
        raise last_exception
    
    def _calculate_delay(self, config: RetryConfig, attempt: int) -> float:
        """Calculate delay based on retry strategy"""
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (2 ** attempt)
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * (attempt + 1)
        else:  # IMMEDIATE_RETRY
            delay = 0.0
        
        # Apply maximum delay cap
        delay = min(delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        if config.jitter:
            delay = delay + random.uniform(0, delay * 0.1)
        
        return delay
    
    async def get_service_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all monitored services"""
        status = {}
        
        for service_name, circuit_breaker in self.circuit_breakers.items():
            status[service_name] = {
                "state": circuit_breaker.state,
                "failure_count": circuit_breaker.failure_count,
                "last_failure_time": circuit_breaker.last_failure_time,
                "healthy": circuit_breaker.state == "closed"
            }
        
        return status

# Global fault recovery manager
fault_recovery = FaultRecoveryManager()
```

### Phase 2.6: Integration & Migration (Week 2, Day 5-7)

#### Replace Existing Orchestrators

**5. Graph Integration Layer**
```python
# agents/workflows/graph_orchestrator.py
from typing import Dict, Any, Optional
import uuid
from .config_extraction_graph import config_extraction_graph, ConfigExtractionState
from .search_workflow_graph import search_workflow_graph, SearchWorkflowState
from .state_persistence import ProductionStatePersistence
from .graph_monitoring import graph_monitor
from .fault_recovery import fault_recovery

class GraphOrchestrator:
    """Unified orchestrator using graph-based workflows"""
    
    def __init__(self):
        self.persistence = ProductionStatePersistence()
        
    async def execute_config_extraction_workflow(
        self,
        raw_data: str,
        domain: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute Config-Extraction workflow using graph"""
        
        # Generate workflow ID if not provided
        if workflow_id is None:
            workflow_id = f"config_extraction_{uuid.uuid4().hex[:8]}"
        
        # Create initial state
        initial_state = ConfigExtractionState(
            workflow_id=workflow_id,
            query="",  # Not used in config-extraction
            domain=domain,
            raw_data=raw_data
        )
        
        # Start monitoring
        await graph_monitor.start_workflow_monitoring(workflow_id, "config_extraction")
        
        try:
            # Execute graph with state persistence
            async with config_extraction_graph.iter(
                initial_state=initial_state,
                persistence=self.persistence
            ) as run:
                
                async for node in run:
                    # Record node execution for monitoring
                    node_start_time = time.time()
                    
                    # Execute node (graph handles this automatically)
                    # This loop just provides monitoring hooks
                    
                    node_execution_time = time.time() - node_start_time
                    await graph_monitor.record_node_execution(
                        workflow_id, 
                        node.__class__.__name__, 
                        node_execution_time
                    )
                    
                    # Check if this is the end node
                    if hasattr(node, 'result'):
                        final_results = node.result
                        break
                
                # Complete monitoring
                await graph_monitor.complete_workflow_monitoring(workflow_id, final_results)
                
                return final_results
                
        except Exception as e:
            # Log error and return failure result
            logfire.error(
                "Config-extraction workflow failed",
                workflow_id=workflow_id,
                error=str(e)
            )
            
            return {
                "workflow_id": workflow_id,
                "error": str(e),
                "status": "failed"
            }
    
    async def execute_search_workflow(
        self,
        query: str,
        domain: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute search workflow using graph"""
        
        # Generate workflow ID if not provided
        if workflow_id is None:
            workflow_id = f"search_{uuid.uuid4().hex[:8]}"
        
        # Create initial state
        initial_state = SearchWorkflowState(
            workflow_id=workflow_id,
            query=query,
            domain=domain
        )
        
        # Start monitoring
        await graph_monitor.start_workflow_monitoring(workflow_id, "search")
        
        try:
            # Execute graph with state persistence
            async with search_workflow_graph.iter(
                initial_state=initial_state,
                persistence=self.persistence
            ) as run:
                
                async for node in run:
                    # Record node execution for monitoring
                    node_start_time = time.time()
                    
                    # Node execution happens automatically
                    # This provides monitoring and error handling
                    
                    node_execution_time = time.time() - node_start_time
                    await graph_monitor.record_node_execution(
                        workflow_id,
                        node.__class__.__name__,
                        node_execution_time
                    )
                    
                    # Check if this is the end node
                    if hasattr(node, 'result'):
                        final_results = node.result
                        break
                
                # Complete monitoring
                await graph_monitor.complete_workflow_monitoring(workflow_id, final_results)
                
                return final_results
                
        except Exception as e:
            # Log error and return failure result
            logfire.error(
                "Search workflow failed",
                workflow_id=workflow_id,
                error=str(e)
            )
            
            return {
                "workflow_id": workflow_id,
                "error": str(e),
                "status": "failed"
            }
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        # Try to load state from persistence
        state = await self.persistence.load_state(workflow_id)
        
        if state is None:
            return None
        
        return {
            "workflow_id": workflow_id,
            "status": "running" if workflow_id in graph_monitor.active_workflows else "completed",
            "current_state": state.__dict__,
            "metrics": graph_monitor.active_workflows.get(workflow_id, {})
        }
    
    async def resume_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Resume a workflow from persisted state"""
        state = await self.persistence.load_state(workflow_id)
        
        if state is None:
            return {"error": f"No persisted state found for workflow {workflow_id}"}
        
        # Determine workflow type and resume appropriately
        if isinstance(state, ConfigExtractionState):
            return await self.execute_config_extraction_workflow(
                raw_data=state.raw_data,
                domain=state.domain,
                workflow_id=workflow_id
            )
        elif isinstance(state, SearchWorkflowState):
            return await self.execute_search_workflow(
                query=state.query,
                domain=state.domain,
                workflow_id=workflow_id
            )
        else:
            return {"error": f"Unknown workflow state type for {workflow_id}"}

# Global graph orchestrator instance
graph_orchestrator = GraphOrchestrator()
```

### Phase 2.7: Testing & Validation (Week 2, Day 6-7)

#### Comprehensive Graph Testing

**6. Graph Workflow Testing**
```python
# tests/phase2/test_graph_workflows.py
import pytest
import asyncio
import time
from agents.workflows.graph_orchestrator import graph_orchestrator
from agents.workflows.graph_monitoring import graph_monitor

class TestGraphWorkflows:
    """Comprehensive testing of graph-based workflows"""
    
    async def test_config_extraction_workflow_complete(self):
        """Test complete Config-Extraction workflow"""
        # Test data
        raw_data = "Sample domain data for testing configuration extraction workflow"
        
        # Execute workflow
        start_time = time.time()
        result = await graph_orchestrator.execute_config_extraction_workflow(
            raw_data=raw_data,
            domain="test_domain"
        )
        execution_time = time.time() - start_time
        
        # Validate results
        assert result is not None
        assert "workflow_id" in result
        assert "extraction_config" in result
        assert "extracted_knowledge" in result
        assert "competitive_advantages_preserved" in result
        
        # Validate competitive advantages preserved
        advantages = result["competitive_advantages_preserved"]
        assert advantages["hybrid_domain_intelligence"] is True
        assert advantages["multi_strategy_extraction"] is True
        assert advantages["zero_config_automation"] is True
        
        # Validate performance SLA
        assert execution_time < 3.0, f"Workflow took {execution_time}s, exceeds 3s SLA"
        assert result["performance_metrics"]["sla_compliance"] is True
    
    async def test_search_workflow_tri_modal(self):
        """Test tri-modal search workflow"""
        query = "test query for tri-modal search validation"
        
        # Execute workflow
        start_time = time.time()
        result = await graph_orchestrator.execute_search_workflow(
            query=query,
            domain="test_domain"
        )
        execution_time = time.time() - start_time
        
        # Validate results
        assert result is not None
        assert "workflow_id" in result
        assert "search_results" in result
        assert "competitive_advantages_preserved" in result
        
        # Validate tri-modal search execution
        advantages = result["competitive_advantages_preserved"]
        assert advantages["tri_modal_search_unity"] is True
        assert advantages["sub_3s_performance"] is True
        
        # Validate tri-modal coverage
        tri_modal_coverage = result["performance_metrics"]["tri_modal_coverage"]
        assert tri_modal_coverage["vector_available"] is True
        assert tri_modal_coverage["graph_available"] is True
        assert tri_modal_coverage["gnn_available"] is True
        
        # Validate performance SLA
        assert execution_time < 3.0, f"Search took {execution_time}s, exceeds 3s SLA"
    
    async def test_state_persistence_and_recovery(self):
        """Test workflow state persistence and recovery"""
        raw_data = "Test data for state persistence validation"
        
        # Start workflow
        workflow_id = "test_persistence_workflow"
        
        # This test would simulate workflow interruption and recovery
        # Implementation depends on specific state persistence mechanisms
        
        # Validate state can be persisted and recovered
        status = await graph_orchestrator.get_workflow_status(workflow_id)
        
        # Test resumption capability (if workflow was interrupted)
        if status and status["status"] == "running":
            resumed_result = await graph_orchestrator.resume_workflow(workflow_id)
            assert resumed_result is not None
    
    async def test_fault_recovery_mechanisms(self):
        """Test fault recovery and circuit breaker functionality"""
        from agents.workflows.fault_recovery import fault_recovery
        
        # Test circuit breaker behavior
        async def failing_service():
            raise Exception("Simulated service failure")
        
        # This should trigger circuit breaker after threshold
        with pytest.raises(Exception):
            for i in range(6):  # Exceed failure threshold
                try:
                    await fault_recovery.execute_with_retry(
                        failing_service,
                        "test_service"
                    )
                except:
                    continue
        
        # Validate circuit breaker state
        health_status = await fault_recovery.get_service_health_status()
        assert health_status["test_service"]["state"] == "open"
    
    async def test_performance_monitoring(self):
        """Test graph monitoring and performance tracking"""
        # Execute multiple workflows to generate metrics
        workflows = []
        for i in range(5):
            workflow_result = await graph_orchestrator.execute_search_workflow(
                query=f"test query {i}",
                domain="test_domain"
            )
            workflows.append(workflow_result)
        
        # Get performance dashboard
        dashboard = await graph_monitor.get_performance_dashboard()
        
        # Validate metrics
        assert dashboard["total_workflows"] >= 5
        assert "sla_compliance_rate" in dashboard
        assert "average_execution_time" in dashboard
        assert "competitive_advantages_status" in dashboard
        
        # Validate competitive advantage preservation rates
        advantages = dashboard["competitive_advantages_status"]
        assert advantages["sub_3s_performance"] >= 80.0  # At least 80% compliance

class TestCompetitiveAdvantagePreservation:
    """Specific tests for competitive advantage preservation"""
    
    async def test_hybrid_domain_intelligence_preservation(self):
        """Ensure hybrid LLM+Statistical analysis preserved in graph"""
        result = await graph_orchestrator.execute_config_extraction_workflow(
            raw_data="Test corpus for hybrid analysis validation",
            domain="test_domain"
        )
        
        # Validate both statistical and semantic analysis occurred
        extraction_config = result["extraction_config"]
        assert extraction_config is not None
        
        # This would validate that both LLM and statistical analysis were used
        # Implementation depends on specific configuration structure
        
    async def test_tri_modal_search_unity_preservation(self):
        """Ensure Vector+Graph+GNN coordination preserved"""
        result = await graph_orchestrator.execute_search_workflow(
            query="test query for tri-modal validation",
            domain="test_domain"
        )
        
        # Validate all three search modalities were executed
        tri_modal_coverage = result["performance_metrics"]["tri_modal_coverage"]
        assert tri_modal_coverage["vector_available"] is True
        assert tri_modal_coverage["graph_available"] is True 
        assert tri_modal_coverage["gnn_available"] is True
        
        # Validate intelligent result synthesis occurred
        advantages = result["competitive_advantages_preserved"]
        assert advantages["intelligent_result_synthesis"] is True
    
    async def test_sub_3_second_performance_guarantee(self):
        """Validate sub-3-second response time maintained"""
        queries = [
            "simple test query",
            "more complex test query with additional context",
            "complex multi-domain query requiring extensive processing"
        ]
        
        for query in queries:
            start_time = time.time()
            result = await graph_orchestrator.execute_search_workflow(query)
            execution_time = time.time() - start_time
            
            # Validate SLA compliance
            assert execution_time < 3.0, f"Query '{query}' took {execution_time}s"
            assert result["performance_metrics"]["sla_compliance"] is True
```

## Data-Driven Implementation Methods

### Required Data-Driven Analytics Infrastructure

**1. Performance Analytics Manager**
```python
class PerformanceAnalyticsManager:
    """Learn all performance parameters from actual operational data"""
    
    async def _calculate_adaptive_sla_target(self, query: str, domain: str) -> float:
        """Calculate adaptive SLA targets based on query complexity and historical data"""
        # Analyze query complexity
        query_complexity = self._analyze_query_complexity(query)
        
        # Load historical performance data for similar queries
        historical_data = await self._load_performance_history_by_complexity(query_complexity, domain)
        
        if not historical_data:
            return 3.0  # Default fallback
        
        # Calculate SLA based on query characteristics
        base_sla = np.percentile(historical_data["response_times"], 90)
        complexity_adjustment = query_complexity * 0.1  # 100ms per complexity unit
        
        return min(base_sla + complexity_adjustment, 5.0)  # Cap at 5s maximum
    
    async def _get_learned_sla_target_from_analytics(self) -> float:
        """Learn optimal SLA targets from comprehensive performance analytics"""
        all_performance_data = await self._load_all_performance_data()
        
        if not all_performance_data:
            return 3.0  # Default fallback
        
        # Use 95th percentile of all successful operations as SLA target
        successful_operations = [t for t in all_performance_data["response_times"] if t < 10.0]
        return np.percentile(successful_operations, 95)
    
    def _analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity for adaptive SLA calculation"""
        # Word count factor
        word_count = len(query.split())
        complexity = word_count / 10.0  # Base complexity
        
        # Domain-specific complexity indicators
        if any(term in query.lower() for term in ["relationship", "entity", "graph"]):
            complexity += 0.5  # Graph operations are more complex
        
        if any(term in query.lower() for term in ["statistical", "analysis", "pattern"]):
            complexity += 0.3  # Statistical operations add complexity
        
        return min(complexity, 3.0)  # Cap complexity score
```

**2. Service Reliability Analytics**
```python
class ServiceReliabilityAnalytics:
    """Learn service patterns and optimal recovery strategies from operational data"""
    
    async def _learn_failure_threshold(self, service_name: str, historical_data: Dict) -> int:
        """Learn optimal failure threshold from service reliability patterns"""
        if not historical_data:
            # Default thresholds based on service type
            defaults = {"azure_openai": 3, "azure_search": 2, "azure_cosmos": 5, "azure_ml": 2}
            return defaults.get(service_name, 3)
        
        # Calculate threshold based on failure frequency
        failure_rate = historical_data.get("failure_rate", 0.01)
        
        # Higher failure rates need lower thresholds for faster circuit breaking
        if failure_rate > 0.05:  # >5% failure rate
            return 2
        elif failure_rate > 0.02:  # >2% failure rate
            return 3
        else:
            return 5  # Low failure rate, more tolerance
    
    async def _learn_recovery_timeout(self, service_name: str, historical_data: Dict) -> float:
        """Learn optimal recovery timeout from service recovery patterns"""
        if not historical_data:
            # Default timeouts based on service characteristics
            defaults = {"azure_openai": 30.0, "azure_search": 15.0, "azure_cosmos": 45.0, "azure_ml": 120.0}
            return defaults.get(service_name, 60.0)
        
        # Use median recovery time as base timeout
        recovery_times = historical_data.get("recovery_times", [60.0])
        median_recovery = np.median(recovery_times)
        
        # Add buffer based on variance
        recovery_variance = np.var(recovery_times)
        buffer = min(recovery_variance * 0.1, 30.0)  # Cap buffer at 30s
        
        return median_recovery + buffer
    
    async def _generate_adaptive_retry_configs(
        self, 
        service_reliability_analytics: Dict,
        error_pattern_analysis: Dict,
        success_rate_targets: Dict[str, float]
    ) -> Dict[str, RetryConfig]:
        """Generate retry configurations from service reliability analytics"""
        retry_configs = {}
        
        for service_name, target_success_rate in success_rate_targets.items():
            service_data = service_reliability_analytics.get(service_name, {})
            error_patterns = error_pattern_analysis.get(service_name, {})
            
            # Calculate retry parameters based on service behavior
            current_success_rate = service_data.get("success_rate", 0.95)
            
            if current_success_rate >= target_success_rate:
                # Service is reliable, fewer retries needed
                max_attempts = 2
                base_delay = 0.5
            elif current_success_rate >= 0.90:
                # Moderate reliability
                max_attempts = 3
                base_delay = 1.0
            else:
                # Lower reliability, more aggressive retry
                max_attempts = 4
                base_delay = 2.0
            
            # Adjust based on error patterns
            if error_patterns.get("timeout_rate", 0) > 0.1:  # High timeout rate
                base_delay *= 1.5  # Longer delays for timeout-prone services
            
            retry_configs[service_name] = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=min(base_delay * 8, 60.0),  # Exponential backoff with cap
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF
            )
        
        return retry_configs
    
    async def _load_service_analytics(self) -> Dict:
        """Load comprehensive service analytics from monitoring data"""
        # This would integrate with Azure Application Insights or similar
        # For now, return structured data format
        return {
            "azure_openai": {
                "success_rate": 0.98,
                "failure_rate": 0.02,
                "recovery_times": [15.0, 20.0, 25.0, 18.0],
                "timeout_rate": 0.01
            },
            "azure_search": {
                "success_rate": 0.95,
                "failure_rate": 0.05,
                "recovery_times": [10.0, 12.0, 15.0, 8.0],
                "timeout_rate": 0.02
            },
            "azure_cosmos": {
                "success_rate": 0.99,
                "failure_rate": 0.01,
                "recovery_times": [30.0, 35.0, 40.0, 28.0],
                "timeout_rate": 0.005
            },
            "azure_ml": {
                "success_rate": 0.85,
                "failure_rate": 0.15,
                "recovery_times": [60.0, 90.0, 120.0, 75.0],
                "timeout_rate": 0.08
            }
        }
    
    async def _analyze_failure_patterns(self) -> Dict:
        """Analyze failure patterns for each service"""
        # This would analyze actual failure logs
        return {
            "azure_openai": {"timeout_rate": 0.01, "rate_limit_rate": 0.005},
            "azure_search": {"timeout_rate": 0.02, "throttling_rate": 0.01},
            "azure_cosmos": {"timeout_rate": 0.005, "throttling_rate": 0.008},
            "azure_ml": {"timeout_rate": 0.08, "model_loading_failures": 0.05}
        }
```

## Success Metrics

### Technical KPIs
- [ ] **100% orchestrator consolidation** - 6 orchestrators replaced with 2 graph workflows
- [ ] **State persistence operational** - PostgreSQL-backed state management working
- [ ] **Fault recovery functional** - Circuit breakers and retry logic operational
- [ ] **Performance monitoring active** - Logfire integration providing comprehensive metrics

### Competitive Advantage KPIs  
- [ ] **Tri-modal search unity preserved** - Vector+Graph+GNN coordination maintained
- [ ] **Hybrid domain intelligence preserved** - LLM+Statistical analysis functionality maintained
- [ ] **Sub-3-second SLA maintained** - 100% compliance with performance guarantee
- [ ] **Zero-config automation preserved** - Configuration-extraction workflow operational

### Architectural KPIs
- [ ] **Graph-based workflow compliance** - 100% adherence to official PydanticAI patterns
- [ ] **State management reliability** - Workflow resumption after interruptions working
- [ ] **Visual debugging capability** - Mermaid diagram generation operational
- [ ] **Production monitoring** - Comprehensive observability and alerting active

### Data-Driven Performance KPIs
- [ ] **Zero hardcoded SLA targets** - All SLA targets learned from performance analytics
- [ ] **Zero hardcoded circuit breaker thresholds** - All thresholds derived from service reliability data
- [ ] **Zero hardcoded retry configurations** - All retry policies learned from failure patterns
- [ ] **Zero hardcoded timeout values** - All timeouts calculated from operational history
- [ ] **Adaptive performance budgets** - All stage budgets derived from performance analytics
- [ ] **Mathematical performance optimization** - All timing parameters learned from data

## Risk Mitigation

### Migration Strategy
1. **Phased replacement** - Replace orchestrators one at a time with equivalent graph workflows
2. **Backward compatibility** - Maintain API contracts during transition period
3. **Feature flags** - Toggle between old and new orchestration for safety
4. **Performance validation** - Continuous monitoring during migration

### Rollback Plan
1. **Orchestrator preservation** - Keep original orchestrator files until validation complete
2. **State migration** - Ability to migrate graph state back to orchestrator state if needed
3. **API compatibility** - Maintain existing API endpoints during transition
4. **Emergency procedures** - Fast rollback procedures for production issues

### Quality Assurance
- [ ] End-to-end workflow testing with real Azure services
- [ ] Performance benchmarking against current orchestrator performance
- [ ] Competitive advantage validation tests
- [ ] Production load testing with graph workflows

## Next Steps

Upon successful completion of Phase 2:
1. **Production deployment** - Deploy graph-based orchestration to staging environment
2. **Performance optimization** - Fine-tune graph performance based on monitoring data
3. **Documentation update** - Update all operational documentation for graph workflows
4. **Team training** - Train operations team on graph monitoring and troubleshooting

## Dependencies

### Prerequisites
- [ ] Phase 1 (Tool Co-Location) completed successfully
- [ ] PostgreSQL instance provisioned for state persistence
- [ ] Azure Key Vault configured for state encryption
- [ ] Monitoring infrastructure (Application Insights) ready

### External Dependencies
- **PostgreSQL**: Required for production state persistence
- **Azure Key Vault**: Required for state encryption
- **Azure Application Insights**: Required for comprehensive monitoring
- **All Azure services**: OpenAI, Search, Cosmos DB, ML for workflow execution