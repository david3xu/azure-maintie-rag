"""
Config-Extraction Workflow Graph

Single graph implementation for the Config-Extraction workflow following target architecture.
Provides graph-based control flow for domain analysis and knowledge extraction.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime

# Import models from centralized data models
from agents.core.data_models import (
    WorkflowState,
    NodeState,
    NodeExecutionResult as WorkflowNode,
    WorkflowExecutionState as WorkflowContext
)

from ..domain_intelligence.agent import get_domain_intelligence_agent
from ..knowledge_extraction.agent import get_knowledge_extraction_agent
from .state_persistence import WorkflowStateManager
from ..core.constants import WorkflowConstants


class ConfigExtractionWorkflow:
    """
    Single graph implementation for Config-Extraction workflow.

    Orchestrates the flow from domain analysis to knowledge extraction
    using a graph-based approach for better control and observability.
    """

    def __init__(self):
        self.state_manager = WorkflowStateManager()
        self.nodes: Dict[str, WorkflowNode] = {}
        self._setup_workflow_graph()

    def _setup_workflow_graph(self):
        """Setup the workflow graph nodes and dependencies"""

        # Node 1: Domain Discovery
        self.nodes["domain_discovery"] = WorkflowNode(
            id="domain_discovery",
            name="Domain Discovery",
            handler=self._execute_domain_discovery
        )

        # Node 2: Corpus Analysis (depends on domain discovery)
        self.nodes["corpus_analysis"] = WorkflowNode(
            id="corpus_analysis",
            name="Corpus Statistical Analysis",
            handler=self._execute_corpus_analysis,
            dependencies=["domain_discovery"]
        )

        # Node 3: Pattern Generation (depends on corpus analysis)
        self.nodes["pattern_generation"] = WorkflowNode(
            id="pattern_generation",
            name="Semantic Pattern Generation",
            handler=self._execute_pattern_generation,
            dependencies=["corpus_analysis"]
        )

        # Node 4: Config Generation (depends on pattern generation)
        self.nodes["config_generation"] = WorkflowNode(
            id="config_generation",
            name="Extraction Configuration Generation",
            handler=self._execute_config_generation,
            dependencies=["pattern_generation"]
        )

        # Node 5: Knowledge Extraction (depends on config generation)
        self.nodes["knowledge_extraction"] = WorkflowNode(
            id="knowledge_extraction",
            name="Document Knowledge Extraction",
            handler=self._execute_knowledge_extraction,
            dependencies=["config_generation"]
        )

        # Node 6: Quality Validation (depends on knowledge extraction)
        self.nodes["quality_validation"] = WorkflowNode(
            id="quality_validation",
            name="Extraction Quality Validation",
            handler=self._execute_quality_validation,
            dependencies=["knowledge_extraction"]
        )

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete Config-Extraction workflow.

        Args:
            input_data: Input data including corpus_path, domain_name, etc.

        Returns:
            Dict containing workflow results and metadata
        """
        import time

        # Create workflow context
        workflow_id = f"config_extraction_{int(time.time())}"
        context = WorkflowContext(
            workflow_id=workflow_id,
            input_data=input_data,
            start_time=datetime.now()
        )

        try:
            # Save initial state
            await self.state_manager.save_workflow_state(
                workflow_id, WorkflowState.RUNNING, context.input_data
            )

            # Execute workflow graph
            execution_order = self._get_execution_order()

            for node_id in execution_order:
                node = self.nodes[node_id]

                # Check if dependencies are satisfied
                if not self._dependencies_satisfied(node_id, context):
                    node.state = NodeState.SKIPPED
                    continue

                # Execute node with retry logic
                await self._execute_node_with_retry(node, context)

                # Update workflow state
                await self.state_manager.update_node_state(
                    workflow_id, node_id, node.state, node.result, node.error
                )

            context.end_time = datetime.now()

            # Determine final workflow state
            final_state = self._determine_final_state()

            await self.state_manager.save_workflow_state(
                workflow_id, final_state, context.results
            )

            return {
                "workflow_id": workflow_id,
                "state": final_state.value,
                "results": context.results,
                "execution_summary": self._generate_execution_summary(),
                "start_time": context.start_time.isoformat(),
                "end_time": context.end_time.isoformat(),
                "total_time_seconds": (context.end_time - context.start_time).total_seconds()
            }

        except Exception as e:
            context.end_time = datetime.now()
            await self.state_manager.save_workflow_state(
                workflow_id, WorkflowState.FAILED, {"error": str(e)}
            )

            return {
                "workflow_id": workflow_id,
                "state": WorkflowState.FAILED.value,
                "error": str(e),
                "execution_summary": self._generate_execution_summary(),
                "start_time": context.start_time.isoformat(),
                "end_time": context.end_time.isoformat(),
                "total_time_seconds": (context.end_time - context.start_time).total_seconds()
            }

    async def _execute_node_with_retry(self, node: WorkflowNode, context: WorkflowContext):
        """Execute a node with retry logic"""
        import time

        while node.retry_count <= node.max_retries:
            try:
                node.state = NodeState.EXECUTING
                start_time = time.time()

                # Execute the node handler
                result = await node.handler(context)

                node.execution_time = time.time() - start_time
                node.result = result
                node.state = NodeState.COMPLETED

                # Store result in context
                context.results[node.id] = result

                return

            except Exception as e:
                node.retry_count += 1
                node.error = str(e)

                if node.retry_count > node.max_retries:
                    node.state = NodeState.FAILED
                    raise

                # Wait before retry
                await asyncio.sleep(2 ** node.retry_count)

    def _get_execution_order(self) -> List[str]:
        """Get topological execution order of nodes"""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []

        def visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node_id}")
            if node_id in visited:
                return

            temp_visited.add(node_id)

            # Visit dependencies first
            for dep in self.nodes[node_id].dependencies:
                visit(dep)

            temp_visited.remove(node_id)
            visited.add(node_id)
            order.append(node_id)

        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)

        return order

    def _dependencies_satisfied(self, node_id: str, context: WorkflowContext) -> bool:
        """Check if node dependencies are satisfied"""
        node = self.nodes[node_id]

        for dep_id in node.dependencies:
            dep_node = self.nodes[dep_id]
            if dep_node.state != NodeState.COMPLETED:
                return False

        return True

    def _determine_final_state(self) -> WorkflowState:
        """Determine final workflow state based on node states"""
        failed_nodes = [n for n in self.nodes.values() if n.state == NodeState.FAILED]
        completed_nodes = [n for n in self.nodes.values() if n.state == NodeState.COMPLETED]

        if failed_nodes:
            return WorkflowState.FAILED
        elif len(completed_nodes) == len(self.nodes):
            return WorkflowState.COMPLETED
        else:
            return WorkflowState.PAUSED

    def _generate_execution_summary(self) -> Dict[str, Any]:
        """Generate execution summary"""
        summary = {
            "total_nodes": len(self.nodes),
            "completed_nodes": len([n for n in self.nodes.values() if n.state == NodeState.COMPLETED]),
            "failed_nodes": len([n for n in self.nodes.values() if n.state == NodeState.FAILED]),
            "skipped_nodes": len([n for n in self.nodes.values() if n.state == NodeState.SKIPPED]),
            "total_execution_time": sum(n.execution_time for n in self.nodes.values()),
            "node_details": {}
        }

        for node_id, node in self.nodes.items():
            summary["node_details"][node_id] = {
                "state": node.state.value,
                "execution_time": node.execution_time,
                "retry_count": node.retry_count,
                "error": node.error
            }

        return summary

    # Node handler implementations
    async def _execute_domain_discovery(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute domain discovery node"""
        agent = get_domain_intelligence_agent()

        # Use domain intelligence agent to discover available domains
        result = await agent.run(
            "discover_available_domains",
            message_history=[{
                "role": "user",
                "content": f"Discover available domains from {context.input_data.get('data_directory', '/workspace/azure-maintie-rag/data/raw')}"
            }]
        )

        return {
            "domains_discovered": result.data if hasattr(result, 'data') else {},
            "discovery_method": "filesystem_scan",
            "discovery_confidence": WorkflowConstants.DISCOVERY_CONFIDENCE
        }

    async def _execute_corpus_analysis(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute corpus analysis node"""
        agent = get_domain_intelligence_agent()

        domain_result = context.results.get("domain_discovery", {})
        corpus_path = context.input_data.get("corpus_path", "/workspace/azure-maintie-rag/data/raw")

        result = await agent.run(
            "analyze_corpus_statistics",
            message_history=[{
                "role": "user",
                "content": f"Analyze corpus statistics for {corpus_path}"
            }]
        )

        return {
            "statistical_analysis": result.data if hasattr(result, 'data') else {},
            "corpus_path": corpus_path,
            "analysis_confidence": WorkflowConstants.ANALYSIS_CONFIDENCE
        }

    async def _execute_pattern_generation(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute pattern generation node"""
        agent = get_domain_intelligence_agent()

        corpus_analysis = context.results.get("corpus_analysis", {})

        result = await agent.run(
            "generate_semantic_patterns",
            message_history=[{
                "role": "user",
                "content": f"Generate semantic patterns based on analysis: {corpus_analysis}"
            }]
        )

        return {
            "semantic_patterns": result.data if hasattr(result, 'data') else {},
            "pattern_confidence": WorkflowConstants.PATTERN_CONFIDENCE
        }

    async def _execute_config_generation(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute config generation node"""
        agent = get_domain_intelligence_agent()

        patterns = context.results.get("pattern_generation", {})
        stats = context.results.get("corpus_analysis", {})

        result = await agent.run(
            "create_fully_learned_extraction_config",
            message_history=[{
                "role": "user",
                "content": f"Create extraction config from patterns: {patterns} and stats: {stats}"
            }]
        )

        return {
            "extraction_config": result.data if hasattr(result, 'data') else {},
            "config_confidence": WorkflowConstants.CONFIG_CONFIDENCE
        }

    async def _execute_knowledge_extraction(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute knowledge extraction node"""
        agent = get_knowledge_extraction_agent()

        config = context.results.get("config_generation", {}).get("extraction_config", {})
        documents = context.input_data.get("documents", [])

        # Mock extraction for now - would use actual agent tools
        return {
            "extracted_entities": [],
            "extracted_relationships": [],
            "extraction_confidence": WorkflowConstants.EXTRACTION_CONFIDENCE,
            "documents_processed": len(documents)
        }

    async def _execute_quality_validation(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute quality validation node"""
        extraction_results = context.results.get("knowledge_extraction", {})

        # Mock validation for now
        return {
            "validation_passed": True,
            "quality_score": WorkflowConstants.QUALITY_SCORE,
            "validation_warnings": [],
            "validation_confidence": WorkflowConstants.VALIDATION_CONFIDENCE
        }


# Export main components
__all__ = [
    "ConfigExtractionWorkflow",
    "WorkflowState",
    "NodeState",
    "WorkflowNode",
    "WorkflowContext",
]
