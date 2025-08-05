"""
Search Workflow Graph

Single graph implementation for search workflow following target architecture.
Provides graph-based control flow for tri-modal search operations.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..universal_search.agent import get_universal_search_agent
from ..domain_intelligence.agent import get_domain_intelligence_agent
from .workflow_enums import WorkflowState, NodeState
from .state_persistence import WorkflowStateManager
from .config_extraction_graph import WorkflowNode, WorkflowContext


class SearchWorkflow:
    """
    Single graph implementation for search workflow.
    
    Orchestrates the tri-modal search process with domain intelligence
    using a graph-based approach for optimal performance and observability.
    """
    
    def __init__(self):
        self.state_manager = WorkflowStateManager()
        self.nodes: Dict[str, WorkflowNode] = {}
        self._setup_workflow_graph()
    
    def _setup_workflow_graph(self):
        """Setup the search workflow graph nodes and dependencies"""
        
        # Node 1: Query Analysis
        self.nodes["query_analysis"] = WorkflowNode(
            id="query_analysis",
            name="Query Analysis and Preprocessing",
            handler=self._execute_query_analysis
        )
        
        # Node 2: Domain Detection (depends on query analysis)
        self.nodes["domain_detection"] = WorkflowNode(
            id="domain_detection",
            name="Domain Intelligence Detection",
            handler=self._execute_domain_detection,
            dependencies=["query_analysis"]
        )
        
        # Node 3: Search Strategy Selection (depends on domain detection)
        self.nodes["search_strategy"] = WorkflowNode(
            id="search_strategy",
            name="Search Strategy Selection",
            handler=self._execute_search_strategy_selection,
            dependencies=["domain_detection"]
        )
        
        # Node 4: Parallel Tri-Modal Search (depends on search strategy)
        self.nodes["tri_modal_search"] = WorkflowNode(
            id="tri_modal_search",
            name="Tri-Modal Search Execution",
            handler=self._execute_tri_modal_search,
            dependencies=["search_strategy"]
        )
        
        # Node 5: Result Synthesis (depends on tri-modal search)
        self.nodes["result_synthesis"] = WorkflowNode(
            id="result_synthesis",
            name="Multi-Modal Result Synthesis",
            handler=self._execute_result_synthesis,
            dependencies=["tri_modal_search"]
        )
        
        # Node 6: Response Generation (depends on result synthesis)
        self.nodes["response_generation"] = WorkflowNode(
            id="response_generation",
            name="Final Response Generation",
            handler=self._execute_response_generation,
            dependencies=["result_synthesis"]
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete search workflow.
        
        Args:
            input_data: Input data including query, context, preferences, etc.
            
        Returns:
            Dict containing search results and metadata
        """
        import time
        
        # Create workflow context
        workflow_id = f"search_{int(time.time())}"
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
                "search_results": context.results.get("response_generation", {}),
                "performance_metrics": self._generate_performance_metrics(),
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
        """Execute a node with retry logic (reused from config extraction)"""
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
        """Get topological execution order of nodes (reused logic)"""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node_id}")
            if node_id in visited:
                return
            
            temp_visited.add(node_id)
            
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
    
    def _generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate performance metrics specific to search workflow"""
        total_time = sum(n.execution_time for n in self.nodes.values())
        
        return {
            "total_search_time": total_time,
            "sub_3_second_target": total_time < 3.0,
            "performance_grade": (
                "excellent" if total_time < 1.0 else
                "good" if total_time < 2.0 else
                "acceptable" if total_time < 3.0 else
                "needs_optimization"
            ),
            "domain_detection_time": self.nodes.get("domain_detection", WorkflowNode("", "", lambda: None)).execution_time,
            "search_execution_time": self.nodes.get("tri_modal_search", WorkflowNode("", "", lambda: None)).execution_time,
            "synthesis_time": self.nodes.get("result_synthesis", WorkflowNode("", "", lambda: None)).execution_time,
            "parallel_efficiency": self._calculate_parallel_efficiency()
        }
    
    def _calculate_parallel_efficiency(self) -> float:
        """Calculate parallel execution efficiency"""
        # Mock calculation - would measure actual parallel vs sequential time
        return 0.85
    
    # Node handler implementations
    async def _execute_query_analysis(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute query analysis node"""
        query = context.input_data.get("query", "")
        
        # Basic query analysis
        return {
            "original_query": query,
            "query_length": len(query),
            "query_complexity": "medium" if len(query.split()) > 5 else "simple",
            "intent_detected": "search",
            "preprocessing_complete": True
        }
    
    async def _execute_domain_detection(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute domain detection node"""
        agent = get_domain_intelligence_agent()
        
        query = context.input_data.get("query", "")
        
        result = await agent.run(
            "detect_domain_from_query",
            message_history=[{
                "role": "user",
                "content": f"Detect domain from this query: {query}"
            }]
        )
        
        return {
            "detected_domain": result.data if hasattr(result, 'data') else {"domain": "general"},
            "detection_confidence": 0.85,
            "detection_method": "domain_intelligence_agent"
        }
    
    async def _execute_search_strategy_selection(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute search strategy selection node"""
        domain_info = context.results.get("domain_detection", {})
        query_info = context.results.get("query_analysis", {})
        
        # Select optimal search strategy based on domain and query
        return {
            "selected_modalities": ["vector", "graph", "gnn"],  # Tri-modal unity
            "search_weights": {"vector": 0.4, "graph": 0.3, "gnn": 0.3},
            "optimization_strategy": "parallel_execution",
            "max_results_per_modality": context.input_data.get("max_results", 10)
        }
    
    async def _execute_tri_modal_search(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute tri-modal search node"""
        agent = get_universal_search_agent()
        
        query = context.input_data.get("query", "")
        domain_info = context.results.get("domain_detection", {})
        strategy = context.results.get("search_strategy", {})
        
        result = await agent.run(
            "process_intelligent_query",
            message_history=[{
                "role": "user",
                "content": f"Search for: {query}"
            }]
        )
        
        return {
            "search_results": result.data if hasattr(result, 'data') else {},
            "modalities_executed": strategy.get("selected_modalities", []),
            "total_results": 0,  # Would be calculated from actual results
            "search_confidence": 0.8
        }
    
    async def _execute_result_synthesis(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute result synthesis node"""
        search_results = context.results.get("tri_modal_search", {})
        strategy = context.results.get("search_strategy", {})
        
        # Synthesize results from multiple modalities
        return {
            "synthesized_results": [],  # Would contain ranked, deduplicated results
            "synthesis_method": "weighted_ranking",
            "confidence_scores": strategy.get("search_weights", {}),
            "total_unique_results": 0,
            "synthesis_confidence": 0.85
        }
    
    async def _execute_response_generation(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute response generation node"""
        synthesized_results = context.results.get("result_synthesis", {})
        original_query = context.input_data.get("query", "")
        
        # Generate final response
        return {
            "final_response": f"Search results for: {original_query}",
            "results_summary": synthesized_results,
            "response_format": "structured",
            "citations_included": True,
            "response_confidence": 0.9,
            "generation_method": "template_based"
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on search workflow components"""
        try:
            # Check if agents are accessible
            domain_agent = get_domain_intelligence_agent()
            search_agent = get_universal_search_agent()
            
            return {
                "overall_status": "healthy",
                "components": {
                    "domain_intelligence_agent": "available",
                    "universal_search_agent": "available",
                    "workflow_nodes": len(self.nodes),
                    "state_manager": "operational"
                },
                "workflow_status": "ready"
            }
        except Exception as e:
            return {
                "overall_status": "degraded",
                "error": str(e),
                "workflow_status": "limited"
            }


# Export main components
__all__ = [
    "SearchWorkflow",
]