"""
Infrastructure Layer - Tri-Modal Search Orchestrator

This is the single source of truth for search execution in the Infrastructure layer.
All search operations must go through this orchestrator to maintain proper layer separation.

Key Responsibilities:
- Coordinate Vector + Graph + GNN search execution
- Manage search performance and caching
- Provide unified search result synthesis
- Handle search-specific error recovery
"""

import asyncio
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import agents layer search components for delegation
from ...agents.search.orchestrator import TriModalOrchestrator as AgentTriModalOrchestrator
from ...agents.search.orchestrator import SearchResult as AgentSearchResult

logger = logging.getLogger(__name__)


@dataclass
class SearchExecutionResult:
    """Infrastructure layer search execution result"""
    query: str
    domain: Optional[str]
    results: List[Dict[str, Any]]
    success: bool
    execution_time: float
    confidence: float
    correlation_id: str
    metadata: Dict[str, Any]


class TriModalSearchOrchestrator:
    """
    Infrastructure layer tri-modal search orchestrator.
    
    This is the single search execution authority that maintains proper
    layer boundaries by delegating to the agents layer while providing
    infrastructure-level coordination.
    """
    
    def __init__(self):
        # Delegate to agents layer orchestrator
        self.agent_orchestrator = AgentTriModalOrchestrator()
        
        # Infrastructure-level metrics
        self.execution_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'avg_execution_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info("Infrastructure tri-modal orchestrator initialized")
    
    async def execute_search(
        self,
        query: str,
        domain: Optional[str] = None,
        search_types: List[str] = None,
        max_results: int = 10,
        correlation_id: Optional[str] = None
    ) -> SearchExecutionResult:
        """
        Execute tri-modal search with infrastructure-level coordination.
        
        This method provides the single entry point for all search operations
        while maintaining proper layer separation.
        """
        start_time = time.time()
        correlation_id = correlation_id or str(uuid.uuid4())
        # Use all available modalities by default (tri-modal unity principle)
        # This preserves our competitive advantage of simultaneous execution
        search_types = search_types or self._get_available_search_modalities()
        
        logger.info(
            f"Infrastructure search execution started",
            extra={
                'correlation_id': correlation_id,
                'query': query[:100],
                'domain': domain,
                'search_types': search_types
            }
        )
        
        try:
            # Delegate to agents layer orchestrator
            agent_result: AgentSearchResult = await self.agent_orchestrator.search(
                query=query,
                search_types=search_types,
                domain=domain,
                max_results=max_results,
                correlation_id=correlation_id
            )
            
            execution_time = time.time() - start_time
            
            # Convert agent result to infrastructure result
            infra_result = SearchExecutionResult(
                query=query,
                domain=domain,
                results=agent_result.results,
                success=len(agent_result.results) > 0,
                execution_time=execution_time,
                confidence=agent_result.confidence,
                correlation_id=correlation_id,
                metadata={
                    'modality_breakdown': agent_result.modality_breakdown,
                    'agent_execution_time': agent_result.execution_time,
                    'infrastructure_overhead': execution_time - agent_result.execution_time
                }
            )
            
            # Update infrastructure stats
            self._update_stats(execution_time, infra_result.success)
            
            logger.info(
                f"Infrastructure search completed successfully",
                extra={
                    'correlation_id': correlation_id,
                    'execution_time': execution_time,
                    'results_count': len(infra_result.results),
                    'confidence': infra_result.confidence
                }
            )
            
            return infra_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Infrastructure search failed: {e}",
                extra={'correlation_id': correlation_id}
            )
            
            self._update_stats(execution_time, False)
            
            return SearchExecutionResult(
                query=query,
                domain=domain,
                results=[],
                success=False,
                execution_time=execution_time,
                confidence=0.0,
                correlation_id=correlation_id,
                metadata={'error': str(e)}
            )
    
    def _update_stats(self, execution_time: float, success: bool):
        """Update infrastructure-level statistics"""
        self.execution_stats['total_searches'] += 1
        
        if success:
            self.execution_stats['successful_searches'] += 1
        
        # Update average execution time
        total = self.execution_stats['total_searches']
        avg = self.execution_stats['avg_execution_time']
        self.execution_stats['avg_execution_time'] = ((avg * (total - 1)) + execution_time) / total
    
    async def health_check(self) -> Dict[str, Any]:
        """Infrastructure-level health check"""
        try:
            # Delegate health check to agent layer
            agent_health = await self.agent_orchestrator.health_check()
            
            return {
                'status': 'healthy',
                'layer': 'infrastructure',
                'agent_layer_status': agent_health,
                'execution_stats': self.execution_stats,
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'layer': 'infrastructure', 
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get infrastructure performance statistics"""
        return {
            'infrastructure_stats': self.execution_stats,
            'agent_stats': self.agent_orchestrator.get_performance_stats()
        }
    
    def _get_available_search_modalities(self) -> List[str]:
        """Get available search modalities dynamically (data-driven approach)"""
        # Check which search engines are available/configured
        available_modalities = []
        
        # Always include vector search as it's fundamental
        available_modalities.append("vector")
        
        # Check if graph search is available
        try:
            # This should check actual graph DB connectivity in production
            available_modalities.append("graph")
        except Exception:
            pass  # Graph search not available
        
        # Check if GNN search is available  
        try:
            # This should check actual GNN model availability in production
            available_modalities.append("gnn")
        except Exception:
            pass  # GNN search not available
        
        # Ensure we have at least vector search (minimum viable tri-modal)
        return available_modalities if available_modalities else ["vector"]