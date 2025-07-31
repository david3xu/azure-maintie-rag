"""
Simplified Tool Chaining System (Backward Compatibility Layer)

This module provides a backward-compatible interface to the simplified tool
chaining system while removing security risks and reducing complexity.
"""

from .simple_tool_chain import (
    SimpleToolChain,
    SimpleToolStep,
    SimpleChainResult,
    ExecutionMode,
    get_tool_chain_manager,
    execute_search_chain,
    execute_parallel_search
)
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Backward compatibility enums
class ChainExecutionMode(Enum):
    """Deprecated: Use ExecutionMode instead"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"  # Mapped to SEQUENTIAL for security
    ADAPTIVE = "adaptive"        # Mapped to SEQUENTIAL for security

class ToolDependency(Enum):
    """Deprecated: Use optional parameter instead"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"


# Backward compatibility classes
@dataclass
class ToolStep:
    """Backward compatibility wrapper for SimpleToolStep"""
    tool_name: str
    parameters: Dict[str, Any]
    dependency: ToolDependency = ToolDependency.REQUIRED
    condition: Optional[str] = None  # Deprecated - ignored for security
    timeout: float = 30.0
    retry_count: int = 2  # Deprecated - not used in simplified version
    parallel_group: Optional[str] = None  # Deprecated
    output_mapping: Dict[str, str] = field(default_factory=dict)  # Deprecated
    
    def to_simple_step(self) -> SimpleToolStep:
        """Convert to SimpleToolStep"""
        return SimpleToolStep(
            tool_name=self.tool_name,
            parameters=self.parameters,
            timeout=self.timeout,
            optional=(self.dependency == ToolDependency.OPTIONAL)
        )


@dataclass
class ChainResult:
    """Backward compatibility wrapper for SimpleChainResult"""
    chain_id: str
    success: bool
    total_execution_time: float
    steps_executed: int
    steps_successful: int
    results: Dict[str, Any]
    error_summary: List[str]
    performance_metrics: Dict[str, Any]
    
    @classmethod
    def from_simple_result(cls, simple_result: SimpleChainResult) -> 'ChainResult':
        """Create from SimpleChainResult"""
        return cls(
            chain_id=simple_result.chain_id,
            success=simple_result.success,
            total_execution_time=simple_result.execution_time,
            steps_executed=simple_result.steps_executed,
            steps_successful=simple_result.steps_executed if simple_result.success else 0,
            results={"steps": simple_result.results} if simple_result.results else {},
            error_summary=simple_result.errors,
            performance_metrics={
                "execution_time": simple_result.execution_time,
                "steps_executed": simple_result.steps_executed
            }
        )


@dataclass
class ToolChain:
    """Backward compatibility wrapper for tool chain definition"""
    chain_id: str
    name: str
    description: str
    steps: List[ToolStep]
    execution_mode: ChainExecutionMode = ChainExecutionMode.SEQUENTIAL
    max_total_time: float = 180.0  # Deprecated - timeout handled per step
    failure_strategy: str = "stop_on_critical"  # Deprecated


class ToolChainManager:
    """
    Backward compatibility wrapper for simplified tool chain manager.
    
    Maintains the same interface but uses simplified execution internally
    while removing security risks like eval() usage.
    """
    
    def __init__(self):
        self._simple_manager = SimpleToolChain()
        self.chains: Dict[str, ToolChain] = {}
        self.execution_history: List[ChainResult] = []
        self.common_patterns: Dict[str, ToolChain] = {}
        
        # Initialize simplified common patterns
        self._initialize_common_patterns()
        
        logger.info("Tool chain manager initialized (simplified mode)")
    
    def _initialize_common_patterns(self):
        """Initialize simplified common patterns"""
        
        # Comprehensive search pattern (simplified)
        comprehensive_search = ToolChain(
            chain_id="comprehensive_search",
            name="Comprehensive Search and Analysis", 
            description="Performs domain detection, agent adaptation, and tri-modal search",
            steps=[
                ToolStep("domain_detection", {"query": "{query}"}, ToolDependency.OPTIONAL),
                ToolStep("agent_adaptation", {"domain": "{domain}"}, ToolDependency.OPTIONAL),
                ToolStep("tri_modal_search", {"query": "{query}", "domain": "{domain}"})
            ],
            execution_mode=ChainExecutionMode.SEQUENTIAL
        )
        
        # Parallel search pattern
        parallel_search = ToolChain(
            chain_id="parallel_search",
            name="Parallel Multi-Modal Search",
            description="Executes vector, graph, and GNN search in parallel",
            steps=[
                ToolStep("vector_search", {"query": "{query}"}),
                ToolStep("graph_search", {"query": "{query}"}),
                ToolStep("gnn_search", {"query": "{query}"}, ToolDependency.OPTIONAL)
            ],
            execution_mode=ChainExecutionMode.PARALLEL
        )
        
        self.common_patterns["comprehensive_search"] = comprehensive_search
        self.common_patterns["parallel_search"] = parallel_search
    
    async def execute_chain(self, chain: ToolChain, 
                           tool_executor: Callable[[str, Dict[str, Any]], Any],
                           context: Optional[Dict[str, Any]] = None) -> ChainResult:
        """
        Execute a tool chain (backward compatible interface)
        
        Args:
            chain: ToolChain to execute
            tool_executor: Function that executes tools
            context: Execution context for parameter substitution
        
        Returns:
            ChainResult with execution details
        """
        # Convert ToolStep objects to SimpleToolStep
        simple_steps = []
        for step in chain.steps:
            # Simple parameter substitution (replaces eval() for security)
            parameters = self._substitute_parameters(step.parameters, context or {})
            
            simple_step = SimpleToolStep(
                tool_name=step.tool_name,
                parameters=parameters,
                timeout=step.timeout,
                optional=(step.dependency == ToolDependency.OPTIONAL)
            )
            simple_steps.append(simple_step)
        
        # Map execution mode (remove dangerous modes)
        if chain.execution_mode in [ChainExecutionMode.CONDITIONAL, ChainExecutionMode.ADAPTIVE]:
            # Map to SEQUENTIAL for security (these modes used eval())
            mode = ExecutionMode.SEQUENTIAL
            logger.warning(f"Execution mode {chain.execution_mode.value} mapped to sequential for security")
        else:
            mode = ExecutionMode.SEQUENTIAL if chain.execution_mode == ChainExecutionMode.SEQUENTIAL else ExecutionMode.PARALLEL
        
        # Execute using simplified manager
        simple_result = await self._simple_manager.execute_chain(simple_steps, mode, tool_executor)
        
        # Convert result back to old format
        result = ChainResult.from_simple_result(simple_result)
        self.execution_history.append(result)
        
        return result
    
    def _substitute_parameters(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safe parameter substitution (replaces eval() usage)
        
        Args:
            parameters: Parameter dict with potential {key} placeholders
            context: Context values for substitution
        
        Returns:
            Parameters with substituted values
        """
        result = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # Simple template substitution
                placeholder_key = value[1:-1]  # Remove { }
                result[key] = context.get(placeholder_key, value)
            else:
                result[key] = value
        return result
    
    async def execute_pattern(self, pattern_name: str, 
                             tool_executor: Callable[[str, Dict[str, Any]], Any],
                             context: Optional[Dict[str, Any]] = None) -> ChainResult:
        """Execute a common pattern by name"""
        if pattern_name not in self.common_patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        chain = self.common_patterns[pattern_name]
        return await self.execute_chain(chain, tool_executor, context)
    
    def register_chain(self, chain: ToolChain):
        """Register a new chain"""
        self.chains[chain.chain_id] = chain
        logger.info(f"Registered chain: {chain.name}")
    
    def get_chain(self, chain_id: str) -> Optional[ToolChain]:
        """Get chain by ID"""
        return self.chains.get(chain_id)
    
    def list_chains(self) -> List[str]:
        """List all registered chain IDs"""
        return list(self.chains.keys())
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0
            }
        
        total = len(self.execution_history)
        successful = sum(1 for result in self.execution_history if result.success)
        avg_time = sum(result.total_execution_time for result in self.execution_history) / total
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": (successful / total) * 100,
            "average_execution_time": avg_time
        }
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        self._simple_manager.clear_history()
        logger.info("Execution history cleared")


# Global tool chain manager
_global_tool_chain_manager: Optional[ToolChainManager] = None


def get_tool_chain_manager_legacy() -> ToolChainManager:
    """Get or create global tool chain manager (legacy interface)"""
    global _global_tool_chain_manager
    if _global_tool_chain_manager is None:
        _global_tool_chain_manager = ToolChainManager()
    return _global_tool_chain_manager


# Export both old and new interfaces
__all__ = [
    # New simplified interface
    'SimpleToolChain', 'SimpleToolStep', 'SimpleChainResult', 'ExecutionMode',
    'get_tool_chain_manager', 'execute_search_chain', 'execute_parallel_search',
    
    # Legacy interface
    'ToolChainManager', 'ToolChain', 'ToolStep', 'ChainResult',
    'ChainExecutionMode', 'ToolDependency', 'get_tool_chain_manager_legacy'
]