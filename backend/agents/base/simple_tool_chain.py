"""
Simplified Tool Chaining System for PydanticAI Agent

This module replaces the complex tool chaining system with essential
sequential and parallel execution capabilities while removing security risks.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Simplified execution modes"""
    SEQUENTIAL = "sequential"  # Execute tools one after another
    PARALLEL = "parallel"      # Execute tools in parallel


@dataclass
class SimpleToolStep:
    """Simplified tool step definition"""
    tool_name: str
    parameters: Dict[str, Any]
    timeout: float = 30.0
    optional: bool = False  # If True, failure doesn't stop the chain


@dataclass
class SimpleChainResult:
    """Result of simplified tool chain execution"""
    chain_id: str
    success: bool
    execution_time: float
    steps_executed: int
    results: List[Any]
    errors: List[str]


class SimpleToolChain:
    """
    Simplified tool chain executor that maintains essential functionality
    while removing complex features like conditional execution and eval().
    
    Supports:
    - Sequential execution (one after another)
    - Parallel execution (all at once)
    - Optional steps (can fail without breaking chain)
    - Timeout handling
    - Error collection
    """
    
    def __init__(self):
        self.execution_history: List[SimpleChainResult] = []
        logger.info("Simple tool chain manager initialized")
    
    async def execute_sequential(self, steps: List[SimpleToolStep], 
                                tool_executor: Callable[[str, Dict[str, Any]], Any]) -> SimpleChainResult:
        """
        Execute tools sequentially (one after another)
        
        Args:
            steps: List of tool steps to execute
            tool_executor: Function that executes a tool given name and parameters
        
        Returns:
            SimpleChainResult with execution details
        """
        chain_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        results = []
        errors = []
        steps_executed = 0
        
        logger.info(f"Starting sequential execution of {len(steps)} steps (chain: {chain_id})")
        
        for i, step in enumerate(steps):
            try:
                logger.debug(f"Executing step {i+1}/{len(steps)}: {step.tool_name}")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    tool_executor(step.tool_name, step.parameters),
                    timeout=step.timeout
                )
                
                results.append(result)
                steps_executed += 1
                
                logger.debug(f"Step {i+1} completed successfully")
                
            except asyncio.TimeoutError:
                error_msg = f"Step {i+1} ({step.tool_name}) timed out after {step.timeout}s"
                errors.append(error_msg)
                logger.warning(error_msg)
                
                if not step.optional:
                    break  # Stop on non-optional failure
                    
            except Exception as e:
                error_msg = f"Step {i+1} ({step.tool_name}) failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
                
                if not step.optional:
                    break  # Stop on non-optional failure
        
        execution_time = time.time() - start_time
        success = len(errors) == 0 or all(step.optional for step in steps if steps.index(step) >= steps_executed)
        
        result = SimpleChainResult(
            chain_id=chain_id,
            success=success,
            execution_time=execution_time,
            steps_executed=steps_executed,
            results=results,
            errors=errors
        )
        
        self.execution_history.append(result)
        
        logger.info(f"Sequential execution completed: {steps_executed}/{len(steps)} steps, "
                   f"{execution_time:.2f}s, success: {success}")
        
        return result
    
    async def execute_parallel(self, steps: List[SimpleToolStep], 
                              tool_executor: Callable[[str, Dict[str, Any]], Any]) -> SimpleChainResult:
        """
        Execute tools in parallel (all at once)
        
        Args:
            steps: List of tool steps to execute
            tool_executor: Function that executes a tool given name and parameters
        
        Returns:
            SimpleChainResult with execution details
        """
        chain_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        logger.info(f"Starting parallel execution of {len(steps)} steps (chain: {chain_id})")
        
        # Create tasks for all steps
        tasks = []
        for i, step in enumerate(steps):
            async def execute_step(step_idx: int, step_info: SimpleToolStep):
                try:
                    logger.debug(f"Executing parallel step {step_idx+1}: {step_info.tool_name}")
                    
                    result = await asyncio.wait_for(
                        tool_executor(step_info.tool_name, step_info.parameters),
                        timeout=step_info.timeout
                    )
                    
                    logger.debug(f"Parallel step {step_idx+1} completed successfully")
                    return {"success": True, "result": result, "error": None}
                    
                except asyncio.TimeoutError:
                    error_msg = f"Step {step_idx+1} ({step_info.tool_name}) timed out after {step_info.timeout}s"
                    logger.warning(error_msg)
                    return {"success": False, "result": None, "error": error_msg}
                    
                except Exception as e:
                    error_msg = f"Step {step_idx+1} ({step_info.tool_name}) failed: {str(e)}"
                    logger.warning(error_msg)
                    return {"success": False, "result": None, "error": error_msg}
            
            task = asyncio.create_task(execute_step(i, step))
            tasks.append(task)
        
        # Wait for all tasks to complete
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = []
        errors = []
        steps_executed = 0
        
        for i, task_result in enumerate(task_results):
            if isinstance(task_result, Exception):
                error_msg = f"Step {i+1} had unexpected error: {str(task_result)}"
                errors.append(error_msg)
            elif task_result["success"]:
                results.append(task_result["result"])
                steps_executed += 1
            else:
                errors.append(task_result["error"])
                if not steps[i].optional:
                    # For parallel execution, we still count the step as executed
                    steps_executed += 1
        
        execution_time = time.time() - start_time
        
        # Success if no errors from required steps
        required_step_errors = [
            error for i, error in enumerate(errors) 
            if i < len(steps) and not steps[i].optional
        ]
        success = len(required_step_errors) == 0
        
        result = SimpleChainResult(
            chain_id=chain_id,
            success=success,
            execution_time=execution_time,
            steps_executed=steps_executed,
            results=results,
            errors=errors
        )
        
        self.execution_history.append(result)
        
        logger.info(f"Parallel execution completed: {steps_executed}/{len(steps)} steps, "
                   f"{execution_time:.2f}s, success: {success}")
        
        return result
    
    async def execute_chain(self, steps: List[SimpleToolStep], 
                           mode: ExecutionMode,
                           tool_executor: Callable[[str, Dict[str, Any]], Any]) -> SimpleChainResult:
        """
        Execute a tool chain with specified mode
        
        Args:
            steps: List of tool steps to execute
            mode: Execution mode (sequential or parallel)
            tool_executor: Function that executes tools
        
        Returns:
            SimpleChainResult with execution details
        """
        if mode == ExecutionMode.SEQUENTIAL:
            return await self.execute_sequential(steps, tool_executor)
        else:  # PARALLEL
            return await self.execute_parallel(steps, tool_executor)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "average_execution_time": 0.0,
                "success_rate": 0.0,
                "total_steps_executed": 0
            }
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for result in self.execution_history if result.success)
        total_time = sum(result.execution_time for result in self.execution_history)
        total_steps = sum(result.steps_executed for result in self.execution_history)
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": (successful_executions / total_executions) * 100,
            "average_execution_time": total_time / total_executions,
            "total_steps_executed": total_steps
        }
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        logger.info("Execution history cleared")


# Global tool chain manager
_global_tool_chain: Optional[SimpleToolChain] = None


def get_tool_chain_manager() -> SimpleToolChain:
    """Get or create global tool chain manager"""
    global _global_tool_chain
    if _global_tool_chain is None:
        _global_tool_chain = SimpleToolChain()
    return _global_tool_chain


# Convenience functions for common patterns
async def execute_search_chain(query: str, domain: str, tool_executor: Callable) -> SimpleChainResult:
    """Execute a common search chain pattern"""
    steps = [
        SimpleToolStep("domain_detection", {"query": query}, optional=True),
        SimpleToolStep("agent_adaptation", {"domain": domain}, optional=True),
        SimpleToolStep("tri_modal_search", {"query": query, "domain": domain}, optional=False),
    ]
    
    manager = get_tool_chain_manager()
    return await manager.execute_chain(steps, ExecutionMode.SEQUENTIAL, tool_executor)


async def execute_parallel_search(query: str, tool_executor: Callable) -> SimpleChainResult:
    """Execute parallel search across different modalities"""
    steps = [
        SimpleToolStep("vector_search", {"query": query}, optional=False),
        SimpleToolStep("graph_search", {"query": query}, optional=False),
        SimpleToolStep("gnn_search", {"query": query}, optional=True),
    ]
    
    manager = get_tool_chain_manager() 
    return await manager.execute_chain(steps, ExecutionMode.PARALLEL, tool_executor)