"""
Reasoning Engine - Core reasoning patterns for intelligent agents.
Implements systematic reasoning chains with transparency and learning capabilities.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import logging

from .agent_interface import ReasoningStep, ReasoningTrace, AgentContext

logger = logging.getLogger(__name__)


class ReasoningPattern(Enum):
    """Supported reasoning patterns - data-driven selection"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_SEARCH = "tree_search"
    MULTI_PERSPECTIVE = "multi_perspective"
    EVIDENCE_SYNTHESIS = "evidence_synthesis"
    ITERATIVE_REFINEMENT = "iterative_refinement"


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning assessment"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ReasoningStep:
    """Individual reasoning step in a chain"""
    step_id: str
    pattern: ReasoningPattern
    description: str
    inputs: Dict[str, Any]
    reasoning_function: Callable
    dependencies: List[str] = field(default_factory=list)
    parallel_allowed: bool = True
    timeout_seconds: float = 30.0
    retry_attempts: int = 2


@dataclass 
class ReasoningResult:
    """Result of a reasoning step execution"""
    step_id: str
    success: bool
    result: Any
    confidence: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ReasoningChain:
    """Complete reasoning chain definition"""
    chain_id: str
    steps: List[ReasoningStep]
    pattern: ReasoningPattern
    parallel_execution: bool = True
    max_total_time: float = 60.0
    confidence_threshold: float = 0.7


class ReasoningEngine:
    """
    Core reasoning engine for intelligent agents.
    
    Provides systematic reasoning patterns with:
    - Transparent reasoning traces
    - Parallel execution capabilities  
    - Confidence tracking and validation
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reasoning engine with data-driven configuration.
        
        Args:
            config: Engine configuration
                - enabled_patterns: List of enabled reasoning patterns
                - default_confidence_threshold: Minimum confidence for results
                - max_parallel_steps: Maximum concurrent reasoning steps
                - performance_targets: Response time and quality targets
                - retry_config: Retry behavior configuration
        """
        self.config = config
        self.enabled_patterns = [
            ReasoningPattern(p) for p in config.get("enabled_patterns", [
                "chain_of_thought", "evidence_synthesis", "multi_perspective"
            ])
        ]
        self.confidence_threshold = config.get("default_confidence_threshold", 0.7)
        self.max_parallel_steps = config.get("max_parallel_steps", 5)
        self.performance_targets = config.get("performance_targets", {
            "max_response_time": 30.0,
            "min_confidence": 0.7
        })
        self.retry_config = config.get("retry_config", {
            "max_attempts": 2,
            "base_delay": 1.0,
            "exponential_backoff": True
        })
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._active_chains: Dict[str, ReasoningChain] = {}
        
    async def execute_reasoning_chain(
        self, 
        chain: ReasoningChain, 
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Execute a complete reasoning chain with transparency and monitoring.
        
        Args:
            chain: Reasoning chain to execute
            context: Agent context for reasoning
            
        Returns:
            Dictionary with:
            - success: bool
            - results: Dict[str, ReasoningResult] 
            - overall_confidence: float
            - execution_trace: List[ReasoningTrace]
            - performance_metrics: Dict[str, Any]
        """
        start_time = time.time()
        execution_trace = []
        results = {}
        
        try:
            self._active_chains[chain.chain_id] = chain
            self.logger.info(f"Starting reasoning chain {chain.chain_id} with {len(chain.steps)} steps")
            
            # Build execution plan based on dependencies
            execution_plan = self._build_execution_plan(chain)
            
            # Execute reasoning steps
            for batch in execution_plan:
                batch_results = await self._execute_step_batch(batch, context, execution_trace)
                results.update(batch_results)
                
                # Check if we should continue based on confidence
                if not self._should_continue_chain(results, chain.confidence_threshold):
                    break
            
            # Calculate overall confidence and success
            overall_confidence = self._calculate_overall_confidence(results)
            success = overall_confidence >= chain.confidence_threshold
            
            execution_time = (time.time() - start_time) * 1000
            
            # Create performance metrics
            performance_metrics = {
                "total_execution_time_ms": execution_time,
                "steps_executed": len(results),
                "parallel_steps": sum(len(batch) for batch in execution_plan if len(batch) > 1),
                "average_step_time_ms": execution_time / max(len(results), 1),
                "overall_confidence": overall_confidence,
                "pattern_used": chain.pattern.value
            }
            
            self.logger.info(
                f"Reasoning chain {chain.chain_id} completed: "
                f"success={success}, confidence={overall_confidence:.3f}, "
                f"time={execution_time:.1f}ms"
            )
            
            return {
                "success": success,
                "results": results,
                "overall_confidence": overall_confidence,
                "execution_trace": execution_trace,
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Reasoning chain {chain.chain_id} failed: {e}")
            
            return {
                "success": False,
                "results": results,
                "overall_confidence": 0.0,
                "execution_trace": execution_trace,
                "performance_metrics": {
                    "total_execution_time_ms": execution_time,
                    "error": str(e)
                },
                "error": str(e)
            }
        finally:
            self._active_chains.pop(chain.chain_id, None)
    
    async def stream_reasoning(
        self, 
        chain: ReasoningChain, 
        context: AgentContext
    ) -> AsyncIterator[ReasoningTrace]:
        """
        Stream reasoning execution in real-time.
        
        Args:
            chain: Reasoning chain to execute
            context: Agent context
            
        Yields:
            ReasoningTrace objects as reasoning progresses
        """
        execution_plan = self._build_execution_plan(chain)
        
        for batch in execution_plan:
            # Start batch execution
            yield ReasoningTrace(
                step=ReasoningStep.PLANNING,
                description=f"Starting batch execution of {len(batch)} reasoning steps",
                inputs={"batch_size": len(batch), "step_ids": [s.step_id for s in batch]},
                outputs={},
                duration_ms=0,
                confidence=0.8
            )
            
            # Execute batch and stream individual step results
            tasks = []
            for step in batch:
                task = asyncio.create_task(self._execute_single_step(step, context))
                tasks.append((step.step_id, task))
            
            for step_id, task in tasks:
                result = await task
                
                yield ReasoningTrace(
                    step=ReasoningStep.EXECUTION,
                    description=f"Completed reasoning step: {step_id}",
                    inputs={"step_id": step_id},
                    outputs={"result": result.result, "success": result.success},
                    duration_ms=result.duration_ms,
                    confidence=result.confidence
                )
    
    def create_chain_of_thought(
        self, 
        query: str, 
        domain: Optional[str] = None
    ) -> ReasoningChain:
        """
        Create a chain-of-thought reasoning chain for systematic analysis.
        
        Args:
            query: Query to analyze
            domain: Optional domain context
            
        Returns:
            ReasoningChain configured for chain-of-thought reasoning
        """
        steps = [
            ReasoningStep(
                step_id="analyze_query",
                pattern=ReasoningPattern.CHAIN_OF_THOUGHT,
                description="Analyze query intent and requirements",
                inputs={"query": query, "domain": domain},
                reasoning_function=self._analyze_query_intent,
                dependencies=[],
                parallel_allowed=False
            ),
            ReasoningStep(
                step_id="identify_concepts",
                pattern=ReasoningPattern.CHAIN_OF_THOUGHT,
                description="Identify key concepts and relationships",
                inputs={"query": query},
                reasoning_function=self._identify_key_concepts,
                dependencies=["analyze_query"],
                parallel_allowed=False
            ),
            ReasoningStep(
                step_id="plan_approach",
                pattern=ReasoningPattern.CHAIN_OF_THOUGHT,
                description="Plan search and reasoning approach",
                inputs={"query": query, "domain": domain},
                reasoning_function=self._plan_search_approach,
                dependencies=["identify_concepts"],
                parallel_allowed=False
            ),
            ReasoningStep(
                step_id="synthesize_strategy",
                pattern=ReasoningPattern.CHAIN_OF_THOUGHT,
                description="Synthesize final reasoning strategy",
                inputs={"query": query},
                reasoning_function=self._synthesize_reasoning_strategy,
                dependencies=["plan_approach"],
                parallel_allowed=False
            )
        ]
        
        return ReasoningChain(
            chain_id=f"cot_{hash(query) % 10000}",
            steps=steps,
            pattern=ReasoningPattern.CHAIN_OF_THOUGHT,
            parallel_execution=False,  # Chain of thought is sequential
            confidence_threshold=self.confidence_threshold
        )
    
    def create_evidence_synthesis_chain(
        self, 
        evidence_sources: List[str], 
        query: str
    ) -> ReasoningChain:
        """
        Create evidence synthesis reasoning chain for multi-source analysis.
        
        Args:
            evidence_sources: List of evidence source identifiers
            query: Query context for synthesis
            
        Returns:
            ReasoningChain configured for evidence synthesis
        """
        steps = []
        
        # Create parallel evidence analysis steps
        for i, source in enumerate(evidence_sources):
            steps.append(ReasoningStep(
                step_id=f"analyze_evidence_{i}",
                pattern=ReasoningPattern.EVIDENCE_SYNTHESIS,
                description=f"Analyze evidence from {source}",
                inputs={"source": source, "query": query},
                reasoning_function=self._analyze_evidence_source,
                dependencies=[],
                parallel_allowed=True
            ))
        
        # Add synthesis step that depends on all evidence analysis
        evidence_step_ids = [f"analyze_evidence_{i}" for i in range(len(evidence_sources))]
        steps.append(ReasoningStep(
            step_id="synthesize_evidence",
            pattern=ReasoningPattern.EVIDENCE_SYNTHESIS,
            description="Synthesize evidence from all sources",
            inputs={"query": query, "num_sources": len(evidence_sources)},
            reasoning_function=self._synthesize_evidence,
            dependencies=evidence_step_ids,
            parallel_allowed=False
        ))
        
        return ReasoningChain(
            chain_id=f"evidence_{hash(query) % 10000}",
            steps=steps,
            pattern=ReasoningPattern.EVIDENCE_SYNTHESIS,
            parallel_execution=True,
            confidence_threshold=self.confidence_threshold
        )
    
    # Private implementation methods
    
    def _build_execution_plan(self, chain: ReasoningChain) -> List[List[ReasoningStep]]:
        """Build execution plan respecting dependencies and parallelization"""
        steps_by_id = {step.step_id: step for step in chain.steps}
        completed = set()
        execution_plan = []
        
        while len(completed) < len(chain.steps):
            # Find steps that can execute now (dependencies satisfied)
            ready_steps = []
            for step in chain.steps:
                if (step.step_id not in completed and 
                    all(dep in completed for dep in step.dependencies)):
                    ready_steps.append(step)
            
            if not ready_steps:
                raise ValueError("Circular dependency detected in reasoning chain")
            
            # Group by parallel execution capability
            if chain.parallel_execution:
                parallel_batch = [s for s in ready_steps if s.parallel_allowed]
                sequential_batch = [s for s in ready_steps if not s.parallel_allowed]
                
                if parallel_batch:
                    execution_plan.append(parallel_batch[:self.max_parallel_steps])
                    completed.update(s.step_id for s in parallel_batch[:self.max_parallel_steps])
                elif sequential_batch:
                    execution_plan.append([sequential_batch[0]])
                    completed.add(sequential_batch[0].step_id)
            else:
                # Sequential execution
                execution_plan.append([ready_steps[0]])
                completed.add(ready_steps[0].step_id)
        
        return execution_plan
    
    async def _execute_step_batch(
        self, 
        batch: List[ReasoningStep], 
        context: AgentContext,
        execution_trace: List[ReasoningTrace]
    ) -> Dict[str, ReasoningResult]:
        """Execute a batch of reasoning steps"""
        if len(batch) == 1:
            # Single step execution
            step = batch[0]
            result = await self._execute_single_step(step, context)
            
            # Add to trace
            execution_trace.append(ReasoningTrace(
                step=ReasoningStep.EXECUTION,
                description=step.description,
                inputs=step.inputs,
                outputs={"result": result.result, "confidence": result.confidence},
                duration_ms=result.duration_ms,
                confidence=result.confidence
            ))
            
            return {step.step_id: result}
        else:
            # Parallel batch execution
            tasks = {
                step.step_id: asyncio.create_task(self._execute_single_step(step, context))
                for step in batch
            }
            
            results = {}
            for step_id, task in tasks.items():
                result = await task
                results[step_id] = result
                
                # Add to trace
                step = next(s for s in batch if s.step_id == step_id)
                execution_trace.append(ReasoningTrace(
                    step=ReasoningStep.EXECUTION,
                    description=step.description,
                    inputs=step.inputs,
                    outputs={"result": result.result, "confidence": result.confidence},
                    duration_ms=result.duration_ms,
                    confidence=result.confidence
                ))
            
            return results
    
    async def _execute_single_step(
        self, 
        step: ReasoningStep, 
        context: AgentContext
    ) -> ReasoningResult:
        """Execute a single reasoning step with error handling and retries"""
        for attempt in range(step.retry_attempts + 1):
            start_time = time.time()
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    step.reasoning_function(step.inputs, context),
                    timeout=step.timeout_seconds
                )
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Validate result and calculate confidence
                confidence = self._calculate_step_confidence(result, step)
                
                return ReasoningResult(
                    step_id=step.step_id,
                    success=True,
                    result=result,
                    confidence=confidence,
                    duration_ms=duration_ms,
                    metadata={
                        "pattern": step.pattern.value,
                        "attempt": attempt + 1
                    }
                )
                
            except asyncio.TimeoutError:
                duration_ms = (time.time() - start_time) * 1000
                error_msg = f"Step {step.step_id} timed out after {step.timeout_seconds}s"
                
                if attempt < step.retry_attempts:
                    await asyncio.sleep(self.retry_config["base_delay"] * (2 ** attempt))
                    continue
                    
                return ReasoningResult(
                    step_id=step.step_id,
                    success=False,
                    result=None,
                    confidence=0.0,
                    duration_ms=duration_ms,
                    error=error_msg
                )
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                error_msg = f"Step {step.step_id} failed: {str(e)}"
                
                if attempt < step.retry_attempts:
                    await asyncio.sleep(self.retry_config["base_delay"] * (2 ** attempt))
                    continue
                
                return ReasoningResult(
                    step_id=step.step_id,
                    success=False,
                    result=None,
                    confidence=0.0,
                    duration_ms=duration_ms,
                    error=error_msg
                )
    
    def _should_continue_chain(
        self, 
        results: Dict[str, ReasoningResult], 
        threshold: float
    ) -> bool:
        """Determine if reasoning chain should continue based on intermediate results"""
        if not results:
            return True
        
        # Check if any critical step failed
        failed_steps = [r for r in results.values() if not r.success]
        if failed_steps:
            # Could implement smart recovery logic here
            return len(failed_steps) / len(results) < 0.5  # Continue if less than 50% failed
        
        # Check confidence levels
        avg_confidence = sum(r.confidence for r in results.values()) / len(results)
        return avg_confidence >= threshold * 0.8  # Continue if within 80% of target
    
    def _calculate_overall_confidence(self, results: Dict[str, ReasoningResult]) -> float:
        """Calculate overall confidence from step results"""
        if not results:
            return 0.0
        
        successful_results = [r for r in results.values() if r.success]
        if not successful_results:
            return 0.0
        
        # Weighted average based on step importance (could be enhanced)
        confidence_sum = sum(r.confidence for r in successful_results)
        success_rate = len(successful_results) / len(results)
        
        return (confidence_sum / len(successful_results)) * success_rate
    
    def _calculate_step_confidence(self, result: Any, step: ReasoningStep) -> float:
        """Calculate confidence for a reasoning step result"""
        # Default confidence calculation - can be enhanced per pattern
        if result is None:
            return 0.0
        
        # Simple heuristics - should be enhanced with domain-specific logic
        if isinstance(result, dict):
            if "confidence" in result:
                return float(result["confidence"])
            return 0.8  # Default for structured results
        elif isinstance(result, (list, tuple)):
            return 0.7 if len(result) > 0 else 0.2
        elif isinstance(result, str):
            return 0.6 if len(result.strip()) > 10 else 0.3
        else:
            return 0.5  # Default for other types
    
    # Default reasoning function implementations
    
    async def _analyze_query_intent(self, inputs: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Default query intent analysis"""
        query = inputs.get("query", "")
        domain = inputs.get("domain")
        
        # Simple intent analysis - should be enhanced with NLP
        intent_keywords = {
            "search": ["find", "search", "what", "where", "who"],
            "analysis": ["analyze", "compare", "evaluate", "assess"],
            "creation": ["create", "generate", "build", "make"],
            "explanation": ["explain", "how", "why", "describe"]
        }
        
        query_lower = query.lower()
        detected_intents = []
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        return {
            "primary_intent": detected_intents[0] if detected_intents else "search",
            "all_intents": detected_intents,
            "query_length": len(query.split()),
            "domain": domain,
            "confidence": 0.8 if detected_intents else 0.5
        }
    
    async def _identify_key_concepts(self, inputs: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Default key concept identification"""
        query = inputs.get("query", "")
        
        # Simple concept extraction - should be enhanced with NER/NLP
        words = query.split()
        concepts = [word for word in words if len(word) > 3 and word.isalpha()]
        
        return {
            "key_concepts": concepts[:5],  # Top 5 concepts
            "concept_count": len(concepts),
            "query_complexity": "high" if len(concepts) > 10 else "medium" if len(concepts) > 5 else "low",
            "confidence": 0.7
        }
    
    async def _plan_search_approach(self, inputs: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Default search approach planning"""
        domain = inputs.get("domain")
        query = inputs.get("query", "")
        
        # Default tri-modal approach
        approach = {
            "vector_search": True,
            "graph_search": True,
            "gnn_enhancement": domain is not None,
            "parallel_execution": True,
            "search_depth": "medium",
            "confidence": 0.8
        }
        
        return approach
    
    async def _synthesize_reasoning_strategy(self, inputs: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Default reasoning strategy synthesis"""
        return {
            "strategy": "tri_modal_search_with_synthesis",
            "execution_order": ["vector", "graph", "gnn"],
            "synthesis_method": "weighted_combination",
            "confidence_threshold": 0.7,
            "confidence": 0.8
        }
    
    async def _analyze_evidence_source(self, inputs: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Default evidence source analysis"""
        source = inputs.get("source", "")
        query = inputs.get("query", "")
        
        return {
            "source": source,
            "relevance_score": 0.7,  # Placeholder
            "evidence_strength": "medium",
            "key_points": [],  # Would extract from actual source
            "confidence": 0.6
        }
    
    async def _synthesize_evidence(self, inputs: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Default evidence synthesis"""
        num_sources = inputs.get("num_sources", 0)
        
        return {
            "synthesis_method": "consensus_ranking",
            "sources_analyzed": num_sources,
            "overall_evidence_strength": "medium",
            "consensus_level": 0.7,
            "confidence": 0.8
        }


__all__ = [
    'ReasoningEngine',
    'ReasoningPattern', 
    'ReasoningChain',
    'ReasoningStep',
    'ReasoningResult',
    'ConfidenceLevel'
]