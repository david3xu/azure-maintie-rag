"""
ReAct Engine - Reason → Act → Observe pattern for tri-modal search coordination.
Implements systematic reasoning with action execution and observation integration.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from .agent_interface import AgentContext, ReasoningTrace, ReasoningStep as BaseReasoningStep

logger = logging.getLogger(__name__)


class ReActStep(Enum):
    """ReAct cycle steps"""
    REASON = "reason"
    ACT = "act"
    OBSERVE = "observe"
    SYNTHESIZE = "synthesize"


class ActionType(Enum):
    """Types of actions the agent can take"""
    VECTOR_SEARCH = "vector_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    GNN_PREDICTION = "gnn_prediction"
    TRI_MODAL_SEARCH = "tri_modal_search"
    CONTEXT_ANALYSIS = "context_analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


@dataclass
class ReActAction:
    """Action to be executed in ReAct cycle"""
    action_id: str
    action_type: ActionType
    parameters: Dict[str, Any]
    expected_outcome: str
    confidence_threshold: float = 0.7
    timeout_seconds: float = 30.0
    retry_attempts: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Result of executing an action"""
    action_id: str
    success: bool
    result: Any
    confidence: float
    execution_time_ms: float
    observations: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningState:
    """Current reasoning state in ReAct cycle"""
    cycle_id: str
    step: ReActStep
    query: str
    domain: Optional[str]
    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    accumulated_confidence: float = 0.0
    goal_achieved: bool = False
    max_cycles: int = 5
    current_cycle: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TriModalReActEngine:
    """
    ReAct (Reason → Act → Observe) engine optimized for tri-modal search coordination.
    
    Implements systematic reasoning cycles that coordinate Vector + Graph + GNN search
    modalities based on query analysis and dynamic strategy adjustment.
    
    Features:
    - Dynamic modality selection based on query characteristics
    - Intelligent action chaining and parallel execution
    - Confidence-based early termination
    - Comprehensive observation tracking
    - Performance-optimized coordination
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ReAct engine with tri-modal search configuration.
        
        Args:
            config: Engine configuration
                - max_reasoning_cycles: Maximum reasoning cycles per query
                - confidence_threshold: Minimum confidence for goal achievement
                - enable_parallel_actions: Enable parallel action execution
                - modality_weights: Weights for different search modalities
                - performance_targets: Response time and quality targets
        """
        self.config = config
        self.max_reasoning_cycles = config.get("max_reasoning_cycles", 5)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.enable_parallel_actions = config.get("enable_parallel_actions", True)
        self.modality_weights = config.get("modality_weights", {
            "vector": 0.4,
            "graph": 0.35,
            "gnn": 0.25
        })
        self.performance_targets = config.get("performance_targets", {
            "max_response_time": 3.0,
            "min_confidence": 0.7
        })
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Active reasoning states
        self._active_states: Dict[str, ReasoningState] = {}
        
        # Performance tracking
        self.metrics = {
            "cycles_executed": 0,
            "actions_executed": 0,
            "successful_cycles": 0,
            "avg_cycle_time": 0.0,
            "modality_usage": {"vector": 0, "graph": 0, "gnn": 0, "tri_modal": 0}
        }
    
    async def execute_react_cycle(
        self, 
        context: AgentContext,
        action_executor: Any,  # Interface for executing actions
        goal_checker: Optional[Any] = None  # Interface for checking goal achievement
    ) -> Dict[str, Any]:
        """
        Execute complete ReAct reasoning cycle for tri-modal search.
        
        Args:
            context: Agent context with query and domain
            action_executor: Interface for executing search actions
            goal_checker: Optional interface for checking goal achievement
            
        Returns:
            Dictionary with:
            - success: bool
            - result: Final reasoning result
            - reasoning_trace: Complete reasoning trace
            - performance_metrics: Execution metrics
            - observations: All observations collected
        """
        start_time = time.time()
        cycle_id = f"react_{int(time.time())}_{hash(context.query) % 10000}"
        
        # Initialize reasoning state
        reasoning_state = ReasoningState(
            cycle_id=cycle_id,
            step=ReActStep.REASON,
            query=context.query,
            domain=context.domain,
            max_cycles=self.max_reasoning_cycles
        )
        
        self._active_states[cycle_id] = reasoning_state
        reasoning_trace = []
        
        try:
            self.logger.info(f"Starting ReAct cycle {cycle_id} for query: {context.query[:100]}...")
            
            # Execute reasoning cycles
            while not reasoning_state.goal_achieved and reasoning_state.current_cycle < reasoning_state.max_cycles:
                reasoning_state.current_cycle += 1
                
                # REASON: Analyze current state and plan next actions
                reasoning_result = await self._reason_step(reasoning_state, context)
                reasoning_trace.append(reasoning_result)
                
                # ACT: Execute planned actions
                action_result = await self._act_step(reasoning_state, context, action_executor)
                reasoning_trace.append(action_result)
                
                # OBSERVE: Process action results and update state
                observation_result = await self._observe_step(reasoning_state, action_result)
                reasoning_trace.append(observation_result)
                
                # Check if goal is achieved
                if goal_checker:
                    goal_achieved = await goal_checker.check_goal(reasoning_state, context)
                else:
                    goal_achieved = await self._default_goal_check(reasoning_state)
                
                reasoning_state.goal_achieved = goal_achieved
                
                # Early termination if confidence threshold met
                if reasoning_state.accumulated_confidence >= self.confidence_threshold:
                    reasoning_state.goal_achieved = True
                    break
            
            # SYNTHESIZE: Final synthesis of all observations
            synthesis_result = await self._synthesize_step(reasoning_state, context)
            reasoning_trace.append(synthesis_result)
            
            execution_time = (time.time() - start_time) * 1000
            success = reasoning_state.goal_achieved
            
            # Update metrics
            self.metrics["cycles_executed"] += reasoning_state.current_cycle
            if success:
                self.metrics["successful_cycles"] += 1
            self._update_cycle_metrics(execution_time)
            
            self.logger.info(
                f"ReAct cycle {cycle_id} completed: "
                f"success={success}, cycles={reasoning_state.current_cycle}, "
                f"confidence={reasoning_state.accumulated_confidence:.3f}, "
                f"time={execution_time:.1f}ms"
            )
            
            return {
                "success": success,
                "result": synthesis_result.outputs.get("synthesis_result", {}),
                "reasoning_trace": reasoning_trace,
                "performance_metrics": {
                    "total_execution_time_ms": execution_time,
                    "cycles_executed": reasoning_state.current_cycle,
                    "actions_executed": len(reasoning_state.actions_taken),
                    "final_confidence": reasoning_state.accumulated_confidence,
                    "goal_achieved": reasoning_state.goal_achieved
                },
                "observations": reasoning_state.observations
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"ReAct cycle {cycle_id} failed: {e}")
            
            return {
                "success": False,
                "result": {},
                "reasoning_trace": reasoning_trace,
                "performance_metrics": {
                    "total_execution_time_ms": execution_time,
                    "error": str(e)
                },
                "observations": reasoning_state.observations,
                "error": str(e)
            }
        finally:
            self._active_states.pop(cycle_id, None)
    
    async def stream_react_cycle(
        self,
        context: AgentContext,
        action_executor: Any,
        goal_checker: Optional[Any] = None
    ) -> AsyncIterator[ReasoningTrace]:
        """
        Stream ReAct cycle execution in real-time.
        
        Args:
            context: Agent context
            action_executor: Action execution interface
            goal_checker: Optional goal checking interface
            
        Yields:
            ReasoningTrace objects as reasoning progresses
        """
        cycle_id = f"react_stream_{int(time.time())}_{hash(context.query) % 10000}"
        
        reasoning_state = ReasoningState(
            cycle_id=cycle_id,
            step=ReActStep.REASON,
            query=context.query,
            domain=context.domain,
            max_cycles=self.max_reasoning_cycles
        )
        
        self._active_states[cycle_id] = reasoning_state
        
        try:
            while not reasoning_state.goal_achieved and reasoning_state.current_cycle < reasoning_state.max_cycles:
                reasoning_state.current_cycle += 1
                
                # Stream reasoning step
                reasoning_result = await self._reason_step(reasoning_state, context)
                yield reasoning_result
                
                # Stream action step
                action_result = await self._act_step(reasoning_state, context, action_executor)
                yield action_result
                
                # Stream observation step
                observation_result = await self._observe_step(reasoning_state, action_result)
                yield observation_result
                
                # Check goal achievement
                if goal_checker:
                    goal_achieved = await goal_checker.check_goal(reasoning_state, context)
                else:
                    goal_achieved = await self._default_goal_check(reasoning_state)
                
                reasoning_state.goal_achieved = goal_achieved
                
                if reasoning_state.accumulated_confidence >= self.confidence_threshold:
                    reasoning_state.goal_achieved = True
                    break
            
            # Stream final synthesis
            synthesis_result = await self._synthesize_step(reasoning_state, context)
            yield synthesis_result
            
        finally:
            self._active_states.pop(cycle_id, None)
    
    async def _reason_step(self, state: ReasoningState, context: AgentContext) -> ReasoningTrace:
        """Execute reasoning step - analyze current state and plan actions"""
        start_time = time.time()
        
        # Analyze query characteristics for modality selection
        query_analysis = await self._analyze_query_for_modalities(context.query, context.domain)
        
        # Determine optimal search strategy based on current observations
        search_strategy = await self._determine_search_strategy(
            query_analysis, 
            state.observations,
            state.current_cycle
        )
        
        # Plan specific actions
        planned_actions = await self._plan_actions(search_strategy, context, state)
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Update state
        state.metadata["current_strategy"] = search_strategy
        state.metadata["planned_actions"] = [a.action_id for a in planned_actions]
        
        return ReasoningTrace(
            step=BaseReasoningStep.ANALYSIS,
            description=f"Reasoning cycle {state.current_cycle}: Analyzed query and planned {len(planned_actions)} actions",
            inputs={
                "query": context.query,
                "domain": context.domain,
                "cycle": state.current_cycle,
                "previous_observations": len(state.observations)
            },
            outputs={
                "query_analysis": query_analysis,
                "search_strategy": search_strategy,
                "planned_actions": [
                    {
                        "action_id": a.action_id,
                        "type": a.action_type.value,
                        "expected_outcome": a.expected_outcome
                    } for a in planned_actions
                ]
            },
            duration_ms=duration_ms,
            confidence=query_analysis.get("confidence", 0.8)
        )
    
    async def _act_step(
        self, 
        state: ReasoningState, 
        context: AgentContext,
        action_executor: Any
    ) -> ReasoningTrace:
        """Execute action step - perform planned search actions"""
        start_time = time.time()
        
        planned_actions = state.metadata.get("planned_actions", [])
        if not planned_actions:
            # Fallback to default tri-modal search
            planned_actions = [
                ReActAction(
                    action_id="fallback_tri_modal",
                    action_type=ActionType.TRI_MODAL_SEARCH,
                    parameters={"query": context.query, "domain": context.domain},
                    expected_outcome="Comprehensive search results"
                )
            ]
        
        # Execute actions (parallel if enabled and appropriate)
        action_results = []
        
        if self.enable_parallel_actions and len(planned_actions) > 1:
            # Parallel execution
            tasks = []
            for action in planned_actions:
                task = asyncio.create_task(self._execute_single_action(action, action_executor))
                tasks.append(task)
            
            action_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for i, result in enumerate(action_results):
                if isinstance(result, Exception):
                    action_results[i] = ActionResult(
                        action_id=planned_actions[i].action_id,
                        success=False,
                        result=None,
                        confidence=0.0,
                        execution_time_ms=0.0,
                        observations={},
                        error=str(result)
                    )
        else:
            # Sequential execution
            for action in planned_actions:
                result = await self._execute_single_action(action, action_executor)
                action_results.append(result)
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Update state
        successful_actions = [r for r in action_results if r.success]
        state.actions_taken.extend([r.action_id for r in successful_actions])
        
        # Update modality usage metrics
        for result in action_results:
            if result.success:
                action_type = result.metadata.get("action_type", "unknown")
                if action_type in self.metrics["modality_usage"]:
                    self.metrics["modality_usage"][action_type] += 1
        
        self.metrics["actions_executed"] += len(action_results)
        
        return ReasoningTrace(
            step=BaseReasoningStep.EXECUTION,
            description=f"Executed {len(action_results)} actions ({len(successful_actions)} successful)",
            inputs={
                "planned_actions": len(planned_actions),
                "execution_mode": "parallel" if self.enable_parallel_actions else "sequential"
            },
            outputs={
                "action_results": [
                    {
                        "action_id": r.action_id,
                        "success": r.success,
                        "confidence": r.confidence,
                        "execution_time_ms": r.execution_time_ms
                    } for r in action_results
                ],
                "successful_actions": len(successful_actions),
                "total_execution_time_ms": duration_ms
            },
            duration_ms=duration_ms,
            confidence=sum(r.confidence for r in successful_actions) / max(1, len(successful_actions))
        )
    
    async def _observe_step(
        self, 
        state: ReasoningState, 
        action_trace: ReasoningTrace
    ) -> ReasoningTrace:
        """Execute observation step - process action results and update state"""
        start_time = time.time()
        
        # Extract observations from action results
        action_results = action_trace.outputs.get("action_results", [])
        new_observations = []
        confidence_sum = 0.0
        
        for action_result in action_results:
            if action_result["success"]:
                observation = {
                    "action_id": action_result["action_id"],
                    "confidence": action_result["confidence"],
                    "execution_time": action_result["execution_time_ms"],
                    "timestamp": time.time()
                }
                new_observations.append(observation)
                confidence_sum += action_result["confidence"]
        
        # Update accumulated confidence (weighted average)
        if new_observations:
            new_confidence = confidence_sum / len(new_observations)
            # Weighted combination with existing confidence
            total_observations = len(state.observations) + len(new_observations)
            state.accumulated_confidence = (
                (state.accumulated_confidence * len(state.observations) + 
                 new_confidence * len(new_observations)) / 
                total_observations
            )
        
        # Add new observations to state
        state.observations.extend(new_observations)
        
        # Analyze observation patterns for insights
        insights = await self._analyze_observations(state.observations)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ReasoningTrace(
            step=BaseReasoningStep.VALIDATION,
            description=f"Processed {len(new_observations)} new observations, total: {len(state.observations)}",
            inputs={
                "new_observations": len(new_observations),
                "action_results": len(action_results)
            },
            outputs={
                "observations_added": len(new_observations),
                "total_observations": len(state.observations),
                "updated_confidence": state.accumulated_confidence,
                "insights": insights
            },
            duration_ms=duration_ms,
            confidence=state.accumulated_confidence
        )
    
    async def _synthesize_step(self, state: ReasoningState, context: AgentContext) -> ReasoningTrace:
        """Execute synthesis step - combine all observations into final result"""
        start_time = time.time()
        
        # Synthesize observations into coherent result
        synthesis_result = await self._synthesize_observations(state.observations, context)
        
        # Calculate final confidence
        final_confidence = min(1.0, state.accumulated_confidence * synthesis_result.get("synthesis_confidence", 1.0))
        
        # Create comprehensive result
        final_result = {
            "query": context.query,
            "domain": context.domain,
            "synthesis": synthesis_result,
            "confidence": final_confidence,
            "reasoning_cycles": state.current_cycle,
            "actions_executed": len(state.actions_taken),
            "observations_collected": len(state.observations),
            "goal_achieved": state.goal_achieved
        }
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ReasoningTrace(
            step=BaseReasoningStep.SYNTHESIS,
            description=f"Synthesized {len(state.observations)} observations into final result",
            inputs={
                "total_observations": len(state.observations),
                "reasoning_cycles": state.current_cycle,
                "accumulated_confidence": state.accumulated_confidence
            },
            outputs={
                "synthesis_result": final_result,
                "final_confidence": final_confidence,
                "goal_achieved": state.goal_achieved
            },
            duration_ms=duration_ms,
            confidence=final_confidence
        )
    
    async def _analyze_query_for_modalities(self, query: str, domain: Optional[str]) -> Dict[str, Any]:
        """Analyze query to determine optimal search modalities"""
        query_lower = query.lower()
        
        # Heuristic analysis for modality selection
        vector_indicators = ["find", "search", "similar", "like", "about", "related"]
        graph_indicators = ["connected", "relationship", "related to", "link", "network", "path"]
        gnn_indicators = ["predict", "recommend", "suggest", "pattern", "trend", "likely"]
        
        vector_score = sum(1 for indicator in vector_indicators if indicator in query_lower)
        graph_score = sum(1 for indicator in graph_indicators if indicator in query_lower)
        gnn_score = sum(1 for indicator in gnn_indicators if indicator in query_lower)
        
        # Normalize scores
        total_score = max(1, vector_score + graph_score + gnn_score)
        
        return {
            "modality_scores": {
                "vector": vector_score / total_score,
                "graph": graph_score / total_score,
                "gnn": gnn_score / total_score
            },
            "recommended_modalities": [
                modality for modality, score in [
                    ("vector", vector_score),
                    ("graph", graph_score),
                    ("gnn", gnn_score)
                ] if score > 0
            ],
            "query_complexity": len(query.split()),
            "domain": domain,
            "confidence": 0.8
        }
    
    async def _determine_search_strategy(
        self, 
        query_analysis: Dict[str, Any],
        observations: List[Dict[str, Any]],
        cycle: int
    ) -> Dict[str, Any]:
        """Determine search strategy based on analysis and previous observations"""
        
        modality_scores = query_analysis["modality_scores"]
        
        # Adjust strategy based on cycle and previous results
        if cycle == 1:
            # First cycle: use query analysis
            strategy = "initial_tri_modal"
            modalities = ["vector", "graph", "gnn"]
        elif len(observations) == 0:
            # No previous results: expand search
            strategy = "expanded_search" 
            modalities = ["vector", "graph", "gnn"]
        else:
            # Subsequent cycles: focus on best-performing modalities
            best_modality = max(modality_scores.keys(), key=lambda k: modality_scores[k])
            strategy = "focused_search"
            modalities = [best_modality]
            
            # Add complementary modality if confidence is low
            avg_confidence = sum(obs.get("confidence", 0) for obs in observations) / len(observations)
            if avg_confidence < 0.7:
                complementary = {"vector": "graph", "graph": "gnn", "gnn": "vector"}
                modalities.append(complementary.get(best_modality, "vector"))
        
        return {
            "strategy_type": strategy,
            "selected_modalities": modalities,
            "cycle": cycle,
            "confidence_threshold": self.confidence_threshold,
            "parallel_execution": self.enable_parallel_actions and len(modalities) > 1
        }
    
    async def _plan_actions(
        self, 
        strategy: Dict[str, Any], 
        context: AgentContext,
        state: ReasoningState
    ) -> List[ReActAction]:
        """Plan specific actions based on strategy"""
        actions = []
        selected_modalities = strategy["selected_modalities"]
        
        # Create actions for each selected modality
        for i, modality in enumerate(selected_modalities):
            if modality == "vector":
                actions.append(ReActAction(
                    action_id=f"vector_search_{state.current_cycle}_{i}",
                    action_type=ActionType.VECTOR_SEARCH,
                    parameters={
                        "query": context.query,
                        "domain": context.domain,
                        "top_k": 10
                    },
                    expected_outcome="Semantically similar documents and content"
                ))
            elif modality == "graph":
                actions.append(ReActAction(
                    action_id=f"graph_traversal_{state.current_cycle}_{i}",
                    action_type=ActionType.GRAPH_TRAVERSAL,
                    parameters={
                        "query": context.query,
                        "domain": context.domain,
                        "max_depth": 3
                    },
                    expected_outcome="Related entities and relationships"
                ))
            elif modality == "gnn":
                actions.append(ReActAction(
                    action_id=f"gnn_prediction_{state.current_cycle}_{i}",
                    action_type=ActionType.GNN_PREDICTION,
                    parameters={
                        "query": context.query,
                        "domain": context.domain,
                        "prediction_type": "relevance"
                    },
                    expected_outcome="Predicted relevant content and patterns"
                ))
        
        # Add synthesis action if multiple modalities
        if len(actions) > 1:
            actions.append(ReActAction(
                action_id=f"synthesis_{state.current_cycle}",
                action_type=ActionType.SYNTHESIS,
                parameters={
                    "input_actions": [a.action_id for a in actions[:-1]],
                    "synthesis_method": "weighted_combination"
                },
                expected_outcome="Synthesized multi-modal results"
            ))
        
        return actions
    
    async def _execute_single_action(self, action: ReActAction, executor: Any) -> ActionResult:
        """Execute a single action with error handling and timeout"""
        start_time = time.time()
        
        try:
            # Execute action based on type
            if action.action_type == ActionType.VECTOR_SEARCH:
                result = await asyncio.wait_for(
                    executor.execute_vector_search(action.parameters),
                    timeout=action.timeout_seconds
                )
            elif action.action_type == ActionType.GRAPH_TRAVERSAL:
                result = await asyncio.wait_for(
                    executor.execute_graph_traversal(action.parameters),
                    timeout=action.timeout_seconds
                )
            elif action.action_type == ActionType.GNN_PREDICTION:
                result = await asyncio.wait_for(
                    executor.execute_gnn_prediction(action.parameters),
                    timeout=action.timeout_seconds
                )
            elif action.action_type == ActionType.TRI_MODAL_SEARCH:
                result = await asyncio.wait_for(
                    executor.execute_tri_modal_search(action.parameters),
                    timeout=action.timeout_seconds
                )
            else:
                result = {"message": f"Action type {action.action_type} not implemented"}
            
            execution_time = (time.time() - start_time) * 1000
            
            # Calculate confidence (simplified)
            confidence = result.get("confidence", 0.7) if isinstance(result, dict) else 0.5
            
            return ActionResult(
                action_id=action.action_id,
                success=True,
                result=result,
                confidence=confidence,
                execution_time_ms=execution_time,
                observations={"action_type": action.action_type.value},
                metadata={"action_type": action.action_type.value}
            )
            
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            return ActionResult(
                action_id=action.action_id,
                success=False,
                result=None,
                confidence=0.0,
                execution_time_ms=execution_time,
                observations={},
                error=f"Action timed out after {action.timeout_seconds}s"
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ActionResult(
                action_id=action.action_id,
                success=False,
                result=None,
                confidence=0.0,
                execution_time_ms=execution_time,
                observations={},
                error=f"Action failed: {str(e)}"
            )
    
    async def _default_goal_check(self, state: ReasoningState) -> bool:
        """Default goal achievement check"""
        # Goal achieved if:
        # 1. High confidence threshold met
        # 2. Sufficient observations collected
        # 3. Recent observations show consistency
        
        if state.accumulated_confidence >= self.confidence_threshold:
            return True
        
        if len(state.observations) >= 3:
            # Check recent observation consistency
            recent_confidences = [
                obs.get("confidence", 0) 
                for obs in state.observations[-3:]
            ]
            if all(conf >= 0.6 for conf in recent_confidences):
                return True
        
        return False
    
    async def _analyze_observations(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze observations for patterns and insights"""
        if not observations:
            return {"pattern_count": 0}
        
        # Calculate confidence trends
        confidences = [obs.get("confidence", 0) for obs in observations]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Analyze execution times
        execution_times = [obs.get("execution_time", 0) for obs in observations]
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        return {
            "observation_count": len(observations),
            "avg_confidence": avg_confidence,
            "confidence_trend": "stable",  # Simplified
            "avg_execution_time": avg_execution_time,
            "pattern_consistency": avg_confidence > 0.7
        }
    
    async def _synthesize_observations(
        self, 
        observations: List[Dict[str, Any]], 
        context: AgentContext
    ) -> Dict[str, Any]:
        """Synthesize all observations into final result"""
        if not observations:
            return {
                "synthesis_method": "no_observations",
                "synthesis_confidence": 0.0,
                "result": "No observations to synthesize"
            }
        
        # Calculate weighted synthesis based on confidence
        total_weight = sum(obs.get("confidence", 0) for obs in observations)
        
        if total_weight == 0:
            return {
                "synthesis_method": "equal_weight",
                "synthesis_confidence": 0.0,
                "result": "All observations had zero confidence"
            }
        
        # Synthesize (simplified implementation)
        synthesis_confidence = total_weight / len(observations)
        
        return {
            "synthesis_method": "confidence_weighted",
            "synthesis_confidence": synthesis_confidence,
            "observation_count": len(observations),
            "avg_confidence": synthesis_confidence,
            "result": f"Synthesized {len(observations)} observations with {synthesis_confidence:.3f} confidence"
        }
    
    def _update_cycle_metrics(self, execution_time_ms: float) -> None:
        """Update performance metrics"""
        current_avg = self.metrics["avg_cycle_time"]
        total_cycles = self.metrics["cycles_executed"]
        
        if total_cycles == 0:
            self.metrics["avg_cycle_time"] = execution_time_ms
        else:
            self.metrics["avg_cycle_time"] = (
                (current_avg * (total_cycles - 1) + execution_time_ms) / total_cycles
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        total_cycles = max(1, self.metrics["cycles_executed"])
        success_rate = self.metrics["successful_cycles"] / total_cycles
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "avg_actions_per_cycle": self.metrics["actions_executed"] / total_cycles,
            "active_reasoning_states": len(self._active_states)
        }


__all__ = [
    'TriModalReActEngine',
    'ReActStep',
    'ActionType',
    'ReActAction',
    'ActionResult',
    'ReasoningState'
]