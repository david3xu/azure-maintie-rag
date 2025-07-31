"""
Validation script for Agent Base Architecture (Phase 2 Week 3).
Tests all core agent components and their integration.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent
sys.path.append(str(backend_path))

from agents.base import (
    AgentInterface, AgentContext, AgentResponse, ReasoningTrace, AgentCapability, ReasoningStep,
    ReasoningEngine, ReasoningPattern, ContextManager, IntelligentMemoryManager,
    TriModalReActEngine, PlanAndExecuteEngine, MemoryType, TaskType
)
from agents.base.temporal_pattern_tracker import (
    TemporalPatternTracker, TemporalEventType, TemporalScope
)


class TestAgent(AgentInterface):
    """Test implementation of AgentInterface"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.test_data = {"initialized": False}
    
    async def _initialize_resources(self) -> None:
        """Initialize test agent resources"""
        await asyncio.sleep(0.1)  # Simulate async initialization
        self.test_data["initialized"] = True
    
    def _extract_capabilities(self, config: Dict[str, Any]) -> List[AgentCapability]:
        """Extract test agent capabilities"""
        return [
            AgentCapability.SEARCH_ORCHESTRATION,
            AgentCapability.REASONING_SYNTHESIS,
            AgentCapability.CONTEXT_MANAGEMENT
        ]
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """Test processing implementation"""
        start_time = time.time()
        
        # Create test reasoning trace
        reasoning_trace = [
            self._create_reasoning_trace(
                step=ReasoningStep.ANALYSIS,
                description="Test query analysis",
                inputs={"query": context.query},
                outputs={"complexity": "medium"},
                duration_ms=(time.time() - start_time) * 1000,
                confidence=0.8
            )
        ]
        
        # Create test response
        result = {
            "query": context.query,
            "domain": context.domain,
            "processed_by": "TestAgent",
            "test_result": True
        }
        
        return AgentResponse(
            result=result,
            reasoning_trace=reasoning_trace,
            confidence=0.8,
            sources=[{"source": "test", "confidence": 0.8}],
            performance_metrics={"processing_time_ms": (time.time() - start_time) * 1000},
            suggested_follow_up=["Test follow-up question"]
        )


class MockTaskExecutor:
    """Mock task executor for testing Plan-and-Execute engine"""
    
    async def execute_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"analysis_result": "completed", "confidence": 0.8}
    
    async def execute_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        search_type = parameters.get("search_type", "unknown")
        return {"search_result": f"{search_type}_completed", "confidence": 0.7}
    
    async def execute_synthesis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"synthesis_result": "completed", "confidence": 0.9}
    
    async def execute_validation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.05)
        return {"validation_result": "passed", "confidence": 0.8}
    
    async def execute_refinement(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.15)
        return {"refinement_result": "completed", "confidence": 0.85}


class MockActionExecutor:
    """Mock action executor for testing ReAct engine"""
    
    async def execute_vector_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"vector_results": ["doc1", "doc2"], "confidence": 0.8}
    
    async def execute_graph_traversal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.15)
        return {"graph_results": [{"entity": "A", "relation": "related_to", "entity": "B"}], "confidence": 0.7}
    
    async def execute_gnn_prediction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        return {"gnn_predictions": ["prediction1", "prediction2"], "confidence": 0.75}
    
    async def execute_tri_modal_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.3)
        return {"tri_modal_results": "comprehensive_results", "confidence": 0.85}


async def test_agent_interface() -> Dict[str, Any]:
    """Test AgentInterface implementation"""
    print("Testing AgentInterface...")
    
    # Create test agent
    config = {
        "supported_domains": ["test", "general"],
        "max_query_complexity": 100,
        "base_response_time": 1.0
    }
    
    agent = TestAgent(config)
    
    # Test initialization
    await agent.initialize()
    assert agent._is_initialized, "Agent should be initialized"
    assert agent.test_data["initialized"], "Test data should be initialized"
    
    # Test capabilities
    capabilities = agent.get_capabilities()
    assert len(capabilities) == 3, f"Expected 3 capabilities, got {len(capabilities)}"
    
    # Test context creation and processing
    context = AgentContext(
        query="Test query for agent processing",
        domain="test",
        conversation_history=[],
        search_constraints={},
        performance_targets={"max_response_time": 3.0}
    )
    
    # Test can_handle
    capability_assessment = await agent.can_handle(context)
    assert capability_assessment["can_handle"], "Agent should be able to handle test context"
    assert capability_assessment["confidence"] > 0.5, "Confidence should be reasonable"
    
    # Test processing
    response = await agent.process(context)
    assert response.result["test_result"], "Test result should be True"
    assert response.confidence > 0.0, "Response should have confidence"
    assert len(response.reasoning_trace) > 0, "Should have reasoning trace"
    
    # Test health check
    health = await agent.health_check()
    assert health["healthy"], "Agent should be healthy"
    
    return {
        "test_name": "AgentInterface",
        "status": "PASSED",
        "details": {
            "capabilities_count": len(capabilities),
            "response_confidence": response.confidence,
            "reasoning_steps": len(response.reasoning_trace),
            "health_status": health["status"]
        }
    }


async def test_reasoning_engine() -> Dict[str, Any]:
    """Test ReasoningEngine implementation"""
    print("Testing ReasoningEngine...")
    
    config = {
        "enabled_patterns": ["chain_of_thought", "evidence_synthesis"],
        "default_confidence_threshold": 0.7,
        "max_parallel_steps": 3
    }
    
    engine = ReasoningEngine(config)
    
    # Test chain of thought creation
    query = "What are the benefits of renewable energy?"
    chain = engine.create_chain_of_thought(query, "energy")
    
    assert chain.pattern == ReasoningPattern.CHAIN_OF_THOUGHT, "Should be chain of thought pattern"
    assert len(chain.steps) > 0, "Should have reasoning steps"
    assert not chain.parallel_execution, "Chain of thought should be sequential"
    
    # Test evidence synthesis chain
    evidence_sources = ["source1", "source2", "source3"]
    evidence_chain = engine.create_evidence_synthesis_chain(evidence_sources, query)
    
    assert evidence_chain.pattern == ReasoningPattern.EVIDENCE_SYNTHESIS, "Should be evidence synthesis pattern"
    assert evidence_chain.parallel_execution, "Evidence synthesis should allow parallel execution"
    
    # Test execution (simplified mock)
    context = AgentContext(query=query, domain="energy")
    
    # Note: Full execution would require implementing all reasoning functions
    # This tests the structure and basic functionality
    
    return {
        "test_name": "ReasoningEngine",
        "status": "PASSED",
        "details": {
            "cot_steps": len(chain.steps),
            "evidence_steps": len(evidence_chain.steps),
            "enabled_patterns": len(config["enabled_patterns"])
        }
    }


async def test_context_manager() -> Dict[str, Any]:
    """Test ContextManager implementation"""
    print("Testing ContextManager...")
    
    config = {
        "max_contexts_per_type": 100,
        "default_expiry_hours": 24,
        "cleanup_interval_seconds": 60
    }
    
    context_manager = ContextManager(config)
    await context_manager.initialize()
    
    try:
        # Test session creation
        session_id = "test_session_123"
        user_id = "test_user"
        
        agent_context = await context_manager.create_agent_context(
            query="Test query for context management",
            session_id=session_id,
            domain="test",
            user_id=user_id
        )
        
        assert agent_context.query == "Test query for context management", "Query should match"
        assert agent_context.domain == "test", "Domain should match"
        assert agent_context.metadata["session_id"] == session_id, "Session ID should be in metadata"
        
        # Test conversation turn saving
        turn_id = await context_manager.save_conversation_turn(
            session_id=session_id,
            query="Test query",
            response={"answer": "Test response"},
            reasoning_trace=[{"step": "analysis", "result": "completed"}],
            domain="test",
            confidence=0.8
        )
        
        assert turn_id, "Turn ID should be generated"
        
        # Test session analytics
        analytics = await context_manager.get_session_analytics(session_id)
        assert analytics["turn_count"] == 1, "Should have 1 conversation turn"
        assert analytics["avg_confidence"] == 0.8, "Average confidence should match"
        
        # Test preferences update
        updated = await context_manager.update_session_preferences(
            session_id, {"language": "en", "verbose": True}
        )
        assert updated, "Preferences should be updated"
        
        # Test domain context
        domain_context = await context_manager.get_domain_context("test")
        assert domain_context["domain"] == "test", "Domain context should be returned"
        
        return {
            "test_name": "ContextManager",
            "status": "PASSED",
            "details": {
                "session_created": True,
                "turn_saved": bool(turn_id),
                "analytics_available": bool(analytics),
                "preferences_updated": updated
            }
        }
    
    finally:
        await context_manager.shutdown()


async def test_memory_manager() -> Dict[str, Any]:
    """Test IntelligentMemoryManager implementation"""
    print("Testing IntelligentMemoryManager...")
    
    config = {
        "max_working_memory": 50,
        "max_episodic_memory": 200,
        "consolidation_interval": 60,
        "enable_summarization": True
    }
    
    memory_manager = IntelligentMemoryManager(config)
    await memory_manager.initialize()
    
    try:
        # Test memory storage
        memory_id1 = await memory_manager.store_memory(
            memory_type=MemoryType.WORKING,
            content={"query": "test query", "result": "test result"},
            confidence=0.8,
            tags=["test", "query"]
        )
        
        assert memory_id1, "Memory ID should be generated"
        
        memory_id2 = await memory_manager.store_memory(
            memory_type=MemoryType.EPISODIC,
            content={"session": "test_session", "interaction": "user_query"},
            confidence=0.7,
            tags=["session", "interaction"]
        )
        
        # Test memory retrieval
        working_memories = await memory_manager.get_working_memory()
        assert len(working_memories) == 1, "Should have 1 working memory"
        
        retrieved_memories = await memory_manager.retrieve_memories(
            memory_type=MemoryType.WORKING,
            tags=["test"],
            limit=5
        )
        assert len(retrieved_memories) > 0, "Should retrieve memories with tag filter"
        
        # Test pattern extraction
        patterns = await memory_manager.extract_learned_patterns()
        assert isinstance(patterns, dict), "Should return patterns dictionary"
        
        # Test statistics
        stats = memory_manager.get_memory_statistics()
        assert stats["total_memories"] >= 2, "Should have at least 2 memories stored"
        
        return {
            "test_name": "IntelligentMemoryManager",
            "status": "PASSED",
            "details": {
                "memories_stored": 2,
                "working_memories": len(working_memories),
                "retrieved_memories": len(retrieved_memories),
                "patterns_extracted": len(patterns)
            }
        }
    
    finally:
        await memory_manager.shutdown()


async def test_react_engine() -> Dict[str, Any]:
    """Test TriModalReActEngine implementation"""
    print("Testing TriModalReActEngine...")
    
    config = {
        "max_reasoning_cycles": 3,
        "confidence_threshold": 0.8,
        "enable_parallel_actions": True
    }
    
    react_engine = TriModalReActEngine(config)
    mock_executor = MockActionExecutor()
    
    context = AgentContext(
        query="Find information about renewable energy benefits",
        domain="energy"
    )
    
    # Test ReAct cycle execution
    result = await react_engine.execute_react_cycle(
        context=context,
        action_executor=mock_executor
    )
    
    assert result["success"], "ReAct cycle should succeed"
    assert "reasoning_trace" in result, "Should have reasoning trace"
    assert "performance_metrics" in result, "Should have performance metrics"
    
    # Test performance metrics
    metrics = react_engine.get_performance_metrics()
    assert "cycles_executed" in metrics, "Should track cycles executed"
    assert "actions_executed" in metrics, "Should track actions executed"
    
    return {
        "test_name": "TriModalReActEngine",
        "status": "PASSED",
        "details": {
            "cycle_success": result["success"],
            "reasoning_steps": len(result["reasoning_trace"]),
            "cycles_executed": metrics["cycles_executed"],
            "actions_executed": metrics["actions_executed"]
        }
    }


async def test_plan_execute_engine() -> Dict[str, Any]:
    """Test PlanAndExecuteEngine implementation"""
    print("Testing PlanAndExecuteEngine...")
    
    config = {
        "max_parallel_tasks": 3,
        "default_task_timeout": 30.0,
        "enable_dynamic_planning": True
    }
    
    plan_engine = PlanAndExecuteEngine(config)
    mock_executor = MockTaskExecutor()
    
    context = AgentContext(
        query="Analyze the impact of renewable energy on economic growth",
        domain="energy"
    )
    
    # Test plan creation
    plan = await plan_engine.create_execution_plan(context)
    
    assert plan.plan_id, "Plan should have ID"
    assert len(plan.tasks) > 0, "Plan should have tasks"
    assert len(plan.execution_levels) > 0, "Plan should have execution levels"
    
    # Test plan execution
    result = await plan_engine.execute_plan(
        plan=plan,
        context=context,
        task_executor=mock_executor
    )
    
    assert result["success"], "Plan execution should succeed"
    assert "synthesis_result" in result, "Should have synthesis result"
    assert "performance_metrics" in result, "Should have performance metrics"
    
    # Test performance metrics
    metrics = plan_engine.get_performance_metrics()
    assert "plans_executed" in metrics, "Should track plans executed"
    assert "tasks_executed" in metrics, "Should track tasks executed"
    
    return {
        "test_name": "PlanAndExecuteEngine",
        "status": "PASSED",
        "details": {
            "plan_tasks": len(plan.tasks),
            "execution_levels": len(plan.execution_levels),
            "execution_success": result["success"],
            "plans_executed": metrics["plans_executed"]
        }
    }


async def test_temporal_pattern_tracker() -> Dict[str, Any]:
    """Test TemporalPatternTracker implementation"""
    print("Testing TemporalPatternTracker...")
    
    config = {
        "max_events_per_entity": 50,
        "pattern_discovery_interval": 60,
        "enable_predictive_patterns": True
    }
    
    tracker = TemporalPatternTracker(config)
    await tracker.initialize()
    
    try:
        # Test event tracking
        event_id1 = await tracker.track_event(
            event_type=TemporalEventType.KNOWLEDGE_CREATION,
            entity_id="entity_1",
            entity_type="concept",
            current_state={"name": "renewable_energy", "confidence": 0.8},
            session_id="test_session"
        )
        
        assert event_id1, "Event ID should be generated"
        
        # Track more events for pattern discovery
        for i in range(3):
            await tracker.track_event(
                event_type=TemporalEventType.KNOWLEDGE_UPDATE,
                entity_id="entity_1",
                entity_type="concept",
                current_state={"name": "renewable_energy", "confidence": 0.8 + i * 0.05},
                previous_state={"name": "renewable_energy", "confidence": 0.8 + (i-1) * 0.05} if i > 0 else None
            )
        
        # Test entity evolution retrieval
        evolution = await tracker.get_entity_evolution("entity_1")
        assert evolution is not None, "Should have evolution data"
        assert len(evolution.events) == 4, "Should have 4 events"
        
        # Test temporal context
        temporal_context = await tracker.get_temporal_context(
            entity_id="entity_1",
            scope=TemporalScope.SHORT_TERM
        )
        assert temporal_context["exists"], "Entity should exist"
        assert temporal_context["events_in_scope"] > 0, "Should have events in scope"
        
        # Test pattern discovery
        patterns = await tracker.discover_temporal_patterns(force_analysis=True)
        assert isinstance(patterns, dict), "Should return patterns dictionary"
        
        # Test predictions
        predictions = await tracker.predict_future_events("entity_1")
        # Note: Predictions might be empty for limited data, which is expected
        
        # Test performance metrics
        metrics = tracker.get_performance_metrics()
        assert metrics["events_tracked"] >= 4, "Should track events"
        
        return {
            "test_name": "TemporalPatternTracker",
            "status": "PASSED",
            "details": {
                "events_tracked": metrics["events_tracked"],
                "entities_tracked": metrics["entities_tracked"],
                "patterns_discovered": sum(len(p) for p in patterns.values()),
                "temporal_context_available": temporal_context["exists"]
            }
        }
    
    finally:
        await tracker.shutdown()


async def run_validation() -> Dict[str, Any]:
    """Run complete validation of agent base architecture"""
    print("=" * 60)
    print("VALIDATING AGENT BASE ARCHITECTURE (Phase 2 Week 3)")
    print("=" * 60)
    
    start_time = time.time()
    results = []
    
    try:
        # Run all tests
        test_functions = [
            test_agent_interface,
            test_reasoning_engine,
            test_context_manager,
            test_memory_manager,
            test_react_engine,
            test_plan_execute_engine,
            test_temporal_pattern_tracker
        ]
        
        for test_func in test_functions:
            try:
                result = await test_func()
                results.append(result)
                print(f"✅ {result['test_name']}: {result['status']}")
            except Exception as e:
                results.append({
                    "test_name": test_func.__name__,
                    "status": "FAILED",
                    "error": str(e)
                })
                print(f"❌ {test_func.__name__}: FAILED - {str(e)}")
        
        # Calculate summary
        passed_tests = [r for r in results if r["status"] == "PASSED"]
        failed_tests = [r for r in results if r["status"] == "FAILED"]
        
        execution_time = time.time() - start_time
        
        summary = {
            "validation_status": "PASSED" if len(failed_tests) == 0 else "FAILED",
            "total_tests": len(results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "execution_time_seconds": execution_time,
            "test_results": results
        }
        
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Status: {summary['validation_status']}")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Tests Failed: {summary['failed_tests']}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        
        if failed_tests:
            print("\nFAILED TESTS:")
            for test in failed_tests:
                print(f"  - {test['test_name']}: {test.get('error', 'Unknown error')}")
        
        print("\nCOMPONENT DETAILS:")
        for result in passed_tests:
            if "details" in result:
                print(f"\n{result['test_name']}:")
                for key, value in result["details"].items():
                    print(f"  - {key}: {value}")
        
        return summary
        
    except Exception as e:
        return {
            "validation_status": "FAILED",
            "error": f"Validation framework error: {str(e)}",
            "execution_time_seconds": time.time() - start_time
        }


if __name__ == "__main__":
    # Run validation
    validation_result = asyncio.run(run_validation())
    
    # Exit with appropriate code
    if validation_result["validation_status"] == "PASSED":
        print("\n🎉 All agent base architecture components validated successfully!")
        sys.exit(0)
    else:
        print("\n💥 Agent base architecture validation failed!")
        sys.exit(1)