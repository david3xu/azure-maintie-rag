#!/usr/bin/env python3
"""
Test script to verify Universal Workflow Manager integration
Ensures output format matches frontend WorkflowStep interface exactly
"""

import asyncio
import json
import logging
from datetime import datetime

from core.workflow.universal_workflow_manager import (
    create_workflow_manager, WorkflowStep, WorkflowStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_workflow_step_creation():
    """Test that WorkflowStep objects match frontend interface exactly"""

    print("🧪 Testing WorkflowStep Creation...")

    # Create a workflow manager
    workflow_manager = create_workflow_manager("test query for pump failure", "maintenance")

    # Start a test step
    step_number = await workflow_manager.start_step(
        step_name="initialize_enhanced_rag",
        user_friendly_name="🔧 Setting up AI system...",
        technology="Enhanced Universal RAG",
        estimated_progress=10,
        technical_data={
            "domain": "maintenance",
            "component": "system_initialization",
            "version": "2.0"
        }
    )

    # Update the step with technical details
    await workflow_manager.update_step(
        step_number,
        "Initializing Universal RAG components...",
        15,
        {
            "status": "initializing_components",
            "modules_loaded": ["knowledge_extractor", "text_processor"],
            "memory_usage": "245MB"
        },
        "Enhanced initialization"
    )

    # Complete the step
    await workflow_manager.complete_step(
        step_number,
        "AI system ready for maintenance domain",
        20,
        {
            "domain": "maintenance",
            "components_initialized": True,
            "performance_score": 0.94
        },
        "Optimized startup sequence"
    )

    # Get the completed step
    step = workflow_manager._get_step(step_number)

    print(f"✅ Created WorkflowStep with ID: {step.query_id}")

    return step


def test_three_layer_disclosure():
    """Test three-layer progressive disclosure"""

    print("\n🔍 Testing Three-Layer Progressive Disclosure...")

    # Create a test step
    step = WorkflowStep(
        query_id="test-query-123",
        step_number=1,
        step_name="process_enhanced_query",
        user_friendly_name="🧠 Processing your question with AI...",
        status=WorkflowStatus.COMPLETED.value,
        progress_percentage=85,
        technology="GPT-4 + Universal RAG",
        details="Generated comprehensive response with 7 sources",
        processing_time_ms=2340.5,
        fix_applied="Advanced RAG processing",
        technical_data={
            "search_results_count": 7,
            "processing_time": 2.34,
            "confidence_indicators": {"relevance": 0.92, "completeness": 0.88},
            "model_used": "gpt-4-turbo",
            "tokens_consumed": 1250
        }
    )

    # Test Layer 1: User-friendly
    layer_1 = step.to_layer_dict(1)
    expected_layer_1_fields = {
        "query_id", "step_number", "status", "progress_percentage",
        "timestamp", "user_friendly_name"
    }

    print(f"📱 Layer 1 (User-friendly): {len(layer_1)} fields")
    assert set(layer_1.keys()) == expected_layer_1_fields, f"Layer 1 fields mismatch: {set(layer_1.keys())}"
    assert layer_1["user_friendly_name"] == "🧠 Processing your question with AI..."
    print("   ✅ User-friendly format correct")

    # Test Layer 2: Technical
    layer_2 = step.to_layer_dict(2)
    expected_layer_2_fields = expected_layer_1_fields | {
        "step_name", "technology", "details", "processing_time_ms"
    }

    print(f"🔧 Layer 2 (Technical): {len(layer_2)} fields")
    assert set(layer_2.keys()) == expected_layer_2_fields, f"Layer 2 fields mismatch: {set(layer_2.keys())}"
    assert layer_2["technology"] == "GPT-4 + Universal RAG"
    assert layer_2["processing_time_ms"] == 2340.5
    print("   ✅ Technical format correct")

    # Test Layer 3: Diagnostic
    layer_3 = step.to_layer_dict(3)
    expected_layer_3_fields = expected_layer_2_fields | {
        "fix_applied", "technical_data"
    }

    print(f"🔬 Layer 3 (Diagnostic): {len(layer_3)} fields")
    assert set(layer_3.keys()) == expected_layer_3_fields, f"Layer 3 fields mismatch: {set(layer_3.keys())}"
    assert layer_3["fix_applied"] == "Advanced RAG processing"
    assert layer_3["technical_data"]["model_used"] == "gpt-4-turbo"
    print("   ✅ Diagnostic format correct")

    return layer_1, layer_2, layer_3


def test_frontend_interface_compatibility():
    """Test exact frontend TypeScript interface compatibility"""

    print("\n🎯 Testing Frontend Interface Compatibility...")

    # Frontend TypeScript interface definition
    frontend_interface = {
        "query_id": "string",
        "step_number": "number",
        "step_name": "string",
        "user_friendly_name": "string",
        "status": "'pending' | 'in_progress' | 'completed' | 'error'",
        "processing_time_ms": "number | undefined",
        "technology": "string",
        "details": "string",
        "fix_applied": "string | undefined",
        "progress_percentage": "number",
        "technical_data": "any | undefined"
    }

    # Create test step matching frontend interface exactly
    step = WorkflowStep(
        query_id="test-frontend-compatibility",
        step_number=2,
        step_name="finalize_response",
        user_friendly_name="✨ Finalizing your answer...",
        status="completed",
        progress_percentage=100,
        technology="Response Formatting",
        details="Response ready! Processed in 1.8s",
        processing_time_ms=1850.2,
        fix_applied="Response optimization",
        technical_data={
            "total_processing_time": 1.85,
            "query_count": 42,
            "average_processing_time": 2.1
        }
    )

    # Convert to dict (what frontend receives)
    step_dict = step.to_dict()

    # Verify all required fields are present
    for field_name in frontend_interface.keys():
        assert field_name in step_dict, f"Missing required field: {field_name}"

    # Verify data types
    assert isinstance(step_dict["query_id"], str)
    assert isinstance(step_dict["step_number"], int)
    assert isinstance(step_dict["step_name"], str)
    assert isinstance(step_dict["user_friendly_name"], str)
    assert step_dict["status"] in ["pending", "in_progress", "completed", "error"]
    assert isinstance(step_dict["processing_time_ms"], (int, float)) or step_dict["processing_time_ms"] is None
    assert isinstance(step_dict["technology"], str)
    assert isinstance(step_dict["details"], str)
    assert isinstance(step_dict["fix_applied"], str) or step_dict["fix_applied"] is None
    assert isinstance(step_dict["progress_percentage"], int)
    assert isinstance(step_dict["technical_data"], dict) or step_dict["technical_data"] is None

    print("   ✅ All frontend interface fields present")
    print("   ✅ All data types match TypeScript interface")

    # Test JSON serialization (SSE requirement)
    json_output = json.dumps(step_dict, indent=2)
    parsed_back = json.loads(json_output)
    assert parsed_back == step_dict, "JSON serialization/deserialization failed"
    print("   ✅ JSON serialization compatible")

    return step_dict


async def test_workflow_manager_integration():
    """Test full workflow manager integration"""

    print("\n🚀 Testing Complete Workflow Manager Integration...")

    # Create workflow manager
    workflow_manager = create_workflow_manager(
        "How do I troubleshoot centrifugal pump vibration issues?",
        "maintenance"
    )

    # Simulate complete workflow
    events_received = []

    async def mock_event_handler(event_type: str, data):
        """Mock event handler to capture events like frontend would"""
        events_received.append({
            "event_type": event_type,
            "data": data.to_dict() if hasattr(data, 'to_dict') else data,
            "timestamp": datetime.now().isoformat()
        })

    # Subscribe to events
    workflow_manager.subscribe_to_events(mock_event_handler)

    # Step 1: System initialization
    step_1 = await workflow_manager.start_step(
        "initialize_system",
        "🔧 Starting up AI systems...",
        "Universal RAG",
        10
    )

    await workflow_manager.complete_step(
        step_1,
        "AI systems ready",
        30,
        {"initialization_time": 0.8},
        "Fast startup"
    )

    # Step 2: Query processing
    step_2 = await workflow_manager.start_step(
        "process_query",
        "🧠 Analyzing your question...",
        "GPT-4 Analysis",
        50
    )

    await workflow_manager.complete_step(
        step_2,
        "Question analyzed successfully",
        70,
        {"concepts_extracted": 15, "entities_found": 8}
    )

    # Step 3: Response generation
    step_3 = await workflow_manager.start_step(
        "generate_response",
        "✨ Creating your answer...",
        "GPT-4 Generation",
        90
    )

    await workflow_manager.complete_step(
        step_3,
        "Comprehensive answer generated",
        100,
        {"response_length": 2400, "sources_cited": 5}
    )

    # Complete workflow
    await workflow_manager.complete_workflow(
        {"success": True, "answer": "Generated response"},
        2.5
    )

    # Verify events
    assert len(events_received) >= 6, f"Expected at least 6 events, got {len(events_received)}"

    # Check event types
    event_types = [event["event_type"] for event in events_received]
    expected_types = ["step_started", "step_completed"] * 3 + ["workflow_completed"]

    print(f"   📡 Captured {len(events_received)} workflow events")
    print(f"   📊 Event types: {set(event_types)}")

    # Verify final summary
    summary = workflow_manager.get_workflow_summary()
    assert summary["status"] == "completed"
    assert summary["total_steps"] == 3
    assert summary["completed_steps"] == 3

    print("   ✅ Workflow events captured correctly")
    print("   ✅ Workflow summary generated")

    return events_received, summary


def print_sample_output():
    """Print sample output for frontend developers"""

    print("\n📋 Sample Output for Frontend Integration:")
    print("=" * 60)

    # Create sample WorkflowStep
    step = WorkflowStep(
        query_id="demo-12345",
        step_number=2,
        step_name="vector_search",
        user_friendly_name="🔍 Searching knowledge base...",
        status="completed",
        progress_percentage=60,
        technology="FAISS Vector Search",
        details="Found 8 relevant documents with 92% confidence",
        processing_time_ms=1240.7,
        fix_applied="Optimized similarity threshold",
        technical_data={
            "documents_found": 8,
            "similarity_scores": [0.92, 0.89, 0.87, 0.85, 0.82, 0.79, 0.76, 0.74],
            "search_time_ms": 145.3,
            "index_size": "50K vectors"
        }
    )

    print("\n🔍 Layer 1 (User-Friendly View):")
    print(json.dumps(step.to_layer_dict(1), indent=2))

    print("\n🔧 Layer 2 (Technical View):")
    print(json.dumps(step.to_layer_dict(2), indent=2))

    print("\n🔬 Layer 3 (Diagnostic View):")
    print(json.dumps(step.to_layer_dict(3), indent=2))


async def main():
    """Run all tests"""

    print("🎯 Universal Workflow Manager Integration Test")
    print("=" * 60)

    try:
        # Test 1: WorkflowStep creation
        step = await test_workflow_step_creation()

        # Test 2: Three-layer disclosure
        layer_1, layer_2, layer_3 = test_three_layer_disclosure()

        # Test 3: Frontend interface compatibility
        step_dict = test_frontend_interface_compatibility()

        # Test 4: Full workflow integration
        events, summary = await test_workflow_manager_integration()

        # Print sample output
        print_sample_output()

        print("\n🎉 All Tests Passed!")
        print("=" * 60)
        print("✅ Universal Workflow Manager is ready for frontend integration")
        print("✅ Three-layer progressive disclosure working correctly")
        print("✅ Frontend TypeScript interface compatibility verified")
        print("✅ Real-time event streaming tested")
        print("✅ All WorkflowStep fields match frontend expectations")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)