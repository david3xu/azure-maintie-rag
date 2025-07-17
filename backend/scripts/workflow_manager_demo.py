#!/usr/bin/env python3
"""
Workflow Manager Integration Demo
Shows how the Universal RAG workflow integrates with the workflow manager
for real-time progress tracking and three-layer progressive disclosure
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import time

# Import workflow manager
from core.workflow.universal_workflow_manager import create_workflow_manager
from core.orchestration.enhanced_rag_universal import get_enhanced_rag_instance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowManagerDemo:
    """
    Demonstrates Universal RAG integration with Workflow Manager
    Shows three-layer progressive disclosure for different user types
    """

    def __init__(self, domain: str = "maintenance"):
        self.domain = domain
        self.workflow_events = []

    async def demonstrate_workflow_integration(self, user_query: str = "How do I fix pump vibration issues?"):
        """Demonstrate workflow manager integration with actual Universal RAG processing"""

        print(f"🔄 WORKFLOW MANAGER INTEGRATION DEMO")
        print(f"🎯 User Query: '{user_query}'")
        print(f"🏷️  Domain: {self.domain}")
        print(f"📅 Started: {datetime.now().isoformat()}")

        # Create workflow manager (real implementation)
        workflow_manager = create_workflow_manager(user_query, self.domain)

        # Subscribe to workflow events
        workflow_manager.subscribe_to_events(self._capture_workflow_event)

        # Get Enhanced RAG instance
        enhanced_rag = get_enhanced_rag_instance(self.domain)

        print(f"\n📡 WORKFLOW EVENTS (Real-time):")
        print(f"{'='*60}")

        try:
            # Initialize components if needed
            if not enhanced_rag.components_initialized:
                print("🔧 Initializing Enhanced RAG components...")
                init_results = await enhanced_rag.initialize_components()
                if not init_results.get("success", False):
                    print(f"❌ Initialization failed: {init_results.get('error', 'Unknown error')}")
                    return

            # Process query with workflow manager integration
            # This will trigger the detailed 7-step workflow
            results = await enhanced_rag.process_query(
                query=user_query,
                max_results=5,
                include_explanations=True,
                enable_safety_warnings=True,
                stream_progress=True,
                workflow_manager=workflow_manager  # Pass workflow manager for detailed steps
            )

            # Complete workflow
            await workflow_manager.complete_workflow(
                results,
                results.get("processing_time", 0)
            )

            print(f"\n✅ Workflow completed successfully!")

            # Demonstrate three-layer progressive disclosure
            await self._demonstrate_progressive_disclosure(workflow_manager)

            # Show final results
            await self._show_final_results(results)

        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            await workflow_manager.fail_workflow(f"Demo failed: {str(e)}")

    async def _capture_workflow_event(self, event_type: str, data: Any):
        """Capture workflow events for real-time display"""

        timestamp = datetime.now().isoformat()

        # Store event
        self.workflow_events.append({
            "timestamp": timestamp,
            "event_type": event_type,
            "data": data.to_dict() if hasattr(data, 'to_dict') else data
        })

        # Display event in real-time
        if event_type == "step_started":
            print(f"🟡 [{timestamp}] Started: {data.user_friendly_name}")
        elif event_type == "step_updated":
            print(f"🔄 [{timestamp}] Updated: {data.user_friendly_name} ({data.progress_percentage:.1f}%)")
        elif event_type == "step_completed":
            print(f"🟢 [{timestamp}] Completed: {data.user_friendly_name} ({data.processing_time_ms:.0f}ms)")
        elif event_type == "step_failed":
            print(f"🔴 [{timestamp}] Failed: {data.user_friendly_name} - {data.error_message}")
        elif event_type == "workflow_completed":
            print(f"🎉 [{timestamp}] Workflow completed in {data.get('total_time', 0):.2f}s")
        elif event_type == "workflow_failed":
            print(f"💥 [{timestamp}] Workflow failed: {data.get('error', 'Unknown error')}")

    async def _demonstrate_progressive_disclosure(self, workflow_manager):
        """Demonstrate three-layer progressive disclosure"""

        print(f"\n🎭 THREE-LAYER PROGRESSIVE DISCLOSURE DEMO")
        print(f"{'='*60}")

        # Layer 1: User-friendly (90% of users)
        print(f"\n📱 LAYER 1: USER-FRIENDLY VIEW")
        print(f"   Target: General users (90%)")
        print(f"   Focus: Simple progress indicators")

        layer_1_steps = workflow_manager.get_steps_for_layer(1)
        print(f"   Steps shown: {len(layer_1_steps)}")

        for step in layer_1_steps:
            status = "✅" if step.get("status") == "completed" else "🔄"
            print(f"   {status} {step.get('user_friendly_name', 'Unknown step')}")

        # Layer 2: Technical (power users)
        print(f"\n🔧 LAYER 2: TECHNICAL VIEW")
        print(f"   Target: Power users and developers")
        print(f"   Focus: Technical details and metrics")

        layer_2_steps = workflow_manager.get_steps_for_layer(2)
        print(f"   Steps shown: {len(layer_2_steps)}")

        for step in layer_2_steps:
            status = "✅" if step.get("status") == "completed" else "🔄"
            tech_data = step.get("technical_data", {})
            print(f"   {status} {step.get('user_friendly_name', 'Unknown step')}")
            print(f"      Technology: {step.get('technology', 'N/A')}")
            print(f"      Progress: {step.get('progress_percentage', 0):.1f}%")
            if tech_data:
                print(f"      Details: {tech_data}")

        # Layer 3: Diagnostic (administrators)
        print(f"\n🔬 LAYER 3: DIAGNOSTIC VIEW")
        print(f"   Target: System administrators")
        print(f"   Focus: Full diagnostics and troubleshooting")

        layer_3_steps = workflow_manager.get_steps_for_layer(3)
        print(f"   Steps shown: {len(layer_3_steps)}")

        for step in layer_3_steps:
            status = "✅" if step.get("status") == "completed" else "🔄"
            print(f"   {status} {step.get('user_friendly_name', 'Unknown step')}")
            print(f"      Step ID: {step.get('step_number', 'N/A')}")
            print(f"      Technology: {step.get('technology', 'N/A')}")
            print(f"      Progress: {step.get('progress_percentage', 0):.1f}%")
            print(f"      Processing Time: {step.get('processing_time_ms', 0):.0f}ms")
            print(f"      Technical Data: {step.get('technical_data', {})}")
            if step.get("fix_applied"):
                print(f"      Fix Applied: {step.get('fix_applied')}")
            print()

    async def _show_final_results(self, results: Dict[str, Any]):
        """Show final workflow results"""

        print(f"\n📊 FINAL RESULTS")
        print(f"{'='*60}")

        print(f"✅ Success: {results.get('success', False)}")
        print(f"⏱️  Processing Time: {results.get('processing_time', 0):.2f} seconds")
        print(f"🏷️  Domain: {results.get('domain', 'N/A')}")

        # Show search results
        search_results = results.get("search_results", [])
        if search_results:
            print(f"🔍 Search Results: {len(search_results)} found")
            for i, result in enumerate(search_results[:3]):  # Show top 3
                if hasattr(result, 'score') and hasattr(result, 'doc_id'):
                    print(f"   {i+1}. {result.doc_id} (Score: {result.score:.3f})")
                    print(f"      {result.content[:100]}...")

        # Show generated response
        response = results.get("generated_response", {})
        if response:
            print(f"\n📝 GENERATED RESPONSE:")
            print(f"{'='*40}")
            print(response.get("generated_response", "No response"))
            print(f"{'='*40}")

        # Show system stats
        system_stats = results.get("system_stats", {})
        if system_stats:
            print(f"\n📈 SYSTEM STATISTICS:")
            for key, value in system_stats.items():
                print(f"   {key}: {value}")

    async def demonstrate_api_integration(self):
        """Demonstrate how this integrates with the streaming API"""

        print(f"\n🌐 API INTEGRATION DEMO")
        print(f"{'='*60}")

        print(f"📡 Streaming API Endpoint: POST /api/v1/query/stream")
        print(f"🔄 Server-Sent Events: GET /api/v1/query/stream/{{query_id}}")
        print(f"📊 Progressive Disclosure: GET /api/v1/workflow/{{query_id}}/steps?layer=1|2|3")

        # Show sample API responses
        print(f"\n📋 SAMPLE API RESPONSES:")

        # Layer 1 API response
        layer_1_response = {
            "success": True,
            "query_id": "demo-query-123",
            "disclosure_layer": 1,
            "steps": [
                {
                    "step_number": 1,
                    "user_friendly_name": "🔧 Setting up AI system...",
                    "status": "completed",
                    "progress_percentage": 100
                },
                {
                    "step_number": 2,
                    "user_friendly_name": "🧠 Processing your question...",
                    "status": "completed",
                    "progress_percentage": 100
                },
                {
                    "step_number": 3,
                    "user_friendly_name": "✨ Generating your answer...",
                    "status": "completed",
                    "progress_percentage": 100
                }
            ],
            "total_steps": 3
        }

        print(f"🔸 Layer 1 Response:")
        print(json.dumps(layer_1_response, indent=2))

        # Layer 2 API response (showing technical details)
        layer_2_response = {
            "success": True,
            "query_id": "demo-query-123",
            "disclosure_layer": 2,
            "steps": [
                {
                    "step_number": 1,
                    "user_friendly_name": "📊 Extracting knowledge from text...",
                    "technology": "Azure OpenAI GPT-4",
                    "status": "completed",
                    "progress_percentage": 100,
                    "technical_data": {
                        "entities_discovered": 15,
                        "relations_discovered": 10,
                        "processing_time": 2.3
                    }
                },
                {
                    "step_number": 2,
                    "user_friendly_name": "🔧 Building searchable vector index...",
                    "technology": "FAISS Engine + 1536D vectors",
                    "status": "completed",
                    "progress_percentage": 100,
                    "technical_data": {
                        "documents_indexed": 7,
                        "vector_dimensions": 1536,
                        "indexing_time": 1.2
                    }
                }
            ],
            "total_steps": 7
        }

        print(f"\n🔸 Layer 2 Response:")
        print(json.dumps(layer_2_response, indent=2))

        print(f"\n🎯 Frontend Integration:")
        print(f"   - React components consume these API responses")
        print(f"   - Real-time updates via Server-Sent Events")
        print(f"   - Progressive disclosure based on user type")
        print(f"   - Transparent workflow for building user trust")

        print(f"\n📱 User Experience:")
        print(f"   - General users: Simple progress indicators")
        print(f"   - Power users: Technical metrics and timings")
        print(f"   - Administrators: Full diagnostic information")

    async def save_demo_results(self):
        """Save demo results for analysis"""

        output_file = f"workflow_manager_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        demo_results = {
            "demo_info": {
                "domain": self.domain,
                "timestamp": datetime.now().isoformat(),
                "total_events": len(self.workflow_events)
            },
            "workflow_events": self.workflow_events,
            "event_types": list(set(event["event_type"] for event in self.workflow_events)),
            "summary": {
                "total_events": len(self.workflow_events),
                "event_types": list(set(event["event_type"] for event in self.workflow_events))
            }
        }

        with open(output_file, 'w') as f:
            json.dump(demo_results, f, indent=2, default=str)

        print(f"\n💾 Demo results saved to: {output_file}")
        return output_file


async def main():
    """Main demo function"""

    # Initialize demo
    demo = WorkflowManagerDemo("maintenance")

    # Run workflow manager integration demo
    await demo.demonstrate_workflow_integration(
        "How do I troubleshoot pump vibration problems and prevent bearing failures?"
    )

    # Demonstrate API integration
    await demo.demonstrate_api_integration()

    # Save results
    await demo.save_demo_results()

    print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())