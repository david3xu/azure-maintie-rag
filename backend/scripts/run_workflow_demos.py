#!/usr/bin/env python3
"""
Run Workflow Demos - Execution Script
Execute both workflow demonstrations to show complete Universal RAG transparency
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run both workflow demonstrations"""

    print(f"🚀 UNIVERSAL RAG WORKFLOW DEMONSTRATIONS")
    print(f"{'='*80}")
    print(f"📅 Started: {datetime.now().isoformat()}")
    print(f"🎯 Purpose: Show transparent, data-driven RAG processing")
    print(f"📋 No assumptions, no hardcoded values, pure codebase-driven")

    # Check if we can import the required modules
    try:
        from universal_rag_workflow_demo import CompletelyFixedUniversalRAGWorkflowDemo
        from workflow_manager_demo import WorkflowManagerDemo

        print(f"\n✅ Successfully imported demo modules")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print(f"📋 Make sure you're running from the project root directory")
        print(f"📋 And that the backend path is correct")
        return

    # Demo 1: Complete Universal RAG Workflow
    print(f"\n" + "="*80)
    print(f"🔹 DEMO 1: COMPLETE UNIVERSAL RAG WORKFLOW")
    print(f"📋 Shows step-by-step processing from raw text to final answer")
    print(f"📊 Displays actual component outputs and metrics")
    print(f"⏱️  Processing time measurements for each step")
    print(f"=" * 80)

    try:
        demo1 = CompletelyFixedUniversalRAGWorkflowDemo("maintenance")
        await demo1.demonstrate_complete_workflow(
            "How do I diagnose and fix pump vibration issues in industrial equipment?"
        )
        print(f"✅ Demo 1 completed successfully!")

    except Exception as e:
        print(f"❌ Demo 1 failed: {e}")
        logger.error(f"Demo 1 error: {e}", exc_info=True)

    # Demo 2: Workflow Manager Integration
    print(f"\n" + "="*80)
    print(f"🔹 DEMO 2: WORKFLOW MANAGER INTEGRATION")
    print(f"📋 Shows real-time progress tracking and three-layer disclosure")
    print(f"📡 Demonstrates streaming API integration")
    print(f"🎭 Progressive disclosure for different user types")
    print(f"=" * 80)

    try:
        demo2 = WorkflowManagerDemo("maintenance")
        await demo2.demonstrate_workflow_integration(
            "What are the best practices for preventing motor bearing failures?"
        )
        print(f"✅ Demo 2 completed successfully!")

    except Exception as e:
        print(f"❌ Demo 2 failed: {e}")
        logger.error(f"Demo 2 error: {e}", exc_info=True)

    # Summary
    print(f"\n" + "="*80)
    print(f"🎯 DEMONSTRATION SUMMARY")
    print(f"{'='*80}")

    print(f"📊 What you've seen:")
    print(f"   🔹 Complete 7-step RAG workflow transparency")
    print(f"   🔹 Real component outputs (no mock data)")
    print(f"   🔹 Actual processing times and metrics")
    print(f"   🔹 Dynamic entity/relation type discovery")
    print(f"   🔹 Pure data-driven approach (no hardcoded values)")
    print(f"   🔹 Three-layer progressive disclosure")
    print(f"   🔹 Real-time streaming workflow events")
    print(f"   🔹 API integration for frontend consumption")

    print(f"\n📱 User Trust Building:")
    print(f"   ✅ Complete transparency at every step")
    print(f"   ✅ Real metrics and processing times")
    print(f"   ✅ Clear indication of what each component does")
    print(f"   ✅ No hidden processing or black boxes")
    print(f"   ✅ Progressive disclosure based on user expertise")

    print(f"\n🔧 Technical Implementation:")
    print(f"   ✅ Based on actual codebase components")
    print(f"   ✅ No assumptions about data structure")
    print(f"   ✅ Pure Universal RAG approach")
    print(f"   ✅ Domain-agnostic processing")
    print(f"   ✅ Real-time workflow manager integration")

    print(f"\n🌟 Key Benefits:")
    print(f"   🔹 Users understand what the system is doing")
    print(f"   🔹 Technical users get detailed metrics")
    print(f"   🔹 Administrators get full diagnostic information")
    print(f"   🔹 Real-time feedback builds confidence")
    print(f"   🔹 Transparent AI processing reduces skepticism")

    print(f"\n📋 Next Steps:")
    print(f"   1. Review generated demo result files")
    print(f"   2. Examine actual component outputs")
    print(f"   3. Test with your own domain text data")
    print(f"   4. Integrate with frontend components")
    print(f"   5. Deploy with streaming API endpoints")

    print(f"\n🎉 DEMONSTRATIONS COMPLETED!")
    print(f"📅 Finished: {datetime.now().isoformat()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    """
    Execute workflow demonstrations

    Usage:
        python run_workflow_demos.py

    Requirements:
        - Run from project root directory
        - Backend components must be available
        - Azure OpenAI credentials configured
    """

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n⏹️  Demos interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo execution failed: {e}")
        logger.error(f"Demo execution error: {e}", exc_info=True)
        sys.exit(1)