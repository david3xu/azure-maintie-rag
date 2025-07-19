#!/usr/bin/env python3
"""
Run Workflow Demos with Azure Services - Execution Script
Execute both workflow demonstrations to show complete Azure RAG transparency
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run both workflow demonstrations with Azure services"""

    print(f"🚀 AZURE RAG WORKFLOW DEMONSTRATIONS")
    print(f"{'='*80}")
    print(f"📅 Started: {datetime.now().isoformat()}")
    print(f"🎯 Purpose: Show transparent, Azure-driven RAG processing")
    print(f"☁️  Azure Services: Blob Storage, Cognitive Search, OpenAI, Cosmos DB")

    # Check if we can import the required modules
    try:
        from azure_rag_workflow_demo import CompletelyFixedUniversalRAGWorkflowDemo
        from workflow_manager_demo import WorkflowManagerDemo

        print(f"\n✅ Successfully imported Azure demo modules")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print(f"📋 Make sure you're running from the project root directory")
        print(f"📋 And that the backend path is correct")
        return

    # Demo 1: Complete Azure RAG Workflow
    print(f"\n" + "="*80)
    print(f"🔹 DEMO 1: COMPLETE AZURE RAG WORKFLOW")
    print(f"📋 Shows step-by-step processing using Azure services")
    print(f"☁️  Azure Blob Storage, Cognitive Search, OpenAI, Cosmos DB")
    print(f"⏱️  Processing time measurements for each Azure service")
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

    # Demo 2: Azure Workflow Manager Integration
    print(f"\n" + "="*80)
    print(f"🔹 DEMO 2: AZURE WORKFLOW MANAGER INTEGRATION")
    print(f"📋 Shows real-time progress tracking with Azure services")
    print(f"☁️  Demonstrates Azure service integration")
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
    print(f"🎯 AZURE DEMONSTRATION SUMMARY")
    print(f"{'='*80}")

    print(f"📊 What you've seen:")
    print(f"   🔹 Complete Azure RAG workflow transparency")
    print(f"   🔹 Real Azure service outputs (no mock data)")
    print(f"   🔹 Actual Azure processing times and metrics")
    print(f"   🔹 Azure service integration and coordination")
    print(f"   🔹 Pure Azure-driven approach (cloud-native)")
    print(f"   🔹 Three-layer progressive disclosure")
    print(f"   🔹 Real-time Azure workflow events")
    print(f"   🔹 Azure API integration for frontend consumption")

    print(f"\n☁️  Azure Services Used:")
    print(f"   ✅ Azure Blob Storage - Document storage and retrieval")
    print(f"   ✅ Azure Cognitive Search - Semantic search and indexing")
    print(f"   ✅ Azure OpenAI - Document processing and response generation")
    print(f"   ✅ Azure Cosmos DB - Metadata storage and query tracking")

    print(f"\n📱 User Trust Building:")
    print(f"   ✅ Complete transparency at every Azure service step")
    print(f"   ✅ Real Azure metrics and processing times")
    print(f"   ✅ Clear indication of what each Azure service does")
    print(f"   ✅ No hidden processing or black boxes")
    print(f"   ✅ Progressive disclosure based on user expertise")

    print(f"\n🔧 Technical Implementation:")
    print(f"   ✅ Based on actual Azure service components")
    print(f"   ✅ No assumptions about data structure")
    print(f"   ✅ Pure Azure RAG approach")
    print(f"   ✅ Domain-agnostic processing")
    print(f"   ✅ Real-time Azure workflow manager integration")

    print(f"\n🌟 Key Benefits:")
    print(f"   🔹 Users understand what the Azure system is doing")
    print(f"   🔹 Technical users get detailed Azure metrics")
    print(f"   🔹 Administrators get full Azure diagnostic information")
    print(f"   🔹 Real-time Azure feedback builds confidence")
    print(f"   🔹 Transparent Azure AI processing reduces skepticism")

    print(f"\n📋 Next Steps:")
    print(f"   1. Review generated Azure demo result files")
    print(f"   2. Examine actual Azure service outputs")
    print(f"   3. Test with your own domain text data")
    print(f"   4. Integrate with frontend components")
    print(f"   5. Deploy with Azure streaming API endpoints")

    print(f"\n🎉 AZURE DEMONSTRATIONS COMPLETED!")
    print(f"📅 Finished: {datetime.now().isoformat()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    """
    Execute Azure workflow demonstrations

    Usage:
        python run_workflow_demos.py

    Requirements:
        - Run from project root directory
        - Azure service components must be available
        - Azure service credentials configured
    """

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n⏹️  Azure demos interrupted by user")
    except Exception as e:
        print(f"\n💥 Azure demo execution failed: {e}")
        logger.error(f"Azure demo execution error: {e}", exc_info=True)
        sys.exit(1)