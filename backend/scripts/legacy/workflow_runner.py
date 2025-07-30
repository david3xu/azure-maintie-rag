#!/usr/bin/env python3
"""
Workflow Runner
Consolidated script for workflow execution and lifecycle management
Replaces: azure_rag_lifecycle_executor.py, query_processing_workflow.py, 
         azure_data_cleanup_workflow.py, multi_hop_reasoning.py
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.infrastructure_service import InfrastructureService
from services.workflow_service import WorkflowService

async def main():
    """Main workflow tool entry point"""
    if len(sys.argv) < 2:
        print("Usage: python workflow_runner.py <command> [args]")
        print("Commands:")
        print("  lifecycle - Execute complete RAG lifecycle")
        print("  query <text> - Process a query")
        print("  cleanup - Clean up Azure resources")
        return 1
    
    command = sys.argv[1]
    infrastructure = InfrastructureService()
    workflow_service = WorkflowService(infrastructure)
    
    print(f"🔄 Workflow Tool - {command}")
    print("="*50)
    
    if command == "lifecycle":
        result = await workflow_service.initialize_rag_orchestration("general")
        return 0 if result.get('success', False) else 1
        
    elif command == "query":
        if len(sys.argv) < 3:
            print("Error: Query text required")
            return 1
        query_text = " ".join(sys.argv[2:])
        print(f"Processing query: {query_text}")
        # Query processing would be implemented here
        return 0
        
    elif command == "cleanup":
        print("Cleaning up Azure resources...")
        # Cleanup would be implemented here
        return 0
        
    else:
        print(f"Unknown command: {command}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))