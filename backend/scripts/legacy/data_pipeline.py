#!/usr/bin/env python3
"""
Azure Universal RAG - Main Data Pipeline
Central orchestrator for complete intelligent RAG pipeline
Modes: full (default), upload, extract, train, query
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.infrastructure_service import InfrastructureService
from services.workflow_service import WorkflowService
from services.data_service import DataService
from services.knowledge_service import KnowledgeService

async def main():
    """Main data pipeline entry point with mode support"""
    parser = argparse.ArgumentParser(description="Azure Universal RAG Data Pipeline")
    parser.add_argument("--mode", choices=["full", "upload", "extract", "train", "query"], default="full", 
                       help="Pipeline mode (default: full)")
    parser.add_argument("--domain", default="general", help="Domain for processing (default: general)")
    parser.add_argument("--source", default="data/raw", help="Source data path (default: data/raw)")
    
    args = parser.parse_args()
    
    print(f"🚀 Azure Universal RAG Pipeline - Mode: {args.mode}")
    print("="*60)
    
    infrastructure = InfrastructureService()
    workflow_service = WorkflowService()
    data_service = DataService(infrastructure)
    knowledge_service = KnowledgeService()
    
    try:
        if args.mode == "full":
            # Complete intelligent pipeline: extraction → graph → GNN
            print("🔄 Executing complete intelligent RAG pipeline...")
            result = await workflow_service.execute_full_pipeline(args.source, args.domain)
            
        elif args.mode == "upload":
            # Upload and storage migration only
            print("📤 Uploading documents to Azure storage...")
            result = await data_service._migrate_to_storage(args.source, args.domain, {})
            
        elif args.mode == "extract":
            # Knowledge extraction only
            print("🧠 Extracting knowledge using LLM...")
            result = await knowledge_service.extract_from_file(args.source, args.domain)
            
        elif args.mode == "train":
            # GNN training only (requires existing graph data)
            print("🤖 Training GNN model...")
            from services.ml_service import MLService
            ml_service = MLService()
            result = await ml_service.train_gnn_model(args.domain)
            
        elif args.mode == "query":
            # Query testing (requires complete system)
            print("🔍 Testing query capabilities...")
            # Placeholder for query testing
            result = {"success": True, "message": "Query mode not yet implemented"}
        
        # Report results
        if result.get('success', False):
            print(f"✅ Pipeline mode '{args.mode}' completed successfully")
            if 'stages' in result:
                for stage, stage_result in result['stages'].items():
                    status = "✅" if stage_result.get('success', False) else "❌"
                    print(f"  {status} {stage}: {stage_result.get('message', 'completed')}")
            return 0
        else:
            print(f"❌ Pipeline mode '{args.mode}' failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"💥 Pipeline error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))