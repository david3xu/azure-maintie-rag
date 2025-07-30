#!/usr/bin/env python3
"""
Full Pipeline Orchestrator - Complete Processing Phase (Data Processing)
Raw Text Data → Azure Services → Knowledge Extraction → PyTorch Geometric → GNN Training

This script orchestrates the complete processing phase of the README architecture:
- Stage 01a: Azure Blob Storage (Raw Text → Blob Storage)
- Stage 01b: Azure Cognitive Search (Text → Document Index)
- Stage 01c: Vector Embeddings (Documents → 1536D Vector Index)
- Stage 02: Knowledge Extraction (Text → Entities/Relations)
- Stage 04: Graph Construction (Entities → PyTorch Geometric Format)
- Stage 05: GNN Training (PyTorch Geometric → Trained Model)

Note: Stage 03 was removed as redundant (covered by Stage 01c)
"""

import sys
import asyncio
import argparse
import json
import importlib
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService

# Import individual stage classes using importlib
storage_module = importlib.import_module('01a_azure_storage')
search_module = importlib.import_module('01b_azure_search')
embedding_module = importlib.import_module('01c_vector_embeddings')
knowledge_extraction_module = importlib.import_module('02_knowledge_extraction')
graph_construction_module = importlib.import_module('04_graph_construction')
gnn_training_module = importlib.import_module('05_gnn_training')

AzureStorageTestStage = storage_module.AzureStorageTestStage
AzureSearchTestStage = search_module.AzureSearchTestStage
VectorEmbeddingStage = embedding_module.VectorEmbeddingStage
KnowledgeExtractionStage = knowledge_extraction_module.KnowledgeExtractionStage
GraphConstructionStage = graph_construction_module.GraphConstructionStage
GNNTrainingStage = gnn_training_module.GNNTrainingStage

logger = logging.getLogger(__name__)

class FullPipelineOrchestrator:
    """Complete Processing Phase Orchestrator - Stages 01a-05, excluding 03"""
    
    def __init__(self):
        self.infrastructure = InfrastructureService()
        
        # Initialize stage classes
        self.azure_storage = AzureStorageTestStage()
        self.azure_search = AzureSearchTestStage()
        self.vector_embeddings = VectorEmbeddingStage()
        self.knowledge_extraction = KnowledgeExtractionStage()
        self.graph_construction = GraphConstructionStage()
        self.gnn_training = GNNTrainingStage()
        
    async def execute_full_pipeline(
        self,
        input_data_path: str,
        domain: str = "general",
        skip_stages: List[str] = None,
        parallel_processing: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete processing pipeline
        
        Args:
            input_data_path: Path to input data files
            domain: Target domain
            skip_stages: List of stages to skip (e.g., ["05"] to skip GNN training)
            parallel_processing: Enable parallel processing where possible
            
        Returns:
            Dict with complete pipeline results
        """
        print("🚀 Full Pipeline Orchestrator - Complete Processing Phase")
        print("=" * 70)
        
        start_time = asyncio.get_event_loop().time()
        skip_stages = skip_stages or []
        
        pipeline_results = {
            "orchestrator": "00_full_pipeline",
            "domain": domain,
            "input_data_path": input_data_path,
            "pipeline_start": datetime.now().isoformat(),
            "stages_executed": [],
            "stages_skipped": skip_stages,
            "stage_results": {},
            "pipeline_metrics": {
                "total_duration": 0,
                "files_uploaded": 0,
                "documents_indexed": 0,
                "vectors_generated": 0,
                "entities_extracted": 0,
                "graph_nodes": 0,
                "model_trained": False
            },
            "success": False
        }
        
        try:
            # Stage 01a: Azure Blob Storage
            if "01a" not in skip_stages:
                print(f"\n🔄 Executing Stage 01a: Azure Blob Storage")
                stage01a_result = await self.azure_storage.execute(
                    source_path=input_data_path,
                    domain=domain
                )
                
                pipeline_results["stage_results"]["01a_azure_storage"] = stage01a_result
                pipeline_results["stages_executed"].append("01a")
                
                if not stage01a_result.get("success"):
                    raise Exception(f"Stage 01a failed: {stage01a_result.get('error')}")
                
                # Extract metrics from Stage 01a
                pipeline_results["pipeline_metrics"]["files_uploaded"] = stage01a_result.get("uploaded_files", 0)
                
                print(f"✅ Stage 01a Complete: {stage01a_result.get('uploaded_files', 0)} files uploaded")
            else:
                print(f"⏭️  Skipping Stage 01a: Azure Blob Storage")
                
            # Stage 01b: Azure Cognitive Search
            if "01b" not in skip_stages:
                print(f"\n🔄 Executing Stage 01b: Azure Cognitive Search")
                stage01b_result = await self.azure_search.execute(
                    source_path=input_data_path,
                    domain=domain
                )
                
                pipeline_results["stage_results"]["01b_azure_search"] = stage01b_result
                pipeline_results["stages_executed"].append("01b")
                
                if not stage01b_result.get("success"):
                    raise Exception(f"Stage 01b failed: {stage01b_result.get('error')}")
                
                # Extract metrics from Stage 01b  
                pipeline_results["pipeline_metrics"]["documents_indexed"] = stage01b_result.get("documents_indexed", 0)
                
                print(f"✅ Stage 01b Complete: {stage01b_result.get('documents_indexed', 0)} documents indexed")
            else:
                print(f"⏭️  Skipping Stage 01b: Azure Cognitive Search")
                
            # Stage 01c: Vector Embeddings
            if "01c" not in skip_stages:
                print(f"\n🔄 Executing Stage 01c: Vector Embeddings")
                stage01c_result = await self.vector_embeddings.execute(
                    domain=domain
                )
                
                pipeline_results["stage_results"]["01c_vector_embeddings"] = stage01c_result
                pipeline_results["stages_executed"].append("01c")
                
                if not stage01c_result.get("success"):
                    raise Exception(f"Stage 01c failed: {stage01c_result.get('error')}")
                
                # Extract metrics from Stage 01c
                pipeline_results["pipeline_metrics"]["vectors_generated"] = stage01c_result.get("embeddings_generated", 0)
                
                print(f"✅ Stage 01c Complete: {stage01c_result.get('embeddings_generated', 0)} vectors generated")
            else:
                print(f"⏭️  Skipping Stage 01c: Vector Embeddings")
            
            # Stage 02: Knowledge Extraction
            if "02" not in skip_stages:
                print(f"\n🔄 Executing Stage 02: Knowledge Extraction")
                # Use the container created in stage 01a if available
                container_name = f"maintie-staging-data-{domain}" if "01a" not in skip_stages else None
                stage02_result = await self.knowledge_extraction.execute(
                    container=container_name,
                    domain=domain,
                    output=f"data/outputs/step02_knowledge_extraction_results.json"
                )
                
                pipeline_results["stage_results"]["02_knowledge_extraction"] = stage02_result
                pipeline_results["stages_executed"].append("02")
                
                if not stage02_result.get("success"):
                    raise Exception(f"Stage 02 failed: {stage02_result.get('error')}")
                
                # Extract metrics from Stage 02
                pipeline_results["pipeline_metrics"]["entities_extracted"] = stage02_result.get("entities_extracted", 0)
                
                print(f"✅ Stage 02 Complete: {stage02_result.get('entities_extracted', 0)} entities extracted")
            else:
                print(f"⏭️  Skipping Stage 02: Knowledge Extraction")
            
            # Stage 04: Graph Construction (PyTorch Geometric)
            if "04" not in skip_stages:
                print(f"\n🔄 Executing Stage 04: Graph Construction")
                # Use extractions container from step 02
                container_name = "extractions" if "02" not in skip_stages else None
                stage04_result = await self.graph_construction.execute(
                    container=container_name,
                    domain=domain
                )
                
                pipeline_results["stage_results"]["04_graph_construction"] = stage04_result
                pipeline_results["stages_executed"].append("04")
                
                if not stage04_result.get("success"):
                    raise Exception(f"Stage 04 failed: {stage04_result.get('error')}")
                
                # Extract metrics from Stage 04
                pytorch_data_info = stage04_result.get("pytorch_data_info", {})
                pipeline_results["pipeline_metrics"]["graph_nodes"] = pytorch_data_info.get("num_nodes", 0)
                
                print(f"✅ Stage 04 Complete: {pytorch_data_info.get('num_nodes', 0)} nodes created")
            else:
                print(f"⏭️  Skipping Stage 04: Graph Construction")
            
            # Stage 05: GNN Training
            if "05" not in skip_stages:
                print(f"\n🔄 Executing Stage 05: GNN Training")
                stage05_result = await self.gnn_training.execute(
                    domain=domain,
                    epochs=50,
                    output=f"data/outputs/step05_gnn_training_results.json"
                )
                
                pipeline_results["stage_results"]["05_gnn_training"] = stage05_result
                pipeline_results["stages_executed"].append("05")
                
                if not stage05_result.get("success"):
                    print(f"⚠️  Stage 05 failed but continuing: {stage05_result.get('error')}")
                    # GNN training failure is not critical for the overall pipeline
                else:
                    pipeline_results["pipeline_metrics"]["model_trained"] = True
                    # Extract metrics from Stage 05
                    training_metrics = stage05_result.get("training_metrics", {})
                    accuracy = training_metrics.get("final_test_accuracy", 0)
                    print(f"✅ Stage 05 Complete: GNN model trained ({accuracy:.1%} accuracy)")
            else:
                print(f"⏭️  Skipping Stage 05: GNN Training")
            
            # Pipeline Success
            total_duration = asyncio.get_event_loop().time() - start_time
            pipeline_results["pipeline_metrics"]["total_duration"] = round(total_duration, 2)
            pipeline_results["pipeline_end"] = datetime.now().isoformat()
            pipeline_results["success"] = True
            
            print(f"\n🎉 Full Pipeline Complete!")
            print(f"   📊 Stages executed: {len(pipeline_results['stages_executed'])}")
            print(f"   📤 Files uploaded: {pipeline_results['pipeline_metrics']['files_uploaded']}")
            print(f"   📄 Documents indexed: {pipeline_results['pipeline_metrics']['documents_indexed']}")
            print(f"   🔍 Vectors generated: {pipeline_results['pipeline_metrics']['vectors_generated']}")
            print(f"   🏷️  Entities extracted: {pipeline_results['pipeline_metrics']['entities_extracted']}")
            print(f"   🕸️  Graph nodes: {pipeline_results['pipeline_metrics']['graph_nodes']}")
            print(f"   🧠 Model trained: {pipeline_results['pipeline_metrics']['model_trained']}")
            print(f"   ⏱️  Total duration: {pipeline_results['pipeline_metrics']['total_duration']}s")
            
            return pipeline_results
            
        except Exception as e:
            pipeline_results["error"] = str(e)
            pipeline_results["pipeline_end"] = datetime.now().isoformat()
            total_duration = asyncio.get_event_loop().time() - start_time
            pipeline_results["pipeline_metrics"]["total_duration"] = round(total_duration, 2)
            
            print(f"\n❌ Pipeline Failed: {e}")
            print(f"   📊 Stages completed: {len(pipeline_results['stages_executed'])}")
            print(f"   ⏱️  Duration: {pipeline_results['pipeline_metrics']['total_duration']}s")
            
            logger.error(f"Full pipeline failed: {e}", exc_info=True)
            return pipeline_results

    async def get_pipeline_status(self, domain: str) -> Dict[str, Any]:
        """Get current status of pipeline components"""
        try:
            status = {
                "domain": domain,
                "component_status": {},
                "readiness": {
                    "data_ready": False,
                    "knowledge_ready": False,
                    "vectors_ready": False,
                    "graph_ready": False,
                    "model_ready": False
                }
            }
            
            # Check each component (simplified)
            # This would normally check actual storage containers and indices
            print(f"📊 Checking pipeline status for domain: {domain}")
            
            return status
            
        except Exception as e:
            print(f"❌ Status check failed: {e}")
            return {"error": str(e)}


async def main():
    """Main entry point for full pipeline orchestrator"""
    parser = argparse.ArgumentParser(
        description="Full Pipeline Orchestrator - Complete Processing Phase (Stages 01a-05, excluding 03)"
    )
    parser.add_argument(
        "input_data_path",
        help="Path to input data files"
    )
    parser.add_argument(
        "--domain", 
        default="general",
        help="Target domain"
    )
    parser.add_argument(
        "--skip-stages",
        nargs="+",
        help="Stages to skip (e.g., --skip-stages 05)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only check pipeline status"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = FullPipelineOrchestrator()
    
    if args.status_only:
        # Get pipeline status
        results = await orchestrator.get_pipeline_status(args.domain)
        print(f"📊 Pipeline Status: {json.dumps(results, indent=2)}")
    else:
        # Execute full pipeline
        results = await orchestrator.execute_full_pipeline(
            input_data_path=args.input_data_path,
            domain=args.domain,
            skip_stages=args.skip_stages,
            parallel_processing=not args.sequential
        )
    
    # Save results if requested
    if args.output and results.get("success"):
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {args.output}")
    
    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))