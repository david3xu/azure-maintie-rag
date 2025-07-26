"""
Azure Prompt Flow Knowledge Extraction Workflow
Universal knowledge extraction using centralized prompt management
Maintains universal principles with enterprise-grade prompt flow orchestration
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.prompt_flow.prompt_flow_integration import AzurePromptFlowIntegrator
from core.azure_storage.storage_factory import StorageFactory
from core.utilities.intelligent_document_processor import UniversalDocumentProcessor
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_prompt_flow_extraction_workflow(domain: str = "general") -> Dict[str, Any]:
    """
    Execute universal knowledge extraction using Azure Prompt Flow
    
    Args:
        domain: Domain name for context (remains universal)
        
    Returns:
        Extraction results and workflow metrics
    """
    workflow_start = datetime.now()
    workflow_results = {
        "success": False,
        "domain": domain,
        "entities_extracted": 0,
        "relations_extracted": 0,
        "documents_processed": 0,
        "extraction_method": "azure_prompt_flow_centralized",
        "workflow_duration": 0.0,
        "errors": []
    }
    
    try:
        logger.info(f"🚀 Starting Azure Prompt Flow Knowledge Extraction Workflow")
        logger.info(f"Domain: {domain} (universal extraction principles)")
        
        # Step 1: Load and process documents
        logger.info("📂 Step 1: Loading and processing documents...")
        
        storage_factory = StorageFactory()
        rag_storage = storage_factory.get_rag_data_storage()
        
        # Get all document chunks from storage
        document_chunks = await rag_storage.list_documents()
        
        if not document_chunks:
            raise ValueError("No document chunks found in storage")
        
        logger.info(f"Found {len(document_chunks)} document chunks")
        
        # Extract text content from chunks
        texts = []
        for chunk in document_chunks[:100]:  # Limit for performance
            try:
                chunk_data = await rag_storage.get_document(chunk)
                if chunk_data and 'content' in chunk_data:
                    texts.append(chunk_data['content'])
            except Exception as e:
                logger.warning(f"Failed to load chunk {chunk}: {e}")
        
        if not texts:
            raise ValueError("No text content extracted from document chunks")
        
        workflow_results["documents_processed"] = len(texts)
        logger.info(f"Extracted text from {len(texts)} document chunks")
        
        # Step 2: Initialize Prompt Flow Integrator
        logger.info("🔧 Step 2: Initializing Azure Prompt Flow integrator...")
        
        prompt_flow_integrator = AzurePromptFlowIntegrator(domain)
        
        # Display prompt templates being used
        templates = prompt_flow_integrator.get_prompt_templates()
        logger.info(f"Using {len(templates)} centralized prompt templates:")
        for template_name in templates.keys():
            logger.info(f"  ✅ {template_name}")
        
        # Step 3: Execute knowledge extraction with Prompt Flow
        logger.info("🧠 Step 3: Executing universal knowledge extraction...")
        logger.info("Using centralized prompts - NO predetermined knowledge!")
        
        extraction_results = await prompt_flow_integrator.extract_knowledge_with_prompt_flow(
            texts=texts,
            max_entities=settings.max_entities_per_document,
            confidence_threshold=settings.extraction_confidence_threshold
        )
        
        if not extraction_results.get("success", False):
            raise Exception(f"Knowledge extraction failed: {extraction_results.get('error', 'Unknown error')}")
        
        # Extract results
        entities = extraction_results.get("entities", [])
        relations = extraction_results.get("relations", [])
        
        workflow_results["entities_extracted"] = len(entities)
        workflow_results["relations_extracted"] = len(relations)
        
        logger.info(f"✅ Extraction completed:")
        logger.info(f"   🔍 Entities extracted: {len(entities)}")
        logger.info(f"   🔗 Relations extracted: {len(relations)}")
        
        # Step 4: Display extraction insights
        logger.info("📊 Step 4: Extraction insights...")
        
        entity_types = list(set(e.get("entity_type", "") for e in entities))
        relation_types = list(set(r.get("relation_type", "") for r in relations))
        
        logger.info(f"Discovered entity types ({len(entity_types)}): {entity_types[:10]}...")
        logger.info(f"Discovered relation types ({len(relation_types)}): {relation_types[:10]}...")
        
        # Step 5: Quality assessment
        quality_assessment = extraction_results.get("quality_assessment", {})
        overall_score = quality_assessment.get("overall_score", 0.0)
        quality_tier = quality_assessment.get("quality_tier", "unknown")
        
        logger.info(f"📈 Quality Assessment: {quality_tier} (score: {overall_score:.3f})")
        
        # Step 6: Success summary
        workflow_duration = (datetime.now() - workflow_start).total_seconds()
        workflow_results["workflow_duration"] = workflow_duration
        workflow_results["success"] = True
        
        logger.info("🎉 Azure Prompt Flow extraction workflow completed successfully!")
        logger.info(f"Total duration: {workflow_duration:.1f} seconds")
        logger.info("=" * 70)
        
        return workflow_results
        
    except Exception as e:
        error_msg = f"Workflow failed: {e}"
        logger.error(error_msg, exc_info=True)
        
        workflow_results["errors"].append(error_msg)
        workflow_results["workflow_duration"] = (datetime.now() - workflow_start).total_seconds()
        
        return workflow_results


async def main():
    """Main entry point for prompt flow extraction workflow"""
    domain = sys.argv[1] if len(sys.argv) > 1 else "general"
    
    try:
        logger.info("🌟 Azure Universal RAG - Prompt Flow Knowledge Extraction")
        logger.info("=" * 70)
        logger.info("🎯 Universal Extraction Principles:")
        logger.info("   ✅ NO predetermined entity types or categories")
        logger.info("   ✅ NO hardcoded domain knowledge")
        logger.info("   ✅ Centralized prompt management with Prompt Flow")
        logger.info("   ✅ Domain-agnostic knowledge discovery")
        logger.info("=" * 70)
        
        # Run the workflow
        results = await run_prompt_flow_extraction_workflow(domain)
        
        # Print final results
        if results["success"]:
            print(f"\n✅ Workflow completed successfully:")
            print(f"   📄 Documents processed: {results['documents_processed']}")
            print(f"   🔍 Entities extracted: {results['entities_extracted']}")
            print(f"   🔗 Relations extracted: {results['relations_extracted']}")
            print(f"   ⏱️  Duration: {results['workflow_duration']:.1f}s")
            print(f"   🎯 Method: {results['extraction_method']}")
        else:
            print(f"\n❌ Workflow failed:")
            for error in results.get("errors", []):
                print(f"   ⚠️  {error}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical workflow failure: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())