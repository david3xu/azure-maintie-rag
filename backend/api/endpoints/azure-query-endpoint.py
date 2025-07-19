"""
Universal Query API Endpoints
New endpoints that use the Universal RAG system for any domain
Replaces maintenance-specific endpoints with universal ones
Enhanced with Universal Workflow Manager for detailed real-time progress tracking
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import time

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import uuid

# Azure service components
from core.orchestration.enhanced_pipeline import (
    AzureRAGEnhancedPipeline as AzureRAGEnhancedPipeline, get_enhanced_rag_instance, initialize_enhanced_rag_system
)
from core.orchestration.rag_orchestration_service import (
    AzureRAGOrchestrationService as AzureRAGOrchestrationService, create_universal_rag_from_texts, create_universal_rag_from_directory
)
from core.workflow.universal_workflow_manager import (
    create_workflow_manager, get_workflow_manager, workflow_registry
)
from config.settings import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["universal-query"])


# Pydantic models
class UniversalQueryRequest(BaseModel):
    """Universal query request model"""
    query: str = Field(..., description="The query to process")
    domain: str = Field(default="general", description="Domain name for the query")
    max_results: int = Field(default=10, description="Maximum number of results to return")
    include_explanations: bool = Field(default=True, description="Whether to include explanations")
    enable_safety_warnings: bool = Field(default=True, description="Whether to enable safety warnings")


class UniversalQueryResponse(BaseModel):
    """Universal query response model"""
    success: bool
    query: str
    domain: str
    generated_response: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    processing_time: float
    system_stats: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None


class StreamingQueryRequest(BaseModel):
    """Streaming query request model"""
    query: str = Field(..., description="The query to process")
    domain: str = Field(default="general", description="Domain name for the query")
    max_results: int = Field(default=10, description="Maximum number of results to return")
    include_explanations: bool = Field(default=True, description="Whether to include explanations")
    enable_safety_warnings: bool = Field(default=True, description="Whether to enable safety warnings")


class StreamingQueryResponse(BaseModel):
    """Streaming query response model"""
    success: bool
    query_id: str
    query: str
    domain: str
    message: str
    timestamp: str
    error: Optional[str] = None


class DomainInitializationRequest(BaseModel):
    """Domain initialization request model"""
    domain: str = Field(..., description="Domain name to initialize")
    text_files: Optional[List[str]] = Field(default=None, description="List of text file paths")
    force_rebuild: bool = Field(default=False, description="Force rebuild even if domain exists")


class BatchQueryRequest(BaseModel):
    """Batch query request model"""
    queries: List[str] = Field(..., description="List of queries to process")
    domain: str = Field(default="general", description="Domain name for queries")
    max_results: int = Field(default=10, description="Maximum results per query")
    include_explanations: bool = Field(default=True, description="Whether to include explanations")


@router.post("/query/universal", response_model=UniversalQueryResponse)
async def process_universal_query(request: UniversalQueryRequest) -> Dict[str, Any]:
    """
    Process a universal query that works with any domain

    This endpoint uses the Enhanced Universal RAG system to process queries
    without requiring domain-specific configuration or hardcoded types.
    """
    logger.info(f"Processing universal query for domain '{request.domain}': {request.query}")

    try:
        # Get Enhanced Universal RAG instance for the domain
        enhanced_rag = get_enhanced_rag_instance(request.domain)

        # Ensure system is initialized
        if not enhanced_rag.components_initialized:
            logger.info(f"Initializing Enhanced Universal RAG for domain: {request.domain}")
            init_results = await enhanced_rag.initialize_components()

            if not init_results.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize Universal RAG system: {init_results.get('error', 'Unknown error')}"
                )

        # Process the query
        results = await enhanced_rag.process_query(
            query=request.query,
            max_results=request.max_results,
            include_explanations=request.include_explanations,
            enable_safety_warnings=request.enable_safety_warnings
        )

        if not results.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Query processing failed: {results.get('error', 'Unknown error')}"
            )

        # Return structured response
        return {
            "success": True,
            "query": results["query"],
            "domain": results["domain"],
            "generated_response": results["generated_response"].to_dict() if hasattr(results["generated_response"], 'to_dict') else results["generated_response"],
            "search_results": [
                result.to_dict() if hasattr(result, 'to_dict') else
                result if isinstance(result, dict) else
                str(result)
                for result in results["search_results"]
            ],
            "processing_time": results["processing_time"],
            "system_stats": results["system_stats"],
            "timestamp": results["timestamp"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Universal query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/streaming", response_model=StreamingQueryResponse)
async def start_streaming_query(request: StreamingQueryRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Start a streaming universal query with detailed real-time progress updates

    Returns immediately with a query_id that can be used to monitor progress
    via the streaming endpoint. Uses Universal Workflow Manager for detailed
    three-layer progressive disclosure.
    """
    logger.info(f"Starting streaming universal query for domain '{request.domain}': {request.query}")

    try:
        # Create workflow manager with detailed tracking
        workflow_manager = create_workflow_manager(request.query, request.domain)
        query_id = workflow_manager.query_id

        # Start background processing with workflow tracking
        background_tasks.add_task(
            _process_streaming_query_with_workflow,
            workflow_manager,
            request
        )

        return {
            "success": True,
            "query_id": query_id,
            "query": request.query,
            "domain": request.domain,
            "message": "Streaming query started with detailed workflow tracking",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to start streaming query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/stream/{query_id}")
async def stream_query_progress(query_id: str) -> StreamingResponse:
    """
    Stream real-time detailed progress updates for a query

    Returns Server-Sent Events (SSE) with detailed WorkflowStep objects
    that match the frontend interface exactly. Supports three-layer
    progressive disclosure.
    """
    workflow_manager = get_workflow_manager(query_id)
    if not workflow_manager:
        logger.warning(f"Workflow not found for query_id: {query_id}")
        raise HTTPException(status_code=404, detail="Query not found")

    async def generate_detailed_progress_events():
        """Generate detailed SSE events with WorkflowStep objects"""
        try:
            # If workflow is already completed, send all steps at once
            if workflow_manager.is_completed:
                logger.info(f"Streaming completed workflow: {query_id}")
                # Send each step as a separate event
                for step in workflow_manager.steps:
                    step_data = step.to_layer_dict(2) if hasattr(step, 'to_layer_dict') else (step.to_dict() if hasattr(step, 'to_dict') else step)
                    event_data = {
                        "event_type": "progress",
                        **step_data
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    await asyncio.sleep(0.1)  # Small delay for frontend processing
                # Send completion event
                completion_data = {
                    "event_type": "workflow_completed",
                    "query_id": query_id,
                    "total_steps": len(workflow_manager.steps),
                    "performance_metrics": workflow_manager.performance_metrics,
                    "success": True
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                return

            # If workflow is still running, stream real-time updates
            last_step_count = 0
            max_wait_time = 300  # 5 minutes timeout
            start_time = time.time()

            while not workflow_manager.is_completed and not workflow_manager.has_error:
                current_time = time.time()
                # Check timeout
                if current_time - start_time > max_wait_time:
                    timeout_data = {
                        "event_type": "timeout",
                        "message": "Workflow timeout - please refresh to check status"
                    }
                    yield f"data: {json.dumps(timeout_data)}\n\n"
                    break
                # Send new steps if available
                current_step_count = len(workflow_manager.steps)
                if current_step_count > last_step_count:
                    for i in range(last_step_count, current_step_count):
                        step = workflow_manager.steps[i]
                        step_data = step.to_layer_dict(2) if hasattr(step, 'to_layer_dict') else (step.to_dict() if hasattr(step, 'to_dict') else step)
                        event_data = {
                            "event_type": "progress",
                            **step_data
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                    last_step_count = current_step_count
                # Send heartbeat
                heartbeat_data = {
                    "event_type": "heartbeat",
                    "timestamp": current_time,
                    "steps_completed": current_step_count
                }
                yield f"data: {json.dumps(heartbeat_data)}\n\n"
                await asyncio.sleep(0.5)  # Check every 500ms
            # Send final completion or error event
            if workflow_manager.is_completed:
                completion_data = {
                    "event_type": "workflow_completed",
                    "query_id": query_id,
                    "total_steps": len(workflow_manager.steps),
                    "performance_metrics": workflow_manager.performance_metrics,
                    "success": True
                }
            else:
                completion_data = {
                    "event_type": "workflow_failed",
                    "query_id": query_id,
                    "error": workflow_manager.error_message or "Unknown error",
                    "success": False
                }
            yield f"data: {json.dumps(completion_data)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error for query {query_id}: {e}", exc_info=True)
            error_data = {
                "event_type": "error",
                "message": f"Streaming error: {str(e)}"
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_detailed_progress_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


@router.post("/domain/initialize")
async def initialize_domain(request: DomainInitializationRequest) -> Dict[str, Any]:
    """
    Initialize a Universal RAG domain from text files

    This endpoint allows explicit initialization of a domain with specific text files.
    """
    logger.info(f"Initializing domain '{request.domain}' with text files")

    try:
        # Convert text file paths to Path objects
        text_files = None
        if request.text_files:
            text_files = [Path(file_path) for file_path in request.text_files]

            # Validate file paths
            invalid_files = [f for f in text_files if not f.exists()]
            if invalid_files:
                raise HTTPException(
                    status_code=400,
                    detail=f"Text files not found: {[str(f) for f in invalid_files]}"
                )

        # Initialize the domain
        init_results = await initialize_enhanced_rag_system(
            domain_name=request.domain,
            text_files=text_files,
            force_rebuild=request.force_rebuild
        )

        if not init_results.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Domain initialization failed: {init_results.get('error', 'Unknown error')}"
            )

        return {
            "success": True,
            "domain": request.domain,
            "message": "Domain initialized successfully",
            "initialization_results": init_results,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Domain initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/batch")
async def process_batch_queries(request: BatchQueryRequest) -> Dict[str, Any]:
    """
    Process multiple queries in batch for efficiency

    Useful for testing or bulk processing scenarios.
    """
    logger.info(f"Processing batch of {len(request.queries)} queries for domain '{request.domain}'")

    try:
        # Get Enhanced Universal RAG instance
        enhanced_rag = get_enhanced_rag_instance(request.domain)

        # Ensure system is initialized
        if not enhanced_rag.components_initialized:
            init_results = await enhanced_rag.initialize_components()
            if not init_results.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize Universal RAG system: {init_results.get('error', 'Unknown error')}"
                )

        # Process batch queries
        batch_results = await enhanced_rag.process_batch_queries(
            queries=request.queries,
            max_results=request.max_results,
            include_explanations=request.include_explanations
        )

        # Calculate batch statistics
        successful_queries = [r for r in batch_results if r.get("success", False)]
        failed_queries = [r for r in batch_results if not r.get("success", False)]
        total_processing_time = sum(r.get("processing_time", 0) for r in batch_results)

        return {
            "success": True,
            "domain": request.domain,
            "total_queries": len(request.queries),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(request.queries) if request.queries else 0,
            "results": batch_results,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domain/{domain_name}/status")
async def get_domain_status(domain_name: str) -> Dict[str, Any]:
    """
    Get status of a specific domain

    Returns information about initialization status, system stats, and discovered types.
    """
    try:
        # Get Enhanced Universal RAG instance
        enhanced_rag = get_enhanced_rag_instance(domain_name)

        # Get comprehensive status
        status = enhanced_rag.get_system_status()

        return {
            "success": True,
            "domain": domain_name,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get domain status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains/list")
async def list_available_domains() -> Dict[str, Any]:
    """
    List all available domains

    Returns a list of domains that have been initialized.
    """
    try:
        # For now, we'll return domains that have processed data
        processed_data_dir = Path(settings.processed_data_dir)

        domains = []
        if processed_data_dir.exists():
            for file_path in processed_data_dir.glob("universal_rag_state_*.json"):
                domain_name = file_path.stem.replace("universal_rag_state_", "")
                domains.append(domain_name)

        return {
            "success": True,
            "domains": sorted(domains),
            "total_domains": len(domains),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to list domains: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/{query_id}/summary")
async def get_workflow_summary(query_id: str) -> Dict[str, Any]:
    """
    Get comprehensive workflow summary for a query

    Returns detailed workflow information including performance metrics,
    step breakdown, and diagnostic information.
    """
    try:
        workflow_manager = get_workflow_manager(query_id)
        if not workflow_manager:
            raise HTTPException(status_code=404, detail="Workflow not found")

        summary = workflow_manager.get_workflow_summary()
        return {
            "success": True,
            "workflow_summary": summary,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/{query_id}/steps")
async def get_workflow_steps(query_id: str, layer: int = 2) -> Dict[str, Any]:
    """
    Get workflow steps for specific disclosure layer

    Args:
        query_id: Query ID
        layer: Disclosure layer (1=user-friendly, 2=technical, 3=diagnostic)

    Returns detailed step information for the specified layer.
    """
    try:
        workflow_manager = get_workflow_manager(query_id)
        if not workflow_manager:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if layer not in [1, 2, 3]:
            raise HTTPException(status_code=400, detail="Layer must be 1, 2, or 3")

        steps = workflow_manager.get_steps_for_layer(layer)
        return {
            "success": True,
            "query_id": query_id,
            "disclosure_layer": layer,
            "steps": steps,
            "total_steps": len(steps),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow steps: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _process_streaming_query_with_workflow(
    workflow_manager,
    request: StreamingQueryRequest
) -> None:
    """
    Background task to process streaming query with detailed workflow tracking

    This function integrates the Universal Workflow Manager with the Enhanced
    Universal RAG system to provide the detailed 7-step workflow from README architecture.
    """
    from core.workflow.universal_workflow_manager import workflow_registry

    try:
        logger.info(f"Starting workflow processing for query: {workflow_manager.query_id}")

        # Get Enhanced RAG instance
        enhanced_rag = get_enhanced_rag_instance(request.domain)

        # Ensure system is initialized
        if not enhanced_rag.components_initialized:
            logger.info(f"Initializing Enhanced Universal RAG for domain: {request.domain}")
            init_results = await enhanced_rag.initialize_components()

            if not init_results.get("success", False):
                await workflow_manager.fail_workflow(
                    f"Failed to initialize Universal RAG system: {init_results.get('error', 'Unknown error')}"
                )
                # Store failed workflow in registry for debugging
                error_results = {
                    "success": False,
                    "error": init_results.get('error', 'Unknown error'),
                    "query": request.query,
                    "domain": request.domain
                }
                workflow_registry.complete_workflow(workflow_manager.query_id, error_results)
                return

        # Process query with 7-step workflow
        results = await enhanced_rag.process_query(
            query=request.query,
            max_results=request.max_results,
            include_explanations=request.include_explanations,
            enable_safety_warnings=request.enable_safety_warnings,
            stream_progress=True,
            workflow_manager=workflow_manager
        )

        if not results.get("success", False):
            await workflow_manager.fail_workflow(
                f"Query processing failed: {results.get('error', 'Unknown error')}"
            )
            # Store failed workflow in registry for debugging
            error_results = {
                "success": False,
                "error": results.get('error', 'Unknown error'),
                "query": request.query,
                "domain": request.domain
            }
            workflow_registry.complete_workflow(workflow_manager.query_id, error_results)
            return

        # Complete the workflow with final results
        await workflow_manager.complete_workflow(
            results,
            results.get("processing_time", 0)
        )

        # Store completed workflow in registry for frontend access
        workflow_registry.complete_workflow(workflow_manager.query_id, results)

        logger.info(f"Workflow completed successfully for query: {workflow_manager.query_id}")

    except Exception as e:
        logger.error(f"Workflow processing failed: {e}", exc_info=True)
        await workflow_manager.fail_workflow(f"Unexpected error: {str(e)}")
        # Even failed workflows should be accessible for debugging
        error_results = {
            "success": False,
            "error": str(e),
            "query": request.query,
            "domain": request.domain
        }
        workflow_registry.complete_workflow(workflow_manager.query_id, error_results)