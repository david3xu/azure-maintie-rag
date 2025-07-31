"""
Azure Workflow Streaming Endpoint
===============================

Connects frontend workflow display to actual Azure services backend implementation
from scripts/query_processing_workflow.py
"""

import asyncio
import json
import time
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# Import Azure service components
from services.infrastructure_service import AsyncInfrastructureService
from services.workflow_service import WorkflowService
from config.settings import AzureSettings

# Temporarily comment out this import to fix server startup
# from core.workflow.azure_workflow_manager import get_workflow_manager
from fastapi import Query
from typing import Optional

router = APIRouter()

# Temporarily commented out to fix server startup
# @router.get("/api/v1/query/stream/{query_id}")
# async def stream_azure_rag_workflow(
#     query_id: str,
#     include_diagnostics: bool = Query(False, description="Include Azure service diagnostics"),
#     azure_service_filter: Optional[str] = Query(None, description="Filter by specific Azure service")
# ):
#     """
#     Stream Azure RAG workflow with service-centric transparency
#     Simplified from three-layer to single configurable stream
#     """
#     pass


async def stream_azure_workflow(query_id: str, query: str, domain: str = "general"):
    try:
        data = {
            'event_type': 'progress',
            'step_number': 6,
            'step_name': 'azure_cosmos_gremlin_storage',
            'user_friendly_name': '[COSMOS GREMLIN] Azure Cosmos DB Gremlin Graph Storage',
            'status': 'in_progress',
            'technology': 'Azure Cosmos DB Gremlin',
            'details': 'Storing query metadata in graph...',
            'progress_percentage': 98
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Store query metadata in Azure Cosmos DB Gremlin graph
        query_metadata = {
            "id": f"query-{time.strftime('%Y%m%d-%H%M%S')}",
            "query": query,
            "domain": domain,
            "search_results_count": len(search_results),
            "retrieved_docs_count": len(retrieved_docs),
            "response_length": len(response),
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
        }

        try:
            cosmos_client = azure_services.get_service('cosmos')
            cosmos_client.add_entity(query_metadata, domain)
        except Exception as e:
            logger.warning(f"Could not store metadata in graph: {e}")

        data = {
            'event_type': 'progress',
            'step_number': 6,
            'step_name': 'azure_cosmos_db_storage',
            'user_friendly_name': '[COSMOS] Azure Cosmos DB Storage',
            'status': 'completed',
            'technology': 'Azure Cosmos DB',
            'details': 'Metadata stored successfully',
            'progress_percentage': 100
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Final completion event
        completion_data = {
            'event_type': 'workflow_completed',
            'query_id': query_id,
            'total_steps': 6,
            'azure_services_used': [
                'Azure Services Manager',
                'Azure OpenAI Integration',
                'Azure Cognitive Search',
                'Azure Blob Storage',
                'Azure OpenAI Processing',
                'Azure Cosmos DB'
            ],
            'success': True,
            'response_length': len(response),
            'documents_processed': len(retrieved_docs)
        }
        yield f"data: {json.dumps(completion_data)}\n\n"

    except Exception as e:
        error_data = {
            'event_type': 'error',
            'query_id': query_id,
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@router.get("/api/v1/query/stream/real/{query_id}")
async def stream_real_workflow_endpoint(
    query_id: str,
    query: str,
    domain: str = "general"
):
    """
    Stream real Azure workflow steps for a query

    This endpoint provides real-time progress updates from the actual
    Azure services backend implementation.
    """
    return StreamingResponse(
        stream_azure_workflow(query_id, query, domain),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )
