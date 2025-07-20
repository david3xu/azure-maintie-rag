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
from integrations.azure_services import AzureServicesManager
from integrations.azure_openai import AzureOpenAIClient
from config.settings import AzureSettings

router = APIRouter()

async def stream_azure_workflow(query_id: str, query: str, domain: str = "general") -> AsyncGenerator[str, None]:
    """
    Stream real Azure workflow steps from query_processing_workflow.py logic

    This connects the frontend WorkflowProgress component to the actual
    Azure services backend implementation, ensuring the UI reflects real processing steps.
    """

    try:
        # Step 1: Azure Services Initialization
        data = {
            'event_type': 'progress',
            'step_number': 1,
            'step_name': 'azure_services_initialization',
            'user_friendly_name': '[AZURE] Azure Services Initialization',
            'status': 'in_progress',
            'technology': 'AzureServicesManager',
            'details': 'Initializing Azure services...',
            'progress_percentage': 14
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Initialize Azure services (real backend logic)
        azure_services = AzureServicesManager()
        await azure_services.initialize()

        data = {
            'event_type': 'progress',
            'step_number': 1,
            'step_name': 'azure_services_initialization',
            'user_friendly_name': '[AZURE] Azure Services Initialization',
            'status': 'completed',
            'technology': 'AzureServicesManager',
            'details': 'Azure services initialized successfully',
            'progress_percentage': 28
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Step 2: Azure OpenAI Integration
        data = {
            'event_type': 'progress',
            'step_number': 2,
            'step_name': 'azure_openai_integration',
            'user_friendly_name': '[OPENAI] Azure OpenAI Integration',
            'status': 'in_progress',
            'technology': 'AzureOpenAIClient',
            'details': 'Setting up Azure OpenAI integration...',
            'progress_percentage': 35
        }
        yield f"data: {json.dumps(data)}\n\n"

        openai_integration = AzureOpenAIClient()

        data = {
            'event_type': 'progress',
            'step_number': 2,
            'step_name': 'azure_openai_integration',
            'user_friendly_name': '[OPENAI] Azure OpenAI Integration',
            'status': 'completed',
            'technology': 'AzureOpenAIClient',
            'details': 'Azure OpenAI integration ready',
            'progress_percentage': 42
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Step 3: Azure Cognitive Search
        data = {
            'event_type': 'progress',
            'step_number': 3,
            'step_name': 'azure_cognitive_search',
            'user_friendly_name': '[SEARCH] Azure Cognitive Search',
            'status': 'in_progress',
            'technology': 'Azure Cognitive Search',
            'details': 'Searching for relevant documents...',
            'progress_percentage': 56
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Execute real Azure search (same as scripts/query_processing_workflow.py)
        index_name = f"rag-index-{domain}"
        search_results = await azure_services.search_client.search_documents(
            index_name, query, top_k=5
        )

        data = {
            'event_type': 'progress',
            'step_number': 3,
            'step_name': 'azure_cognitive_search',
            'user_friendly_name': '[SEARCH] Azure Cognitive Search',
            'status': 'completed',
            'technology': 'Azure Cognitive Search',
            'details': f'Found {len(search_results)} relevant documents',
            'progress_percentage': 64
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Step 4: Azure Blob Storage Retrieval
        data = {
            'event_type': 'progress',
            'step_number': 4,
            'step_name': 'azure_blob_storage_retrieval',
            'user_friendly_name': '[STORAGE] Azure Blob Storage Retrieval',
            'status': 'in_progress',
            'technology': 'Azure Blob Storage',
            'details': 'Retrieving document content...',
            'progress_percentage': 72
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Retrieve documents from Azure Blob Storage
        container_name = f"rag-data-{domain}"
        retrieved_docs = []

        for i, result in enumerate(search_results[:3]):
            blob_name = f"document_{i}.txt"
            try:
                content = await azure_services.storage_client.download_text(container_name, blob_name)
                retrieved_docs.append(content)
            except Exception as e:
                logger.warning(f"Could not retrieve document {i}: {e}")

        data = {
            'event_type': 'progress',
            'step_number': 4,
            'step_name': 'azure_blob_storage_retrieval',
            'user_friendly_name': '[STORAGE] Azure Blob Storage Retrieval',
            'status': 'completed',
            'technology': 'Azure Blob Storage',
            'details': f'Retrieved {len(retrieved_docs)} documents',
            'progress_percentage': 80
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Step 5: Azure OpenAI Processing
        data = {
            'event_type': 'progress',
            'step_number': 5,
            'step_name': 'azure_openai_processing',
            'user_friendly_name': '[OPENAI] Azure OpenAI Processing',
            'status': 'in_progress',
            'technology': 'Azure OpenAI',
            'details': 'Generating response with Azure OpenAI...',
            'progress_percentage': 88
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Generate response using Azure OpenAI
        response = await openai_integration.generate_response(
            query, retrieved_docs, domain
        )

        data = {
            'event_type': 'progress',
            'step_number': 5,
            'step_name': 'azure_openai_processing',
            'user_friendly_name': '[OPENAI] Azure OpenAI Processing',
            'status': 'completed',
            'technology': 'Azure OpenAI',
            'details': 'Response generated successfully',
            'progress_percentage': 96
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Step 6: Azure Cosmos DB Gremlin Graph Storage
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
            await azure_services.cosmos_client.add_entity(query_metadata, domain)
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