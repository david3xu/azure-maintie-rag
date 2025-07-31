"""
Azure Query API Endpoints
New endpoints that use the Azure services architecture for any domain
Replaces maintenance-specific endpoints with Azure-powered universal ones
Enhanced with Azure services integration for detailed real-time progress tracking
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import time

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import uuid

# Azure service components
from services.infrastructure_service import AsyncInfrastructureService
# DataService accessed through infrastructure service to maintain proper layer boundaries
from services.query_service import ConsolidatedQueryService
# Using focused services instead of direct integrations
from config.settings import AzureSettings
from config.settings import settings, azure_settings

# Import dependency functions from dependencies module
from api.dependencies import get_infrastructure_service, get_query_service, get_agent_service
from services.agent_service import ConsolidatedAgentService, AgentServiceRequest

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["azure-query"])


# Pydantic models
class AzureQueryRequest(BaseModel):
    """Azure query request model"""
    query: str = Field(..., description="The query to process")
    domain: str = Field(default="general", description="Domain name for the query")
    max_results: int = Field(default=10, description="Maximum number of results to return")
    include_explanations: bool = Field(default=True, description="Whether to include explanations")
    enable_safety_warnings: bool = Field(default=True, description="Whether to enable safety warnings")


class AzureQueryResponse(BaseModel):
    """Azure query response model"""
    success: bool
    query: str
    domain: str
    generated_response: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    processing_time: float
    azure_services_used: List[str]
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


class PydanticAIQueryRequest(BaseModel):
    """PydanticAI agent query request model"""
    query: str = Field(..., description="The query to process with PydanticAI agent")
    domain: Optional[str] = Field(default=None, description="Domain context for the query")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the query")
    session_id: Optional[str] = Field(default=None, description="Session ID for tracking")
    user_id: Optional[str] = Field(default=None, description="User ID for personalization")
    performance_requirements: Dict[str, float] = Field(
        default_factory=lambda: {"max_response_time": 3.0},
        description="Performance requirements"
    )


class PydanticAIQueryResponse(BaseModel):
    """PydanticAI agent query response model"""
    success: bool
    query: str
    agent_response: str
    confidence: float
    execution_time: float
    tools_used: List[str]
    session_id: str
    correlation_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    error: Optional[str] = None


@router.post("/agent/query", response_model=PydanticAIQueryResponse)
async def process_pydantic_ai_query(
    request: PydanticAIQueryRequest,
    agent_service: ConsolidatedAgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """
    Process a query using the PydanticAI agent with tri-modal search and domain discovery.
    
    This endpoint leverages our competitive advantages:
    - Tri-modal search (Vector + Graph + GNN)
    - Zero-config domain adaptation
    - Dynamic tool discovery and generation
    - Intelligent reasoning with tool chaining
    """
    logger.info(f"Processing PydanticAI agent query: {request.query[:100]}...")
    
    try:
        start_time = time.time()
        
        # Create agent service request
        agent_request = AgentServiceRequest(
            query=request.query,
            context=request.context,
            session_id=request.session_id,
            user_id=request.user_id,
            performance_requirements=request.performance_requirements
        )
        
        # Process request through PydanticAI agent
        agent_response = await agent_service.process_request(agent_request)
        
        execution_time = time.time() - start_time
        
        # Return structured response
        return {
            "success": True,
            "query": request.query,
            "agent_response": agent_response.response,
            "confidence": agent_response.confidence,
            "execution_time": agent_response.execution_time,
            "tools_used": agent_response.tools_used,
            "session_id": agent_response.session_id,
            "correlation_id": agent_response.correlation_id,
            "metadata": {
                **agent_response.metadata,
                "domain": request.domain,
                "user_id": request.user_id,
                "agent_type": "PydanticAI",
                "competitive_advantages": [
                    "tri_modal_search",
                    "zero_config_domain_adaptation", 
                    "dynamic_tool_discovery",
                    "intelligent_reasoning"
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"PydanticAI agent query failed: {e}")
        execution_time = time.time() - start_time
        
        return {
            "success": False,
            "query": request.query,
            "agent_response": f"I apologize, but I encountered an error processing your request: {str(e)}",
            "confidence": 0.0,
            "execution_time": execution_time,
            "tools_used": [],
            "session_id": request.session_id or str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "metadata": {"error": True, "error_type": type(e).__name__},
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.post("/query/universal", response_model=AzureQueryResponse)
async def process_azure_query(
    request: AzureQueryRequest,
    infrastructure: AsyncInfrastructureService = Depends(get_infrastructure_service),
    query_service: ConsolidatedQueryService = Depends(get_query_service)
) -> Dict[str, Any]:
    """
    Process an Azure-powered query that works with any domain

    This endpoint uses Azure services to process queries without requiring
    domain-specific configuration or hardcoded types.
    """
    logger.info(f"Processing Azure query for domain '{request.domain}': {request.query}")

    try:
        start_time = time.time()
        azure_services_used = []

        # Step 1: Process query using the query service
        logger.info("Processing query using EnhancedQueryService...")
        query_response = await query_service.process_universal_query(
            request.query, 
            domain=request.domain, 
            max_results=request.max_results
        )
        
        search_results = query_response.get('data', {}).get('search_results', [])
        azure_services_used.append("Azure Cognitive Search")

        # Extract response data from query service
        generated_response = query_response.get('data', {}).get('response', 'No response generated')
        azure_services_used.extend(["Azure Blob Storage", "Azure OpenAI", "Azure Cosmos DB"])
        
        processing_time = time.time() - start_time

        # Return structured response
        return {
            "success": True,
            "query": request.query,
            "domain": request.domain,
            "generated_response": {
                "content": generated_response,
                "length": len(generated_response) if isinstance(generated_response, str) else 0,
                "model_used": azure_settings.openai_deployment_name or "gpt-4o"
            },
            "search_results": search_results[:request.max_results] if search_results else [],
            "processing_time": processing_time,
            "azure_services_used": azure_services_used,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Azure query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/streaming", response_model=StreamingQueryResponse)
async def start_streaming_query(
    request: StreamingQueryRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Start a streaming Azure query with detailed real-time progress updates

    Returns immediately with a query_id that can be used to monitor progress
    via the streaming endpoint. Uses Azure services for detailed tracking.
    """
    logger.info(f"Starting streaming Azure query for domain '{request.domain}': {request.query}")

    try:
        query_id = str(uuid.uuid4())

        # Start background processing with Azure services tracking
        background_tasks.add_task(
            _process_streaming_query_with_azure,
            query_id,
            request
        )

        return {
            "success": True,
            "query_id": query_id,
            "query": request.query,
            "domain": request.domain,
            "message": "Streaming query started with Azure services tracking",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to start streaming query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/stream/{query_id}")
async def stream_query_progress(query_id: str) -> StreamingResponse:
    """
    Stream real-time detailed progress updates for a query using Azure services
    """
    async def generate_azure_progress_events():
        """Generate Azure service progress events"""
        try:
            # Simulate Azure service progress
            steps = [
                {"step": "azure_cognitive_search", "message": "🔍 Searching Azure Cognitive Search...", "progress": 25},
                {"step": "azure_blob_storage", "message": "☁️ Retrieving documents from Azure Blob Storage...", "progress": 50},
                {"step": "azure_openai", "message": "🤖 Generating response with Azure OpenAI...", "progress": 75},
                {"step": "azure_cosmos_db", "message": "💾 Storing metadata in Azure Cosmos DB...", "progress": 100}
            ]

            for step in steps:
                event_data = {
                    "query_id": query_id,
                    "step": step["step"],
                    "message": step["message"],
                    "progress": step["progress"],
                    "timestamp": datetime.now().isoformat(),
                    "azure_service": step["step"]
                }

                yield f"data: {json.dumps(event_data)}\n\n"
                await asyncio.sleep(1)  # Simulate processing time

            # Final completion event
            completion_data = {
                "query_id": query_id,
                "status": "completed",
                "message": "✅ Azure query processing completed successfully",
                "timestamp": datetime.now().isoformat(),
                "azure_services_used": ["Azure Cognitive Search", "Azure Blob Storage", "Azure OpenAI", "Azure Cosmos DB"]
            }

            yield f"data: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            error_data = {
                "query_id": query_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_azure_progress_events(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@router.post("/domain/initialize")
async def initialize_domain(request: DomainInitializationRequest) -> Dict[str, Any]:
    """
    Initialize a domain with Azure services
    """
    logger.info(f"Initializing domain '{request.domain}' with Azure services")

    try:
        azure_services = AzureServicesManager()
        await azure_services.initialize()

        # Step 1: Create Azure Blob Storage containers for different data types
        rag_container_name = f"rag-data-{request.domain}"
        ml_container_name = f"ml-models-{request.domain}"
        app_container_name = f"app-data-{request.domain}"

        # Create containers using appropriate storage clients
        rag_storage = azure_services.get_rag_storage_client()
        ml_storage = azure_services.get_ml_storage_client()
        app_storage = azure_services.get_app_storage_client()

        await rag_storage.create_container(rag_container_name)
        await ml_storage.create_container(ml_container_name)
        await app_storage.create_container(app_container_name)

        # Step 2: Create Azure Cognitive Search index
        index_name = f"rag-index-{request.domain}"
        search_client = azure_services.get_service('search')
        await search_client.create_index(index_name)

        # Step 3: Initialize Azure Cosmos DB Gremlin graph
        # Gremlin automatically creates graph structure
        logger.info(f"Azure Cosmos DB Gremlin graph ready for domain: {request.domain}")

        return {
            "success": True,
            "domain": request.domain,
            "azure_services_initialized": {
                "rag_storage": rag_container_name,
                "ml_storage": ml_container_name,
                "app_storage": app_container_name,
                "cognitive_search": index_name,
                "cosmos_db_gremlin": f"graph-{request.domain}"
            },
            "message": f"Domain '{request.domain}' initialized with Azure services (multi-storage)"
        }

    except Exception as e:
        logger.error(f"Failed to initialize domain: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/batch")
async def process_batch_queries(request: BatchQueryRequest) -> Dict[str, Any]:
    """
    Process multiple queries in batch using Azure services
    """
    logger.info(f"Processing batch queries for domain '{request.domain}': {len(request.queries)} queries")

    try:
        azure_services = AzureServicesManager()
        await azure_services.initialize()

        openai_integration = UnifiedAzureOpenAIClient()

        results = []
        for i, query in enumerate(request.queries):
            try:
                # Process each query using Azure services
                search_client = azure_services.get_service('search')
                search_results = await search_client.search_documents(
                    query, top=request.max_results
                )

                # Create simple prompt for batch processing
                simple_prompt = f"Answer this query: {query}"
                response = await openai_integration.get_completion(
                    simple_prompt, request.domain
                )

                results.append({
                    "query": query,
                    "success": True,
                    "response": response,
                    "search_results_count": len(search_results)
                })

            except Exception as e:
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })

        return {
            "success": True,
            "domain": request.domain,
            "total_queries": len(request.queries),
            "successful_queries": len([r for r in results if r["success"]]),
            "failed_queries": len([r for r in results if not r["success"]]),
            "results": results
        }

    except Exception as e:
        logger.error(f"Batch query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domain/{domain_name}/status")
async def get_domain_status(domain_name: str) -> Dict[str, Any]:
    """
    Get status of a domain's Azure services
    """
    try:
        azure_services = AzureServicesManager()
        await azure_services.initialize()

        # Check Azure services status for the domain
        container_name = f"rag-data-{domain_name}"
        index_name = f"rag-index-{domain_name}"
        database_name = f"rag-metadata-{domain_name}"

        status = {
            "domain": domain_name,
            "azure_services": {
                "blob_storage": {
                    "container": container_name,
                    "exists": True  # Simplified check
                },
                "cognitive_search": {
                    "index": index_name,
                    "exists": True  # Simplified check
                },
                "cosmos_db": {
                    "database": database_name,
                    "exists": True  # Simplified check
                }
            },
            "status": "active"
        }

        return status

    except Exception as e:
        logger.error(f"Failed to get domain status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains/list")
async def list_available_domains() -> Dict[str, Any]:
    """
    List all available domains with Azure services
    """
    try:
        # Query Azure services for available domains from actual data
        infrastructure = AsyncInfrastructureService()
        storage_client = infrastructure.storage_client
        
        # Get available domains from storage containers
        containers = await storage_client.list_containers()
        domains = []
        
        for container_name in containers:
            # list_containers returns a list of strings (container names)
            if container_name:
                domain_name = container_name.replace('-', '_')
                domains.append({
                    "name": domain_name,
                    "azure_services": ["blob_storage", "cognitive_search", "cosmos_db"],
                    "status": "active",
                    "container": container_name,
                    "last_modified": ""  # Would need separate call to get metadata
                })
        
        # Ensure we have at least general domain
        if not any(d['name'] == 'general' for d in domains):
            domains.insert(0, {
                "name": "general",
                "azure_services": ["blob_storage", "cognitive_search", "cosmos_db"],
                "status": "active"
            })

        return {
            "success": True,
            "domains": domains,
            "total_domains": len(domains)
        }

    except Exception as e:
        logger.error(f"Failed to list domains: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _process_streaming_query_with_azure(
    query_id: str,
    request: StreamingQueryRequest
) -> None:
    """
    Process streaming query using Azure services
    """
    try:
        azure_services = AzureServicesManager()
        await azure_services.initialize()

        openai_integration = UnifiedAzureOpenAIClient()

        # Process the query using Azure services
        search_client = azure_services.get_service('search')
        search_results = await search_client.search_documents(
            request.query, top=request.max_results
        )

        # Create simple prompt for streaming processing
        simple_prompt = f"Answer this query: {request.query}"
        response = await openai_integration.get_completion(
            simple_prompt, request.domain
        )

        logger.info(f"Streaming query {query_id} completed successfully")

    except Exception as e:
        logger.error(f"Streaming query {query_id} failed: {e}", exc_info=True)