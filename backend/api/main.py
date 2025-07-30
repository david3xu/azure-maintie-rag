"""
FastAPI application for Azure Universal RAG
Production-ready API with Azure services integration, streaming, and real-time progress
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse
import uvicorn

# Azure service components
from services.infrastructure_service import InfrastructureService
from services.data_service import DataService
# Removed integrations import - using focused services
from config.settings import AzureSettings
from config.settings import settings
from api.endpoints import health_endpoint
from api.endpoints import query_endpoint
from api.dependencies import (
    set_infrastructure_service, 
    set_data_service, 
    set_workflow_service,
    set_query_service, 
    set_azure_settings
)
from api.middleware import configure_middleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with Azure services initialization"""
    # Startup
    logger.info("Starting Azure Universal RAG API...")

    try:
        # Initialize Azure services using focused services
        logger.info("Initializing Azure services...")
        from services.workflow_service import WorkflowService
        from services.query_service import QueryService
        
        # Initialize services in proper dependency order
        infrastructure = InfrastructureService()
        data_service = DataService(infrastructure)
        workflow_service = WorkflowService(infrastructure)
        query_service = QueryService()

        # Validate services
        validation = infrastructure.validate_configuration()
        if not validation.get('valid', False):
            logger.warning(f"Azure services validation issues: {validation}")
            # Continue for development - don't fail completely

        azure_settings = AzureSettings()

        # Store in app state
        app.state.infrastructure_service = infrastructure
        app.state.data_service = data_service
        app.state.workflow_service = workflow_service
        app.state.query_service = query_service
        app.state.azure_settings = azure_settings

        # Set global dependencies for endpoints
        set_infrastructure_service(infrastructure)
        set_data_service(data_service)
        set_workflow_service(workflow_service)
        set_query_service(query_service)
        set_azure_settings(azure_settings)

        logger.info("Azure Universal RAG API startup completed")
        yield

    except Exception as e:
        logger.error(f"Azure services initialization failed: {e}")
        raise


# Create FastAPI app with Azure services support
app = FastAPI(
    title="Azure Universal RAG API",
    description="Azure-powered Retrieval-Augmented Generation system that works with any domain",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Setup middleware from middleware.py
configure_middleware(app)

# Include routers
app.include_router(health_endpoint.router)
app.include_router(query_endpoint.router)

# Import and include workflow stream router
from api.streaming.workflow_stream import router as workflow_stream_router
app.include_router(workflow_stream_router)

# Import and include knowledge graph demo router for supervisor demo
from api.endpoints.graph_endpoint import router as graph_router
app.include_router(graph_router)

# Import and include simple demo router for supervisor demo (no async issues)
from api.endpoints.demo_endpoint import router as demo_router
app.include_router(demo_router)

# Import and include Gremlin demo router for real-time queries
from api.endpoints.gremlin_endpoint import router as gremlin_router
app.include_router(gremlin_router)

# Import and include GNN operations router
from api.endpoints.gnn_endpoint import router as gnn_router
app.include_router(gnn_router)

# Import and include workflow evidence router
from api.endpoints.workflow_endpoint import router as workflow_router
app.include_router(workflow_router)

# Import and include unified search demo endpoint (Crown Jewel Demo)
from api.endpoints.unified_search_endpoint import router as unified_search_router
app.include_router(unified_search_router)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Azure Universal RAG API",
        "version": "2.0.0",
        "description": "Azure-powered Retrieval-Augmented Generation system that works with any domain",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "azure_query": "/api/v1/query/universal",
        "streaming_query": "/api/v1/query/streaming",
        "domain_management": "/api/v1/domain/",
                    "features": [
                "Azure services integration",
                "Real-time streaming queries",
                "Azure Cognitive Search",
                "Azure OpenAI processing",
                "Multi-storage architecture (RAG, ML, App)",
                "Azure Cosmos DB metadata",
                "Multi-domain batch processing"
            ]
    }

# System info endpoint
@app.get("/api/v1/info")
async def get_system_info():
    """Get system information and Azure services status"""
    try:
        # Get services status
        infrastructure = getattr(app.state, 'infrastructure_service', None)
        azure_settings = getattr(app.state, 'azure_settings', None)

        azure_status = {
            "initialized": infrastructure is not None,
            "location": azure_settings.azure_region if azure_settings else None,
            "resource_prefix": azure_settings.azure_resource_prefix if azure_settings else None,
            "services": {
                "storage": infrastructure.storage_client is not None if infrastructure else False,
                "cognitive_search": infrastructure.search_client is not None if infrastructure else False,
                "cosmos_db_gremlin": infrastructure.cosmos_client is not None if infrastructure else False,
                "openai": infrastructure.openai_client is not None if infrastructure else False,
                "machine_learning": infrastructure.ml_client is not None if infrastructure else False
            }
        }

        return {
            "api_version": "2.0.0",
            "system_type": "Azure Universal RAG",
            "azure_status": azure_status,
            "features": {
                "azure_services_integration": True,
                "streaming_queries": True,
                "real_time_progress": True,
                "azure_cognitive_search": True,
                "azure_openai": True,
                "azure_blob_storage": True,
                "azure_cosmos_db": True,
                "multi_domain_batch": True
            },
            "endpoints": {
                "azure_query": "/api/v1/query/universal",
                "streaming_query": "/api/v1/query/streaming",
                "batch_query": "/api/v1/query/batch",
                "domain_initialization": "/api/v1/domain/initialize",
                "domain_status": "/api/v1/domain/{domain_name}/status",
                "workflow_summary": "/api/v1/workflow/{query_id}/summary",
                "workflow_steps": "/api/v1/workflow/{query_id}/steps"
            }
        }

    except Exception as e:
        logger.error(f"Failed to get system info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Dependency functions moved to api/dependencies.py to avoid circular imports


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
