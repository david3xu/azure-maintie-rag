"""
FastAPI application for Azure Universal RAG
Production-ready API with Azure services integration, streaming, and real-time progress
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Azure service components
from azure.integrations.azure_services import AzureServicesManager
from azure.integrations.azure_openai import AzureOpenAIIntegration
from config.settings import AzureSettings
from config.settings import settings
from api.endpoints import health, azure_query_endpoint

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
        # Initialize Azure services
        logger.info("Initializing Azure services...")
        azure_services = AzureServicesManager()
        await azure_services.initialize()

        openai_integration = AzureOpenAIIntegration()
        azure_settings = AzureSettings()

        # Store Azure services in app state
        app.state.azure_services = azure_services
        app.state.openai_integration = openai_integration
        app.state.azure_settings = azure_settings

        logger.info("Azure services initialized successfully")
        logger.info(f"Azure location: {azure_settings.azure_location}")
        logger.info(f"Azure resource prefix: {azure_settings.azure_resource_prefix}")

    except Exception as e:
        logger.error(f"Failed to initialize Azure services: {e}", exc_info=True)
        app.state.azure_services = None
        app.state.openai_integration = None
        app.state.azure_settings = None

    yield

    # Shutdown
    logger.info("Shutting down Azure Universal RAG API...")


# Create FastAPI app with Azure services support
app = FastAPI(
    title="Azure Universal RAG API",
    description="Azure-powered Retrieval-Augmented Generation system that works with any domain",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security (if configured)
if hasattr(settings, 'trusted_hosts') and settings.trusted_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_hosts
    )

# Include routers
app.include_router(health.router)
app.include_router(azure_query_endpoint.router)

# Import and include workflow stream router
from api.workflow_stream import router as workflow_stream_router
app.include_router(workflow_stream_router)

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

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()

    # Skip logging for health checks and static files
    if request.url.path in ["/api/v1/health", "/favicon.ico"]:
        response = await call_next(request)
        return response

    logger.info(f"Request: {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {e} - {process_time:.3f}s")
        raise

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
            "Azure Blob Storage",
            "Azure Cosmos DB metadata",
            "Multi-domain batch processing"
        ]
    }

# System info endpoint
@app.get("/api/v1/info")
async def get_system_info():
    """Get system information and Azure services status"""
    try:
        # Get Azure services status
        azure_services = getattr(app.state, 'azure_services', None)
        azure_settings = getattr(app.state, 'azure_settings', None)

        azure_status = {
            "initialized": azure_services is not None,
            "location": azure_settings.azure_location if azure_settings else None,
            "resource_prefix": azure_settings.azure_resource_prefix if azure_settings else None,
            "services": {
                "blob_storage": azure_services.storage_client is not None if azure_services else False,
                "cognitive_search": azure_services.search_client is not None if azure_services else False,
                "cosmos_db_gremlin": azure_services.cosmos_client is not None if azure_services else False,
                "machine_learning": azure_services.ml_client is not None if azure_services else False
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


async def get_azure_services() -> AzureServicesManager:
    """Get Azure services instance"""
    azure_services = getattr(app.state, 'azure_services', None)
    if not azure_services:
        raise HTTPException(status_code=503, detail="Azure services not initialized")
    return azure_services


async def get_openai_integration() -> AzureOpenAIIntegration:
    """Get Azure OpenAI integration instance"""
    openai_integration = getattr(app.state, 'openai_integration', None)
    if not openai_integration:
        raise HTTPException(status_code=503, detail="Azure OpenAI integration not initialized")
    return openai_integration


async def get_azure_settings() -> AzureSettings:
    """Get Azure settings instance"""
    azure_settings = getattr(app.state, 'azure_settings', None)
    if not azure_settings:
        raise HTTPException(status_code=503, detail="Azure settings not initialized")
    return azure_settings


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
