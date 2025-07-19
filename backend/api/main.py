"""
FastAPI application for Universal Enhanced RAG
Production-ready API with universal domain support, streaming, and real-time progress
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

# Universal components
from core.orchestration.enhanced_pipeline import (
    get_enhanced_rag_instance, initialize_enhanced_rag_system
)
from config.settings import settings
from api.endpoints import health, universal_query

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with Universal RAG initialization"""
    # Startup
    logger.info("Starting Universal Enhanced RAG API...")

    try:
        # Initialize Universal RAG system for default domain
        logger.info("Initializing Universal RAG system...")
        init_results = await initialize_enhanced_rag_system(
            domain_name="general",
            force_rebuild=False
        )

        app.state.initialization_results = init_results

        if not init_results.get("success", False):
            logger.warning("Universal RAG system initialization incomplete - some features may not work")
            logger.warning(f"Initialization error: {init_results.get('error', 'Unknown error')}")
        else:
            logger.info("Universal RAG system initialized successfully")
            logger.info(f"System stats: {init_results['system_stats']}")

        # Store the Universal RAG instance
        app.state.universal_rag_system = get_enhanced_rag_instance("general")

    except Exception as e:
        logger.error(f"Failed to initialize Universal RAG system: {e}", exc_info=True)
        app.state.universal_rag_system = None
        app.state.initialization_results = {"success": False, "error": str(e)}

    yield

    # Shutdown
    logger.info("Shutting down Universal Enhanced RAG API...")


# Create FastAPI app with Universal RAG support
app = FastAPI(
    title="Universal Enhanced RAG API",
    description="Universal Retrieval-Augmented Generation system that works with any domain",
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
app.include_router(universal_query.router)

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
        "message": "Universal Enhanced RAG API",
        "version": "2.0.0",
        "description": "Universal Retrieval-Augmented Generation system that works with any domain",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "universal_query": "/api/v1/query/universal",
        "streaming_query": "/api/v1/query/streaming",
        "domain_management": "/api/v1/domain/",
        "features": [
            "Universal domain support",
            "Real-time streaming queries",
            "Dynamic entity/relation discovery",
            "No configuration files required",
            "Pure text file processing",
            "Multi-domain batch processing"
        ]
    }

# System info endpoint
@app.get("/api/v1/info")
async def get_system_info():
    """Get system information and status"""
    try:
        # Get initialization results
        init_results = getattr(app.state, 'initialization_results', {})

        # Get current system status if available
        system_status = {}
        if hasattr(app.state, 'universal_rag_system') and app.state.universal_rag_system:
            system_status = app.state.universal_rag_system.get_system_status()

        return {
            "api_version": "2.0.0",
            "system_type": "Universal Enhanced RAG",
            "initialization_status": init_results.get("success", False),
            "initialization_error": init_results.get("error"),
            "system_stats": init_results.get("system_stats", {}),
            "discovered_types": init_results.get("discovered_types", {}),
            "current_status": system_status,
            "features": {
                "universal_domain_support": True,
                "streaming_queries": True,
                "real_time_progress": True,
                "dynamic_type_discovery": True,
                "multi_domain_batch": True
            },
            "endpoints": {
                "universal_query": "/api/v1/query/universal",
                "streaming_query": "/api/v1/query/streaming",
                "batch_query": "/api/v1/query/batch",
                "domain_initialization": "/api/v1/domain/initialize",
                "domain_status": "/api/v1/domain/{domain_name}/status",
                "list_domains": "/api/v1/domains/list"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Dependency to get Universal RAG instance
async def get_universal_rag_instance(domain: str = "general"):
    """Dependency to get Universal RAG instance"""
    try:
        return get_enhanced_rag_instance(domain)
    except Exception as e:
        logger.error(f"Failed to get Universal RAG instance: {e}")
        raise HTTPException(status_code=500, detail="Universal RAG system not available")


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )
