"""
FastAPI application for MaintIE Enhanced RAG
Production-ready API with authentication, monitoring, and error handling
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

from src.pipeline.enhanced_rag import get_rag_instance, initialize_rag_system
from config.settings import settings


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting MaintIE Enhanced RAG API...")

    try:
        # Initialize RAG system
        init_results = initialize_rag_system()
        app.state.initialization_results = init_results

        if not init_results.get("data_transformer", False):
            logger.warning("RAG system initialization incomplete - some features may not work")
        else:
            logger.info("RAG system initialized successfully")

        app.state.rag_system = get_rag_instance()

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
        app.state.rag_system = None
        app.state.initialization_results = {"error": str(e)}

    yield

    # Shutdown
    logger.info("Shutting down MaintIE Enhanced RAG API...")


# Create FastAPI application
app = FastAPI(
    title="MaintIE Enhanced RAG API",
    description="Enterprise maintenance intelligence powered by enhanced RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url}")

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")

    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


def get_rag_system():
    """Dependency to get RAG system instance"""
    if not hasattr(app.state, 'rag_system') or app.state.rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not available - please check system initialization"
        )
    return app.state.rag_system


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MaintIE Enhanced RAG API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "query": "/api/v1/query",
            "health": "/api/v1/health",
            "metrics": "/api/v1/metrics",
            "docs": "/docs"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """System health check endpoint"""
    try:
        # Check if RAG system is available
        if not hasattr(app.state, 'rag_system') or app.state.rag_system is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "RAG system not initialized",
                    "timestamp": time.time()
                }
            )

        # Get detailed health status
        rag_system = app.state.rag_system
        health_status = rag_system.validate_pipeline_health()
        system_status = rag_system.get_system_status()

        # Determine overall health
        overall_status = "healthy"
        if health_status["overall_status"] == "unhealthy":
            status_code = 503
            overall_status = "unhealthy"
        elif health_status["overall_status"] == "degraded":
            status_code = 200
            overall_status = "degraded"
        else:
            status_code = 200

        response_data = {
            "status": overall_status,
            "timestamp": time.time(),
            "components": health_status["components"],
            "system_stats": {
                "queries_processed": system_status["total_queries_processed"],
                "average_response_time": system_status["average_processing_time"],
                "components_initialized": system_status["components_initialized"]
            },
            "issues": health_status.get("issues", []),
            "recommendations": health_status.get("recommendations", [])
        }

        return JSONResponse(status_code=status_code, content=response_data)

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "timestamp": time.time()
            }
        )


@app.get("/api/v1/metrics")
async def get_metrics(rag_system=Depends(get_rag_system)):
    """Get system performance metrics"""
    try:
        performance_metrics = rag_system.get_performance_metrics()
        system_status = rag_system.get_system_status()

        metrics = {
            "timestamp": time.time(),
            "performance": performance_metrics,
            "system": system_status,
            "api_info": {
                "version": "1.0.0",
                "environment": settings.environment
            }
        }

        return metrics

    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


@app.get("/api/v1/system/status")
async def get_system_status(rag_system=Depends(get_rag_system)):
    """Get detailed system status"""
    try:
        status = rag_system.get_system_status()

        # Add initialization results if available
        if hasattr(app.state, 'initialization_results'):
            status["initialization"] = app.state.initialization_results

        return status

    except Exception as e:
        logger.error(f"Error retrieving system status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving system status: {str(e)}")


# Include query endpoints
from api.endpoints.query import router as query_router
app.include_router(query_router, prefix="/api/v1", tags=["Query Processing"])


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
