"""
Enhanced health checks for production monitoring
Simple, professional implementation
"""

import logging
import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from src.pipeline.rag_structured import MaintIEStructuredRAG

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton or factory for RAG instance (adjust as needed)
rag_instance = MaintIEStructuredRAG()

def get_rag_instance():
    return rag_instance

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check for production monitoring"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {},
        "performance": {},
        "cache": {}
    }
    try:
        rag_system = get_rag_instance()
        health_status["checks"]["rag_system"] = "healthy" if rag_system else "unhealthy"
        if rag_system:
            structured_rag = rag_system
            health_status["checks"]["structured_pipeline"] = (
                "healthy" if getattr(structured_rag, 'components_initialized', True) else "initializing"
            )
            graph_enabled = getattr(structured_rag, 'graph_operations_enabled', False)
            health_status["checks"]["graph_operations"] = (
                "enabled" if graph_enabled else "disabled"
            )
            caching_enabled = getattr(structured_rag, 'caching_enabled', False)
            health_status["checks"]["response_caching"] = (
                "enabled" if caching_enabled else "disabled"
            )
            if (caching_enabled and hasattr(structured_rag, 'response_cache') and structured_rag.response_cache):
                health_status["cache"] = structured_rag.response_cache.get_cache_stats()
            if hasattr(structured_rag, 'query_count'):
                health_status["performance"] = {
                    "total_queries": structured_rag.query_count,
                    "average_processing_time": getattr(structured_rag, 'average_processing_time', 0)
                }
        unhealthy_checks = [
            check for check, status in health_status["checks"].items()
            if status == "unhealthy"
        ]
        if unhealthy_checks:
            health_status["status"] = "unhealthy"
            health_status["issues"] = unhealthy_checks
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@router.get("/health/cache")
async def cache_health() -> Dict[str, Any]:
    """Specific cache health and statistics"""
    try:
        rag_system = get_rag_instance()
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system not available")
        structured_rag = rag_system
        if (not getattr(structured_rag, 'caching_enabled', False) or
            not hasattr(structured_rag, 'response_cache')):
            return {"status": "disabled", "message": "Caching not enabled"}
        cache_stats = structured_rag.response_cache.get_cache_stats()
        cache_stats["status"] = "healthy"
        cache_stats["timestamp"] = time.time()
        return cache_stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "status": "error"}
        )

@router.post("/health/cache/clear")
async def clear_cache() -> Dict[str, Any]:
    """Clear response cache (admin endpoint)"""
    try:
        rag_system = get_rag_instance()
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system not available")
        structured_rag = rag_system
        if (not getattr(structured_rag, 'caching_enabled', False) or
            not hasattr(structured_rag, 'response_cache')):
            raise HTTPException(status_code=400, detail="Caching not available")
        success = structured_rag.response_cache.clear_cache()
        return {
            "success": success,
            "message": "Cache cleared successfully" if success else "Failed to clear cache",
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "success": False}
        )