"""
Universal Query Orchestrator - Compatibility Module
===================================================

Backward compatibility wrapper for query orchestration.
"""

from typing import Any, Dict
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Query request model for backward compatibility."""
    query: str
    query_type: str = "analysis"
    parameters: Dict[str, Any] = {}


class UniversalQueryOrchestrator:
    """Universal query orchestrator for backward compatibility."""
    
    async def orchestrate_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Orchestrate a query request."""
        return {
            "query": request.query,
            "type": request.query_type,
            "result": "placeholder_result"
        }


# Global orchestrator instance
query_orchestrator = UniversalQueryOrchestrator()


# Orchestrated functions for backward compatibility
async def generate_analysis_query_orchestrated(query: str, **kwargs) -> str:
    """Generate analysis query via orchestrator."""
    from agents.shared.query_tools import generate_analysis_query
    from agents.core.universal_deps import get_universal_deps
    
    # Create a minimal RunContext-like object
    class MockRunContext:
        def __init__(self):
            pass
    
    # Try to use the real function, fall back to simple passthrough
    try:
        deps = await get_universal_deps()
        mock_ctx = MockRunContext()
        mock_ctx.deps = deps
        return await generate_analysis_query(mock_ctx, query, **kwargs)
    except Exception:
        return query


async def generate_gremlin_query_orchestrated(query: str, **kwargs) -> str:
    """Generate Gremlin query via orchestrator."""
    from agents.shared.query_tools import generate_gremlin_query
    from agents.core.universal_deps import get_universal_deps
    
    class MockRunContext:
        def __init__(self):
            pass
    
    try:
        deps = await get_universal_deps()
        mock_ctx = MockRunContext()
        mock_ctx.deps = deps
        return await generate_gremlin_query(mock_ctx, query, **kwargs)
    except Exception:
        return f"g.V().has('text', containing('{query}'))"


async def generate_search_query_orchestrated(query: str, **kwargs) -> str:
    """Generate search query via orchestrator."""
    return query


# Re-export for backward compatibility
__all__ = [
    "QueryRequest",
    "UniversalQueryOrchestrator", 
    "query_orchestrator",
    "generate_analysis_query_orchestrated",
    "generate_gremlin_query_orchestrated",
    "generate_search_query_orchestrated",
]