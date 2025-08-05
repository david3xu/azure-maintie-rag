"""
Simple Search API - CODING_STANDARDS Compliant
Clean search endpoints without over-engineering.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

# Use consolidated agent approach
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.universal_search.agent import UniversalSearchAgent
from agents.core.azure_service_container import ConsolidatedAzureServices

# Create router
router = APIRouter(prefix="/api/v1", tags=["search"])

# Simple models
class SearchRequest(BaseModel):
    query: str
    # TODO: max_results should be query-complexity driven from Config-Extraction workflow
    # HARDCODED: max_results = 10 should be domain-specific and adaptive
    max_results: int = None  # Must be loaded from domain analysis
    domain: Optional[str] = None

class SearchResponse(BaseModel):
    success: bool
    query: str
    results: list
    execution_time: float
    timestamp: str
    error: Optional[str] = None


@router.post("/search", response_model=SearchResponse)
async def search_content(request: SearchRequest) -> Dict[str, Any]:
    """Simple universal search endpoint"""
    
    try:
        start_time = time.time()
        
        # Initialize services and agent
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()
        
        search_agent = UniversalSearchAgent(azure_services)
        
        # TODO: All search parameters should be loaded from Config-Extraction workflow
        # HARDCODED VALUES REMOVED to force workflow integration
        
        # Validate that max_results is provided (no more hardcoded defaults)
        if request.max_results is None:
            raise ValueError(
                "max_results parameter is required. "
                "Hardcoded default removed to force domain-specific optimization from Config-Extraction workflow."
            )
        
        # TODO: Default domain should be from domain intelligence detection
        # HARDCODED: "general" domain should be replaced with domain detection
        detected_domain = request.domain or "NEEDS_DOMAIN_DETECTION"  # TODO: Use domain intelligence
        
        search_request = {
            "query": request.query,
            "limit": request.max_results,  # Now required - no hardcoded default
            "domain": detected_domain,  # TODO: Must use domain detection
            "config_source": "API_ENDPOINT_PLACEHOLDER"  # TODO: Mark for workflow integration
        }
        
        # Execute search
        result = await search_agent.process_query(search_request)
        
        execution_time = time.time() - start_time
        
        if result.get("success"):
            return {
                "success": True,
                "query": request.query,
                "results": result.get("results", []),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "query": request.query,
                "results": [],
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "error": result.get("error", "Search failed")
            }
            
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "success": False,
            "query": request.query,
            "results": [],
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.get("/search/health")
async def search_health_check() -> Dict[str, Any]:
    """Simple search health check"""
    
    try:
        # Test basic service availability
        azure_services = ConsolidatedAzureServices()
        service_status = azure_services.get_service_status()
        
        return {
            "status": "healthy" if service_status["overall_health"] else "degraded",
            "services_ready": f"{service_status['successful_services']}/{service_status['total_services']}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }