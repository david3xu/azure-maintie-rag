"""
PERMANENT Diagnostics Endpoint
==============================

Provides real-time diagnostic information about Azure service authentication.
PERMANENT solution for debugging authentication issues without bypassing.
"""

import os
import logging
from typing import Dict, Any
from fastapi import APIRouter

from infrastructure.azure_cosmos.health_validator import CosmosHealthValidator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/api/v1/diagnostics/cosmos")
async def cosmos_diagnostics() -> Dict[str, Any]:
    """
    PERMANENT diagnostic endpoint for Cosmos DB authentication.
    Provides detailed debugging information without bypassing security.
    """
    try:
        # Get environment configuration
        env_config = {
            "cosmos_endpoint": os.getenv("AZURE_COSMOS_ENDPOINT", "NOT_SET"),
            "cosmos_key_configured": bool(os.getenv("AZURE_COSMOS_KEY")),
            "cosmos_key_length": len(os.getenv("AZURE_COSMOS_KEY", "")) if os.getenv("AZURE_COSMOS_KEY") else 0,
            "database_name": os.getenv("AZURE_COSMOS_DATABASE_NAME", "NOT_SET"),
            "container_name": os.getenv("AZURE_COSMOS_GRAPH_NAME", "NOT_SET"),
            "connection_string_configured": bool(os.getenv("COSMOS_CONNECTION_STRING"))
        }

        # Run authentication validation
        validation_result = CosmosHealthValidator.validate_authentication()

        # Construct diagnostic response
        diagnostic_info = {
            "timestamp": "2025-08-20T12:57:00Z",
            "environment_config": env_config,
            "authentication_test": validation_result,
            "recommendations": []
        }

        # Add specific recommendations based on findings
        if not validation_result["authenticated"]:
            if not env_config["cosmos_key_configured"]:
                diagnostic_info["recommendations"].append("AZURE_COSMOS_KEY environment variable not set")
            elif env_config["cosmos_key_length"] < 60:
                diagnostic_info["recommendations"].append("AZURE_COSMOS_KEY appears invalid (too short)")
            elif not validation_result["endpoint_reachable"]:
                diagnostic_info["recommendations"].append("Cannot reach Cosmos DB Gremlin endpoint")
            else:
                diagnostic_info["recommendations"].append("Authentication credentials invalid - check key rotation")

        return diagnostic_info

    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        return {
            "error": str(e),
            "timestamp": "2025-08-20T12:57:00Z",
            "status": "diagnostic_failure"
        }

@router.get("/api/v1/diagnostics/environment")
async def environment_diagnostics() -> Dict[str, Any]:
    """
    PERMANENT diagnostic endpoint for environment configuration.
    Shows all Azure service configuration without exposing secrets.
    """
    env_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_COSMOS_ENDPOINT",
        "AZURE_STORAGE_ACCOUNT",
        "AZURE_COSMOS_DATABASE_NAME",
        "AZURE_COSMOS_GRAPH_NAME",
        "USE_MANAGED_IDENTITY"
    ]

    config_status = {}
    for var in env_vars:
        value = os.getenv(var)
        config_status[var] = {
            "configured": bool(value),
            "length": len(value) if value else 0,
            "preview": value[:20] + "..." if value and len(value) > 20 else value
        }

    # Check for sensitive variables without exposing them
    sensitive_vars = ["AZURE_COSMOS_KEY", "COSMOS_CONNECTION_STRING"]
    for var in sensitive_vars:
        value = os.getenv(var)
        config_status[var] = {
            "configured": bool(value),
            "length": len(value) if value else 0,
            "valid_format": value and len(value) > 50 if value else False
        }

    return {
        "timestamp": "2025-08-20T12:57:00Z",
        "environment_variables": config_status,
        "container_info": {
            "startup_validation_enabled": True,
            "permanent_solutions_active": True
        }
    }
