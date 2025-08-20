"""
Permanent Cosmos DB Health Validator
===================================

PERMANENT SOLUTION for validating Cosmos DB authentication on startup.
This ensures authentication issues are caught immediately, not during runtime.
Following the rule: NO temporary fixes, PERMANENT solutions only.
"""

import logging
import os
from typing import Dict, Any
from gremlin_python.driver import client, serializer

logger = logging.getLogger(__name__)


class CosmosHealthValidator:
    """Permanent health validation for Cosmos DB Gremlin authentication"""

    @staticmethod
    def validate_authentication() -> Dict[str, Any]:
        """
        PERMANENT validation of Cosmos DB Gremlin authentication.
        This is called on startup to ensure authentication works.

        Returns:
            Dict with validation results - NO fake success patterns
        """
        validation_result = {
            "authenticated": False,
            "endpoint_reachable": False,
            "database_accessible": False,
            "error_message": None
        }

        try:
            # Get required environment variables
            cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
            cosmos_key = os.getenv("AZURE_COSMOS_KEY")
            database_name = os.getenv("AZURE_COSMOS_DATABASE_NAME", "maintie-rag-prod")
            container_name = os.getenv("AZURE_COSMOS_GRAPH_NAME", "knowledge-graph-prod")

            # Validate environment configuration
            if not cosmos_endpoint:
                validation_result["error_message"] = "AZURE_COSMOS_ENDPOINT not configured"
                return validation_result

            if not cosmos_key:
                validation_result["error_message"] = "AZURE_COSMOS_KEY not configured"
                return validation_result

            # Extract account name and create Gremlin endpoint
            if "documents.azure.com" not in cosmos_endpoint:
                validation_result["error_message"] = f"Invalid Cosmos endpoint format: {cosmos_endpoint}"
                return validation_result

            account_name = cosmos_endpoint.replace("https://", "").replace(":443/", "").split(".")[0]
            gremlin_endpoint = f"wss://{account_name}.gremlin.cosmos.azure.com:443/"

            logger.info(f"Testing Gremlin endpoint: {gremlin_endpoint}")
            logger.info(f"Database: {database_name}, Container: {container_name}")

            # Create test client with exact same configuration as production
            test_client = client.Client(
                gremlin_endpoint,
                "g",
                username=f"/dbs/{database_name}/colls/{container_name}",
                password=cosmos_key,
                message_serializer=serializer.GraphSONSerializersV2d0(),
            )

            validation_result["endpoint_reachable"] = True

            # Test basic authentication with simple query
            try:
                result = test_client.submit("g.V().count()").all().result()
                validation_result["authenticated"] = True
                validation_result["database_accessible"] = True
                logger.info(f"✅ Cosmos DB authentication validated. Vertex count: {result}")

            except Exception as query_error:
                validation_result["error_message"] = f"Authentication failed: {str(query_error)}"
                logger.error(f"❌ Cosmos DB query failed: {query_error}")

            finally:
                test_client.close()

        except Exception as e:
            validation_result["error_message"] = f"Validation error: {str(e)}"
            logger.error(f"❌ Cosmos DB validation failed: {e}")

        return validation_result

    @staticmethod
    def ensure_database_ready() -> bool:
        """
        PERMANENT check to ensure database and container exist.
        Called on startup to verify infrastructure is properly deployed.
        """
        try:
            validation = CosmosHealthValidator.validate_authentication()

            if not validation["authenticated"]:
                logger.error(f"❌ Database not ready: {validation['error_message']}")
                return False

            logger.info("✅ Cosmos DB authentication and database access validated")
            return True

        except Exception as e:
            logger.error(f"❌ Database readiness check failed: {e}")
            return False
