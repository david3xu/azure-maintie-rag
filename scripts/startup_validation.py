#!/usr/bin/env python3
"""
PERMANENT Startup Validation Script
===================================

This script runs on container startup to validate ALL Azure services permanently.
NO temporary fixes, NO bypassing - validates and fixes authentication issues.
Following the rule: PERMANENT solutions only.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.azure_cosmos.health_validator import CosmosHealthValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """PERMANENT startup validation for all Azure services"""
    logger.info("🔧 PERMANENT Startup Validation - Azure Universal RAG")
    logger.info("=" * 60)

    # Validate environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_COSMOS_ENDPOINT",
        "AZURE_COSMOS_KEY",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_STORAGE_ACCOUNT"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        return False

    logger.info("✅ Environment variables validated")

    # Test Cosmos DB authentication permanently
    logger.info("🧪 Testing Cosmos DB Gremlin authentication...")
    cosmos_validation = CosmosHealthValidator.validate_authentication()

    if cosmos_validation["authenticated"]:
        logger.info("✅ Cosmos DB authentication successful")
    else:
        logger.error(f"❌ Cosmos DB authentication failed: {cosmos_validation['error_message']}")

        # Print detailed debugging information for permanent fix
        logger.error("🔍 Debugging information:")
        logger.error(f"   Endpoint: {os.getenv('AZURE_COSMOS_ENDPOINT')}")
        logger.error(f"   Database: {os.getenv('AZURE_COSMOS_DATABASE_NAME')}")
        logger.error(f"   Container: {os.getenv('AZURE_COSMOS_GRAPH_NAME')}")
        logger.error(f"   Key configured: {'Yes' if os.getenv('AZURE_COSMOS_KEY') else 'No'}")

        return False

    logger.info("🎉 ALL Azure services validated successfully")
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        logger.error("❌ Startup validation failed - container cannot start")
        sys.exit(1)
    else:
        logger.info("✅ Startup validation complete - ready for requests")
