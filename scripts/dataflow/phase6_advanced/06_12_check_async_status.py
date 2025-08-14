#!/usr/bin/env python3
"""
Check Async GNN Deployment Status
=================================

Simple script to check the status of async GNN deployment.
Follows QUICK FAIL principles - returns REAL status only.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.dataflow.phase6_advanced.06_11_gnn_async_bootstrap import GNNAsyncBootstrap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Check async deployment status"""
    logger.info("🔍 Checking GNN async deployment status...")
    
    bootstrap = GNNAsyncBootstrap()
    status = await bootstrap.check_deployment_status()
    
    print(f"📊 GNN Deployment Status: {status['status'].upper()}")
    print(f"💬 Message: {status['message']}")
    
    if status.get('endpoint_name'):
        print(f"📍 Endpoint: {status['endpoint_name']}")
    
    if status.get('elapsed_seconds'):
        print(f"⏰ Elapsed: {status['elapsed_seconds']:.0f} seconds")
    
    if status.get('scoring_uri'):
        print(f"🔗 Scoring URI: {status['scoring_uri']}")
    
    # Pretty print JSON for debugging
    print(f"\n🔧 Full status:")
    print(json.dumps(status, indent=2))
    
    # Return appropriate exit code
    if status['status'] in ['ready']:
        return 0
    elif status['status'] in ['pending', 'not_started']:
        return 2  # Still in progress
    else:
        return 1  # Error

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)