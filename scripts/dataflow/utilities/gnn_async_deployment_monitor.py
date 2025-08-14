#!/usr/bin/env python3
"""
GNN Async Deployment Monitor - FAIL FAST Pattern
=================================================

This script monitors GNN deployment status and FAILS FAST until the model is ready.
NO fake success, NO placeholders, NO fallbacks.

Usage:
    python gnn_async_deployment_monitor.py [--wait-for-ready] [--timeout 600]
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from infrastructure.azure_ml.gnn_inference_client import GNNInferenceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNDeploymentMonitor:
    """Monitor GNN deployment with STRICT FAIL-FAST pattern."""
    
    def __init__(self):
        """Initialize the monitor."""
        self.gnn_client = GNNInferenceClient()
        self.deployment_status_file = Path("gnn_deployment_result.json")
        self.start_time = time.time()
        
    async def check_deployment_status(self) -> Dict[str, Any]:
        """
        Check if GNN deployment is ready.
        FAILS FAST if not ready - no fake success.
        """
        logger.info("🔍 Checking GNN deployment status...")
        
        # Check environment variables (set by deployment pipeline)
        endpoint_name = os.getenv('GNN_ENDPOINT_NAME')
        scoring_uri = os.getenv('GNN_SCORING_URI')
        
        if not endpoint_name:
            # FAIL FAST - No fake success
            raise RuntimeError(
                "❌ GNN_ENDPOINT_NAME not set - GNN deployment has not started yet.\n"
                "   Run: python scripts/dataflow/phase6_advanced/06_10_gnn_deployment_pipeline.py"
            )
        
        if not scoring_uri:
            # Deployment started but not complete
            raise RuntimeError(
                f"⏳ GNN endpoint '{endpoint_name}' is deploying but not ready yet.\n"
                "   Scoring URI not available. Deployment in progress..."
            )
        
        # Try to validate the endpoint is actually working
        try:
            await self.gnn_client.initialize()
            
            # Test with real prediction
            test_data = {
                "query": "Azure AI services test",
                "entities": [
                    {"text": "Azure AI", "type": "service"},
                    {"text": "test", "type": "action"}
                ]
            }
            
            result = await self.gnn_client.predict(test_data)
            
            if result.get("predictions"):
                logger.info(f"✅ GNN endpoint is READY and responding!")
                return {
                    "status": "ready",
                    "endpoint_name": endpoint_name,
                    "scoring_uri": scoring_uri,
                    "test_predictions": len(result["predictions"]),
                    "deployment_time": time.time() - self.start_time
                }
            else:
                # Endpoint exists but not returning valid predictions
                raise RuntimeError(
                    f"❌ GNN endpoint '{endpoint_name}' exists but returned no predictions.\n"
                    "   Endpoint may still be initializing..."
                )
                
        except Exception as e:
            # FAIL FAST with clear error message
            raise RuntimeError(
                f"❌ GNN endpoint validation failed: {e}\n"
                f"   Endpoint: {endpoint_name}\n"
                f"   Status: NOT READY for production use"
            ) from e
    
    async def wait_for_deployment(self, timeout: int = 600) -> Dict[str, Any]:
        """
        Wait for GNN deployment to complete.
        FAILS after timeout - no infinite waiting.
        """
        logger.info(f"⏳ Waiting for GNN deployment (timeout: {timeout}s)...")
        
        start_time = time.time()
        check_interval = 30  # Check every 30 seconds
        
        while (time.time() - start_time) < timeout:
            try:
                # Try to check status
                status = await self.check_deployment_status()
                
                if status["status"] == "ready":
                    logger.info(f"✅ GNN deployment ready after {status['deployment_time']:.1f}s")
                    
                    # Save status for other scripts
                    with open(self.deployment_status_file, "w") as f:
                        json.dump(status, f, indent=2)
                    
                    return status
                    
            except RuntimeError as e:
                # Expected failures while waiting
                logger.info(f"⏳ Not ready yet: {str(e).split(':', 1)[0]}")
                
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                
                if remaining > 0:
                    logger.info(f"   Waiting {min(check_interval, remaining):.0f}s before retry...")
                    await asyncio.sleep(min(check_interval, remaining))
                else:
                    break
        
        # TIMEOUT - FAIL FAST
        elapsed = time.time() - start_time
        raise TimeoutError(
            f"❌ GNN deployment did not complete within {timeout}s timeout.\n"
            f"   Elapsed time: {elapsed:.1f}s\n"
            f"   Action required: Check Azure ML deployment logs for issues"
        )
    
    async def trigger_async_deployment(self) -> Dict[str, Any]:
        """
        Trigger GNN deployment asynchronously if not already running.
        Returns immediately after starting deployment.
        """
        logger.info("🚀 Triggering async GNN deployment...")
        
        # Check if deployment already exists
        if os.getenv('GNN_ENDPOINT_NAME'):
            logger.info("ℹ️  GNN deployment already triggered or complete")
            return {
                "status": "already_triggered",
                "endpoint_name": os.getenv('GNN_ENDPOINT_NAME')
            }
        
        # Start deployment in background
        import subprocess
        
        deployment_script = Path(__file__).parent.parent / "phase6_advanced" / "06_10_gnn_deployment_pipeline.py"
        
        if not deployment_script.exists():
            raise FileNotFoundError(
                f"GNN deployment script not found: {deployment_script}\n"
                "Cannot trigger async deployment without deployment script"
            )
        
        # Start deployment process in background
        process = subprocess.Popen(
            [sys.executable, str(deployment_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy()
        )
        
        logger.info(f"✅ GNN deployment triggered (PID: {process.pid})")
        logger.info("   Deployment will continue in background...")
        logger.info("   Run with --wait-for-ready to monitor completion")
        
        return {
            "status": "triggered",
            "process_id": process.pid,
            "deployment_script": str(deployment_script)
        }


async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GNN Async Deployment Monitor")
    parser.add_argument(
        "--wait-for-ready",
        action="store_true",
        help="Wait for deployment to complete (blocks until ready)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds when waiting (default: 600)"
    )
    parser.add_argument(
        "--trigger-deployment",
        action="store_true",
        help="Trigger async deployment if not already running"
    )
    
    args = parser.parse_args()
    
    print("🔍 GNN ASYNC DEPLOYMENT MONITOR - FAIL FAST PATTERN")
    print("=" * 60)
    
    monitor = GNNDeploymentMonitor()
    
    try:
        if args.trigger_deployment:
            # Trigger deployment and exit
            result = await monitor.trigger_async_deployment()
            print(f"Status: {result['status']}")
            if result["status"] == "triggered":
                print(f"Process ID: {result['process_id']}")
            return 0
        
        if args.wait_for_ready:
            # Wait for deployment to complete
            result = await monitor.wait_for_deployment(timeout=args.timeout)
            print()
            print("✅ GNN DEPLOYMENT READY FOR PRODUCTION")
            print(f"Endpoint: {result['endpoint_name']}")
            print(f"Scoring URI: {result['scoring_uri']}")
            print(f"Deployment time: {result['deployment_time']:.1f}s")
            return 0
        else:
            # Just check current status
            result = await monitor.check_deployment_status()
            print()
            print("✅ GNN DEPLOYMENT STATUS: READY")
            print(f"Endpoint: {result['endpoint_name']}")
            print(f"Scoring URI: {result['scoring_uri']}")
            return 0
            
    except (RuntimeError, TimeoutError) as e:
        print()
        print(str(e))
        print()
        print("❌ GNN DEPLOYMENT NOT READY")
        print("Next steps:")
        print("  1. Run deployment: python scripts/dataflow/phase6_advanced/06_10_gnn_deployment_pipeline.py")
        print("  2. Wait for ready: python scripts/dataflow/utilities/gnn_async_deployment_monitor.py --wait-for-ready")
        return 1
    except Exception as e:
        print()
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)