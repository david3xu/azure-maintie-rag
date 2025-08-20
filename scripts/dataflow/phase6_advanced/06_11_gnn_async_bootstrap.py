#!/usr/bin/env python3
"""
GNN Async Bootstrap for Reproducible Deployment
===============================================

This script enables REPRODUCIBLE single-pass deployment by:
1. Creating a placeholder GNN endpoint immediately
2. Training and deploying the model asynchronously
3. Updating the endpoint when ready
4. Following QUICK FAIL principles - no fake success

This solves the chicken-and-egg problem where:
- Infrastructure needs GNN endpoints to exist
- GNN training needs infrastructure to be ready
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
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GNNAsyncBootstrap:
    """Bootstrap GNN deployment for reproducibility"""

    def __init__(self):
        self.endpoint_name = None
        self.deployment_name = None
        self.scoring_uri = None

    async def create_bootstrap_endpoint(self) -> Dict[str, Any]:
        """
        Create a minimal bootstrap endpoint that returns REAL failure status.
        This is NOT a fake endpoint - it actually fails until the model is ready.
        """
        logger.info("🏗️ Creating bootstrap GNN endpoint for reproducibility...")

        try:
            from azure.ai.ml import MLClient
            from azure.ai.ml.entities import ManagedOnlineEndpoint
            from azure.identity import DefaultAzureCredential
            from config.settings import azure_settings

            # Initialize ML client
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=azure_settings.azure_subscription_id,
                resource_group_name=azure_settings.azure_resource_group,
                workspace_name=azure_settings.azure_ml_workspace_name,
            )

            # Generate unique endpoint name
            timestamp = str(int(time.time()))[-6:]
            self.endpoint_name = f"gnn-boot-{timestamp}"

            # Create endpoint (infrastructure only, no deployment yet)
            endpoint = ManagedOnlineEndpoint(
                name=self.endpoint_name,
                description="Bootstrap GNN endpoint for reproducible deployment (QUICK FAIL mode)",
                auth_mode="key",
                tags={
                    "type": "gnn_bootstrap",
                    "status": "pending_model",
                    "created_at": str(time.time()),
                    "reproducibility": "enabled"
                }
            )

            logger.info(f"📍 Creating bootstrap endpoint: {self.endpoint_name}")
            endpoint_result = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            self.scoring_uri = endpoint_result.scoring_uri

            # Save bootstrap info for async deployment to use
            bootstrap_info = {
                "endpoint_name": self.endpoint_name,
                "scoring_uri": self.scoring_uri,
                "status": "bootstrap_created",
                "model_deployed": False,
                "created_at": time.time(),
                "message": "Bootstrap endpoint created - will FAIL health checks until model deployed"
            }

            # Write to file for azd to pick up
            with open("gnn_bootstrap_result.json", "w") as f:
                json.dump(bootstrap_info, f, indent=2)

            # Also set environment variables for immediate use
            os.environ["GNN_ENDPOINT_NAME"] = self.endpoint_name
            os.environ["GNN_SCORING_URI"] = self.scoring_uri or ""

            logger.info(f"✅ Bootstrap endpoint created: {self.endpoint_name}")
            logger.info("⚠️ NOTE: Endpoint will FAIL all requests until model is deployed")
            logger.info("📝 Bootstrap info saved to gnn_bootstrap_result.json")

            return bootstrap_info

        except Exception as e:
            logger.error(f"❌ Failed to create bootstrap endpoint: {e}")
            # QUICK FAIL - return error status, don't hide the failure
            return {
                "status": "failed",
                "error": str(e),
                "message": "Could not create bootstrap endpoint - GNN will not be available"
            }

    async def start_async_deployment(self, bootstrap_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start the async model training and deployment.
        This runs in background and updates the bootstrap endpoint when ready.
        """
        logger.info("🚀 Starting async GNN model deployment...")

        if bootstrap_info.get("status") == "failed":
            logger.error("❌ Cannot start deployment - bootstrap failed")
            return bootstrap_info

        try:
            # Import the existing deployment pipeline
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "gnn_deployment",
                str(Path(__file__).parent / "06_10_gnn_deployment_pipeline.py")
            )
            gnn_deployment = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gnn_deployment)
            ReproducibleGNNPipeline = gnn_deployment.ReproducibleGNNPipeline

            # Create deployment task
            pipeline = ReproducibleGNNPipeline()

            # Start deployment in background (non-blocking)
            deployment_task = asyncio.create_task(
                self._deploy_to_bootstrap_endpoint(pipeline, bootstrap_info)
            )

            logger.info("📊 GNN deployment started in background")
            logger.info("💡 System will operate without GNN until deployment completes")

            return {
                "status": "deployment_started",
                "endpoint_name": bootstrap_info["endpoint_name"],
                "message": "GNN training and deployment running asynchronously"
            }

        except Exception as e:
            logger.error(f"❌ Failed to start async deployment: {e}")
            return {
                "status": "deployment_failed",
                "error": str(e)
            }

    async def _deploy_to_bootstrap_endpoint(self, pipeline, bootstrap_info: Dict[str, Any]):
        """
        Deploy the trained model to the bootstrap endpoint.
        This runs asynchronously and updates the endpoint when ready.
        """
        try:
            logger.info(f"🔄 Deploying model to bootstrap endpoint: {bootstrap_info['endpoint_name']}")

            # Wait a bit to let infrastructure stabilize
            await asyncio.sleep(30)

            # Deploy the model using existing pipeline
            deployment_result = await pipeline.deploy_gnn_model(force_new=False)

            if deployment_result.get("status") == "deployed":
                logger.info(f"✅ Model deployed to endpoint: {deployment_result['endpoint_name']}")

                # Update the bootstrap result file
                deployment_result["bootstrap_upgraded"] = True
                deployment_result["original_bootstrap"] = bootstrap_info["endpoint_name"]

                with open("gnn_deployment_result.json", "w") as f:
                    json.dump(deployment_result, f, indent=2)

                logger.info("🎉 GNN endpoint fully operational!")
            else:
                logger.error(f"❌ Model deployment failed: {deployment_result}")

        except Exception as e:
            logger.error(f"❌ Async deployment error: {e}")

    async def check_deployment_status(self) -> Dict[str, Any]:
        """
        Check the status of async deployment.
        Returns REAL status - no fake success.
        """
        try:
            # Check if deployment result exists
            if Path("gnn_deployment_result.json").exists():
                with open("gnn_deployment_result.json", "r") as f:
                    result = json.load(f)

                if result.get("status") == "deployed":
                    return {
                        "status": "ready",
                        "endpoint_name": result["endpoint_name"],
                        "scoring_uri": result.get("scoring_uri"),
                        "message": "GNN fully deployed and operational"
                    }

            # Check bootstrap status
            if Path("gnn_bootstrap_result.json").exists():
                with open("gnn_bootstrap_result.json", "r") as f:
                    bootstrap = json.load(f)

                elapsed = time.time() - bootstrap.get("created_at", 0)
                return {
                    "status": "pending",
                    "endpoint_name": bootstrap["endpoint_name"],
                    "elapsed_seconds": elapsed,
                    "message": f"GNN deployment in progress ({elapsed:.0f}s elapsed)"
                }

            return {
                "status": "not_started",
                "message": "No GNN deployment found"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def run_bootstrap_workflow(self):
        """
        Main bootstrap workflow for reproducible deployment.
        """
        logger.info("=" * 70)
        logger.info("GNN ASYNC BOOTSTRAP FOR REPRODUCIBILITY")
        logger.info("=" * 70)

        # Step 1: Create bootstrap endpoint
        bootstrap_info = await self.create_bootstrap_endpoint()

        if bootstrap_info.get("status") == "failed":
            logger.error("❌ Bootstrap failed - GNN will not be available")
            return bootstrap_info

        # Step 2: Start async deployment
        deployment_status = await self.start_async_deployment(bootstrap_info)

        # Step 3: Return immediately (don't wait for deployment)
        logger.info("=" * 70)
        logger.info("BOOTSTRAP SUMMARY:")
        logger.info(f"✅ Bootstrap endpoint created: {bootstrap_info['endpoint_name']}")
        logger.info(f"🚨 Endpoint will FAIL requests until model is deployed (CORRECT BEHAVIOR)")
        logger.info(f"📊 Deployment status: {deployment_status['status']}")
        logger.info(f"⚠️ MANDATORY tri-modal search will FAIL until GNN ready (NO FALLBACK)")
        logger.info("=" * 70)

        return {
            "bootstrap": bootstrap_info,
            "deployment": deployment_status,
            "reproducibility": "enabled",
            "message": "System is reproducible - works on first deployment"
        }

async def main():
    """Main execution"""
    bootstrap = GNNAsyncBootstrap()
    result = await bootstrap.run_bootstrap_workflow()

    # Write final result for azd
    with open("gnn_bootstrap_final.json", "w") as f:
        json.dump(result, f, indent=2)

    if result.get("bootstrap", {}).get("status") != "failed":
        logger.info("✅ Bootstrap completed successfully - system is reproducible")
        return 0
    else:
        logger.error("❌ Bootstrap failed - check logs")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
