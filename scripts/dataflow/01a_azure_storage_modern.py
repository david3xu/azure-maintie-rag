#!/usr/bin/env python3
"""
Modern Azure Blob Storage Test - Step 01a
Updated for Consolidated Agent Architecture v2.0

Raw Text Data → Azure Blob Storage
Tests ONLY Azure Blob Storage upload functionality with modern architecture
"""

import sys
import asyncio
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Modern imports - consolidated architecture
from services.infrastructure_service import AsyncInfrastructureService
from agents.tools.consolidated_tools import ConsolidatedToolManager
from config.settings import settings, azure_settings

logger = logging.getLogger(__name__)

class ModernAzureStorageStage:
    """Modern Azure Storage stage with consolidated agent architecture"""

    def __init__(self):
        self.infrastructure = None
        self.tool_manager = None
        self.stage_name = "01a_azure_storage"

    async def initialize(self):
        """Initialize with async infrastructure"""
        try:
            logger.info("Initializing Modern Azure Storage Stage...")

            # Initialize async infrastructure service
            self.infrastructure = AsyncInfrastructureService()
            await self.infrastructure.initialize()

            # Initialize consolidated tool manager
            self.tool_manager = ConsolidatedToolManager()

            logger.info("✅ Modern Azure Storage Stage initialized")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Modern Azure Storage Stage: {e}")
            return False

    async def execute(self, input_path: str, container_name: str, verify_upload: bool = True) -> Dict[str, Any]:
        """Execute storage upload with consolidated tools"""
        start_time = time.time()

        try:
            results = {
                "stage": self.stage_name,
                "status": "in_progress",
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "files_processed": [],
                "total_files": 0,
                "total_size_bytes": 0,
                "errors": []
            }

            logger.info(f"🔄 Starting Azure Storage upload...")
            logger.info(f"   Input path: {input_path}")
            logger.info(f"   Container: {container_name}")

            # Check if input path exists
            input_dir = Path(input_path)
            if not input_dir.exists():
                raise FileNotFoundError(f"Input directory not found: {input_path}")

            # Get all text files from input directory
            text_files = list(input_dir.glob("*.txt"))
            results["total_files"] = len(text_files)

            if not text_files:
                logger.warning("⚠️ No .txt files found in input directory")
                results["status"] = "completed_no_files"
                return results

            logger.info(f"📁 Found {len(text_files)} text files to upload")

            # Get storage client from infrastructure
            if not self.infrastructure.storage_client:
                raise RuntimeError("Storage client not available in infrastructure service")

            storage_client = self.infrastructure.storage_client

            # Create container if it doesn't exist
            try:
                await storage_client.create_container(container_name)
                logger.info(f"✅ Container '{container_name}' created/verified")
            except Exception as e:
                logger.info(f"Container may already exist: {e}")

            # Upload each file
            for file_path in text_files:
                try:
                    logger.info(f"📤 Uploading: {file_path.name}")

                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    file_size = len(content.encode('utf-8'))

                    # Upload to storage
                    blob_name = f"dataflow_test/{file_path.name}"
                    await storage_client.upload_blob(
                        container_name=container_name,
                        blob_name=blob_name,
                        data=content,
                        overwrite=True
                    )

                    # Verify upload if requested
                    if verify_upload:
                        try:
                            blob_props = await storage_client.get_blob_properties(
                                container_name=container_name,
                                blob_name=blob_name
                            )
                            verified = blob_props.size == file_size
                        except:
                            verified = False
                    else:
                        verified = None

                    file_result = {
                        "filename": file_path.name,
                        "blob_name": blob_name,
                        "size_bytes": file_size,
                        "upload_verified": verified,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                    }

                    results["files_processed"].append(file_result)
                    results["total_size_bytes"] += file_size

                    logger.info(f"✅ Upload completed: {file_path.name} ({file_size} bytes)")

                except Exception as e:
                    error_msg = f"Failed to upload {file_path.name}: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    results["errors"].append(error_msg)

            # Calculate final results
            execution_time = time.time() - start_time
            successful_uploads = len(results["files_processed"])

            results.update({
                "status": "completed" if not results["errors"] else "completed_with_errors",
                "successful_uploads": successful_uploads,
                "failed_uploads": len(results["errors"]),
                "execution_time_seconds": round(execution_time, 2),
                "throughput_mb_per_second": round((results["total_size_bytes"] / 1024 / 1024) / execution_time, 2) if execution_time > 0 else 0,
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            })

            logger.info(f"🎉 Azure Storage stage completed!")
            logger.info(f"   Successful uploads: {successful_uploads}/{results['total_files']}")
            logger.info(f"   Total size: {results['total_size_bytes']} bytes")
            logger.info(f"   Execution time: {execution_time:.2f}s")

            return results

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Azure Storage stage failed: {str(e)}"
            logger.error(f"❌ {error_msg}")

            results.update({
                "status": "failed",
                "error": error_msg,
                "execution_time_seconds": round(execution_time, 2),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            })

            return results

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.infrastructure:
                # Infrastructure service handles its own cleanup
                pass
            logger.info("✅ Modern Azure Storage Stage cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

async def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description="Modern Azure Storage Upload - Step 01a")
    parser.add_argument("--input-path", required=True, help="Input directory path")
    parser.add_argument("--container-name", required=True, help="Azure storage container name")
    parser.add_argument("--verify-upload", action="store_true", help="Verify uploads after completion")
    parser.add_argument("--output-path", help="Output path for results JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and execute stage
    stage = ModernAzureStorageStage()

    try:
        # Initialize
        if not await stage.initialize():
            logger.error("❌ Failed to initialize stage")
            return 1

        # Execute
        results = await stage.execute(
            input_path=args.input_path,
            container_name=args.container_name,
            verify_upload=args.verify_upload
        )

        # Save results if output path specified
        if args.output_path:
            output_file = Path(args.output_path) / f"{stage.stage_name}_results.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"📝 Results saved to: {output_file}")

        # Print summary
        print("\n" + "="*60)
        print("AZURE STORAGE UPLOAD SUMMARY")
        print("="*60)
        print(f"Status: {results['status']}")
        print(f"Files processed: {results.get('successful_uploads', 0)}/{results['total_files']}")
        print(f"Total size: {results['total_size_bytes']} bytes")
        print(f"Execution time: {results['execution_time_seconds']}s")

        if results.get('errors'):
            print(f"Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")

        # Return success/failure code
        return 0 if results['status'] in ['completed', 'completed_with_errors'] else 1

    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        return 1

    finally:
        await stage.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
