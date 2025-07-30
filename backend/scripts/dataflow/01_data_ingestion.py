#!/usr/bin/env python3
"""
Data Ingestion - Stage 1 of README Data Flow
Raw Text Data → Azure Blob Storage

This script implements the first stage of the README architecture:
- Uses existing DataService to migrate raw data to Azure services  
- Processes data from data/raw directory
- Uploads to Blob Storage, Search Index, and Cosmos DB
- Prepares data for knowledge extraction stage
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService
from services.data_service import DataService
from config.settings import azure_settings

logger = logging.getLogger(__name__)

class DataIngestionStage:
    """Stage 1: Raw Text Data → Azure Blob Storage (using existing DataService)"""
    
    def __init__(self):
        self.infrastructure = InfrastructureService()
        self.data_service = DataService(self.infrastructure)
        
    async def execute(self, source_path: str, domain: str = "general") -> Dict[str, Any]:
        """
        Execute data ingestion stage using existing DataService
        
        Args:
            source_path: Path to raw text files
            domain: Domain for processing
            
        Returns:
            Dict with ingestion results and metadata
        """
        print("🔄 Stage 1: Data Ingestion - Raw Text → Azure Blob Storage")
        print("=" * 60)
        
        start_time = asyncio.get_event_loop().time()
        
        results = {
            "stage": "01_data_ingestion",
            "source_path": str(source_path),
            "domain": domain,
            "success": False
        }
        
        try:
            # Use the existing DataService migrate_data_to_azure method
            print(f"🚀 Using DataService to migrate data from: {source_path}")
            migration_result = await self.data_service.migrate_data_to_azure(source_path, domain)
            
            # Extract relevant information from migration result
            results.update({
                "migration_result": migration_result,
                "migrations_completed": migration_result.get("summary", {}).get("successful_migrations", 0),
                "total_migrations": migration_result.get("summary", {}).get("total_migrations", 0),
                "status": migration_result.get("status", "unknown")
            })
            
            # ENFORCE Azure-only mode: Success ONLY if ALL Azure services work
            # NO partial success, NO degraded mode, NO local fallbacks
            status = migration_result.get("status")
            success = status == "completed"  # ONLY full completion is success
            
            # Explicitly fail for partial success or degraded modes
            if status in ["functional_degraded", "partial_success"]:
                success = False
                print(f"❌ REJECTING partial success - Azure-only mode requires ALL services working")
                
            results["success"] = success
            
            # Duration
            duration = asyncio.get_event_loop().time() - start_time
            results["duration_seconds"] = round(duration, 2)
            
            if success:
                print(f"✅ Stage 1 Complete:")
                print(f"   🚀 Migrations: {results['migrations_completed']}/{results['total_migrations']}")
                print(f"   📊 Status: {results['status']}")
                print(f"   ⏱️  Duration: {results['duration_seconds']}s")
            else:
                print(f"⚠️  Stage 1 Partial: {migration_result.get('status', 'unknown')}")
                if migration_result.get("error"):
                    print(f"   ❌ Error: {migration_result.get('error')}")
                
            return results
            
        except Exception as e:
            results["error"] = str(e)
            results["duration_seconds"] = round(asyncio.get_event_loop().time() - start_time, 2)
            print(f"❌ Stage 1 Failed: {e}")
            logger.error(f"Data ingestion failed: {e}", exc_info=True)
            return results


async def main():
    """Main entry point for data ingestion stage"""
    parser = argparse.ArgumentParser(
        description="Stage 1: Data Ingestion - Raw Text → Azure Blob Storage"
    )
    parser.add_argument(
        "--source", 
        required=True,
        help="Path to raw text files"
    )
    parser.add_argument(
        "--domain", 
        default="general",
        help="Domain for processing"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Execute stage
    stage = DataIngestionStage()
    results = await stage.execute(
        source_path=args.source,
        domain=args.domain
    )
    
    # Save results if requested
    if args.output and results.get("success"):
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {args.output}")
    
    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))