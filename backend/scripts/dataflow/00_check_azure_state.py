#!/usr/bin/env python3
"""
Azure State Checker - Pre-execution Data State Validation
Checks current state of all Azure services before running dataflow scripts

This script validates:
- Azure Blob Storage state (containers, documents)
- Azure Cognitive Search state (indices, document count)
- Azure Cosmos DB state (databases, graphs, vertices/edges)
- Raw data availability
- Service connectivity
"""

import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService
from services.data_service import DataService

logger = logging.getLogger(__name__)

class AzureStateChecker:
    """Check current state of all Azure services and data"""
    
    def __init__(self):
        self.infrastructure = InfrastructureService()
        self.data_service = DataService(self.infrastructure)
        
    async def check_complete_state(self, domain: str = "maintenance") -> Dict[str, Any]:
        """
        Check complete Azure data state for a domain
        
        Args:
            domain: Domain to check (maintenance, general, etc.)
            
        Returns:
            Dict with complete state information
        """
        print("🔍 Azure State Checker - Validating Current Data State")
        print("=" * 60)
        
        start_time = datetime.now()
        
        state_report = {
            "check_timestamp": start_time.isoformat(),
            "domain": domain,
            "raw_data_state": {},
            "azure_services_state": {},
            "recommendations": [],
            "ready_for_ingestion": False,
            "requires_processing": True
        }
        
        try:
            print(f"🎯 Checking state for domain: {domain}")
            
            # Check raw data availability
            print(f"\n📁 Checking raw data availability...")
            raw_data_state = await self._check_raw_data_state()
            state_report["raw_data_state"] = raw_data_state
            
            if not raw_data_state.get("has_data", False):
                state_report["recommendations"].append("❌ No raw data found - cannot proceed with ingestion")
                return state_report
            
            print(f"✅ Raw data found: {raw_data_state.get('file_count', 0)} files")
            
            # Check Azure services state using DataService
            print(f"\n☁️  Checking Azure services state...")
            azure_state = await self.data_service.validate_domain_data_state(domain)
            state_report["azure_services_state"] = azure_state
            
            # Analyze current state
            print(f"\n📊 Azure Services Status:")
            print(f"   📦 Storage data: {'✅' if azure_state.get('has_storage_data') else '❌'} ({azure_state.get('storage_blob_count', 0)} blobs)")
            print(f"   🔍 Search data: {'✅' if azure_state.get('has_search_data') else '❌'} ({azure_state.get('search_document_count', 0)} documents)")
            print(f"   🕸️  Cosmos data: {'✅' if azure_state.get('has_cosmos_data') else '❌'} ({azure_state.get('cosmos_vertex_count', 0)} vertices)")
            print(f"   🚀 Data sources ready: {azure_state.get('data_sources_ready', 0)}/3")
            
            # Determine processing requirements
            requires_processing = azure_state.get('requires_processing', True)
            ready_for_ingestion = raw_data_state.get("has_data", False)
            
            state_report["requires_processing"] = requires_processing
            state_report["ready_for_ingestion"] = ready_for_ingestion
            
            # Generate recommendations
            recommendations = []
            if not requires_processing:
                recommendations.append("✅ All Azure services have data - system is ready for queries")
                recommendations.append("ℹ️  Run ingestion only if you want to refresh/update data")
            else:
                missing_services = []
                if not azure_state.get('has_storage_data'):
                    missing_services.append("Storage")
                if not azure_state.get('has_search_data'):
                    missing_services.append("Search")
                if not azure_state.get('has_cosmos_data'):
                    missing_services.append("Cosmos")
                
                if missing_services:
                    recommendations.append(f"🔄 Missing data in: {', '.join(missing_services)}")
                    recommendations.append("✅ Ready to run 01_data_ingestion.py to populate Azure services")
                
            state_report["recommendations"] = recommendations
            
            # Final status
            duration = (datetime.now() - start_time).total_seconds()
            state_report["check_duration_seconds"] = round(duration, 2)
            
            print(f"\n📋 Summary:")
            print(f"   🎯 Domain: {domain}")
            print(f"   📁 Raw data available: {'✅' if ready_for_ingestion else '❌'}")
            print(f"   ☁️  Requires processing: {'✅' if requires_processing else '❌'}")
            print(f"   ⏱️  Check duration: {state_report['check_duration_seconds']}s")
            
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
            
            return state_report
            
        except Exception as e:
            print(f"❌ State check failed: {e}")
            state_report["error"] = str(e)
            logger.error(f"Azure state check failed: {e}", exc_info=True)
            return state_report
    
    async def _check_raw_data_state(self) -> Dict[str, Any]:
        """Check raw data directory state"""
        try:
            raw_data_path = Path(__file__).parent.parent.parent / "data" / "raw"
            
            if not raw_data_path.exists():
                return {
                    "has_data": False,
                    "directory_exists": False,
                    "file_count": 0,
                    "total_size": 0,
                    "files": []
                }
            
            # Get all files in raw data directory
            files = list(raw_data_path.glob("*.md"))
            file_details = []
            total_size = 0
            
            for file_path in files:
                file_size = file_path.stat().st_size
                total_size += file_size
                
                # Get basic file info
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    record_count = len(content.split('<id>')) - 1 if '<id>' in content else 0
                
                file_details.append({
                    "filename": file_path.name,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "maintenance_records": record_count if record_count > 0 else "unknown",
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                })
            
            return {
                "has_data": len(files) > 0,
                "directory_exists": True,
                "file_count": len(files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files": file_details
            }
            
        except Exception as e:
            return {
                "has_data": False,
                "error": str(e)
            }


async def main():
    """Main entry point for Azure state checker"""
    parser = argparse.ArgumentParser(
        description="Azure State Checker - Validate current data state before processing"
    )
    parser.add_argument(
        "--domain", 
        default="maintenance",
        help="Domain to check"
    )
    parser.add_argument(
        "--output",
        help="Save state report to JSON file"
    )
    
    args = parser.parse_args()
    
    # Execute state check
    checker = AzureStateChecker()
    state_report = await checker.check_complete_state(domain=args.domain)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(state_report, f, indent=2)
        print(f"\n📄 State report saved to: {args.output}")
    
    # Return appropriate exit code
    return 0 if state_report.get("ready_for_ingestion", False) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))