#!/usr/bin/env python3
"""
Data Processing Tool
Consolidated script for data upload, preparation, and knowledge extraction
Replaces: data_preparation_workflow.py, data_upload_workflow.py, knowledge_extraction_workflow.py,
         azure_kg_bulk_loader.py, prepare_raw_data.py, full_dataset_extraction.py
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.infrastructure_service import InfrastructureService
from services.data_service import DataService

async def main():
    """Main data processing tool entry point"""
    infrastructure = InfrastructureService()
    data_service = DataService(infrastructure)
    
    print("📊 Data Processing Tool")
    print("="*50)
    
    domain = "general"
    
    # Check current data state
    print("1. Checking current data state...")
    data_state = await data_service.validate_domain_data_state(domain)
    
    if data_state.get('requires_processing', True):
        print("2. Processing required - starting data pipeline...")
        
        # Upload and process data
        migration_result = await data_service.migrate_data_to_azure("data/raw", domain)
        
        if migration_result.get('success', False):
            print("✅ Data processing completed successfully")
            return 0
        else:
            print(f"❌ Data processing failed: {migration_result.get('error', 'Unknown error')}")
            return 1
    else:
        print("✅ Data already processed and available")
        return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))