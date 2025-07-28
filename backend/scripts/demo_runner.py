#!/usr/bin/env python3
"""
Demo Tool
Consolidated script for demo execution and testing
Replaces: azure-rag-demo-script.py, azure-rag-workflow-demo.py, demo_quick_loader.py
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.infrastructure_service import InfrastructureService
from services.data_service import DataService

async def main():
    """Main demo tool entry point"""
    infrastructure = InfrastructureService()
    data_service = DataService(infrastructure)
    
    print("🎯 Demo Tool")
    print("="*50)
    
    # Quick demo execution
    print("Running Azure RAG demonstration...")
    
    # Check system readiness
    data_state = await data_service.validate_domain_data_state("general")
    
    if data_state.get('requires_processing', True):
        print("⚠️  System requires data processing before demo")
        print("    Run: python data_processing_tool.py")
        return 1
    
    print("✅ System ready for demonstration")
    print("📝 Demo scenarios:")
    print("    1. Knowledge extraction demo")
    print("    2. Query processing demo") 
    print("    3. Multi-hop reasoning demo")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))