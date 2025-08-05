#!/usr/bin/env python3
"""
Simple Query Pipeline - CODING_STANDARDS Compliant
Clean query processing script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.universal_search.agent import UniversalSearchAgent
from agents.core.azure_service_container import ConsolidatedAzureServices


async def process_query(query: str, domain: str = "general"):
    """Simple query processing pipeline"""
    print(f"🔍 Processing Query: '{query}'")
    
    try:
        # Initialize services
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()
        
        # Initialize search agent
        search_agent = UniversalSearchAgent(azure_services)
        
        # Process query
        result = await search_agent.process_query({
            "query": query,
            "domain": domain,
            "limit": 10
        })
        
        if result.get("success"):
            results = result.get("results", [])
            print(f"✅ Found {len(results)} results")
            
            for i, result_item in enumerate(results[:3], 1):
                title = result_item.get("title", "No title")[:50]
                score = result_item.get("score", 0)
                print(f"   {i}. {title}... (score: {score:.2f})")
            
            return True
        else:
            print(f"❌ Query failed: {result.get('error')}")
            return False
        
    except Exception as e:
        print(f"❌ Query processing failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple query pipeline")
    parser.add_argument("query", help="Query to process")
    parser.add_argument("--domain", default="general", help="Query domain")
    args = parser.parse_args()
    
    result = asyncio.run(process_query(args.query, args.domain))
    sys.exit(0 if result else 1)