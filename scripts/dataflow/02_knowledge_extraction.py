#!/usr/bin/env python3
"""
Simple Knowledge Extraction - CODING_STANDARDS Compliant
Clean knowledge extraction script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.knowledge_extraction.agent import KnowledgeExtractionAgent
from agents.core.azure_service_container import ConsolidatedAzureServices


async def extract_knowledge(text: str):
    """Simple knowledge extraction"""
    print("🧠 Knowledge Extraction")
    
    try:
        # Initialize services
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()
        
        # Initialize extraction agent
        extraction_agent = KnowledgeExtractionAgent(azure_services)
        
        print(f"📄 Processing {len(text)} characters of text")
        
        # Extract knowledge
        result = await extraction_agent.process_document({
            "content": text,
            "domain": "general"
        })
        
        if result.get("success"):
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])
            
            print(f"✅ Extracted {len(entities)} entities, {len(relationships)} relationships")
            return {"entities": entities, "relationships": relationships}
        else:
            print(f"❌ Extraction failed: {result.get('error')}")
            return None
        
    except Exception as e:
        print(f"❌ Knowledge extraction failed: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple knowledge extraction")
    parser.add_argument("--text", required=True, help="Text to extract knowledge from")
    args = parser.parse_args()
    
    result = asyncio.run(extract_knowledge(args.text))
    sys.exit(0 if result else 1)