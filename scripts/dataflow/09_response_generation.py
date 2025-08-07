#!/usr/bin/env python3
"""
Simple Response Generation - CODING_STANDARDS Compliant
Clean response generation script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.universal_search.agent import UniversalSearchAgent
from agents.core.azure_service_container import ConsolidatedAzureServices


async def generate_response(query: str, context: str = ""):
    """Simple response generation"""
    print(f"💭 Response Generation: '{query}'")

    try:
        # Initialize services
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()

        # Get Azure OpenAI client
        openai_client = azure_services.openai_client

        # Simple response prompt
        prompt = f"""Answer this question: {query}

Context: {context}

Provide a clear, helpful answer based on the available context."""

        # Generate response
        result = await openai_client.get_completion(prompt, max_tokens=500)

        if result.get("success"):
            response = result.get("content", "")
            print(f"✅ Generated response ({len(response)} characters)")
            return {"response": response, "query": query}
        else:
            print(f"❌ Response generation failed: {result.get('error')}")
            return None

    except Exception as e:
        print(f"❌ Response generation failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple response generation")
    parser.add_argument("query", help="Query to generate response for")
    parser.add_argument("--context", default="", help="Context for response")
    args = parser.parse_args()

    result = asyncio.run(generate_response(args.query, args.context))
    sys.exit(0 if result else 1)
