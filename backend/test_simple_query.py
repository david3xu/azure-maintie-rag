#!/usr/bin/env python3
"""
Simple test to isolate the serialization issue
"""

import asyncio
import sys
import os
import json

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestration.enhanced_rag_universal import get_enhanced_rag_instance

async def test_simple_query():
    """Test the query processing with simple serialization"""
    try:
        print("Testing Enhanced Universal RAG query processing...")

        # Get the enhanced RAG instance
        print("Getting enhanced RAG instance...")
        enhanced_rag = get_enhanced_rag_instance("general")
        print(f"Enhanced RAG instance: {enhanced_rag}")
        print(f"Components initialized: {enhanced_rag.components_initialized}")

        # Test initialization if needed
        if not enhanced_rag.components_initialized:
            print("Initializing components...")
            init_results = await enhanced_rag.initialize_components()
            print(f"Initialization results: {init_results}")

            if not init_results.get("success", False):
                print(f"Initialization failed: {init_results.get('error', 'Unknown error')}")
                return

        # Test query processing
        print("Processing test query...")
        results = await enhanced_rag.process_query(
            query="what are common issues?",
            max_results=10,
            include_explanations=True,
            enable_safety_warnings=True
        )

        print(f"Query results type: {type(results)}")
        print(f"Query results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")

        if results.get("success", False):
            print("✅ Query processing successful!")

            # Debug the generated_response
            print(f"Generated response type: {type(results['generated_response'])}")
            print(f"Generated response: {results['generated_response']}")

            # Test serialization
            print("Testing serialization...")
            try:
                # Test the exact response format from the API
                response = {
                    "success": True,
                    "query": results["query"],
                    "domain": results["domain"],
                    "generated_response": results["generated_response"].to_dict() if hasattr(results["generated_response"], 'to_dict') else results["generated_response"],
                    "search_results": [
                        result.to_dict() if hasattr(result, 'to_dict') else
                        result if isinstance(result, dict) else
                        str(result)
                        for result in results["search_results"]
                    ],
                    "processing_time": results["processing_time"],
                    "system_stats": results["system_stats"],
                    "timestamp": results["timestamp"]
                }

                # Try to serialize
                json_str = json.dumps(response, indent=2)
                print("✅ Serialization successful!")
                print(f"Response length: {len(json_str)} characters")

            except Exception as e:
                print(f"❌ Serialization failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ Query processing failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_query())