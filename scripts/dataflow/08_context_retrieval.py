#!/usr/bin/env python3
"""
Context Retrieval - Stage 8 of README Data Flow
Search Results → Context Preparation

This script implements the context preparation stage of the README architecture:
- Uses existing QueryService which already handles context consolidation
- Takes unified search results and prepares structured context
- Leverages QueryService's built-in context preparation and ranking
- Prepares structured context for response generation stage
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService
from services.query_service import QueryService

logger = logging.getLogger(__name__)

class ContextRetrievalStage:
    """Stage 8: Search Results → Context Preparation (using existing QueryService)"""

    def __init__(self):
        self.infrastructure = InfrastructureService()
        self.query_service = QueryService()

    async def execute(
        self,
        query: str,
        domain: str = "maintenance",
        max_context_items: int = 15
    ) -> Dict[str, Any]:
        """
        Execute context retrieval stage using existing QueryService

        Args:
            query: User's natural language query
            domain: Target domain for context retrieval
            max_context_items: Maximum number of context items to include

        Returns:
            Dict with structured context and citations for response generation
        """
        print("📋 Stage 8: Context Retrieval - Search Results → Context Preparation")
        print("=" * 65)

        start_time = asyncio.get_event_loop().time()

        results = {
            "stage": "08_context_retrieval",
            "query": query,
            "domain": domain,
            "max_context_items": max_context_items,
            "context": {},
            "citations": [],
            "success": False
        }

        try:
            # Validate query
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")

            print(f"📋 Retrieving context for query: \"{query[:100]}{'...' if len(query) > 100 else ''}\"")
            print("   🎯 **AZURE CONTEXT RETRIEVAL SYSTEM** - Preparing structured context for response generation")

            # Show model configuration for transparency
            from config.async_pattern_manager import get_pattern_manager

from config.discovery_infrastructure_naming import get_discovery_naming
from config.dynamic_ml_config import get_dynamic_ml_config

            prompts = await self._get_prompts_async(domain)
            print(f"   🤖 **CONTEXT MODEL**: {prompts.model_name} (temp: {prompts.temperature}, max_tokens: {prompts.max_tokens})")

            print("🔍 Step 1: Universal Query Processing for Context...")

            # Use QueryService's process_universal_query which includes context preparation
            # This method internally handles search, context consolidation, and preparation
            universal_result = await self.query_service.process_universal_query(
                query=query,
                domain=domain,
                max_results=max_context_items
            )

            if not universal_result.get("success"):
                raise Exception(f"Context retrieval failed: {universal_result.get('error')}")

            # Extract context information from the universal query result
            search_data = universal_result.get("data", {})
            context_sources = search_data.get("context_sources", 0)
            processing_time = search_data.get("processing_time", 0.0)

            print(f"   ✅ **UNIVERSAL QUERY COMPLETED**: {context_sources} context sources discovered")
            print(f"   ⚡ **PROCESSING TIME**: {processing_time:.2f}s")

            print("📊 Step 2: Detailed Semantic Search for Context Items...")

            # Get detailed semantic search for more granular context items
            semantic_result = await self.query_service.semantic_search(
                query=query,
                search_type="hybrid",
                filters={"domain": domain}
            )

            # Prepare structured context
            context = {
                "query": query,
                "domain": domain,
                "response": search_data.get("response", ""),
                "context_sources": search_data.get("context_sources", 0),
                "processing_time": search_data.get("processing_time", 0.0),
                "items": []
            }

            citations = []

            if semantic_result.get("success"):
                semantic_data = semantic_result.get("data", {})
                detailed_results = semantic_data.get("results", {})

                # Show detailed search results
                doc_count = len(detailed_results.get("documents", []))
                graph_count = len(detailed_results.get("graph", []))
                entity_count = len(detailed_results.get("entities", []))

                print(f"   ✅ **SEMANTIC SEARCH COMPLETED**: Multi-source context assembled")
                print(f"   📄 **DOCUMENT CONTEXT**: {doc_count} documents for context preparation")
                print(f"   🕸️  **GRAPH CONTEXT**: {graph_count} knowledge graph entities")
                print(f"   🧠 **ENTITY CONTEXT**: {entity_count} related entities")

                print("🔗 Step 3: Context Item Assembly and Citation Generation...")

                # Process documents for context
                documents = detailed_results.get("documents", [])
                for i, doc in enumerate(documents[:max_context_items//3]):
                    context_item = {
                        "item_id": f"doc_{i+1}",
                        "source_type": "document",
                        "content": doc.get("content", ""),
                        "title": doc.get("title", f"Document {i+1}"),
                        "relevance_score": doc.get("score", 0.0),
                        "rank": i+1
                    }
                    context["items"].append(context_item)

                    # Create citation
                    citations.append({
                        "citation_id": f"[{i+1}]",
                        "source_type": "document",
                        "title": doc.get("title", f"Document {i+1}"),
                        "content_preview": doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", ""),
                        "metadata": doc.get("metadata", {})
                    })

                # Process graph entities for context
                graph_entities = detailed_results.get("graph", [])
                for i, entity in enumerate(graph_entities[:max_context_items//3]):
                    # Handle both string and dict entities
                    if isinstance(entity, str):
                        entity_text = entity
                        entity_type = "entity"
                        entity_metadata = {"name": entity}
                    else:
                        entity_text = entity.get("text", entity.get("name", str(entity)))
                        entity_type = entity.get("type", "entity")
                        entity_metadata = entity

                    context_item = {
                        "item_id": f"graph_{i+1}",
                        "source_type": "knowledge_graph",
                        "content": entity_text,
                        "entity_type": entity_type,
                        "relevance_score": 0.8,  # Default graph relevance
                        "rank": len(context["items"]) + 1
                    }
                    context["items"].append(context_item)

                    # Create citation
                    content_preview = entity_text[:200] + "..." if len(entity_text) > 200 else entity_text
                    citations.append({
                        "citation_id": f"[KG{i+1}]",
                        "source_type": "knowledge_graph",
                        "entity_type": entity_type,
                        "content_preview": content_preview,
                        "metadata": entity_metadata
                    })

                # Process related entities for context
                entities = detailed_results.get("entities", [])
                for i, entity in enumerate(entities[:max_context_items//3]):
                    # Handle both string and dict entities
                    if isinstance(entity, str):
                        entity_text = entity
                        entity_type = "entity"
                        entity_metadata = {"name": entity}
                    else:
                        entity_text = entity.get("text", entity.get("name", str(entity)))
                        entity_type = entity.get("entity_type", entity.get("type", "entity"))
                        entity_metadata = entity

                    context_item = {
                        "item_id": f"entity_{i+1}",
                        "source_type": "entity_search",
                        "content": entity_text,
                        "entity_type": entity_type,
                        "relevance_score": 0.7,  # Default entity relevance
                        "rank": len(context["items"]) + 1
                    }
                    context["items"].append(context_item)

                    # Create citation
                    content_preview = entity_text[:200] + "..." if len(entity_text) > 200 else entity_text
                    citations.append({
                        "citation_id": f"[E{i+1}]",
                        "source_type": "entity_search",
                        "entity_type": entity_type,
                        "content_preview": content_preview,
                        "metadata": entity_metadata
                    })

            # Update results
            results["context"] = context
            results["citations"] = citations
            results["total_context_items"] = len(context["items"])
            results["total_citations"] = len(citations)

            # Success
            duration = asyncio.get_event_loop().time() - start_time
            results["duration_seconds"] = round(duration, 2)
            results["success"] = True

            print(f"✅ Stage 8 Complete:")
            print(f"   📋 Query: {query[:50]}{'...' if len(query) > 50 else ''}")
            print(f"   🎯 Domain: {domain}")
            print(f"   📋 Total context items: {results['total_context_items']}")
            print(f"   📄 Document items: {len([item for item in context['items'] if item['source_type'] == 'document'])}")
            print(f"   🕸️  Graph items: {len([item for item in context['items'] if item['source_type'] == 'knowledge_graph'])}")
            print(f"   🔗 Entity items: {len([item for item in context['items'] if item['source_type'] == 'entity_search'])}")
            print(f"   📖 Total citations: {results['total_citations']}")
            print(f"   ⏱️  Duration: {results['duration_seconds']}s")

            return results

        except Exception as e:
            results["error"] = str(e)
            results["duration_seconds"] = round(asyncio.get_event_loop().time() - start_time, 2)
            print(f"❌ Stage 8 Failed: {e}")
            logger.error(f"Context retrieval failed: {e}", exc_info=True)
            return results


async def main():
    """Main entry point for context retrieval stage"""
    parser = argparse.ArgumentParser(
        description="Stage 8: Context Retrieval - Search Results → Context Preparation"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User's natural language query"
    )
    parser.add_argument(
        "--domain",
        default="maintenance",
        help="Target domain for context retrieval"
    )
    parser.add_argument(
        "--max-context-items",
        type=int,
        default=15,
        help="Maximum number of context items to include"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Execute stage
    stage = ContextRetrievalStage()
    results = await stage.execute(
        query=args.query,
        domain=args.domain,
        max_context_items=args.max_context_items
    )

    # Save results if requested
    if args.output and results.get("success"):
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {args.output}")

    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
