#!/usr/bin/env python3
"""
Response Generation - Stage 9 of README Data Flow
Context → Azure OpenAI Response Generation → Final Answer with Citations

This script implements the final stage of the query phase:
- Uses existing QueryService which already handles response generation
- Takes user query and generates comprehensive response using existing service
- Leverages QueryService's built-in response generation with proper citations
- Returns final structured response for user consumption
"""

import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService
from services.query_service import QueryService

logger = logging.getLogger(__name__)

class ResponseGenerationStage:
    """Stage 9: Context → Azure OpenAI Response Generation → Final Answer with Citations (using existing QueryService)"""
    
    def __init__(self):
        self.infrastructure = InfrastructureService()
        self.query_service = QueryService()
        
    async def execute(
        self, 
        query: str,
        domain: str = "maintenance",
        max_results: int = 15
    ) -> Dict[str, Any]:
        """
        Execute response generation stage using existing QueryService
        
        Args:
            query: User's natural language query
            domain: Target domain for response generation
            max_results: Maximum number of results to use for context
            
        Returns:
            Dict with final response, citations, and metadata
        """
        print("💬 Stage 9: Response Generation - Context → Azure OpenAI → Final Answer")
        print("=" * 70)
        
        start_time = asyncio.get_event_loop().time()
        
        results = {
            "stage": "09_response_generation",
            "query": query,
            "domain": domain,
            "response_data": {},
            "final_answer": "",
            "citations": [],
            "metadata": {},
            "success": False
        }
        
        try:
            # Validate query
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            print(f"💬 Generating response for query: \"{query[:100]}{'...' if len(query) > 100 else ''}\"")
            print("   🎯 **AZURE UNIVERSAL RAG RESPONSE GENERATION** - Final stage of the Universal RAG pipeline")
            
            # Show model configuration for transparency
            from config.domain_patterns import DomainPatternManager
            prompts = DomainPatternManager.get_prompts(domain)
            print(f"   🤖 **RESPONSE MODEL**: {prompts.model_name} (temp: {prompts.temperature}, max_tokens: {prompts.max_tokens})")
            
            print("🚀 Step 1: Universal Query Processing for Response Generation...")
            
            # Use QueryService's process_universal_query for complete response generation
            # This method handles search, context retrieval, and response generation
            universal_result = await self.query_service.process_universal_query(
                query=query, 
                domain=domain, 
                max_results=max_results
            )
            
            if not universal_result.get("success"):
                raise Exception(f"Response generation failed: {universal_result.get('error')}")
            
            # Extract comprehensive response data
            response_data = universal_result.get("data", {})
            context_sources = response_data.get("context_sources", 0)
            processing_time = response_data.get("processing_time", 0.0)
            
            print(f"   ✅ **UNIVERSAL QUERY COMPLETED**: {context_sources} context sources processed")
            print(f"   ⚡ **PROCESSING TIME**: {processing_time:.2f}s")
            
            print("📊 Step 2: Detailed Semantic Search for Citation Assembly...")
            
            # Get additional detailed results for citation tracking
            semantic_result = await self.query_service.semantic_search(
                query=query,
                search_type="hybrid",
                filters={"domain": domain}
            )
            
            # Prepare final response structure
            final_answer = response_data.get("response", "")
            
            # Create citations from semantic search results
            citations = []
            if semantic_result.get("success"):
                semantic_data = semantic_result.get("data", {})
                detailed_results = semantic_data.get("results", {})
                
                # Show detailed search results
                doc_count = len(detailed_results.get("documents", []))
                graph_count = len(detailed_results.get("graph", []))
                entity_count = len(detailed_results.get("entities", []))
                
                print(f"   ✅ **SEMANTIC SEARCH COMPLETED**: Multi-source citation sources assembled")
                print(f"   📄 **DOCUMENT SOURCES**: {doc_count} documents for citation generation")
                print(f"   🕸️  **GRAPH SOURCES**: {graph_count} knowledge graph entities")
                print(f"   🧠 **ENTITY SOURCES**: {entity_count} related entities")
                
                print("🔗 Step 3: Citation Generation and Response Assembly...")
                
                # Document citations
                documents = detailed_results.get("documents", [])
                for i, doc in enumerate(documents[:5]):  # Limit to top 5 documents
                    citations.append({
                        "citation_id": f"[{i+1}]",
                        "source_type": "document",
                        "title": doc.get("title", f"Document {i+1}"),
                        "content_preview": doc.get("content", "")[:300] + "..." if len(doc.get("content", "")) > 300 else doc.get("content", ""),
                        "relevance_score": doc.get("score", 0.0),
                        "metadata": doc.get("metadata", {})
                    })
                
                # Knowledge graph citations
                graph_entities = detailed_results.get("graph", [])
                for i, entity in enumerate(graph_entities[:3]):  # Limit to top 3 entities
                    # Handle both string and dict entities
                    if isinstance(entity, str):
                        entity_text = entity
                        entity_type = "entity"
                        entity_metadata = {"name": entity}
                    else:
                        entity_text = entity.get("text", entity.get("name", str(entity)))
                        entity_type = entity.get("type", "entity")
                        entity_metadata = entity
                    
                    citations.append({
                        "citation_id": f"[KG{i+1}]",
                        "source_type": "knowledge_graph",
                        "entity_type": entity_type,
                        "content_preview": entity_text[:300] + "..." if len(entity_text) > 300 else entity_text,
                        "relevance_score": 0.8,
                        "metadata": entity_metadata
                    })
                
                # Entity search citations
                entities = detailed_results.get("entities", [])
                for i, entity in enumerate(entities[:3]):  # Limit to top 3 entities
                    # Handle both string and dict entities
                    if isinstance(entity, str):
                        entity_text = entity
                        entity_type = "entity"
                        entity_metadata = {"name": entity}
                    else:
                        entity_text = entity.get("text", entity.get("name", str(entity)))
                        entity_type = entity.get("entity_type", entity.get("type", "entity"))
                        entity_metadata = entity
                    
                    citations.append({
                        "citation_id": f"[E{i+1}]",
                        "source_type": "entity_search",
                        "entity_type": entity_type,
                        "content_preview": entity_text[:300] + "..." if len(entity_text) > 300 else entity_text,
                        "relevance_score": 0.7,
                        "metadata": entity_metadata
                    })
            
            # Prepare metadata
            metadata = {
                "query": query,
                "domain": domain,
                "context_sources": response_data.get("context_sources", 0),
                "processing_time": response_data.get("processing_time", 0.0),
                "query_analysis": response_data.get("query_analysis", {}),
                "timestamp": response_data.get("timestamp", ""),
                "total_citations": len(citations),
                "response_length": len(final_answer),
                "generation_method": "azure_openai_rag"
            }
            
            # Update results
            results.update({
                "response_data": response_data,
                "final_answer": final_answer,
                "citations": citations,
                "metadata": metadata,
                "total_citations": len(citations),
                "response_length": len(final_answer)
            })
            
            # Success
            duration = asyncio.get_event_loop().time() - start_time
            results["duration_seconds"] = round(duration, 2)
            results["success"] = True
            
            print(f"")
            print(f"   🎯 **AZURE UNIVERSAL RAG RESPONSE SUMMARY**:")
            print(f"   📊 Total citations generated: {len(citations)} from {doc_count + graph_count + entity_count} sources")
            print(f"   ⚡ Response generation: {processing_time:.2f}s")
            print(f"   🎯 **FINAL ANSWER DELIVERY**: Complete response with full citation tracking")
            
            print(f"✅ Stage 9 Complete:")
            print(f"   💬 Query: {query[:50]}{'...' if len(query) > 50 else ''}")
            print(f"   🎯 Domain: {domain}")
            print(f"   📝 Response length: {results['response_length']} characters")
            print(f"   📊 Context sources: {metadata['context_sources']}")
            print(f"   📖 Total citations: {results['total_citations']}")
            print(f"   📄 Document citations: {len([c for c in citations if c['source_type'] == 'document'])}")
            print(f"   🕸️  Graph citations: {len([c for c in citations if c['source_type'] == 'knowledge_graph'])}")
            print(f"   🔗 Entity citations: {len([c for c in citations if c['source_type'] == 'entity_search'])}")
            print(f"   ⏱️  Duration: {results['duration_seconds']}s")
            print(f"\n📝 Final Answer Preview:")
            print(f"   {final_answer[:200]}{'...' if len(final_answer) > 200 else ''}")
            
            return results
            
        except Exception as e:
            results["error"] = str(e)
            results["duration_seconds"] = round(asyncio.get_event_loop().time() - start_time, 2)
            print(f"❌ Stage 9 Failed: {e}")
            logger.error(f"Response generation failed: {e}", exc_info=True)
            return results


async def main():
    """Main entry point for response generation stage"""
    parser = argparse.ArgumentParser(
        description="Stage 9: Response Generation - Context → Azure OpenAI → Final Answer"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User's natural language query"
    )
    parser.add_argument(
        "--domain", 
        default="maintenance",
        help="Target domain for response generation"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=15,
        help="Maximum number of results to use for context"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Execute stage
    stage = ResponseGenerationStage()
    results = await stage.execute(
        query=args.query,
        domain=args.domain,
        max_results=args.max_results
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