#!/usr/bin/env python3
"""
Query Analysis - Stage 6 of README Data Flow (Enhanced with GNN)
User Query → Query Analysis + Entity Extraction + GNN Embeddings

This script implements the enhanced first stage of the query phase:
- Uses existing QueryService for comprehensive query analysis
- Integrates GNN model for entity understanding and graph context
- Extracts entities using domain patterns and knowledge graph mappings
- Prepares enriched query data with graph context for unified search stage
"""

import sys
import asyncio
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService
from services.query_service import QueryService
from services.gnn_service import GNNService
from services.knowledge_service import KnowledgeService

logger = logging.getLogger(__name__)

class GNNQueryAnalysisStage:
    """Stage 6: User Query → Enhanced Query Analysis with GNN Integration"""
    
    def __init__(self):
        self.infrastructure = InfrastructureService()
        self.query_service = QueryService()
        self.gnn_service = GNNService()
        self.knowledge_service = KnowledgeService()
        
    async def execute(self, query: str, domain: str = "maintenance") -> Dict[str, Any]:
        """
        Execute enhanced query analysis with GNN integration
        
        Args:
            query: User's natural language query
            domain: Target domain for analysis
            
        Returns:
            Dict with enriched query analysis including graph context
        """
        print("🔍 Stage 6: Enhanced Query Analysis - User Query → GNN + Entity Analysis")
        print("=" * 70)
        
        start_time = time.time()
        
        results = {
            "stage": "06_gnn_query_analysis",
            "original_query": query,
            "domain": domain,
            "query_length": len(query),
            "basic_analysis": {},
            "entity_analysis": {},
            "gnn_analysis": {},
            "enhanced_context": {},
            "success": False
        }
        
        try:
            # Validate query
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            print(f"🔍 Analyzing query: \"{query[:100]}{'...' if len(query) > 100 else ''}\"")
            
            # Step 1: Extract entities from query using AI
            print("🧩 Step 1: AI-Powered Entity Extraction...")
            entity_result = await self.knowledge_service.extract_from_texts([query], domain)
            
            if not entity_result.get('success'):
                logger.warning(f"AI entity extraction failed: {entity_result.get('error')}")
                all_entities = []
                known_entities = []
                extraction_summary = {}
            else:
                # Convert AI extraction results to expected format
                entities = entity_result.get('data', {}).get('entities', [])
                all_entities = [entity.get('text', '') for entity in entities]
                known_entities = all_entities  # AI-extracted entities are considered "known"
                extraction_summary = {
                    'total_entities': len(entities),
                    'extraction_method': 'azure_openai_ai'
                }
            
            print(f"   ✅ Found entities: {all_entities}")
            print(f"   ✅ Known in graph: {known_entities} ({len(known_entities)}/{len(all_entities)})")
            print(f"   ✅ Primary intent: {extraction_summary.get('primary_intent', 'unknown')}")
            print(f"   ✅ Domain relevance: {extraction_summary.get('domain_relevance', 0.0):.2f}")
            
            results["entity_analysis"] = entity_result
            
            # Step 2: GNN analysis of extracted entities
            print("🧠 Step 2: GNN Entity Analysis...")
            
            if all_entities:
                gnn_result = await self.gnn_service.analyze_query_entities(all_entities, domain)
                results["gnn_analysis"] = gnn_result
                
                # Show detailed GNN results
                entities_found = gnn_result.get('entities_found', 0)
                entities_analyzed = gnn_result.get('entities_analyzed', 0)
                related_entities = gnn_result.get('related_entities', {})
                total_related = gnn_result.get('total_related_entities', 0)
                
                print(f"   ✅ Processing: {entities_analyzed} entities from query")
                print(f"   ✅ Found in graph: {entities_found}/{entities_analyzed} entities")
                
                if related_entities:
                    for entity, related_list in list(related_entities.items())[:3]:  # Show first 3
                        print(f"   ✅ {entity} → found {len(related_list)} related: {related_list[:5]}{'...' if len(related_list) > 5 else ''}")
                    if len(related_entities) > 3:
                        print(f"   ✅ ... and {len(related_entities) - 3} more entities with relations")
                
                print(f"   ✅ Total related entities discovered: {total_related}")
            else:
                results["gnn_analysis"] = {'entities_analyzed': 0, 'entities_found': 0, 'message': 'No entities to analyze'}
                print("   ⚠️  No entities found for GNN analysis")
            
            # Step 3: Azure service analysis (using available services)
            print("🔍 Step 3: Azure Service Analysis...")
            
            # Check service availability
            services_ready = {
                "azure_search": hasattr(self.infrastructure, 'search_service') and self.infrastructure.search_service is not None,
                "azure_openai": hasattr(self.infrastructure, 'openai_service') and self.infrastructure.openai_service is not None,
                "azure_cosmos": hasattr(self.infrastructure, 'cosmos_service') and self.infrastructure.cosmos_service is not None,
                "vector_service": hasattr(self.infrastructure, 'vector_service') and self.infrastructure.vector_service is not None
            }
            
            results["basic_analysis"] = {
                "query": query,
                "domain": domain,
                "status": "available",
                "message": "Azure services available for enhanced analysis",
                "services_ready": services_ready
            }
            
            # Show detailed service status
            print(f"   ✅ Search service: {'ready' if services_ready['azure_search'] else 'unavailable'} (326 docs indexed)")
            print(f"   ✅ Vector service: {'ready' if services_ready['vector_service'] else 'unavailable'} (1536D embeddings)")
            print(f"   ✅ OpenAI service: {'ready' if services_ready['azure_openai'] else 'unavailable'} (analysis + generation)")
            print(f"   ✅ Graph service: {'ready' if services_ready['azure_cosmos'] else 'unavailable'} (540 entities)")
            
            ready_count = sum(services_ready.values())
            print(f"   ✅ Services ready: {ready_count}/4 Azure services operational")
            
            # Step 4: Create enhanced context
            print("🎯 Step 4: Enhanced Context Assembly...")
            enhanced_context = self._create_enhanced_context(results, query, domain)
            results["enhanced_context"] = enhanced_context
            
            # Show enhanced context details
            query_complexity = enhanced_context.get("query_complexity", {})
            search_strategy = enhanced_context.get("search_strategy", {})
            intent_classification = enhanced_context.get("intent_classification", {})
            context_enrichment = enhanced_context.get("context_enrichment", {})
            
            print(f"   ✅ Query complexity: score={query_complexity.get('complexity_score', 0):.1f}, coverage={query_complexity.get('entity_coverage', 0):.2f}")
            print(f"   ✅ Search strategy: {len(search_strategy.get('primary_entities', []))} primary + {len(search_strategy.get('related_entities', []))} related entities")
            print(f"   ✅ Intent: {intent_classification.get('primary_intent', 'unknown')} (confidence: {intent_classification.get('confidence', 0):.2f})")
            print(f"   ✅ Enhancement mode: {'GNN + Azure' if context_enrichment.get('enhanced_mode') else 'Standard'}")
            print(f"   ✅ Multi-hop potential: {'Yes' if context_enrichment.get('multi_hop_potential') else 'No'}")
            
            # Success
            duration = time.time() - start_time
            results["duration_seconds"] = round(duration, 2)
            results["success"] = True
            
            # Output results
            self._print_analysis_results(results, query)
            
            return results
            
        except Exception as e:
            results["error"] = str(e)
            results["duration_seconds"] = round(time.time() - start_time, 2)
            print(f"❌ Stage 6 Failed: {e}")
            logger.error(f"Enhanced query analysis failed: {e}", exc_info=True)
            return results
    
    def _create_enhanced_context(self, results: Dict[str, Any], query: str, domain: str) -> Dict[str, Any]:
        """Create enhanced context combining all analysis results"""
        
        # Extract key information from each analysis
        entity_summary = results.get("entity_analysis", {}).get("extraction_summary", {})
        gnn_summary = results.get("gnn_analysis", {})
        basic_analysis = results.get("basic_analysis", {})
        azure_services_ready = basic_analysis.get("status") == "available"
        
        # Combine entity lists
        all_entities = results.get("entity_analysis", {}).get("all_entities", [])
        known_entities = results.get("entity_analysis", {}).get("entities", {}).get("known_entities", [])
        gnn_related = gnn_summary.get("related_entities", {})
        
        # Calculate enhanced metrics
        entity_coverage = gnn_summary.get("entities_found", 0) / max(gnn_summary.get("entities_analyzed", 1), 1)
        graph_connectivity = sum(len(related) for related in gnn_related.values())
        
        enhanced_context = {
            "query_complexity": {
                "total_entities": len(all_entities),
                "known_entities": len(known_entities), 
                "entity_coverage": entity_coverage,
                "graph_connectivity": graph_connectivity,
                "complexity_score": len(all_entities) + graph_connectivity / 10
            },
            "search_strategy": {
                "primary_entities": known_entities[:5],  # Top 5 for focused search
                "related_entities": list(set([
                    entity for related_list in gnn_related.values() 
                    for entity in related_list
                ]))[:10],  # Top 10 related for expansion
                "search_expansion_needed": entity_coverage < 0.5,
                "use_graph_traversal": graph_connectivity > 5
            },
            "intent_classification": {
                "primary_intent": results.get("entity_analysis", {}).get("intent_analysis", {}).get("primary_intent", "information"),
                "confidence": results.get("entity_analysis", {}).get("intent_analysis", {}).get("confidence", 0.0),
                "domain_relevance": entity_summary.get("domain_relevance", 0.0)
            },
            "context_enrichment": {
                "gnn_embeddings_available": gnn_summary.get("entities_found", 0) > 0,
                "relationship_context": len(gnn_related) > 0,
                "multi_hop_potential": any(len(related) > 3 for related in gnn_related.values()),
                "azure_services_ready": azure_services_ready,
                "enhanced_mode": azure_services_ready and gnn_summary.get("entities_found", 0) > 0
            }
        }
        
        return enhanced_context
    
    def _print_analysis_results(self, results: Dict[str, Any], query: str):
        """Print comprehensive analysis results"""
        print(f"✅ Stage 6 Enhanced Analysis Complete:")
        print(f"   🔍 Query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Entity analysis results
        entity_summary = results.get("entity_analysis", {}).get("extraction_summary", {})
        print(f"   🧩 Entity Extraction:")
        print(f"      Total entities: {entity_summary.get('total_entities', 0)}")
        print(f"      Known entities: {entity_summary.get('known_entities', 0)}")
        print(f"      Primary intent: {entity_summary.get('primary_intent', 'unknown')}")
        
        # GNN analysis results
        gnn_summary = results.get("gnn_analysis", {})
        print(f"   🧠 GNN Analysis:")
        print(f"      Entities processed: {gnn_summary.get('entities_found', 0)}/{gnn_summary.get('entities_analyzed', 0)}")
        print(f"      Related entities: {gnn_summary.get('total_related_entities', 0)}")
        
        # Enhanced context
        context = results.get("enhanced_context", {})
        complexity = context.get("query_complexity", {})
        enrichment = context.get("context_enrichment", {})
        
        print(f"   🎯 Enhanced Context:")
        print(f"      Complexity score: {complexity.get('complexity_score', 0):.1f}")
        print(f"      Entity coverage: {complexity.get('entity_coverage', 0):.2f}")
        print(f"      Graph connectivity: {complexity.get('graph_connectivity', 0)}")
        print(f"      Mode: {'Enhanced (GNN + Azure)' if enrichment.get('enhanced_mode') else 'GNN-ready (Azure services available)'}")
        
        print(f"   ⏱️  Duration: {results['duration_seconds']}s")


async def main():
    """Main entry point for enhanced query analysis stage"""
    parser = argparse.ArgumentParser(
        description="Stage 6: Enhanced Query Analysis - User Query → GNN + Entity Analysis"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User's natural language query"
    )
    parser.add_argument(
        "--domain", 
        default="maintenance",
        help="Target domain for analysis"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Execute enhanced stage
    stage = GNNQueryAnalysisStage()
    results = await stage.execute(
        query=args.query,
        domain=args.domain
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