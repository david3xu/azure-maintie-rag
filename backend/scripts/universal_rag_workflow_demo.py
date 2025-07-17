#!/usr/bin/env python3
"""
Universal RAG Workflow Demonstration Script
Shows transparent step-by-step processing from user query to final answer
Based on real codebase - no assumptions, no hardcoded values, pure data-driven
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Import actual components from the codebase
from core.orchestration.universal_rag_orchestrator_complete import (
    UniversalRAGOrchestrator, create_universal_rag_from_texts
)
from core.extraction.universal_knowledge_extractor import UniversalKnowledgeExtractor
from core.knowledge.universal_text_processor import UniversalTextProcessor
from core.enhancement.universal_query_analyzer import UniversalQueryAnalyzer
from core.retrieval.universal_vector_search import UniversalVectorSearch
from core.generation.universal_llm_interface import UniversalLLMInterface
from core.models.universal_models import UniversalEntity, UniversalRelation, UniversalDocument
from core.workflow.universal_workflow_manager import create_workflow_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalRAGWorkflowDemo:
    """
    Demonstrates the complete Universal RAG workflow with transparency
    Shows actual output at each step to build user trust
    """

    def __init__(self, domain: str = "maintenance"):
        self.domain = domain
        self.demo_texts = [
            "Pump impeller damage can cause vibration and reduced flow rate. Regular inspection prevents costly repairs.",
            "Bearing lubrication must be checked monthly. Proper oil levels ensure smooth operation and extend equipment life.",
            "Motor overheating may indicate blocked ventilation or electrical issues. Check cooling systems and connections.",
            "Valve seals deteriorate over time, causing leaks. Replace seals when pressure drops are observed.",
            "Pressure gauge calibration is critical for accurate readings. Calibrate instruments every six months."
        ]

        self.results = {}
        self.start_time = None

    def print_step_header(self, step_num: int, title: str, description: str):
        """Print formatted step header"""
        print(f"\n{'='*80}")
        print(f"🔹 STEP {step_num}: {title}")
        print(f"📋 {description}")
        print(f"{'='*80}")

    def print_step_output(self, data: Dict[str, Any], max_items: int = 5):
        """Print step output in readable format"""
        print(f"📊 STEP OUTPUT:")
        print(f"⏱️  Processing Time: {data.get('processing_time', 'N/A')} seconds")
        print(f"✅ Success: {data.get('success', 'N/A')}")

        if 'error' in data:
            print(f"❌ Error: {data['error']}")
            return

        # Show key metrics
        if 'total_entities' in data:
            print(f"🏷️  Entities Extracted: {data['total_entities']}")
        if 'total_relations' in data:
            print(f"🔗 Relations Extracted: {data['total_relations']}")
        if 'total_documents' in data:
            print(f"📄 Documents Processed: {data['total_documents']}")
        if 'index_built' in data:
            print(f"🔧 Index Built: {data['index_built']}")
        if 'search_results' in data:
            print(f"🔍 Search Results: {len(data['search_results'])}")

        # Show discovered types (dynamic, not hardcoded)
        if 'discovered_types' in data:
            types = data['discovered_types']
            if 'entity_types' in types:
                print(f"🏷️  Entity Types Discovered: {types['entity_types'][:max_items]}")
            if 'relation_types' in types:
                print(f"🔗 Relation Types Discovered: {types['relation_types'][:max_items]}")

        # Show actual data samples
        if 'entities' in data and isinstance(data['entities'], dict):
            entity_samples = list(data['entities'].items())[:max_items]
            print(f"📋 Entity Samples:")
            for entity_id, entity_data in entity_samples:
                if isinstance(entity_data, dict):
                    print(f"   - {entity_id}: {entity_data.get('text', 'N/A')} [{entity_data.get('type', 'unknown')}]")

        if 'relations' in data and isinstance(data['relations'], list):
            relation_samples = data['relations'][:max_items]
            print(f"📋 Relation Samples:")
            for relation in relation_samples:
                if isinstance(relation, dict):
                    print(f"   - {relation.get('subject', 'N/A')} → {relation.get('predicate', 'N/A')} → {relation.get('object', 'N/A')}")

        if 'search_results' in data and isinstance(data['search_results'], list):
            print(f"📋 Search Result Samples:")
            for i, result in enumerate(data['search_results'][:max_items]):
                if hasattr(result, 'score') and hasattr(result, 'doc_id'):
                    print(f"   - {i+1}. {result.doc_id}: Score {result.score:.3f}")
                    print(f"      Content: {result.content[:100]}...")

    async def demonstrate_complete_workflow(self, user_query: str = "How do I fix pump vibration issues?"):
        """Demonstrate complete workflow from user query to final answer"""

        print(f"🚀 UNIVERSAL RAG WORKFLOW DEMONSTRATION")
        print(f"🎯 User Query: '{user_query}'")
        print(f"🏷️  Domain: {self.domain}")
        print(f"📅 Started: {datetime.now().isoformat()}")

        self.start_time = time.time()

        # Step 1: Data Ingestion (Text Processing)
        await self._step_1_data_ingestion()

        # Step 2: Knowledge Extraction (LLM Processing)
        await self._step_2_knowledge_extraction()

        # Step 3: Vector Indexing (FAISS Search Preparation)
        await self._step_3_vector_indexing()

        # Step 4: Graph Construction (Knowledge Graph)
        await self._step_4_graph_construction()

        # Step 5: Query Processing (Query Analysis)
        await self._step_5_query_processing(user_query)

        # Step 6: Retrieval (Multi-modal Search)
        await self._step_6_retrieval(user_query)

        # Step 7: Generation (LLM Response Generation)
        await self._step_7_generation(user_query)

        # Final Summary
        await self._final_summary()

    async def _step_1_data_ingestion(self):
        """Step 1: Data Ingestion - Process raw text into universal documents"""

        self.print_step_header(
            1,
            "Data Ingestion",
            "Process raw text files into universal documents for analysis"
        )

        print(f"📥 Input: {len(self.demo_texts)} raw text samples")
        for i, text in enumerate(self.demo_texts[:3]):  # Show first 3
            print(f"   {i+1}. {text[:80]}...")

        # Use actual text processor
        text_processor = UniversalTextProcessor(self.domain)

        step_start = time.time()
        try:
            # Update: Use extract_universal_knowledge instead of process_texts_to_documents
            knowledge_stats = text_processor.extract_universal_knowledge()

            step_result = {
                "success": True,
                "processing_time": time.time() - step_start,
                "total_documents": knowledge_stats.get("total_documents", 0),
                "total_entities": knowledge_stats.get("total_entities", 0),
                "total_relations": knowledge_stats.get("total_relations", 0),
                "discovered_entity_types": knowledge_stats.get("discovered_entity_types", []),
                "method": "extract_universal_knowledge"
            }

            self.results["step_1"] = step_result
            self.print_step_output(step_result)

        except Exception as e:
            step_result = {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - step_start
            }
            self.results["step_1"] = step_result
            self.print_step_output(step_result)

    async def _step_2_knowledge_extraction(self):
        """Step 2: Knowledge Extraction - Extract entities and relations using LLM"""

        self.print_step_header(
            2,
            "Knowledge Extraction",
            "Extract entities and relations from text using Azure OpenAI GPT-4"
        )

        # Use actual knowledge extractor
        extractor = UniversalKnowledgeExtractor(self.domain)

        step_start = time.time()
        try:
            # Extract knowledge from texts (actual method from codebase)
            extraction_results = await extractor.extract_knowledge_from_texts(self.demo_texts)

            # Get extracted knowledge structure
            knowledge_data = extractor.get_extracted_knowledge()

            step_result = {
                "success": extraction_results.get("success", False),
                "processing_time": time.time() - step_start,
                "total_entities": len(knowledge_data.get("entities", {})),
                "total_relations": len(knowledge_data.get("relations", [])),
                "discovered_types": knowledge_data.get("discovered_types", {}),
                "entities": knowledge_data.get("entities", {}),
                "relations": knowledge_data.get("relations", []),
                "extraction_stats": knowledge_data.get("statistics", {}),
                "method": "universal_knowledge_extractor"
            }

            self.results["step_2"] = step_result
            self.print_step_output(step_result)

        except Exception as e:
            step_result = {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - step_start
            }
            self.results["step_2"] = step_result
            self.print_step_output(step_result)

    async def _step_3_vector_indexing(self):
        """Step 3: Vector Indexing - Build FAISS vector index for semantic search"""

        self.print_step_header(
            3,
            "Vector Indexing",
            "Build FAISS vector index from documents for semantic search"
        )

        # Use actual vector search
        vector_search = UniversalVectorSearch(self.domain)

        step_start = time.time()
        try:
            # Prepare documents for indexing
            documents = []
            for i, text in enumerate(self.demo_texts):
                documents.append({
                    "doc_id": f"doc_{i}",
                    "content": text,
                    "title": f"Document {i}",
                    "metadata": {"source": "demo", "domain": self.domain}
                })

            # Build index (actual method from codebase)
            index_result = vector_search.build_index_from_documents(documents)

            # Get index statistics
            index_stats = vector_search.get_index_statistics()

            step_result = {
                "success": index_result.get("success", False),
                "processing_time": time.time() - step_start,
                "total_documents": index_stats.get("total_documents", 0),
                "total_embeddings": index_stats.get("total_embeddings", 0),
                "embedding_dimension": index_stats.get("embedding_dimension", 0),
                "index_built": index_stats.get("index_available", False),
                "method": "universal_vector_search",
                "index_stats": index_stats
            }

            self.results["step_3"] = step_result
            self.results["vector_search"] = vector_search  # Save for later use
            self.print_step_output(step_result)

        except Exception as e:
            step_result = {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - step_start
            }
            self.results["step_3"] = step_result
            self.print_step_output(step_result)

    async def _step_4_graph_construction(self):
        """Step 4: Graph Construction - Build knowledge graph from entities and relations"""

        self.print_step_header(
            4,
            "Graph Construction",
            "Build knowledge graph from extracted entities and relations"
        )

        step_start = time.time()
        try:
            # Use extracted knowledge from step 2
            step_2_result = self.results.get("step_2", {})
            entities = step_2_result.get("entities", {})
            relations = step_2_result.get("relations", [])

            # Build graph statistics
            graph_nodes = len(entities)
            graph_edges = len(relations)

            # Create graph structure (simplified for demo)
            graph_data = {
                "nodes": list(entities.keys()),
                "edges": [(r.get("subject", ""), r.get("object", "")) for r in relations if isinstance(r, dict)],
                "node_types": set(e.get("type", "unknown") for e in entities.values() if isinstance(e, dict)),
                "edge_types": set(r.get("predicate", "unknown") for r in relations if isinstance(r, dict))
            }

            step_result = {
                "success": True,
                "processing_time": time.time() - step_start,
                "graph_nodes": graph_nodes,
                "graph_edges": graph_edges,
                "unique_node_types": len(graph_data["node_types"]),
                "unique_edge_types": len(graph_data["edge_types"]),
                "graph_data": graph_data,
                "method": "knowledge_graph_construction"
            }

            self.results["step_4"] = step_result
            self.print_step_output(step_result)

        except Exception as e:
            step_result = {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - step_start
            }
            self.results["step_4"] = step_result
            self.print_step_output(step_result)

    async def _step_5_query_processing(self, user_query: str):
        """Step 5: Query Processing - Analyze and enhance user query"""

        self.print_step_header(
            5,
            "Query Processing",
            "Analyze user query and extract concepts for better retrieval"
        )

        print(f"🔍 Processing Query: '{user_query}'")

        # Use actual query analyzer
        query_analyzer = UniversalQueryAnalyzer(self.domain)

        step_start = time.time()
        try:
            # Update: Use analyze_query_universal and enhance_query_universal
            analysis_result = query_analyzer.analyze_query_universal(user_query)
            enhanced_query = query_analyzer.enhance_query_universal(user_query)

            step_result = {
                "success": True,
                "processing_time": time.time() - step_start,
                "original_query": user_query,
                "analysis_result": analysis_result,
                "enhanced_query": enhanced_query,
                "concepts_identified": len(getattr(analysis_result, 'concepts_detected', [])),
                "query_type": getattr(analysis_result, 'query_type', 'unknown'),
                "method": "universal_query_analyzer"
            }

            self.results["step_5"] = step_result
            self.print_step_output(step_result)

        except Exception as e:
            step_result = {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - step_start,
                "original_query": user_query
            }
            self.results["step_5"] = step_result
            self.print_step_output(step_result)

    async def _step_6_retrieval(self, user_query: str):
        """Step 6: Retrieval - Search knowledge base for relevant information"""

        self.print_step_header(
            6,
            "Retrieval",
            "Search vector index and knowledge graph for relevant information"
        )

        step_start = time.time()
        try:
            # Use vector search from step 3
            vector_search = self.results.get("vector_search")
            if not vector_search:
                raise Exception("Vector search not available from step 3")

            # Get enhanced query from step 5
            step_5_result = self.results.get("step_5", {})
            enhanced_query = step_5_result.get("enhanced_query", {})
            search_query = enhanced_query.get("enhanced_text", user_query)

            # Perform search (actual method from codebase)
            search_results = vector_search.search_universal(search_query, top_k=5)

            step_result = {
                "success": True,
                "processing_time": time.time() - step_start,
                "search_query": search_query,
                "search_results": search_results,
                "results_count": len(search_results),
                "top_score": search_results[0].score if search_results else 0,
                "method": "universal_vector_search"
            }

            self.results["step_6"] = step_result
            self.print_step_output(step_result)

        except Exception as e:
            step_result = {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - step_start
            }
            self.results["step_6"] = step_result
            self.print_step_output(step_result)

    async def _step_7_generation(self, user_query: str):
        """Step 7: Generation - Generate final response using LLM"""

        self.print_step_header(
            7,
            "Generation",
            "Generate comprehensive answer using Azure OpenAI GPT-4"
        )

        # Use actual LLM interface
        llm_interface = UniversalLLMInterface(self.domain)

        step_start = time.time()
        try:
            # Get search results from step 6
            step_6_result = self.results.get("step_6", {})
            search_results = step_6_result.get("search_results", [])

            # Get enhanced query from step 5
            step_5_result = self.results.get("step_5", {})
            enhanced_query = step_5_result.get("enhanced_query", {})

            # Update: Use generate_universal_response instead of generate_response
            response = llm_interface.generate_universal_response(
                query=user_query,
                search_results=search_results,
                enhanced_query=enhanced_query
            )

            step_result = {
                "success": True,
                "processing_time": time.time() - step_start,
                "response": response,
                "response_length": len(getattr(response, 'answer', '')),
                "citations_included": len(getattr(response, 'citations', [])),
                "model_used": "gpt-4",
                "method": "universal_llm_interface"
            }

            self.results["step_7"] = step_result
            self.print_step_output(step_result)

            # Show the actual generated response
            print(f"\n📝 GENERATED RESPONSE:")
            print(f"{'='*60}")
            print(getattr(response, 'answer', 'No response generated'))
            print(f"{'='*60}")

        except Exception as e:
            step_result = {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - step_start
            }
            self.results["step_7"] = step_result
            self.print_step_output(step_result)

    async def _final_summary(self):
        """Show final workflow summary"""

        print(f"\n{'='*80}")
        print(f"🎯 WORKFLOW SUMMARY")
        print(f"{'='*80}")

        total_time = time.time() - self.start_time
        successful_steps = sum(1 for step in self.results.values() if step.get("success", False))
        total_steps = len(self.results)

        print(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
        print(f"✅ Successful Steps: {successful_steps}/{total_steps}")
        print(f"🏷️  Domain: {self.domain}")
        print(f"📅 Completed: {datetime.now().isoformat()}")

        # Show step-by-step performance
        print(f"\n📊 STEP PERFORMANCE:")
        for step_name, step_data in self.results.items():
            status = "✅" if step_data.get("success", False) else "❌"
            time_taken = step_data.get("processing_time", 0)
            print(f"   {status} {step_name}: {time_taken:.2f}s")

        # Show key metrics
        print(f"\n📈 KEY METRICS:")

        step_2 = self.results.get("step_2", {})
        if step_2.get("success"):
            print(f"   🏷️  Entities Discovered: {step_2.get('total_entities', 0)}")
            print(f"   🔗 Relations Discovered: {step_2.get('total_relations', 0)}")

        step_3 = self.results.get("step_3", {})
        if step_3.get("success"):
            print(f"   🔧 Documents Indexed: {step_3.get('total_documents', 0)}")
            print(f"   📐 Vector Dimensions: {step_3.get('embedding_dimension', 0)}")

        step_6 = self.results.get("step_6", {})
        if step_6.get("success"):
            print(f"   🔍 Search Results: {step_6.get('results_count', 0)}")
            print(f"   🎯 Top Score: {step_6.get('top_score', 0):.3f}")

        step_7 = self.results.get("step_7", {})
        if step_7.get("success"):
            print(f"   📝 Response Length: {step_7.get('response_length', 0)} characters")
            print(f"   📚 Citations: {step_7.get('citations_included', 0)}")

        print(f"\n🎉 WORKFLOW COMPLETED!")
        print(f"{'='*80}")


async def main():
    """Main demonstration function"""

    # Initialize demo
    demo = UniversalRAGWorkflowDemo("maintenance")

    # Run complete workflow demonstration
    await demo.demonstrate_complete_workflow(
        "How do I troubleshoot pump vibration problems and fix bearing issues?"
    )

    # Save results for analysis
    output_file = f"workflow_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for step_name, step_data in demo.results.items():
            json_results[step_name] = {
                key: value for key, value in step_data.items()
                if key not in ['vector_search', 'search_results']  # Skip non-serializable objects
            }
        json.dump(json_results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())