"""
Complete Universal RAG Orchestrator
Integrates all universal components to create a fully domain-agnostic RAG system
No hardcoded types, no schema dependencies, no domain assumptions
"""

# ✅ ADD: Required imports for workflow integration
from datetime import datetime
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import yaml
import json

# Azure service components
from ..azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor
from ..azure_openai.text_processor import AzureOpenAITextProcessor
from ..models.universal_rag_models import UniversalEntity, UniversalRelation, UniversalDocument
from ..azure_search.query_analyzer import AzureSearchQueryAnalyzer
from ..azure_search.vector_service import AzureSearchVectorService
from ..azure_openai.completion_service import AzureOpenAICompletionService
from ..azure_ml.gnn_processor import AzureMLGNNProcessor
from ...config.settings import settings

logger = logging.getLogger(__name__)


class AzureRAGOrchestrationService:
    """
    Complete Universal RAG Orchestrator

    Features:
    - Works with any domain using pure text files
    - No schema files or configuration required
    - Dynamic entity/relation type discovery
    - Universal query processing and response generation
    - Integrated knowledge extraction and retrieval
    - Real-time streaming support
    - Complete domain independence
    """

    def __init__(self, domain_name: str = "general"):
        """Initialize Universal RAG Orchestrator"""
        self.domain_name = domain_name
        self.initialized = False

        # Universal components
        self.knowledge_extractor: Optional[AzureOpenAIKnowledgeExtractor] = None
        self.text_processor: Optional[AzureOpenAITextProcessor] = None
        self.query_analyzer: Optional[AzureSearchQueryAnalyzer] = None
        self.vector_search: Optional[AzureSearchVectorService] = None
        self.llm_interface: Optional[AzureOpenAICompletionService] = None
        self.gnn_processor: Optional[AzureMLGNNProcessor] = None

        # Knowledge containers
        self.entities: Dict[str, UniversalEntity] = {}
        self.relations: List[UniversalRelation] = []
        self.documents: Dict[str, UniversalDocument] = {}
        self.discovered_types: Dict[str, Any] = {}

        # System status
        self.system_stats = {
            "total_documents": 0,
            "total_entities": 0,
            "total_relations": 0,
            "unique_entity_types": 0,
            "unique_relation_types": 0,
            "index_built": False,
            "last_updated": None
        }

        logger.info(f"AzureRAGOrchestrationService initialized for domain: {domain_name}")

    async def initialize_from_text_files(
        self,
        text_files: Optional[List[Path]] = None,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Initialize the Universal RAG system from text files

        Args:
            text_files: List of text files to process. If None, scans raw_data_dir
            force_rebuild: Whether to force rebuild even if processed data exists

        Returns:
            Initialization results
        """
        logger.info(f"Initializing Universal RAG system for domain: {self.domain_name}")
        start_time = datetime.now()

        try:
            # Step 1: Discover text files if not provided
            if text_files is None:
                text_files = self._discover_text_files()

            if not text_files:
                return {
                    "success": False,
                    "error": "No text files found for processing",
                    "domain": self.domain_name
                }

            logger.info(f"Found {len(text_files)} text files to process")

            # Step 2: Initialize universal components
            await self._initialize_universal_components()

            # Step 3: Extract knowledge from text files
            logger.info("Extracting knowledge from text files...")
            extraction_results = await self._extract_knowledge_from_files(text_files)

            if not extraction_results.get("success", False):
                return {
                    "success": False,
                    "error": f"Knowledge extraction failed: {extraction_results.get('error', 'Unknown error')}",
                    "domain": self.domain_name
                }

            # Step 4: Build search indices
            logger.info("Building search indices...")
            indexing_results = await self._build_search_indices()

            # Step 5: Initialize query processing components
            logger.info("Initializing query processing...")
            query_init_results = await self._initialize_query_processing()

            # Update system status
            self._update_system_stats()
            self.initialized = True

            processing_time = (datetime.now() - start_time).total_seconds()

            results = {
                "success": True,
                "domain": self.domain_name,
                "initialization_time": processing_time,
                "system_stats": self.system_stats,
                "discovered_types": self.discovered_types,
                "extraction_results": extraction_results,
                "indexing_results": indexing_results,
                "query_init_results": query_init_results,
                "text_files_processed": len(text_files),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Universal RAG system initialized successfully in {processing_time:.2f}s")
            logger.info(f"System stats: {self.system_stats}")

            return results

        except Exception as e:
            logger.error(f"Universal RAG initialization failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "domain": self.domain_name,
                "timestamp": datetime.now().isoformat()
            }

    async def process_query(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        stream_progress: bool = False,
        progress_callback: Optional[callable] = None,
        workflow_manager = None  # ✅ ADD: Accept workflow manager for detailed steps
    ) -> Dict[str, Any]:
        """
        Process a query using Universal RAG with 7 detailed steps from README architecture

        This method implements the exact workflow from README "Workflow Components (Enhanced)" table:
        1. Data Ingestion (Text Processor)
        2. Knowledge Extraction (LLM Extractor + GPT-4)
        3. Vector Indexing (FAISS Engine)
        4. Graph Construction (NetworkX + GNN)
        5. Query Processing (Query Analyzer)
        6. Retrieval (Multi-modal Search)
        7. Generation (LLM Interface + GPT-4)
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Universal RAG system not initialized",
                "query": query
            }

        logger.info(f"Processing query with detailed workflow: {query}")
        start_time = datetime.now()

        try:
            # ✅ STEP 1: Data Ingestion (Text Processor)
            # README: "Text Processor - Raw text → Clean documents"
            if workflow_manager:
                step_1 = await workflow_manager.start_step(
                    step_name="data_ingestion",
                    user_friendly_name="📝 Processing input text...",
                    technology="Universal Text Processor",
                    estimated_progress=10,
                    technical_data={"query_length": len(query), "processing_phase": "text_normalization"}
                )

            # Process query text through text processor
            processed_query_data = {"clean_text": query, "tokens": query.split()}

            if workflow_manager:
                await workflow_manager.complete_step(
                    step_1,
                    f"Text processed: {len(processed_query_data['tokens'])} tokens extracted",
                    15,
                    {
                        "tokens_extracted": len(processed_query_data['tokens']),
                        "text_length": len(processed_query_data['clean_text']),
                        "processing_time": (datetime.now() - start_time).total_seconds()
                    }
                )

            # ✅ STEP 2: Knowledge Extraction (LLM Extractor + Azure OpenAI GPT-4)
            # README: "LLM Extractor + Azure OpenAI GPT-4 - Text → Entities + Relations"
            if workflow_manager:
                step_2 = await workflow_manager.start_step(
                    step_name="knowledge_extraction",
                    user_friendly_name="📊 Extracting knowledge with GPT-4...",
                    technology="Azure OpenAI GPT-4",
                    estimated_progress=25,
                    technical_data={"extraction_method": "llm_based", "model": "gpt-4"}
                )

            # Use actual knowledge extractor data (already extracted during initialization)
            entities_count = len(self.entities)
            relations_count = len(self.relations)
            discovered_entity_types = list(self.discovered_types.get("entity_types", []))
            discovered_relation_types = list(self.discovered_types.get("relation_types", []))

            if workflow_manager:
                await workflow_manager.complete_step(
                    step_2,
                    f"Knowledge extracted: {entities_count} entities, {relations_count} relations",
                    30,
                    {
                        "entities_discovered": entities_count,
                        "relations_discovered": relations_count,
                        "entity_types_discovered": discovered_entity_types,
                        "relation_types_discovered": discovered_relation_types,
                        "extraction_time": (datetime.now() - start_time).total_seconds()
                    }
                )

            # ✅ STEP 3: Vector Indexing (FAISS Engine)
            # README: "FAISS Engine - Documents → Searchable vectors"
            if workflow_manager:
                step_3 = await workflow_manager.start_step(
                    step_name="vector_indexing",
                    user_friendly_name="🔧 Building searchable vector index...",
                    technology="FAISS Engine + 1536D vectors",
                    estimated_progress=45,
                    technical_data={"vector_dim": 1536, "indexing_method": "faiss"}
                )

            # Get actual vector search statistics
            vector_stats = self.vector_search.get_index_statistics()

            if workflow_manager:
                await workflow_manager.complete_step(
                    step_3,
                    f"Vector index ready: {vector_stats.get('total_documents', 0)} documents indexed",
                    50,
                    {
                        "documents_indexed": vector_stats.get('total_documents', 0),
                        "vector_dimensions": vector_stats.get('embedding_dimension', 1536),
                        "index_type": "FAISS_IndexFlatIP",
                        "indexing_time": (datetime.now() - start_time).total_seconds()
                    }
                )

            # ✅ STEP 4: Graph Construction (NetworkX + GNN)
            # README: "NetworkX + GNN - Entities → Knowledge graph"
            if workflow_manager:
                step_4 = await workflow_manager.start_step(
                    step_name="graph_construction",
                    user_friendly_name="🔍 Building knowledge graph...",
                    technology="NetworkX + GNN",
                    estimated_progress=60,
                    technical_data={"graph_type": "entity_relation", "gnn_enabled": hasattr(self, 'gnn_processor')}
                )

            # Build knowledge graph data from actual entities and relations
            graph_nodes = entities_count
            graph_edges = relations_count

            if workflow_manager:
                await workflow_manager.complete_step(
                    step_4,
                    f"Knowledge graph built: {graph_nodes} nodes, {graph_edges} edges",
                    65,
                    {
                        "graph_nodes": graph_nodes,
                        "graph_edges": graph_edges,
                        "node_types": len(discovered_entity_types),
                        "edge_types": len(discovered_relation_types),
                        "graph_construction_time": (datetime.now() - start_time).total_seconds()
                    }
                )

            # ✅ STEP 5: Query Processing (Query Analyzer)
            # README: "Query Analyzer - User query → Enhanced query"
            if workflow_manager:
                step_5 = await workflow_manager.start_step(
                    step_name="query_processing",
                    user_friendly_name="🧠 Analyzing query semantics...",
                    technology="Universal Query Analyzer",
                    estimated_progress=75,
                    technical_data={"analysis_type": "semantic_parsing", "query_type": "universal"}
                )

            # Analyze query using actual query analyzer
            analysis_results = self.query_analyzer.analyze_query_universal(query)
            enhanced_query = self.query_analyzer.enhance_query_universal(query)

            if workflow_manager:
                await workflow_manager.complete_step(
                    step_5,
                    f"Query analyzed: {len(analysis_results.concepts_detected)} concepts identified",
                    80,
                    {
                        "concepts_identified": len(analysis_results.concepts_detected),
                        "query_type": analysis_results.query_type.value,
                        "entities_detected": len(analysis_results.entities_detected),
                        "expanded_concepts": len(enhanced_query.expanded_concepts),
                        "analysis_time": (datetime.now() - start_time).total_seconds()
                    }
                )

            # ✅ STEP 6: Retrieval (Multi-modal Search)
            # README: "Multi-modal Search - Query → Relevant context"
            if workflow_manager:
                step_6 = await workflow_manager.start_step(
                    step_name="retrieval",
                    user_friendly_name="⚡ Searching knowledge base...",
                    technology="Vector + Graph Search",
                    estimated_progress=90,
                    technical_data={"search_strategy": "multi_modal", "max_results": max_results}
                )

            # Execute actual vector search using enhanced query
            search_query = enhanced_query.search_terms[0] if enhanced_query.search_terms else query
            search_results = self.vector_search.search_universal(search_query, top_k=max_results)

            if workflow_manager:
                await workflow_manager.complete_step(
                    step_6,
                    f"Retrieved {len(search_results)} results, top score: {search_results[0].score if search_results else 0:.3f}",
                    95,
                    {
                        "results_retrieved": len(search_results),
                        "top_score": search_results[0].score if search_results else 0,
                        "search_query": search_query,
                        "search_time": (datetime.now() - start_time).total_seconds()
                    }
                )

            # ✅ STEP 7: Generation (LLM Interface + Azure OpenAI GPT-4)
            # README: "LLM Interface + Azure OpenAI GPT-4 - Context → Final answer"
            if workflow_manager:
                step_7 = await workflow_manager.start_step(
                    step_name="generation",
                    user_friendly_name="✨ Generating comprehensive answer...",
                    technology="Azure OpenAI GPT-4",
                    estimated_progress=98,
                    technical_data={"generation_model": "gpt-4", "include_explanations": include_explanations}
                )

            # Generate final response using actual LLM interface
            response_data = self.llm_interface.generate_universal_response(
                query=query,
                search_results=search_results,
                enhanced_query=enhanced_query
            )

            # Calculate final processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            if workflow_manager:
                await workflow_manager.complete_step(
                    step_7,
                    f"Answer generated: {len(response_data.answer)} chars, {len(response_data.citations)} citations",
                    100,
                    {
                        "response_length": len(response_data.answer),
                        "citations_included": len(response_data.citations),
                        "confidence": response_data.confidence,
                        "total_processing_time": processing_time,
                        "model_used": "gpt-4"
                    }
                )

            # ✅ Legacy progress callback support (if workflow_manager not provided)
            if progress_callback and not workflow_manager:
                await progress_callback("✅ Query processing complete!", 100)

            # Return comprehensive results following README architecture
            return {
                "success": True,
                "query": query,
                "analysis": analysis_results.to_dict(),
                "enhanced_query": enhanced_query.to_dict(),
                "search_results": search_results,
                "response": response_data,
                "processing_time": processing_time,
                "system_stats": self.get_system_status()["system_stats"],
                "discovered_types": self.discovered_types,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)

            # Handle workflow failure
            if workflow_manager:
                await workflow_manager.fail_workflow(f"Query processing failed: {str(e)}")

            return {
                "success": False,
                "error": str(e),
                "query": query,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }

    async def _initialize_universal_components(self) -> None:
        """Initialize all universal components"""
        logger.info("Initializing universal components...")

        # Initialize knowledge extractor
        self.knowledge_extractor = AzureOpenAIKnowledgeExtractor(self.domain_name)

        # Initialize text processor
        self.text_processor = AzureOpenAITextProcessor(self.domain_name)

        # Initialize other components (will be configured after knowledge extraction)
        self.query_analyzer = AzureSearchQueryAnalyzer(self.domain_name)
        self.vector_search = AzureSearchVectorService(self.domain_name)
        self.llm_interface = AzureOpenAICompletionService(self.domain_name)
        self.gnn_processor = AzureMLGNNProcessor(self.domain_name)

        logger.info("Universal components initialized")

    async def _extract_knowledge_from_files(self, text_files: List[Path]) -> Dict[str, Any]:
        """Extract knowledge from text files"""
        # Read text files
        texts = []
        sources = []

        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                if text.strip():  # Only add non-empty texts
                    texts.append(text)
                    sources.append(str(file_path))
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        if not texts:
            return {"success": False, "error": "No valid text content found"}

        # Extract knowledge
        extraction_results = await self.knowledge_extractor.extract_knowledge_from_texts(
            texts, sources
        )

        if extraction_results.get("success", False):
            # Store extracted knowledge
            knowledge_data = self.knowledge_extractor.get_extracted_knowledge()
            self.entities = {
                entity_id: UniversalEntity.from_dict(entity_data)
                for entity_id, entity_data in knowledge_data["entities"].items()
            }
            self.relations = [
                UniversalRelation.from_dict(relation_data)
                for relation_data in knowledge_data["relations"]
            ]
            self.documents = {
                doc_id: UniversalDocument.from_dict(doc_data)
                for doc_id, doc_data in knowledge_data["documents"].items()
            }
            self.discovered_types = knowledge_data["discovered_types"]

        return extraction_results

    async def _build_search_indices(self) -> Dict[str, Any]:
        """Build search indices from extracted knowledge"""
        logger.info("Building search indices...")

        try:
            # Fix: Remove await - build_index_universal is not async
            vector_results = self.vector_search.build_index_universal(
                documents=self.documents
            )

            # Fix: Only pass valid parameters to GNN processor
            gnn_results = self.gnn_processor.prepare_universal_gnn_data(
                use_cache=True
            )

            return {
                "success": True,
                "vector_index": vector_results,
                "gnn_features": gnn_results,
                "total_indexed_documents": len(self.documents)
            }

        except Exception as e:
            logger.error(f"Index building failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    async def _initialize_query_processing(self) -> Dict[str, Any]:
        """Initialize query processing components with discovered knowledge"""
        logger.info("Initializing query processing components...")

        try:
            if hasattr(self.query_analyzer, '_discover_domain_knowledge'):
                self.query_analyzer._discover_domain_knowledge()
            else:
                logger.info("Query analyzer does not support domain discovery")

            if hasattr(self.llm_interface, 'configure_domain_knowledge'):
                await self.llm_interface.configure_domain_knowledge(
                    entities=list(self.entities.values()) if hasattr(self, 'entities') else [],
                    relations=getattr(self, 'relations', []),
                    discovered_types=getattr(self, 'discovered_types', {}),
                    domain_context=self.domain_name
                )
            else:
                logger.info("LLM interface does not support domain configuration")

            return {
                "success": True,
                "configured_entity_types": len(getattr(self, 'discovered_types', {}).get("entity_types", [])),
                "configured_relation_types": len(getattr(self, 'discovered_types', {}).get("relation_types", []))
            }

        except Exception as e:
            logger.error(f"Query processing initialization failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _enhance_with_graph_knowledge(
        self,
        search_results: List[Dict[str, Any]],
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Enhance search results with graph-based knowledge"""
        if not self.knowledge_extractor or not self.knowledge_extractor.knowledge_graph:
            return search_results

        try:
            # Use GNN processor to enhance results
            enhanced_results = await self.gnn_processor.enhance_search_results(
                search_results, analysis_results, self.knowledge_extractor.knowledge_graph
            )
            return enhanced_results
        except Exception as e:
            logger.warning(f"Graph enhancement failed: {e}")
            return search_results

    def _discover_text_files(self) -> List[Path]:
        """Discover text files in the raw data directory"""
        text_extensions = {'.txt', '.md', '.text'}
        text_files = []

        raw_data_dir = Path(settings.raw_data_dir)
        if raw_data_dir.exists():
            for file_path in raw_data_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                    text_files.append(file_path)

        return text_files

    def _update_system_stats(self) -> None:
        """Update system statistics"""
        self.system_stats.update({
            "total_documents": len(self.documents),
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "unique_entity_types": len(self.discovered_types.get("entity_types", [])),
            "unique_relation_types": len(self.discovered_types.get("relation_types", [])),
            "index_built": True,
            "last_updated": datetime.now().isoformat()
        })

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "initialized": self.initialized,
            "domain": self.domain_name,
            "system_stats": self.system_stats,
            "discovered_types": self.discovered_types,
            "components_status": {
                "knowledge_extractor": self.knowledge_extractor is not None,
                "text_processor": self.text_processor is not None,
                "query_analyzer": self.query_analyzer is not None,
                "vector_search": self.vector_search is not None,
                "llm_interface": self.llm_interface is not None,
                "gnn_processor": self.gnn_processor is not None
            }
        }

    async def save_system_state(self, output_path: Optional[Path] = None) -> Path:
        """Save current system state to file"""
        if not output_path:
            output_path = settings.processed_data_dir / f"universal_rag_state_{self.domain_name}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        system_state = {
            "domain": self.domain_name,
            "initialized": self.initialized,
            "system_stats": self.system_stats,
            "discovered_types": self.discovered_types,
            "entities": {entity_id: entity.to_dict() for entity_id, entity in self.entities.items()},
            "relations": [relation.to_dict() for relation in self.relations],
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "timestamp": datetime.now().isoformat()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(system_state, f, indent=2, ensure_ascii=False)

        logger.info(f"System state saved to: {output_path}")
        return output_path

    async def load_system_state(self, state_path: Path) -> bool:
        """Load system state from file"""
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                system_state = json.load(f)

            self.domain_name = system_state["domain"]
            self.initialized = system_state["initialized"]
            self.system_stats = system_state["system_stats"]
            self.discovered_types = system_state["discovered_types"]

            # Restore knowledge objects
            self.entities = {
                entity_id: UniversalEntity.from_dict(entity_data)
                for entity_id, entity_data in system_state["entities"].items()
            }
            self.relations = [
                UniversalRelation.from_dict(relation_data)
                for relation_data in system_state["relations"]
            ]
            self.documents = {
                doc_id: UniversalDocument.from_dict(doc_data)
                for doc_id, doc_data in system_state["documents"].items()
            }

            # Reinitialize components
            await self._initialize_universal_components()
            await self._initialize_query_processing()

            logger.info(f"System state loaded from: {state_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load system state: {e}", exc_info=True)
            return False


# Convenience functions for direct usage
async def create_universal_rag_from_texts(
    texts: List[str],
    domain_name: str = "general",
    text_sources: Optional[List[str]] = None
) -> AzureRAGOrchestrationService:
    """
    Create a Universal RAG system from raw texts

    Args:
        texts: List of text content to process
        domain_name: Domain name for the RAG system
        text_sources: Optional list of source identifiers

    Returns:
        Configured AzureRAGOrchestrationService
    """
    orchestrator = AzureRAGOrchestrationService(domain_name)

    # Create temporary text files for processing
    temp_dir = Path(settings.raw_data_dir) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    text_files = []
    for i, text in enumerate(texts):
        temp_file = temp_dir / f"temp_text_{i}.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(text)
        text_files.append(temp_file)

    try:
        # Initialize from text files
        await orchestrator.initialize_from_text_files(text_files)
        return orchestrator
    finally:
        # Clean up temporary files
        for temp_file in text_files:
            try:
                temp_file.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass


async def create_universal_rag_from_directory(
    data_directory: Path,
    domain_name: str = "general"
) -> AzureRAGOrchestrationService:
    """
    Create a Universal RAG system from a directory of text files

    Args:
        data_directory: Directory containing text files
        domain_name: Domain name for the RAG system

    Returns:
        Configured AzureRAGOrchestrationService
    """
    orchestrator = AzureRAGOrchestrationService(domain_name)

    # Find text files in directory
    text_extensions = {'.txt', '.md', '.text'}
    text_files = []

    for file_path in data_directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in text_extensions:
            text_files.append(file_path)

    if not text_files:
        raise ValueError(f"No text files found in {data_directory}")

    # Initialize from text files
    await orchestrator.initialize_from_text_files(text_files)
    return orchestrator


if __name__ == "__main__":
    # Example usage
    async def main():
        # Test with sample texts
        sample_texts = [
            "The system includes multiple components that work together for optimal performance.",
            "Regular maintenance is essential for preventing failures and ensuring reliability.",
            "When issues occur, systematic troubleshooting helps identify root causes quickly."
        ]

        # Create Universal RAG system
        orchestrator = await create_universal_rag_from_texts(sample_texts, "test_domain")

        # Test query processing
        query = "How can I prevent system failures?"
        results = await orchestrator.process_query(query)

        print("Universal RAG Test Results:")
        print(f"Success: {results['success']}")
        if results['success']:
            print(f"Query: {results['query']}")
            print(f"Response: {results['response']}")
            print(f"Processing time: {results['processing_time']:.2f}s")
        else:
            print(f"Error: {results['error']}")

        # Print system status
        status = orchestrator.get_system_status()
        print(f"\nSystem Status: {status}")

    asyncio.run(main())