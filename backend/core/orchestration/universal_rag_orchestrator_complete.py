"""
Complete Universal RAG Orchestrator
Integrates all universal components to create a fully domain-agnostic RAG system
No hardcoded types, no schema dependencies, no domain assumptions
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import yaml
import json

# Universal components only - no domain-specific imports
from core.extraction.universal_knowledge_extractor import UniversalKnowledgeExtractor
from core.knowledge.universal_text_processor import UniversalTextProcessor
from core.models.universal_models import UniversalEntity, UniversalRelation, UniversalDocument
from core.enhancement.universal_query_analyzer import UniversalQueryAnalyzer
from core.retrieval.universal_vector_search import UniversalVectorSearch
from core.generation.universal_llm_interface import UniversalLLMInterface
from core.gnn.universal_gnn_processor import UniversalGNNDataProcessor
from config.settings import settings

logger = logging.getLogger(__name__)


class UniversalRAGOrchestrator:
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
        self.knowledge_extractor: Optional[UniversalKnowledgeExtractor] = None
        self.text_processor: Optional[UniversalTextProcessor] = None
        self.query_analyzer: Optional[UniversalQueryAnalyzer] = None
        self.vector_search: Optional[UniversalVectorSearch] = None
        self.llm_interface: Optional[UniversalLLMInterface] = None
        self.gnn_processor: Optional[UniversalGNNDataProcessor] = None

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

        logger.info(f"UniversalRAGOrchestrator initialized for domain: {domain_name}")

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
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process a query using the Universal RAG system

        Args:
            query: User query to process
            max_results: Maximum number of results to return
            include_explanations: Whether to include explanations in response
            stream_progress: Whether to stream progress updates
            progress_callback: Optional callback for progress updates

        Returns:
            Query processing results
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Universal RAG system not initialized",
                "query": query
            }

        logger.info(f"Processing query: {query}")
        start_time = datetime.now()

        try:
            # Step 1: Analyze query
            if progress_callback:
                await progress_callback("Analyzing query...", 10)

            analysis_results = await self.query_analyzer.analyze_query(query)
            enhanced_query = await self.query_analyzer.enhance_query(analysis_results)

            # Step 2: Retrieve relevant documents
            if progress_callback:
                await progress_callback("Searching knowledge base...", 30)

            search_results = await self.vector_search.search_documents(
                enhanced_query.get("enhanced_text", query),
                max_results=max_results
            )

            # Step 3: Apply graph-based enhancement (if available)
            if progress_callback:
                await progress_callback("Enhancing with graph knowledge...", 50)

            enhanced_results = await self._enhance_with_graph_knowledge(
                search_results, analysis_results
            )

            # Step 4: Generate response
            if progress_callback:
                await progress_callback("Generating response...", 70)

            response = await self.llm_interface.generate_response(
                query=query,
                enhanced_query=enhanced_query,
                search_results=enhanced_results,
                include_explanations=include_explanations,
                domain_context=self.domain_name
            )

            if progress_callback:
                await progress_callback("Finalizing response...", 100)

            processing_time = (datetime.now() - start_time).total_seconds()

            # Compile final results
            results = {
                "success": True,
                "query": query,
                "analysis": analysis_results,
                "enhanced_query": enhanced_query,
                "search_results": enhanced_results,
                "response": response,
                "processing_time": processing_time,
                "domain": self.domain_name,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "domain": self.domain_name,
                "timestamp": datetime.now().isoformat()
            }

    async def _initialize_universal_components(self) -> None:
        """Initialize all universal components"""
        logger.info("Initializing universal components...")

        # Initialize knowledge extractor
        self.knowledge_extractor = UniversalKnowledgeExtractor(self.domain_name)

        # Initialize text processor
        self.text_processor = UniversalTextProcessor(self.domain_name)

        # Initialize other components (will be configured after knowledge extraction)
        self.query_analyzer = UniversalQueryAnalyzer(self.domain_name)
        self.vector_search = UniversalVectorSearch(self.domain_name)
        self.llm_interface = UniversalLLMInterface(self.domain_name)
        self.gnn_processor = UniversalGNNDataProcessor(self.domain_name)

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
        """Build search indices for documents and entities"""
        logger.info("Building search indices...")

        try:
            # Build vector index for documents
            documents_list = list(self.documents.values())
            vector_results = await self.vector_search.build_index_from_documents(documents_list)

            # Build GNN features if we have a knowledge graph
            gnn_results = {}
            if self.knowledge_extractor and self.knowledge_extractor.knowledge_graph:
                gnn_results = await self.gnn_processor.prepare_gnn_data(
                    entities=list(self.entities.values()),
                    relations=self.relations,
                    documents=list(self.documents.values())
                )

            return {
                "success": True,
                "vector_index": vector_results,
                "gnn_features": gnn_results,
                "total_indexed_documents": len(documents_list)
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
            # Configure query analyzer with discovered types
            await self.query_analyzer.configure_domain_knowledge(
                entity_types=list(self.discovered_types.get("entity_types", [])),
                relation_types=list(self.discovered_types.get("relation_types", [])),
                domain_context=self.domain_name
            )

            # Configure LLM interface with domain knowledge
            await self.llm_interface.configure_domain_knowledge(
                entities=list(self.entities.values()),
                relations=self.relations,
                discovered_types=self.discovered_types,
                domain_context=self.domain_name
            )

            return {
                "success": True,
                "configured_entity_types": len(self.discovered_types.get("entity_types", [])),
                "configured_relation_types": len(self.discovered_types.get("relation_types", []))
            }

        except Exception as e:
            logger.error(f"Query processing initialization failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

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
) -> UniversalRAGOrchestrator:
    """
    Create a Universal RAG system from raw texts

    Args:
        texts: List of text content to process
        domain_name: Domain name for the RAG system
        text_sources: Optional list of source identifiers

    Returns:
        Configured UniversalRAGOrchestrator
    """
    orchestrator = UniversalRAGOrchestrator(domain_name)

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
) -> UniversalRAGOrchestrator:
    """
    Create a Universal RAG system from a directory of text files

    Args:
        data_directory: Directory containing text files
        domain_name: Domain name for the RAG system

    Returns:
        Configured UniversalRAGOrchestrator
    """
    orchestrator = UniversalRAGOrchestrator(domain_name)

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