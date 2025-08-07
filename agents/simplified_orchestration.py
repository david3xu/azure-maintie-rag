"""
Simplified Agent Orchestration - PydanticAI Best Practices
==========================================================

This replaces the complex dual-graph workflow orchestration with simple,
direct agent-to-agent communication using PydanticAI patterns.

Key simplifications:
- Direct agent composition instead of workflow graphs
- Simple async/await coordination instead of state machines  
- Clean data passing using PydanticAI result types
- No complex orchestrators or state management
"""

import asyncio
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel

from .domain_intelligence.simplified_agent import get_domain_agent, DomainDeps
from .knowledge_extraction.simplified_agent import get_extraction_agent, ExtractionDeps  
from .universal_search.simplified_agent import get_search_agent, SearchDeps

class RAGOrchestrationResult(BaseModel):
    """Results from complete RAG pipeline orchestration"""
    domain_analysis: Dict[str, Any]
    extraction_results: Dict[str, Any]
    search_results: Dict[str, Any]
    total_execution_time: float
    success: bool
    error_message: Optional[str] = None

class SimplifiedRAGOrchestrator:
    """
    Simplified RAG orchestrator using direct agent composition.
    
    This replaces the complex DualGraphOrchestrator with simple,
    direct agent coordination following PydanticAI best practices.
    """
    
    def __init__(self):
        # No complex initialization - agents created on demand
        pass
    
    async def process_document_corpus(
        self, 
        corpus_path: str, 
        domain_name: Optional[str] = None
    ) -> RAGOrchestrationResult:
        """
        Process document corpus through domain analysis and knowledge extraction.
        
        This replaces the complex config-extraction workflow with simple agent coordination.
        """
        start_time = time.time()
        
        try:
            # Step 1: Domain Analysis (simple agent call)
            domain_agent = get_domain_agent()
            domain_deps = DomainDeps(data_directory=corpus_path)
            
            domain_result = await domain_agent.run(
                f"Analyze domain for corpus at {corpus_path}",
                deps=domain_deps
            )
            
            detected_domain = domain_result.data.detected_domain
            
            # Step 2: Knowledge Extraction (direct agent coordination)
            extraction_agent = get_extraction_agent()  
            extraction_deps = ExtractionDeps(
                confidence_threshold=0.8,
                max_entities_per_chunk=15,
                enable_relationships=True
            )
            
            # Read sample documents for extraction
            from pathlib import Path
            corpus_dir = Path(corpus_path)
            sample_files = list(corpus_dir.glob("*.md"))[:3]  # Process first 3 files
            
            extraction_results = []
            for file_path in sample_files:
                try:
                    content = file_path.read_text(encoding='utf-8')[:2000]  # First 2000 chars
                    result = await extraction_agent.run(
                        f"Extract knowledge from: {content}",
                        deps=extraction_deps
                    )
                    extraction_results.append(result.data.model_dump())
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
            
            execution_time = time.time() - start_time
            
            return RAGOrchestrationResult(
                domain_analysis=domain_result.data.model_dump(),
                extraction_results={"results": extraction_results, "files_processed": len(extraction_results)},
                search_results={},  # Not included in document processing
                total_execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return RAGOrchestrationResult(
                domain_analysis={},
                extraction_results={},
                search_results={},
                total_execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    async def execute_universal_search(
        self, 
        query: str, 
        max_results: int = 10
    ) -> RAGOrchestrationResult:
        """
        Execute universal search with optional domain detection.
        
        This replaces the complex search workflow with simple agent coordination.
        """
        start_time = time.time()
        
        try:
            # Step 1: Optional domain detection for query optimization
            domain_agent = get_domain_agent()
            domain_deps = DomainDeps()
            
            # Simple domain detection based on query
            domain_result = await domain_agent.run(
                f"What domain does this query belong to: {query}",
                deps=domain_deps
            )
            
            # Step 2: Universal search (direct agent call)  
            search_agent = get_search_agent()
            search_deps = SearchDeps(
                max_results=max_results,
                similarity_threshold=0.7,
                enable_vector_search=True,
                enable_graph_search=True
            )
            
            search_result = await search_agent.run(
                f"Search for: {query}",
                deps=search_deps
            )
            
            execution_time = time.time() - start_time
            
            return RAGOrchestrationResult(
                domain_analysis=domain_result.data.model_dump(),
                extraction_results={},  # Not included in search
                search_results=search_result.data.model_dump(),
                total_execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return RAGOrchestrationResult(
                domain_analysis={},
                extraction_results={},  
                search_results={},
                total_execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    async def full_rag_pipeline(
        self,
        corpus_path: str,
        query: str,
        domain_name: Optional[str] = None
    ) -> RAGOrchestrationResult:
        """
        Execute complete RAG pipeline: domain analysis -> extraction -> search.
        
        This demonstrates simple agent composition without complex orchestration.
        """
        start_time = time.time()
        
        try:
            # Step 1: Process corpus (domain + extraction)
            corpus_result = await self.process_document_corpus(corpus_path, domain_name)
            
            if not corpus_result.success:
                return corpus_result
            
            # Step 2: Execute search using learned domain knowledge
            search_result = await self.execute_universal_search(query)
            
            if not search_result.success:
                return search_result
            
            # Step 3: Combine results
            total_time = time.time() - start_time
            
            return RAGOrchestrationResult(
                domain_analysis=corpus_result.domain_analysis,
                extraction_results=corpus_result.extraction_results,
                search_results=search_result.search_results,
                total_execution_time=total_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return RAGOrchestrationResult(
                domain_analysis={},
                extraction_results={},
                search_results={},
                total_execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

# Simple factory function (no global state)
def create_rag_orchestrator() -> SimplifiedRAGOrchestrator:
    """Create RAG orchestrator with no complex initialization"""
    return SimplifiedRAGOrchestrator()

# Simple API functions for common use cases
async def analyze_corpus(corpus_path: str, domain_name: Optional[str] = None) -> Dict[str, Any]:
    """Simple corpus analysis API"""
    orchestrator = create_rag_orchestrator()
    result = await orchestrator.process_document_corpus(corpus_path, domain_name)
    return result.model_dump()

async def search_knowledge(query: str, max_results: int = 10) -> Dict[str, Any]:
    """Simple knowledge search API"""
    orchestrator = create_rag_orchestrator() 
    result = await orchestrator.execute_universal_search(query, max_results)
    return result.model_dump()

async def complete_rag(corpus_path: str, query: str) -> Dict[str, Any]:
    """Complete RAG pipeline API"""
    orchestrator = create_rag_orchestrator()
    result = await orchestrator.full_rag_pipeline(corpus_path, query)
    return result.model_dump()

# Export simplified interface
__all__ = [
    "SimplifiedRAGOrchestrator",
    "create_rag_orchestrator", 
    "analyze_corpus",
    "search_knowledge",
    "complete_rag",
    "RAGOrchestrationResult"
]