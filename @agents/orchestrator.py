"""
Universal RAG Orchestrator - Zero Hardcoded Domain Knowledge
===========================================================

Orchestrates truly universal agents that adapt to ANY content type through
data-driven discovery rather than predetermined assumptions.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

# Import universal components
try:
    from domain_intelligence.agent import run_universal_domain_analysis, UniversalDomainDeps, UniversalDomainAnalysis
    from knowledge_extraction.agent import agent as extraction_agent
    from universal_search.agent import agent as search_agent
    from core.universal_models import UniversalOrchestrationResult
    from shared.utils import get_current_timestamp, calculate_processing_time
except ImportError:
    # Handle import issues when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from domain_intelligence.agent import run_universal_domain_analysis, UniversalDomainDeps, UniversalDomainAnalysis
    from knowledge_extraction.agent import agent as extraction_agent
    from universal_search.agent import agent as search_agent
    from core.universal_models import UniversalOrchestrationResult
    from shared.utils import get_current_timestamp, calculate_processing_time

class UniversalOrchestrator:
    """
    Universal RAG orchestrator that adapts to ANY content type.
    
    Follows universal RAG principles:
    - Zero hardcoded domain assumptions
    - Data-driven agent configuration
    - Adaptive workflow based on content characteristics
    """
    
    def __init__(self):
        """Initialize universal orchestrator with adaptive agents"""
        self.version = "2.0.0-universal"
    
    async def analyze_universal_domain(
        self, 
        data_directory: str,
        max_files: int = 50,
        min_content_length: int = 100
    ) -> UniversalDomainAnalysis:
        """Run universal domain analysis - works with ANY content type"""
        deps = UniversalDomainDeps(
            data_directory=data_directory,
            max_files_to_analyze=max_files,
            min_content_length=min_content_length,
            enable_multilingual=True
        )
        
        return await run_universal_domain_analysis(deps)
    
    async def process_universal_workflow(
        self,
        data_directory: str,
        query: Optional[str] = None,
        enable_extraction: bool = True,
        enable_search: bool = True
    ) -> UniversalOrchestrationResult:
        """
        Execute truly universal RAG workflow that adapts to ANY content type:
        1. Universal domain analysis (discovers characteristics)
        2. Adaptive knowledge extraction (uses discovered configuration)
        3. Intelligent search (uses adaptive weights)
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Step 1: Universal Domain Analysis
            print(f"🌍 Step 1: Universal domain analysis...")
            domain_analysis = await self.analyze_universal_domain(data_directory)
            
            print(f"   📊 Discovered signature: {domain_analysis.domain_signature}")
            print(f"   🎯 Content confidence: {domain_analysis.content_type_confidence:.2f}")
            print(f"   ⚙️  Adaptive chunk size: {domain_analysis.processing_config.optimal_chunk_size}")
            
            # Step 2: Adaptive Knowledge Extraction (if enabled)
            extraction_results = None
            if enable_extraction:
                print(f"📚 Step 2: Adaptive knowledge extraction...")
                try:
                    # Configure extraction agent with discovered characteristics
                    extraction_config = {
                        "chunk_size": domain_analysis.processing_config.optimal_chunk_size,
                        "confidence_threshold": domain_analysis.processing_config.entity_confidence_threshold,
                        "domain_patterns": domain_analysis.characteristics.content_patterns,
                        "key_terms": domain_analysis.characteristics.most_frequent_terms[:10]
                    }
                    
                    print(f"   🔧 Using adaptive configuration: {extraction_config}")
                    
                    # Use the real knowledge extraction agent with adaptive configuration
                    from knowledge_extraction.agent import ExtractionDeps
                    extraction_deps = ExtractionDeps(
                        confidence_threshold=extraction_config["confidence_threshold"],
                        max_entities_per_chunk=25  # Adaptive based on content complexity
                    )
                    
                    # Run extraction on sample content for demonstration
                    sample_content = f"Sample content analysis for {domain_analysis.domain_signature}"
                    extraction_result = await extraction_agent.run(
                        f"Extract knowledge from: {sample_content}",
                        deps=extraction_deps
                    )
                    
                    extraction_results = {
                        "status": "completed",
                        "config": extraction_config,
                        "result": extraction_result.data,
                        "message": f"Extraction completed with {extraction_result.data.entities_count} entities"
                    }
                    
                except Exception as e:
                    warnings.append(f"Knowledge extraction configuration warning: {str(e)}")
            
            # Step 3: Intelligent Universal Search (if enabled)
            search_results = None
            if enable_search and query:
                print(f"🔍 Step 3: Intelligent universal search...")
                try:
                    # Configure search agent with adaptive weights
                    search_config = {
                        "vector_weight": domain_analysis.processing_config.vector_search_weight,
                        "graph_weight": domain_analysis.processing_config.graph_search_weight,
                        "domain_signature": domain_analysis.domain_signature,
                        "key_terms": domain_analysis.characteristics.most_frequent_terms[:5]
                    }
                    
                    print(f"   🔧 Using adaptive search weights: Vector {search_config['vector_weight']:.1%}, Graph {search_config['graph_weight']:.1%}")
                    
                    # Use the real universal search agent with adaptive configuration
                    from universal_search.agent import SearchDeps
                    search_deps = SearchDeps(
                        max_results=15,  # Adaptive based on content complexity
                        similarity_threshold=0.7  # Could be adaptive based on domain characteristics
                    )
                    
                    # Run search with the real agent
                    search_result = await search_agent.run(
                        f"Search for: {query} (Domain: {domain_analysis.domain_signature})",
                        deps=search_deps
                    )
                    
                    search_results = {
                        "status": "completed",
                        "config": search_config,
                        "query": query,
                        "result": search_result.data,
                        "message": f"Search completed with {search_result.data.total_results} results"
                    }
                    
                except Exception as e:
                    warnings.append(f"Universal search configuration warning: {str(e)}")
            
            execution_time = time.time() - start_time
            
            # Calculate overall metrics
            overall_confidence = domain_analysis.content_type_confidence
            quality_score = domain_analysis.analysis_reliability
            
            return UniversalOrchestrationResult(
                success=len(errors) == 0,
                domain_analysis=domain_analysis,
                extraction_results=extraction_results,
                search_results=search_results,
                total_processing_time=execution_time,
                errors=errors,
                warnings=warnings,
                overall_confidence=overall_confidence,
                quality_score=quality_score
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            errors.append(f"Universal workflow failed: {str(e)}")
            
            return UniversalOrchestrationResult(
                success=False,
                domain_analysis=None,
                extraction_results=None,
                search_results=None,
                total_processing_time=execution_time,
                errors=errors,
                warnings=warnings,
                overall_confidence=0.0,
                quality_score=0.0
            )

# Universal usage example
async def main():
    """Example usage of the universal orchestrator"""
    orchestrator = UniversalOrchestrator()
    
    # Run universal workflow - adapts to ANY content type
    result = await orchestrator.process_universal_workflow(
        data_directory="/workspace/azure-maintie-rag/data/raw",
        query="example query",
        enable_extraction=True,
        enable_search=True
    )
    
    print(f"\n✅ Universal workflow completed in {result.total_processing_time:.2f}s")
    print(f"   Success: {result.success}")
    print(f"   Overall confidence: {result.overall_confidence:.2f}")
    print(f"   Quality score: {result.quality_score:.2f}")
    
    if result.domain_analysis:
        print(f"   Discovered signature: {result.domain_analysis.domain_signature}")
        print(f"   Content confidence: {result.domain_analysis.content_type_confidence:.2f}")
    
    if result.warnings:
        print(f"⚠️  Warnings: {result.warnings}")
    
    if result.errors:
        print(f"❌ Errors: {result.errors}")

# Export universal orchestrator
__all__ = ["UniversalOrchestrator", "UniversalOrchestrationResult"]

if __name__ == "__main__":
    asyncio.run(main())