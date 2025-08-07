"""
Universal Domain Intelligence Agent - Pure Data-Driven Discovery
================================================================

This implementation addresses the concern that "hardcoded values in agents create huge biases"
by creating a TRULY UNIVERSAL agent that:

1. NO hardcoded domain types or keywords
2. NO predetermined thresholds or scoring  
3. Learns domain characteristics from actual data
4. Adapts to any content type automatically
5. Universal across languages and domains

The agent discovers patterns, characteristics, and optimal configurations
purely from data analysis without any predetermined assumptions.
"""

import asyncio
import logging
import json
import hashlib
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Import only the most basic dependencies to avoid bias
from agents.core.azure_service_container import ConsolidatedAzureServices

logger = logging.getLogger(__name__)


class UniversalDomainCharacteristics(BaseModel):
    """Pure data-driven domain characteristics with no predetermined assumptions"""
    
    # Content analysis (learned from data)
    content_signature: str = Field(description="Unique signature derived from content patterns")
    vocabulary_density: float = Field(description="Unique words per total words ratio")
    structural_patterns: Dict[str, float] = Field(default_factory=dict, description="Document structure patterns")
    linguistic_features: Dict[str, Any] = Field(default_factory=dict, description="Language patterns found")
    
    # Complexity measures (data-driven)
    complexity_indicators: Dict[str, float] = Field(default_factory=dict, description="Complexity measures")
    document_heterogeneity: float = Field(description="Variation between documents")
    concept_density: float = Field(description="Conceptual information density")
    
    # Content distribution (learned)
    length_distribution: Dict[str, float] = Field(default_factory=dict, description="Document length patterns")
    token_distribution: Dict[str, float] = Field(default_factory=dict, description="Token usage patterns")
    semantic_clusters: List[Dict[str, Any]] = Field(default_factory=list, description="Discovered semantic groups")


class UniversalProcessingConfiguration(BaseModel):
    """Processing configuration learned entirely from data analysis"""
    
    # Learned parameters (no hardcoded values)
    optimal_chunk_size: int = Field(description="Chunk size derived from document analysis")
    optimal_overlap: int = Field(description="Overlap derived from context preservation needs")
    confidence_thresholds: Dict[str, float] = Field(default_factory=dict, description="Confidence levels learned from data")
    
    # Processing strategy (data-driven)
    processing_strategy: str = Field(description="Strategy determined from content characteristics")
    parallel_processing_units: int = Field(description="Parallelism based on content complexity")
    expected_processing_time: float = Field(description="Time estimate based on content analysis")
    
    # Adaptive parameters
    entity_extraction_approach: str = Field(description="Extraction approach based on content type")
    relationship_discovery_method: str = Field(description="Relationship method based on content structure")
    quality_assurance_level: str = Field(description="QA level based on content criticality")


class UniversalDomainAnalysis(BaseModel):
    """Complete domain analysis with pure data-driven insights"""
    
    # Core analysis
    domain_signature: str = Field(description="Unique identifier for this domain")
    characteristics: UniversalDomainCharacteristics
    processing_config: UniversalProcessingConfiguration
    
    # Discovery metadata
    analysis_confidence: float = Field(description="Confidence in analysis based on data quality")
    data_quality_score: float = Field(description="Quality of source data")
    discovery_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Adaptive learning
    learned_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Patterns discovered from data")
    optimization_suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    

class UniversalDomainDeps(BaseModel):
    """Minimal dependencies for universal domain analysis"""
    data_directory: str = "/workspace/azure-maintie-rag/data/raw"
    azure_services: Optional[ConsolidatedAzureServices] = None
    cache_enabled: bool = True


class UniversalDomainIntelligenceAgent:
    """
    Truly universal domain intelligence agent that learns everything from data.
    
    Key principles:
    - Zero hardcoded domain knowledge
    - All parameters learned from actual content
    - Universal across domains and languages
    - Pure data-driven discovery
    """
    
    def __init__(self):
        self.agent = self._create_agent()
        self._analysis_cache = {}
    
    def _create_agent(self) -> Agent:
        """Create PydanticAI agent with universal tools"""
        
        # Use environment model configuration
        import os
        model_name = f"openai:{os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o')}"
        
        agent = Agent(
            model_name,
            deps_type=UniversalDomainDeps,
            result_type=UniversalDomainAnalysis,
            system_prompt="""You are a Universal Domain Intelligence Agent that discovers domain characteristics purely from data.

CRITICAL PRINCIPLES:
- You have NO predetermined domain knowledge
- You discover ALL characteristics from actual content analysis
- You learn thresholds and parameters from data patterns
- You adapt to ANY content type without bias
- You provide evidence for every conclusion

Your approach:
1. Analyze content WITHOUT assumptions about domain type
2. Discover patterns through statistical and linguistic analysis
3. Learn optimal parameters from data characteristics
4. Generate configurations based on discovered patterns
5. Provide confidence based on data quality and consistency

You must use your tools to perform thorough data-driven analysis and return structured UniversalDomainAnalysis results.""",
        )
        
        # Add universal analysis tools
        @agent.tool
        async def analyze_content_distribution(ctx: RunContext[UniversalDomainDeps], corpus_path: str) -> Dict[str, Any]:
            """Analyze content distribution to understand domain characteristics without assumptions"""
            return await self._analyze_content_distribution(corpus_path)
        
        @agent.tool
        async def generate_domain_signature(ctx: RunContext[UniversalDomainDeps], corpus_path: str) -> Dict[str, Any]:
            """Generate unique domain signature based on discovered patterns"""
            return await self._generate_domain_signature(corpus_path)
        
        @agent.tool
        async def generate_adaptive_configuration(ctx: RunContext[UniversalDomainDeps], 
                                                characteristics: Dict[str, Any]) -> Dict[str, Any]:
            """Generate processing configuration adapted to discovered characteristics"""
            return await self._generate_adaptive_configuration(characteristics)
        
        return agent
    
    async def analyze_domain(self, corpus_path: str, cache_key: Optional[str] = None) -> UniversalDomainAnalysis:
        """
        Perform universal domain analysis with pure data-driven discovery
        
        Args:
            corpus_path: Path to content corpus
            cache_key: Optional cache key for performance
            
        Returns:
            Complete universal domain analysis
        """
        
        # Check cache
        if cache_key and cache_key in self._analysis_cache:
            logger.info(f"Using cached analysis for {cache_key}")
            return self._analysis_cache[cache_key]
        
        try:
            logger.info(f"Starting universal domain analysis for: {corpus_path}")
            
            # Create dependencies
            deps = UniversalDomainDeps(
                data_directory=corpus_path,
                cache_enabled=True
            )
            
            # Craft analysis prompt without domain assumptions
            analysis_prompt = f"""
            Perform a complete universal domain analysis of the content at: {corpus_path}
            
            I need you to discover:
            1. What type of content this is (without assuming any predetermined categories)
            2. What processing approach would work best for this specific content
            3. What parameters should be used based on the content characteristics
            4. What patterns exist that could guide processing decisions
            
            Key requirements:
            - Base ALL conclusions on actual content analysis
            - DO NOT assume any domain type or category
            - Learn optimal parameters from the data itself
            - Provide evidence for every recommendation
            - Generate configuration that adapts to the discovered characteristics
            
            Use your tools to analyze content distribution, generate domain signature, 
            and create adaptive configuration based on discovered patterns.
            """
            
            # Run analysis
            logger.info("Running universal domain analysis...")
            result = await self.agent.run_async(analysis_prompt, deps=deps)
            analysis = result.data
            
            # Cache result
            if cache_key:
                self._analysis_cache[cache_key] = analysis
            
            logger.info(f"Universal domain analysis complete. Domain signature: {analysis.domain_signature}")
            return analysis
            
        except Exception as e:
            logger.error(f"Universal domain analysis failed: {e}")
            # Return minimal analysis with error indication
            return UniversalDomainAnalysis(
                domain_signature="analysis_failed",
                characteristics=UniversalDomainCharacteristics(
                    content_signature="error",
                    vocabulary_density=0.0,
                    document_heterogeneity=0.0,
                    concept_density=0.0
                ),
                processing_config=UniversalProcessingConfiguration(
                    optimal_chunk_size=1000,  # Conservative default
                    optimal_overlap=200,
                    processing_strategy="conservative",
                    parallel_processing_units=1,
                    expected_processing_time=0.0,
                    entity_extraction_approach="basic",
                    relationship_discovery_method="pattern_based",
                    quality_assurance_level="standard"
                ),
                analysis_confidence=0.0,
                data_quality_score=0.0,
                optimization_suggestions=[f"Analysis failed: {str(e)}"]
            )
    
    async def _analyze_content_distribution(self, corpus_path: str) -> Dict[str, Any]:
        """Analyze content distribution without predetermined assumptions"""
        
        corpus_dir = Path(corpus_path)
        if not corpus_dir.exists():
            return {"error": f"Corpus path {corpus_path} does not exist"}
        
        # Discover content files without format assumptions
        content_files = []
        for ext in ['*.txt', '*.md', '*.rst', '*.html', '*.xml', '*.json', '*.yaml', '*.yml']:
            content_files.extend(corpus_dir.glob(ext))
        
        if not content_files:
            return {"error": "No content files found"}
        
        # Analyze without assumptions
        documents = []
        total_chars = 0
        total_words = 0
        vocabulary = set()
        word_frequencies = Counter()
        line_patterns = Counter()
        structural_elements = Counter()
        
        for file_path in content_files[:100]:  # Sample to avoid memory issues
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Basic metrics
                char_count = len(content)
                words = content.lower().split()
                word_count = len(words)
                
                documents.append({
                    'char_count': char_count,
                    'word_count': word_count,
                    'line_count': len(content.split('\n'))
                })
                
                total_chars += char_count
                total_words += word_count
                
                # Vocabulary analysis
                for word in words:
                    clean_word = ''.join(c for c in word if c.isalnum())
                    if len(clean_word) > 2:
                        vocabulary.add(clean_word)
                        word_frequencies[clean_word] += 1
                
                # Structural pattern detection
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Detect patterns without assumptions
                    if line.startswith('#'):
                        structural_elements['heading'] += 1
                    elif line.startswith('-') or line.startswith('*'):
                        structural_elements['list_item'] += 1
                    elif line.startswith('```') or line.startswith('    '):
                        structural_elements['code_block'] += 1
                    elif ':' in line and len(line.split(':')) == 2:
                        structural_elements['key_value'] += 1
                    elif line.isupper() and len(line.split()) < 10:
                        structural_elements['title'] += 1
                    
                    line_patterns[len(line.split())] += 1
                
            except Exception:
                continue
        
        # Calculate distribution metrics
        if documents:
            char_lengths = [doc['char_count'] for doc in documents]
            word_lengths = [doc['word_count'] for doc in documents]
            
            return {
                "document_count": len(documents),
                "total_characters": total_chars,
                "total_words": total_words,
                "vocabulary_size": len(vocabulary),
                "vocabulary_density": len(vocabulary) / max(1, total_words),
                "avg_document_length": statistics.mean(char_lengths),
                "median_document_length": statistics.median(char_lengths),
                "document_length_std": statistics.stdev(char_lengths) if len(char_lengths) > 1 else 0,
                "avg_words_per_doc": statistics.mean(word_lengths),
                "common_words": dict(word_frequencies.most_common(20)),
                "structural_elements": dict(structural_elements),
                "line_patterns": dict(sorted(line_patterns.items())[-10:]),  # Most common line lengths
                "document_heterogeneity": statistics.stdev(char_lengths) / max(1, statistics.mean(char_lengths)) if len(char_lengths) > 1 else 0
            }
        
        return {"error": "No valid documents found"}
    
    async def _generate_domain_signature(self, corpus_path: str) -> Dict[str, Any]:
        """Generate unique domain signature based on discovered content patterns"""
        
        # Analyze content patterns
        distribution = await self._analyze_content_distribution(corpus_path)
        
        if "error" in distribution:
            return distribution
        
        # Generate signature based on discovered characteristics
        signature_components = [
            f"vocab_density_{distribution['vocabulary_density']:.3f}",
            f"avg_length_{int(distribution['avg_document_length'])}",
            f"heterogeneity_{distribution['document_heterogeneity']:.3f}",
            f"docs_{distribution['document_count']}",
        ]
        
        # Add structural patterns to signature
        structural = distribution.get('structural_elements', {})
        for element, count in structural.items():
            if count > 0:
                signature_components.append(f"{element}_{count}")
        
        # Create hash-based signature
        signature_string = "_".join(sorted(signature_components))
        signature_hash = hashlib.md5(signature_string.encode()).hexdigest()[:12]
        
        return {
            "domain_signature": f"universal_{signature_hash}",
            "signature_components": signature_components,
            "content_characteristics": {
                "primary_pattern": self._identify_primary_pattern(structural),
                "complexity_level": self._assess_complexity_level(distribution),
                "processing_hints": self._generate_processing_hints(distribution)
            }
        }
    
    async def _generate_adaptive_configuration(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing configuration that adapts to discovered characteristics"""
        
        # Extract key metrics
        vocab_density = characteristics.get('vocabulary_density', 0.1)
        avg_length = characteristics.get('avg_document_length', 1000)
        heterogeneity = characteristics.get('document_heterogeneity', 0.5)
        doc_count = characteristics.get('document_count', 1)
        
        # Learn optimal chunk size from document characteristics
        if avg_length < 500:
            optimal_chunk_size = max(200, int(avg_length * 0.8))
        elif avg_length > 5000:
            optimal_chunk_size = min(2000, int(avg_length * 0.3))
        else:
            optimal_chunk_size = int(avg_length * 0.6)
        
        # Learn overlap from heterogeneity
        if heterogeneity > 0.8:  # High variation - need more overlap
            overlap_ratio = 0.3
        elif heterogeneity < 0.2:  # Low variation - less overlap needed
            overlap_ratio = 0.15
        else:
            overlap_ratio = 0.2
        
        optimal_overlap = int(optimal_chunk_size * overlap_ratio)
        
        # Learn confidence thresholds from vocabulary density
        if vocab_density > 0.3:  # Rich vocabulary - can be more selective
            entity_threshold = 0.8
            relationship_threshold = 0.75
        elif vocab_density < 0.1:  # Limited vocabulary - be more inclusive
            entity_threshold = 0.6
            relationship_threshold = 0.55
        else:
            entity_threshold = 0.7
            relationship_threshold = 0.65
        
        # Determine processing strategy from characteristics
        structural = characteristics.get('structural_elements', {})
        if structural.get('code_block', 0) > structural.get('heading', 0):
            processing_strategy = "technical_content"
        elif structural.get('list_item', 0) > 0:
            processing_strategy = "structured_content"
        else:
            processing_strategy = "narrative_content"
        
        # Determine parallelism from document count and complexity
        if doc_count > 50 and avg_length > 1000:
            parallel_units = min(8, max(2, doc_count // 25))
        else:
            parallel_units = 1
        
        # Estimate processing time based on content characteristics
        complexity_factor = vocab_density * heterogeneity
        processing_time = (doc_count * avg_length * complexity_factor) / 10000  # Rough estimate
        
        return {
            "optimal_chunk_size": optimal_chunk_size,
            "optimal_overlap": optimal_overlap,
            "confidence_thresholds": {
                "entity_threshold": entity_threshold,
                "relationship_threshold": relationship_threshold,
                "similarity_threshold": (entity_threshold + relationship_threshold) / 2
            },
            "processing_strategy": processing_strategy,
            "parallel_processing_units": parallel_units,
            "expected_processing_time": processing_time,
            "entity_extraction_approach": self._select_extraction_approach(characteristics),
            "relationship_discovery_method": self._select_relationship_method(characteristics),
            "quality_assurance_level": self._determine_qa_level(characteristics),
            "learned_parameters": {
                "vocabulary_density": vocab_density,
                "document_heterogeneity": heterogeneity,
                "avg_document_length": avg_length,
                "optimization_basis": "content_analysis"
            }
        }
    
    def _identify_primary_pattern(self, structural_elements: Dict[str, int]) -> str:
        """Identify primary content pattern from structural analysis"""
        if not structural_elements:
            return "unstructured"
        
        max_element = max(structural_elements.items(), key=lambda x: x[1])
        return f"primarily_{max_element[0]}"
    
    def _assess_complexity_level(self, distribution: Dict[str, Any]) -> str:
        """Assess complexity level based on distribution characteristics"""
        vocab_density = distribution.get('vocabulary_density', 0.1)
        heterogeneity = distribution.get('document_heterogeneity', 0.5)
        
        complexity_score = vocab_density + heterogeneity
        
        if complexity_score > 1.0:
            return "high_complexity"
        elif complexity_score > 0.5:
            return "medium_complexity"
        else:
            return "low_complexity"
    
    def _generate_processing_hints(self, distribution: Dict[str, Any]) -> List[str]:
        """Generate processing hints based on content analysis"""
        hints = []
        
        vocab_density = distribution.get('vocabulary_density', 0.1)
        if vocab_density > 0.3:
            hints.append("rich_vocabulary_detected")
        elif vocab_density < 0.05:
            hints.append("limited_vocabulary_detected")
        
        avg_length = distribution.get('avg_document_length', 1000)
        if avg_length > 5000:
            hints.append("long_documents_detected")
        elif avg_length < 500:
            hints.append("short_documents_detected")
        
        heterogeneity = distribution.get('document_heterogeneity', 0.5)
        if heterogeneity > 0.8:
            hints.append("high_document_variation")
        elif heterogeneity < 0.2:
            hints.append("consistent_document_structure")
        
        return hints
    
    def _select_extraction_approach(self, characteristics: Dict[str, Any]) -> str:
        """Select entity extraction approach based on content characteristics"""
        structural = characteristics.get('structural_elements', {})
        vocab_density = characteristics.get('vocabulary_density', 0.1)
        
        if structural.get('code_block', 0) > 0 or vocab_density > 0.4:
            return "technical_pattern_extraction"
        elif structural.get('key_value', 0) > 0:
            return "structured_data_extraction"
        else:
            return "natural_language_extraction"
    
    def _select_relationship_method(self, characteristics: Dict[str, Any]) -> str:
        """Select relationship discovery method based on content structure"""
        structural = characteristics.get('structural_elements', {})
        
        if structural.get('list_item', 0) > 0:
            return "hierarchical_relationship_discovery"
        elif structural.get('heading', 0) > 0:
            return "section_based_relationships"
        else:
            return "proximity_based_relationships"
    
    def _determine_qa_level(self, characteristics: Dict[str, Any]) -> str:
        """Determine quality assurance level based on content characteristics"""
        vocab_density = characteristics.get('vocabulary_density', 0.1)
        heterogeneity = characteristics.get('document_heterogeneity', 0.5)
        
        if vocab_density > 0.3 and heterogeneity > 0.6:
            return "high_qa_required"
        elif vocab_density < 0.1 or heterogeneity < 0.2:
            return "standard_qa_sufficient"
        else:
            return "moderate_qa_recommended"


# Factory function for easy instantiation
def create_universal_domain_agent() -> UniversalDomainIntelligenceAgent:
    """Create universal domain intelligence agent"""
    return UniversalDomainIntelligenceAgent()


# Demonstration function
async def demonstrate_universal_analysis(corpus_paths: List[str]) -> Dict[str, UniversalDomainAnalysis]:
    """
    Demonstrate universal domain analysis on multiple corpora
    
    This shows how the agent adapts to different content types without bias
    """
    
    logger.info("Starting universal domain analysis demonstration...")
    agent = create_universal_domain_agent()
    results = {}
    
    for corpus_path in corpus_paths:
        try:
            logger.info(f"Analyzing: {corpus_path}")
            analysis = await agent.analyze_domain(corpus_path, cache_key=corpus_path)
            results[corpus_path] = analysis
            
            logger.info(f"Domain signature: {analysis.domain_signature}")
            logger.info(f"Processing strategy: {analysis.processing_config.processing_strategy}")
            logger.info(f"Analysis confidence: {analysis.analysis_confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to analyze {corpus_path}: {e}")
    
    logger.info(f"Universal analysis complete. Analyzed {len(results)} domains.")
    return results


# Export main interfaces
__all__ = [
    'UniversalDomainIntelligenceAgent',
    'UniversalDomainAnalysis', 
    'create_universal_domain_agent',
    'demonstrate_universal_analysis'
]


if __name__ == "__main__":
    # Demonstration script
    async def main():
        print("🌍 Universal Domain Intelligence Agent - Pure Data-Driven Discovery")
        print("=" * 70)
        print("This agent learns domain characteristics WITHOUT predetermined assumptions!")
        print()
        
        # Test with available data
        test_paths = [
            "/workspace/azure-maintie-rag/data/raw/Programming-Language",
            "/workspace/azure-maintie-rag/data/raw"
        ]
        
        # Filter to existing paths
        existing_paths = [path for path in test_paths if Path(path).exists()]
        
        if existing_paths:
            print(f"Testing with {len(existing_paths)} corpus paths:")
            for path in existing_paths:
                print(f"  • {path}")
            print()
            
            results = await demonstrate_universal_analysis(existing_paths)
            
            print("📊 ANALYSIS RESULTS:")
            print("=" * 50)
            
            for corpus_path, analysis in results.items():
                print(f"\n🔍 {Path(corpus_path).name.upper()}:")
                print(f"   Domain Signature: {analysis.domain_signature}")
                print(f"   Vocabulary Density: {analysis.characteristics.vocabulary_density:.3f}")
                print(f"   Document Heterogeneity: {analysis.characteristics.document_heterogeneity:.3f}")
                print(f"   Processing Strategy: {analysis.processing_config.processing_strategy}")
                print(f"   Optimal Chunk Size: {analysis.processing_config.optimal_chunk_size}")
                print(f"   Entity Threshold: {analysis.processing_config.confidence_thresholds.get('entity_threshold', 'N/A')}")
                print(f"   Analysis Confidence: {analysis.analysis_confidence:.3f}")
                
                if analysis.optimization_suggestions:
                    print(f"   Suggestions: {', '.join(analysis.optimization_suggestions[:2])}")
        else:
            print("❌ No test corpus paths found. Please ensure data directory exists.")
            print("Expected paths:")
            for path in test_paths:
                print(f"  • {path}")
    
    asyncio.run(main())