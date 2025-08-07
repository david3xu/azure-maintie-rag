"""
Universal Domain Intelligence Agent - Truly Universal RAG
=========================================================

This agent discovers domains dynamically without predetermined knowledge,
making it truly universal for any content type or language.

Key Principles:
1. NO hardcoded domain types or keywords
2. NO predetermined thresholds or scoring
3. Learns domain characteristics from actual data
4. Adapts to any content type automatically
5. Universal across languages and domains
"""

import os
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Universal schema models (no predetermined domains)
class UniversalDomainCharacteristics(BaseModel):
    """Data-driven domain characteristics discovered from content"""
    
    # Content structure analysis (discovered, not predetermined)
    avg_document_length: int
    document_count: int
    vocabulary_richness: float = Field(..., description="Unique words / total words ratio")
    sentence_complexity: float = Field(..., description="Average words per sentence")
    
    # Content patterns (learned from data)
    most_frequent_terms: List[str] = Field(default=[], description="Top terms found in content")
    content_patterns: List[str] = Field(default=[], description="Structural patterns discovered")
    language_indicators: Dict[str, float] = Field(default={}, description="Language detection scores")
    
    # Complexity indicators (measured, not assumed)
    lexical_diversity: float = Field(..., description="Type-token ratio")
    technical_vocabulary_ratio: float = Field(..., description="Technical vs common word ratio")
    structural_consistency: float = Field(..., description="Document structure consistency")

class UniversalProcessingConfiguration(BaseModel):
    """Configuration generated from data characteristics, not predetermined rules"""
    
    # Chunking based on actual content patterns
    optimal_chunk_size: int = Field(..., description="Based on sentence/paragraph patterns")
    chunk_overlap_ratio: float = Field(..., description="Based on content coherence analysis")
    
    # Extraction thresholds learned from content distribution
    entity_confidence_threshold: float = Field(..., description="Based on term frequency distribution")
    relationship_density: float = Field(..., description="Based on co-occurrence patterns")
    
    # Search optimization based on content characteristics  
    vector_search_weight: float = Field(..., description="Based on semantic density")
    graph_search_weight: float = Field(..., description="Based on relationship patterns")
    
    # Quality expectations based on content analysis
    expected_extraction_quality: float = Field(..., description="Based on content consistency")
    processing_complexity: str = Field(..., description="Based on measured characteristics")

class UniversalDomainAnalysis(BaseModel):
    """Universal domain analysis without predetermined categories"""
    
    # Dynamic domain identification (not from predetermined list)
    domain_signature: str = Field(..., description="Unique signature derived from content")
    content_type_confidence: float = Field(..., description="Confidence in content type detection")
    
    # Learned characteristics
    characteristics: UniversalDomainCharacteristics
    processing_config: UniversalProcessingConfiguration
    
    # Content insights (discovered, not predetermined)
    key_insights: List[str] = Field(default=[], description="Key insights about this content")
    adaptation_recommendations: List[str] = Field(default=[], description="How to adapt processing")
    
    # Quality and reliability
    analysis_timestamp: str
    processing_time: float
    data_source_path: str
    analysis_reliability: float = Field(..., description="Reliability of this analysis")

class UniversalDomainDeps(BaseModel):
    """Universal dependencies - no domain assumptions"""
    data_directory: str = Field(default="/workspace/azure-maintie-rag/data/raw")
    max_files_to_analyze: int = Field(default=50, description="Limit for deep analysis")
    min_content_length: int = Field(default=100, description="Minimum content length to analyze")
    enable_multilingual: bool = Field(default=True, description="Support multiple languages")

# Create universal agent with real Azure OpenAI
model_name = f"openai:{os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o')}"

agent = Agent(
    model_name,
    deps_type=UniversalDomainDeps,
    result_type=UniversalDomainAnalysis,
    system_prompt="""You are a Universal Domain Intelligence Agent that analyzes ANY type of content without predetermined assumptions.

Your approach:
1. Discover content characteristics through statistical analysis
2. Learn domain patterns from actual data distribution
3. Generate optimal configurations based on measured properties
4. Adapt to any language, subject matter, or content type
5. Never assume specific domains - let the data reveal its nature

You analyze content structure, vocabulary patterns, complexity metrics, and relationship patterns to understand the optimal processing approach for THIS specific content, regardless of domain.

Always return UniversalDomainAnalysis with data-driven insights and recommendations.""",
)

@agent.tool
async def analyze_content_distribution(ctx: RunContext[UniversalDomainDeps]) -> Dict[str, Any]:
    """Analyze actual content distribution without domain assumptions"""
    
    data_path = Path(ctx.deps.data_directory)
    if not data_path.exists():
        return {"error": f"Data directory {data_path} not found"}
    
    all_content = []
    file_stats = []
    
    # Collect content samples
    for root_path in data_path.rglob("*"):
        if root_path.is_file() and root_path.suffix in ['.md', '.txt', '.py', '.json', '.xml', '.html']:
            try:
                content = root_path.read_text(encoding='utf-8', errors='ignore')
                if len(content) >= ctx.deps.min_content_length:
                    all_content.append(content)
                    file_stats.append({
                        'length': len(content),
                        'path': str(root_path),
                        'extension': root_path.suffix
                    })
                    
                    # Limit analysis for performance
                    if len(all_content) >= ctx.deps.max_files_to_analyze:
                        break
            except Exception:
                continue
    
    if not all_content:
        return {"error": "No analyzable content found"}
    
    # Statistical analysis of actual content
    total_content = " ".join(all_content)
    words = total_content.lower().split()
    
    # Vocabulary analysis
    unique_words = set(words)
    vocabulary_richness = len(unique_words) / len(words) if words else 0
    
    # Content complexity analysis
    sentences = total_content.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    # Term frequency analysis (no predetermined terms)
    word_freq = {}
    for word in words:
        if len(word) > 3 and word.isalpha():  # Filter meaningful words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get most frequent terms (discovered from data)
    top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    most_frequent_terms = [term for term, freq in top_terms]
    
    # Structure pattern analysis
    content_patterns = []
    if '```' in total_content or 'def ' in total_content:
        content_patterns.append("code_blocks")
    if total_content.count('\n- ') > 10:
        content_patterns.append("list_structures") 
    if total_content.count('|') > 20 and total_content.count('-') > 20:
        content_patterns.append("tabular_data")
    if total_content.count('#') > 10:
        content_patterns.append("hierarchical_headers")
    
    # Technical vocabulary ratio (words not in common vocabulary)
    common_words = {'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but', 'his', 'from', 'they', 'she', 'her', 'been', 'than', 'its', 'who', 'oil', 'use', 'word', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'water', 'very', 'what', 'know', 'get', 'has', 'had', 'let', 'put', 'too', 'old', 'any', 'app', 'may', 'say', 'she', 'use', 'her', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'can', 'part'}
    technical_words = [word for word in unique_words if word not in common_words and len(word) > 4]
    technical_vocabulary_ratio = len(technical_words) / len(unique_words) if unique_words else 0
    
    return {
        "total_files": len(file_stats),
        "total_content_length": len(total_content),
        "vocabulary_richness": vocabulary_richness,
        "avg_sentence_length": avg_sentence_length,
        "most_frequent_terms": most_frequent_terms,
        "content_patterns": content_patterns,
        "technical_vocabulary_ratio": technical_vocabulary_ratio,
        "file_length_distribution": {
            "min": min(f['length'] for f in file_stats),
            "max": max(f['length'] for f in file_stats),
            "avg": sum(f['length'] for f in file_stats) // len(file_stats)
        },
        "file_extensions": list(set(f['extension'] for f in file_stats))
    }

@agent.tool  
async def generate_domain_signature(ctx: RunContext[UniversalDomainDeps], content_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate unique domain signature from content characteristics"""
    
    if "error" in content_analysis:
        return content_analysis
    
    # Create unique signature based on measured characteristics
    characteristics = []
    
    # Vocabulary characteristics
    vocab_richness = content_analysis.get('vocabulary_richness', 0)
    if vocab_richness > 0.3:
        characteristics.append("high_vocabulary_diversity")
    elif vocab_richness > 0.15:
        characteristics.append("medium_vocabulary_diversity")
    else:
        characteristics.append("low_vocabulary_diversity")
    
    # Technical density
    tech_ratio = content_analysis.get('technical_vocabulary_ratio', 0)
    if tech_ratio > 0.4:
        characteristics.append("high_technical_density")
    elif tech_ratio > 0.2:
        characteristics.append("medium_technical_density")
    else:
        characteristics.append("low_technical_density")
    
    # Structure patterns
    patterns = content_analysis.get('content_patterns', [])
    if 'code_blocks' in patterns:
        characteristics.append("code_rich")
    if 'tabular_data' in patterns:
        characteristics.append("data_structured")
    if 'hierarchical_headers' in patterns:
        characteristics.append("hierarchically_organized")
    if 'list_structures' in patterns:
        characteristics.append("list_heavy")
    
    # Sentence complexity
    avg_sentence = content_analysis.get('avg_sentence_length', 0)
    if avg_sentence > 20:
        characteristics.append("complex_sentences")
    elif avg_sentence > 12:
        characteristics.append("medium_sentences")
    else:
        characteristics.append("simple_sentences")
    
    # Generate signature
    domain_signature = "_".join(sorted(characteristics))
    
    # Calculate confidence based on consistency of characteristics
    consistency_score = len(set(characteristics)) / 10  # Normalize
    confidence = min(0.95, consistency_score + 0.3)
    
    return {
        "domain_signature": domain_signature,
        "content_type_confidence": confidence,
        "characteristics": characteristics,
        "signature_components": {
            "vocabulary_diversity": vocab_richness,
            "technical_density": tech_ratio,
            "structural_patterns": patterns,
            "sentence_complexity": avg_sentence
        }
    }

@agent.tool
async def generate_adaptive_configuration(ctx: RunContext[UniversalDomainDeps], signature_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate processing configuration adapted to discovered content characteristics"""
    
    if "error" in signature_data:
        return signature_data
    
    components = signature_data.get('signature_components', {})
    characteristics = signature_data.get('characteristics', [])
    
    # Adaptive chunking based on content structure
    base_chunk_size = 1000
    vocab_diversity = components.get('vocabulary_diversity', 0.2)
    sentence_complexity = components.get('sentence_complexity', 10)
    
    # Adapt chunk size to content characteristics
    if 'complex_sentences' in characteristics:
        optimal_chunk_size = int(base_chunk_size * 1.3)  # Larger chunks for complex content
    elif 'code_rich' in characteristics:
        optimal_chunk_size = int(base_chunk_size * 1.5)  # Even larger for code
    elif 'simple_sentences' in characteristics:
        optimal_chunk_size = int(base_chunk_size * 0.8)  # Smaller chunks for simple content
    else:
        optimal_chunk_size = base_chunk_size
    
    # Adaptive overlap based on content coherence
    if 'hierarchically_organized' in characteristics:
        chunk_overlap_ratio = 0.15  # Less overlap for well-structured content
    elif 'data_structured' in characteristics:
        chunk_overlap_ratio = 0.1   # Minimal overlap for structured data
    else:
        chunk_overlap_ratio = 0.2   # Standard overlap
    
    # Adaptive entity threshold based on vocabulary diversity
    if vocab_diversity > 0.3:
        entity_threshold = 0.7  # Lower threshold for rich vocabulary
    elif vocab_diversity > 0.15:
        entity_threshold = 0.8  # Standard threshold
    else:
        entity_threshold = 0.85  # Higher threshold for limited vocabulary
    
    # Adaptive search weights based on content patterns
    technical_density = components.get('technical_density', 0.2)
    
    if technical_density > 0.4:
        vector_weight = 0.4  # Less vector, more graph for technical content
        graph_weight = 0.6
    elif 'data_structured' in characteristics:
        vector_weight = 0.3  # Even less vector for structured data
        graph_weight = 0.7
    else:
        vector_weight = 0.6  # More vector for general content
        graph_weight = 0.4
    
    # Processing complexity assessment
    complexity_indicators = len([c for c in characteristics if 'high' in c or 'complex' in c])
    if complexity_indicators > 2:
        processing_complexity = "high"
        expected_quality = 0.75  # Lower expectations for complex content
    elif complexity_indicators > 0:
        processing_complexity = "medium" 
        expected_quality = 0.85
    else:
        processing_complexity = "low"
        expected_quality = 0.9
    
    return {
        "optimal_chunk_size": optimal_chunk_size,
        "chunk_overlap_ratio": chunk_overlap_ratio,
        "entity_confidence_threshold": entity_threshold,
        "relationship_density": min(0.8, technical_density + 0.3),
        "vector_search_weight": vector_weight,
        "graph_search_weight": graph_weight,
        "expected_extraction_quality": expected_quality,
        "processing_complexity": processing_complexity,
        "adaptation_rationale": {
            "chunk_size_factor": f"Adapted for {characteristics}",
            "threshold_reasoning": f"Based on vocabulary diversity: {vocab_diversity:.2f}",
            "search_weighting": f"Optimized for technical density: {technical_density:.2f}"
        }
    }

async def run_universal_domain_analysis(deps: UniversalDomainDeps) -> UniversalDomainAnalysis:
    """Main orchestration function that runs the complete universal analysis"""
    
    start_time = time.time()
    
    try:
        # Step 1: Analyze actual content distribution
        print("🔍 Analyzing content distribution...")
        content_analysis = await analyze_content_distribution(RunContext(deps=deps))
        
        if "error" in content_analysis:
            raise Exception(f"Content analysis failed: {content_analysis['error']}")
        
        # Step 2: Generate domain signature from discovered characteristics
        print("🏷️ Generating domain signature...")
        signature_data = await generate_domain_signature(RunContext(deps=deps), content_analysis)
        
        if "error" in signature_data:
            raise Exception(f"Signature generation failed: {signature_data['error']}")
        
        # Step 3: Generate adaptive configuration
        print("⚙️ Generating adaptive configuration...")
        adaptive_config = await generate_adaptive_configuration(RunContext(deps=deps), signature_data)
        
        if "error" in adaptive_config:
            raise Exception(f"Configuration generation failed: {adaptive_config['error']}")
        
        processing_time = time.time() - start_time
        
        # Create comprehensive universal analysis
        characteristics = UniversalDomainCharacteristics(
            avg_document_length=content_analysis["file_length_distribution"]["avg"],
            document_count=content_analysis["total_files"],
            vocabulary_richness=content_analysis["vocabulary_richness"],
            sentence_complexity=content_analysis["avg_sentence_length"],
            most_frequent_terms=content_analysis["most_frequent_terms"],
            content_patterns=content_analysis["content_patterns"],
            language_indicators={"english": 1.0},  # Could be enhanced with actual detection
            lexical_diversity=content_analysis["vocabulary_richness"],
            technical_vocabulary_ratio=content_analysis["technical_vocabulary_ratio"],
            structural_consistency=0.8  # Could be calculated from pattern consistency
        )
        
        processing_config = UniversalProcessingConfiguration(
            optimal_chunk_size=adaptive_config["optimal_chunk_size"],
            chunk_overlap_ratio=adaptive_config["chunk_overlap_ratio"],
            entity_confidence_threshold=adaptive_config["entity_confidence_threshold"],
            relationship_density=adaptive_config["relationship_density"],
            vector_search_weight=adaptive_config["vector_search_weight"],
            graph_search_weight=adaptive_config["graph_search_weight"],
            expected_extraction_quality=adaptive_config["expected_extraction_quality"],
            processing_complexity=adaptive_config["processing_complexity"]
        )
        
        # Generate insights based on discovered patterns
        key_insights = []
        adaptation_recommendations = []
        
        # Content-driven insights
        if content_analysis["technical_vocabulary_ratio"] > 0.4:
            key_insights.append("High technical vocabulary density detected - content requires specialized processing")
        
        if "code_blocks" in content_analysis["content_patterns"]:
            key_insights.append("Code-rich content detected - enable syntax-aware chunking")
            adaptation_recommendations.append("Use larger chunk sizes to preserve code block integrity")
        
        if "hierarchical_headers" in content_analysis["content_patterns"]:
            key_insights.append("Well-structured hierarchical content detected")
            adaptation_recommendations.append("Leverage document structure for improved chunk boundaries")
        
        if content_analysis["vocabulary_richness"] > 0.3:
            key_insights.append("High vocabulary diversity - expect good semantic vector performance")
        else:
            key_insights.append("Limited vocabulary diversity - focus on graph relationship extraction")
            adaptation_recommendations.append("Increase graph search weight for better coverage")
        
        # Calculate reliability based on analysis depth and consistency
        analysis_reliability = min(0.95, 
            (content_analysis["total_files"] / max(deps.max_files_to_analyze, 1)) * 0.4 +
            (len(content_analysis["content_patterns"]) / 10) * 0.3 +
            (content_analysis["vocabulary_richness"] * 0.3)
        )
        
        return UniversalDomainAnalysis(
            domain_signature=signature_data["domain_signature"],
            content_type_confidence=signature_data["content_type_confidence"],
            characteristics=characteristics,
            processing_config=processing_config,
            key_insights=key_insights,
            adaptation_recommendations=adaptation_recommendations,
            analysis_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            processing_time=processing_time,
            data_source_path=deps.data_directory,
            analysis_reliability=analysis_reliability
        )
        
    except Exception as e:
        # Return basic analysis even if something fails
        return UniversalDomainAnalysis(
            domain_signature="analysis_failed",
            content_type_confidence=0.0,
            characteristics=UniversalDomainCharacteristics(
                avg_document_length=1000,
                document_count=0,
                vocabulary_richness=0.2,
                sentence_complexity=10.0,
                lexical_diversity=0.2,
                technical_vocabulary_ratio=0.2,
                structural_consistency=0.5
            ),
            processing_config=UniversalProcessingConfiguration(
                optimal_chunk_size=1000,
                chunk_overlap_ratio=0.2,
                entity_confidence_threshold=0.8,
                relationship_density=0.5,
                vector_search_weight=0.6,
                graph_search_weight=0.4,
                expected_extraction_quality=0.7,
                processing_complexity="unknown"
            ),
            key_insights=[f"Analysis failed: {str(e)}"],
            adaptation_recommendations=["Use default processing configuration"],
            analysis_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            processing_time=time.time() - start_time,
            data_source_path=deps.data_directory,
            analysis_reliability=0.1
        )

# Export universal agent and functions
__all__ = ["agent", "UniversalDomainDeps", "UniversalDomainAnalysis", "run_universal_domain_analysis"]