"""
Hybrid Domain Analyzer - LLM + Statistical Models

This module combines Large Language Models with statistical analysis for optimal
domain configuration generation. Uses Azure OpenAI for high-level token extraction
and mathematical models for precise configuration parameter optimization.

Architecture:
1. LLM Stage: Extract semantic concepts, entities, and domain characteristics
2. Statistical Stage: Analyze patterns, frequencies, and optimize configuration parameters  
3. Config Stage: Generate precise ExtractionConfiguration based on both approaches

Benefits:
- LLM understands context and semantics
- Statistics provide precise numerical parameters
- Combined approach eliminates both hardcoded values and semantic gaps
"""

import time
import statistics
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import asyncio

# Azure OpenAI integration
try:
    from openai import AsyncAzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    AsyncAzureOpenAI = None

logger = logging.getLogger(__name__)


@dataclass
class LLMExtraction:
    """High-level semantic extraction from LLM"""
    domain_concepts: List[str]
    key_entities: List[str] 
    semantic_relationships: List[Tuple[str, str, str]]
    domain_classification: str
    confidence_assessment: str
    processing_complexity: str
    recommended_strategies: List[str]
    technical_vocabulary: List[str]
    content_structure_analysis: Dict[str, Any]


@dataclass
class StatisticalFeatures:
    """Statistical features for config optimization"""
    optimal_chunk_size: int
    chunk_overlap_ratio: float
    entity_density: float
    relationship_density: float
    vocabulary_complexity: float
    processing_load_estimate: float
    confidence_thresholds: Dict[str, float]
    performance_parameters: Dict[str, Any]


@dataclass
class HybridAnalysis:
    """Combined LLM + Statistical analysis"""
    llm_extraction: LLMExtraction
    statistical_features: StatisticalFeatures
    hybrid_confidence: float
    config_recommendations: Dict[str, Any]
    optimization_parameters: Dict[str, float]
    source_file: Optional[str] = None
    processing_time: float = 0.0


class HybridDomainAnalyzer:
    """
    Hybrid analyzer combining LLM semantic understanding with statistical precision.
    
    Two-stage process:
    1. LLM Stage: Azure OpenAI extracts semantic concepts and domain understanding
    2. Statistical Stage: Mathematical analysis optimizes configuration parameters
    """
    
    def __init__(self):
        """Initialize hybrid analyzer with both LLM and statistical components"""
        # Azure OpenAI client for semantic analysis
        self.azure_client = self._initialize_azure_client()
        
        # Statistical analysis components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=0.01,
            max_df=0.95
        )
        
        # Configuration optimization models
        self.config_optimizer = ConfigurationOptimizer()
        
        # Performance tracking
        self.analysis_count = 0
        self.llm_calls = 0
        self.total_processing_time = 0.0
        
        logger.info("Hybrid domain analyzer initialized with LLM + Statistical foundation")

    def _initialize_azure_client(self) -> Optional[AsyncAzureOpenAI]:
        """Initialize Azure OpenAI client if available"""
        if not AZURE_OPENAI_AVAILABLE:
            logger.warning("Azure OpenAI not available, falling back to statistical-only analysis")
            return None
        
        try:
            import os
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
            
            if endpoint and api_key:
                return AsyncAzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=api_version
                )
            else:
                logger.warning("Azure OpenAI credentials not found, using statistical-only mode")
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            return None

    async def analyze_domain_hybrid(self, file_path: Path) -> HybridAnalysis:
        """
        Perform hybrid analysis combining LLM semantic extraction with statistical optimization.
        
        Process:
        1. Extract text content
        2. LLM Stage: Semantic analysis and concept extraction  
        3. Statistical Stage: Parameter optimization and feature analysis
        4. Hybrid Stage: Combine results for optimal configuration
        """
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return self._create_empty_analysis(str(file_path), time.time() - start_time)
        
        if not text.strip():
            return self._create_empty_analysis(str(file_path), time.time() - start_time)
        
        # Stage 1: LLM Semantic Extraction
        llm_extraction = await self._extract_semantics_with_llm(text)
        
        # Stage 2: Statistical Feature Analysis
        statistical_features = await self._analyze_statistical_features(text, llm_extraction)
        
        # Stage 3: Hybrid Optimization
        hybrid_confidence = self._calculate_hybrid_confidence(llm_extraction, statistical_features)
        config_recommendations = self._generate_config_recommendations(llm_extraction, statistical_features)
        optimization_parameters = self._optimize_parameters(llm_extraction, statistical_features)
        
        processing_time = time.time() - start_time
        self.analysis_count += 1
        self.total_processing_time += processing_time
        
        return HybridAnalysis(
            llm_extraction=llm_extraction,
            statistical_features=statistical_features,
            hybrid_confidence=hybrid_confidence,
            config_recommendations=config_recommendations,
            optimization_parameters=optimization_parameters,
            source_file=str(file_path),
            processing_time=processing_time
        )

    async def _extract_semantics_with_llm(self, text: str) -> LLMExtraction:
        """Use Azure OpenAI to extract high-level semantic concepts"""
        
        if not self.azure_client:
            # Fallback to rule-based extraction if LLM not available
            return self._fallback_semantic_extraction(text)
        
        try:
            self.llm_calls += 1
            
            # Construct prompt for semantic extraction
            prompt = self._create_semantic_extraction_prompt(text)
            
            response = await self.azure_client.chat.completions.create(
                model="gpt-4",  # or your deployment name
                messages=[
                    {"role": "system", "content": "You are a domain analysis expert. Extract semantic concepts and domain characteristics from text for configuration optimization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=1000
            )
            
            # Parse LLM response
            llm_response = response.choices[0].message.content
            return self._parse_llm_response(llm_response)
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}, falling back to rule-based")
            return self._fallback_semantic_extraction(text)

    def _create_semantic_extraction_prompt(self, text: str) -> str:
        """Create structured prompt for LLM semantic extraction"""
        # Truncate text to avoid token limits
        text_sample = text[:3000] if len(text) > 3000 else text
        
        return f"""
Analyze this text and extract semantic information for domain configuration:

TEXT:
{text_sample}

Please provide a JSON response with the following structure:
{{
    "domain_concepts": ["list of 5-10 main domain concepts"],
    "key_entities": ["list of 5-15 important entities/objects"],
    "semantic_relationships": [["entity1", "relationship", "entity2"], ...],
    "domain_classification": "technical|process|academic|general|maintenance",
    "confidence_assessment": "high|medium|low confidence in classification",
    "processing_complexity": "high|medium|low complexity for extraction",
    "recommended_strategies": ["list of recommended extraction strategies"],
    "technical_vocabulary": ["list of 10-20 technical terms"],
    "content_structure_analysis": {{
        "has_procedures": true|false,
        "has_technical_specs": true|false,
        "has_relationships": true|false,
        "document_type": "manual|specification|guide|reference|other"
    }}
}}

Focus on concepts that would be important for knowledge extraction configuration.
"""

    def _parse_llm_response(self, response: str) -> LLMExtraction:
        """Parse LLM JSON response into structured data"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Convert semantic relationships to tuples
                relationships = []
                for rel in data.get('semantic_relationships', []):
                    if len(rel) >= 3:
                        relationships.append((rel[0], rel[1], rel[2]))
                
                return LLMExtraction(
                    domain_concepts=data.get('domain_concepts', []),
                    key_entities=data.get('key_entities', []),
                    semantic_relationships=relationships,
                    domain_classification=data.get('domain_classification', 'general'),
                    confidence_assessment=data.get('confidence_assessment', 'medium'),
                    processing_complexity=data.get('processing_complexity', 'medium'),
                    recommended_strategies=data.get('recommended_strategies', []),
                    technical_vocabulary=data.get('technical_vocabulary', []),
                    content_structure_analysis=data.get('content_structure_analysis', {})
                )
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._fallback_semantic_extraction("")

    def _fallback_semantic_extraction(self, text: str) -> LLMExtraction:
        """Fallback semantic extraction when LLM is not available"""
        words = text.lower().split()
        word_freq = Counter(words)
        
        # Simple keyword-based extraction
        common_concepts = [word for word, count in word_freq.most_common(10) if len(word) > 4]
        technical_terms = [word for word in words if word.isupper() or '_' in word or any(char.isdigit() for char in word)]
        
        return LLMExtraction(
            domain_concepts=common_concepts[:8],
            key_entities=technical_terms[:10],
            semantic_relationships=[],
            domain_classification="general",
            confidence_assessment="low",
            processing_complexity="medium",
            recommended_strategies=["general_extraction"],
            technical_vocabulary=technical_terms[:15],
            content_structure_analysis={"has_procedures": False, "has_technical_specs": False, "has_relationships": False, "document_type": "other"}
        )

    async def _analyze_statistical_features(self, text: str, llm_extraction: LLMExtraction) -> StatisticalFeatures:
        """Analyze statistical features and optimize configuration parameters"""
        
        # Basic text statistics
        words = text.split()
        word_count = len(words)
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        # Calculate optimal chunk size based on content analysis
        optimal_chunk_size = self._calculate_optimal_chunk_size(word_count, llm_extraction)
        
        # Calculate chunk overlap ratio
        chunk_overlap_ratio = self._calculate_optimal_overlap_ratio(llm_extraction)
        
        # Entity and relationship density analysis
        entity_density = len(llm_extraction.key_entities) / max(word_count, 1) * 1000  # per 1000 words
        relationship_density = len(llm_extraction.semantic_relationships) / max(word_count, 1) * 1000
        
        # Vocabulary complexity analysis
        vocabulary_complexity = self._analyze_vocabulary_complexity(text, llm_extraction)
        
        # Processing load estimate
        processing_load_estimate = self._estimate_processing_load(llm_extraction, word_count)
        
        # Confidence thresholds based on LLM assessment
        confidence_thresholds = self._calculate_confidence_thresholds(llm_extraction)
        
        # Performance parameters
        performance_parameters = self._optimize_performance_parameters(llm_extraction, word_count)
        
        return StatisticalFeatures(
            optimal_chunk_size=optimal_chunk_size,
            chunk_overlap_ratio=chunk_overlap_ratio,
            entity_density=entity_density,
            relationship_density=relationship_density,
            vocabulary_complexity=vocabulary_complexity,
            processing_load_estimate=processing_load_estimate,
            confidence_thresholds=confidence_thresholds,
            performance_parameters=performance_parameters
        )

    def _calculate_optimal_chunk_size(self, word_count: int, llm_extraction: LLMExtraction) -> int:
        """Calculate optimal chunk size based on content characteristics"""
        base_size = 1000  # Default chunk size
        
        # Adjust based on complexity
        complexity_multiplier = {
            "high": 0.7,    # Smaller chunks for complex content
            "medium": 1.0,  # Standard chunks
            "low": 1.3      # Larger chunks for simple content
        }.get(llm_extraction.processing_complexity, 1.0)
        
        # Adjust based on entity density
        entity_density = len(llm_extraction.key_entities) / max(word_count, 1) * 1000
        if entity_density > 20:  # High entity density
            complexity_multiplier *= 0.8
        elif entity_density < 5:  # Low entity density
            complexity_multiplier *= 1.2
        
        # Adjust based on document type
        if llm_extraction.content_structure_analysis.get("has_technical_specs", False):
            complexity_multiplier *= 0.8  # Smaller chunks for technical specs
        
        optimal_size = int(base_size * complexity_multiplier)
        
        # Ensure reasonable bounds
        return max(500, min(2000, optimal_size))

    def _calculate_optimal_overlap_ratio(self, llm_extraction: LLMExtraction) -> float:
        """Calculate optimal chunk overlap ratio"""
        base_ratio = 0.2  # 20% default overlap
        
        # Increase overlap for high relationship density
        if len(llm_extraction.semantic_relationships) > 5:
            base_ratio = 0.25
        
        # Increase overlap for procedural content
        if llm_extraction.content_structure_analysis.get("has_procedures", False):
            base_ratio = 0.3
        
        # Decrease overlap for simple content
        if llm_extraction.processing_complexity == "low":
            base_ratio = 0.15
        
        return min(0.4, max(0.1, base_ratio))

    def _analyze_vocabulary_complexity(self, text: str, llm_extraction: LLMExtraction) -> float:
        """Analyze vocabulary complexity for parameter optimization"""
        words = text.split()
        unique_words = len(set(word.lower() for word in words))
        
        # Type-token ratio
        ttr = unique_words / max(len(words), 1)
        
        # Technical vocabulary density
        tech_density = len(llm_extraction.technical_vocabulary) / max(len(words), 1) * 1000
        
        # Combine metrics
        complexity_score = ttr * 0.6 + min(tech_density / 50, 1.0) * 0.4
        
        return complexity_score

    def _estimate_processing_load(self, llm_extraction: LLMExtraction, word_count: int) -> float:
        """Estimate processing load for performance optimization"""
        base_load = word_count / 1000  # Base load per 1000 words
        
        # Adjust for complexity
        complexity_factor = {
            "high": 1.5,
            "medium": 1.0,
            "low": 0.7
        }.get(llm_extraction.processing_complexity, 1.0)
        
        # Adjust for entity density
        entity_factor = 1.0 + (len(llm_extraction.key_entities) / 100)
        
        # Adjust for relationship complexity
        relationship_factor = 1.0 + (len(llm_extraction.semantic_relationships) / 50)
        
        total_load = base_load * complexity_factor * entity_factor * relationship_factor
        
        return total_load

    def _calculate_confidence_thresholds(self, llm_extraction: LLMExtraction) -> Dict[str, float]:
        """Calculate confidence thresholds based on LLM assessment"""
        base_thresholds = {
            "entity_confidence": 0.7,
            "relationship_confidence": 0.6,
            "overall_confidence": 0.65
        }
        
        # Adjust based on LLM confidence assessment
        confidence_multiplier = {
            "high": 0.9,    # Lower thresholds for high-confidence content
            "medium": 1.0,  # Standard thresholds
            "low": 1.1      # Higher thresholds for low-confidence content
        }.get(llm_extraction.confidence_assessment, 1.0)
        
        # Adjust based on domain classification
        domain_adjustments = {
            "technical": 0.95,  # Slightly lower thresholds for technical content
            "academic": 1.0,    # Standard thresholds
            "process": 0.9,     # Lower thresholds for process documentation
            "general": 1.05     # Slightly higher thresholds for general content
        }
        
        domain_multiplier = domain_adjustments.get(llm_extraction.domain_classification, 1.0)
        
        final_multiplier = confidence_multiplier * domain_multiplier
        
        return {
            key: max(0.5, min(0.9, value * final_multiplier))
            for key, value in base_thresholds.items()
        }

    def _optimize_performance_parameters(self, llm_extraction: LLMExtraction, word_count: int) -> Dict[str, Any]:
        """Optimize performance parameters based on analysis"""
        
        # Concurrency settings
        max_concurrent = min(10, max(2, word_count // 2000))  # Scale with content size
        
        # Timeout settings based on complexity
        timeout_base = 30  # seconds
        complexity_factor = {
            "high": 1.5,
            "medium": 1.0,
            "low": 0.8
        }.get(llm_extraction.processing_complexity, 1.0)
        
        extraction_timeout = int(timeout_base * complexity_factor)
        
        # Quality settings
        quality_settings = {
            "enable_validation": True,
            "min_entity_count": max(3, len(llm_extraction.key_entities) // 2),
            "min_relationship_count": max(2, len(llm_extraction.semantic_relationships) // 2),
            "enable_caching": True,
            "cache_ttl": 3600
        }
        
        return {
            "max_concurrent_chunks": max_concurrent,
            "extraction_timeout_seconds": extraction_timeout,
            "quality_settings": quality_settings,
            "performance_mode": "balanced"
        }

    def _calculate_hybrid_confidence(self, llm_extraction: LLMExtraction, statistical_features: StatisticalFeatures) -> float:
        """Calculate overall confidence combining LLM and statistical analysis"""
        
        # LLM confidence score
        llm_confidence = {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.5
        }.get(llm_extraction.confidence_assessment, 0.6)
        
        # Statistical confidence based on feature quality
        stat_confidence = min(1.0, (
            statistical_features.entity_density / 20 * 0.3 +
            statistical_features.vocabulary_complexity * 0.3 +
            (1.0 - statistical_features.processing_load_estimate / 10) * 0.4
        ))
        
        # Combine with weighted average
        hybrid_confidence = llm_confidence * 0.6 + stat_confidence * 0.4
        
        return max(0.1, min(1.0, hybrid_confidence))

    def _generate_config_recommendations(self, llm_extraction: LLMExtraction, statistical_features: StatisticalFeatures) -> Dict[str, Any]:
        """Generate configuration recommendations based on hybrid analysis"""
        
        return {
            "extraction_strategy": self._recommend_extraction_strategy(llm_extraction),
            "entity_types_focus": llm_extraction.key_entities[:20],
            "relationship_patterns": [f"{r[0]} {r[1]} {r[2]}" for r in llm_extraction.semantic_relationships[:15]],
            "technical_vocabulary": llm_extraction.technical_vocabulary,
            "processing_parameters": {
                "chunk_size": statistical_features.optimal_chunk_size,
                "chunk_overlap": int(statistical_features.optimal_chunk_size * statistical_features.chunk_overlap_ratio),
                "parallel_processing": statistical_features.processing_load_estimate > 2.0,
                "complexity_handling": llm_extraction.processing_complexity
            },
            "quality_thresholds": statistical_features.confidence_thresholds,
            "performance_optimization": statistical_features.performance_parameters
        }

    def _recommend_extraction_strategy(self, llm_extraction: LLMExtraction) -> str:
        """Recommend extraction strategy based on LLM analysis"""
        
        # Use LLM recommendations if available
        if llm_extraction.recommended_strategies:
            primary_strategy = llm_extraction.recommended_strategies[0]
            
            # Map to standard strategies
            strategy_mapping = {
                "technical_extraction": "TECHNICAL_CONTENT",
                "process_extraction": "STRUCTURED_DATA", 
                "general_extraction": "MIXED_CONTENT",
                "academic_extraction": "CONVERSATIONAL"
            }
            
            return strategy_mapping.get(primary_strategy, "MIXED_CONTENT")
        
        # Fallback based on domain classification
        domain_strategies = {
            "technical": "TECHNICAL_CONTENT",
            "process": "STRUCTURED_DATA",
            "academic": "CONVERSATIONAL",
            "general": "MIXED_CONTENT"
        }
        
        return domain_strategies.get(llm_extraction.domain_classification, "MIXED_CONTENT")

    def _optimize_parameters(self, llm_extraction: LLMExtraction, statistical_features: StatisticalFeatures) -> Dict[str, float]:
        """Optimize numerical parameters for configuration"""
        
        return {
            "entity_confidence_threshold": statistical_features.confidence_thresholds["entity_confidence"],
            "relationship_confidence_threshold": statistical_features.confidence_thresholds["relationship_confidence"], 
            "chunk_size_ratio": statistical_features.optimal_chunk_size / 1000.0,
            "overlap_ratio": statistical_features.chunk_overlap_ratio,
            "processing_complexity_score": self._complexity_to_score(llm_extraction.processing_complexity),
            "vocabulary_complexity_score": statistical_features.vocabulary_complexity,
            "load_factor": min(1.0, statistical_features.processing_load_estimate / 5.0)
        }

    def _complexity_to_score(self, complexity: str) -> float:
        """Convert complexity assessment to numerical score"""
        return {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }.get(complexity, 0.5)

    def _create_empty_analysis(self, file_path: str, processing_time: float) -> HybridAnalysis:
        """Create empty analysis for error cases"""
        empty_llm = LLMExtraction(
            domain_concepts=[], key_entities=[], semantic_relationships=[],
            domain_classification="general", confidence_assessment="low",
            processing_complexity="low", recommended_strategies=[],
            technical_vocabulary=[], content_structure_analysis={}
        )
        
        empty_stats = StatisticalFeatures(
            optimal_chunk_size=1000, chunk_overlap_ratio=0.2, entity_density=0.0,
            relationship_density=0.0, vocabulary_complexity=0.0, processing_load_estimate=0.0,
            confidence_thresholds={"entity_confidence": 0.7, "relationship_confidence": 0.6},
            performance_parameters={}
        )
        
        return HybridAnalysis(
            llm_extraction=empty_llm,
            statistical_features=empty_stats, 
            hybrid_confidence=0.1,
            config_recommendations={},
            optimization_parameters={},
            source_file=file_path,
            processing_time=processing_time
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        avg_processing_time = self.total_processing_time / max(self.analysis_count, 1)
        
        return {
            "analyses_performed": self.analysis_count,
            "llm_calls_made": self.llm_calls,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "azure_openai_available": self.azure_client is not None,
            "hybrid_mode_active": True
        }


class ConfigurationOptimizer:
    """Helper class for configuration parameter optimization"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_for_domain(self, domain: str, analysis: HybridAnalysis) -> Dict[str, Any]:
        """Optimize configuration parameters for specific domain"""
        # Domain-specific optimization logic
        optimizations = {
            "technical": self._optimize_technical_domain(analysis),
            "process": self._optimize_process_domain(analysis),
            "academic": self._optimize_academic_domain(analysis),
            "general": self._optimize_general_domain(analysis)
        }
        
        domain_type = analysis.llm_extraction.domain_classification
        return optimizations.get(domain_type, optimizations["general"])
    
    def _optimize_technical_domain(self, analysis: HybridAnalysis) -> Dict[str, Any]:
        """Optimize for technical content"""
        return {
            "entity_focus": "technical_terms",
            "relationship_focus": "implementation_dependencies", 
            "chunk_strategy": "precision_focused",
            "validation_level": "strict"
        }
    
    def _optimize_process_domain(self, analysis: HybridAnalysis) -> Dict[str, Any]:
        """Optimize for process documentation"""
        return {
            "entity_focus": "procedural_steps",
            "relationship_focus": "sequential_flow",
            "chunk_strategy": "flow_preserving",
            "validation_level": "moderate"
        }
    
    def _optimize_academic_domain(self, analysis: HybridAnalysis) -> Dict[str, Any]:
        """Optimize for academic content"""
        return {
            "entity_focus": "conceptual_entities",
            "relationship_focus": "theoretical_connections",
            "chunk_strategy": "context_preserving", 
            "validation_level": "comprehensive"
        }
    
    def _optimize_general_domain(self, analysis: HybridAnalysis) -> Dict[str, Any]:
        """Optimize for general content"""
        return {
            "entity_focus": "balanced",
            "relationship_focus": "contextual",
            "chunk_strategy": "adaptive",
            "validation_level": "balanced"
        }