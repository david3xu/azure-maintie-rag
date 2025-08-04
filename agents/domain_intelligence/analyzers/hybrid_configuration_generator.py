"""
Hybrid Configuration Generator - LLM-Powered Configuration Generation

This module simplifies the HybridDomainAnalyzer to focus solely on its unique value:
LLM-powered configuration generation using unified content analysis.

Key responsibilities:
- LLM semantic understanding and extraction
- Configuration parameter optimization  
- Azure OpenAI integration for domain insights
- Parameter generation based on content analysis

Uses UnifiedContentAnalyzer for all statistical analysis, eliminating redundancy
while preserving the unique LLM-based configuration generation capabilities.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configuration imports
from config.centralized_config import (
    get_model_config, 
    get_processing_config, 
    get_domain_analysis_config,
    get_ml_config,
    get_entity_extraction_config,
    get_hybrid_domain_analyzer_config
)

# Import unified content analyzer
from .unified_content_analyzer import UnifiedContentAnalyzer, UnifiedAnalysis

# Azure OpenAI integration
try:
    from openai import AsyncAzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMExtraction:
    """LLM semantic extraction results"""
    
    domain_characteristics: List[str]
    key_concepts: List[str]
    entity_types: List[str]
    relationship_patterns: List[str]
    processing_complexity: str  # low, medium, high
    content_structure: str  # structured, semi_structured, unstructured
    semantic_density: float
    confidence_score: float
    reasoning: str


@dataclass
class ConfigurationRecommendations:
    """Configuration recommendations based on LLM + statistical analysis"""
    
    # Processing configuration
    optimal_chunk_size: int
    entity_extraction_thresholds: Dict[str, float]
    relationship_confidence_min: float
    processing_timeout: int
    
    # ML configuration  
    vector_dimensions: int
    clustering_parameters: Dict[str, Any]
    model_parameters: Dict[str, Any]
    
    # Domain-specific tuning
    complexity_adjustments: Dict[str, float]
    quality_thresholds: Dict[str, float]
    performance_targets: Dict[str, float]
    
    # Metadata
    generation_confidence: float
    reasoning: str
    source_analysis: str
    generation_time: float


class HybridConfigurationGenerator:
    """
    LLM-powered configuration generator using unified content analysis.
    
    Simplified from HybridDomainAnalyzer to focus only on unique value:
    - LLM semantic understanding 
    - Configuration parameter optimization
    - Azure OpenAI integration
    
    Uses UnifiedContentAnalyzer for all statistical analysis, eliminating
    redundant calculations while preserving LLM-based insights.
    """

    def __init__(self, content_analyzer: Optional[UnifiedContentAnalyzer] = None):
        """Initialize with unified content analyzer dependency"""
        # Use provided analyzer or create new one
        self.content_analyzer = content_analyzer or UnifiedContentAnalyzer()
        
        # Get centralized configuration
        self.hybrid_config = get_hybrid_domain_analyzer_config()
        self.model_config = get_model_config()
        self.processing_config = get_processing_config()
        self.ml_config = get_ml_config()
        
        # Initialize Azure OpenAI client
        self.openai_client = None
        if AZURE_OPENAI_AVAILABLE:
            try:
                self.openai_client = AsyncAzureOpenAI(
                    api_key=self.model_config.azure_openai_api_key,
                    api_version=self.model_config.azure_openai_api_version,
                    azure_endpoint=self.model_config.azure_openai_endpoint,
                )
                logger.info("Azure OpenAI client initialized for configuration generation")
            except Exception as e:
                logger.warning(f"Azure OpenAI initialization failed: {e}")
                self.openai_client = None
        
        # Performance tracking
        self.generation_stats = {
            "configurations_generated": 0,
            "llm_extractions": 0,
            "avg_generation_time": 0.0,
            "total_generation_time": 0.0,
        }

        logger.info("Hybrid configuration generator initialized with unified analysis")

    async def generate_configuration(self, file_path: Path) -> ConfigurationRecommendations:
        """
        Generate configuration recommendations using LLM + unified statistical analysis.
        
        Single processing pipeline:
        1. Use UnifiedContentAnalyzer for all statistical analysis
        2. Use LLM for semantic understanding and insights  
        3. Combine both for optimal configuration generation
        """
        start_time = time.time()

        try:
            # Step 1: Get unified content analysis (single pass for all statistics)
            logger.info(f"Starting unified content analysis for {file_path}")
            content_analysis = self.content_analyzer.analyze_content_complete(file_path)
            
            # Step 2: LLM semantic extraction (unique value-add)
            logger.info("Performing LLM semantic extraction")
            llm_analysis = await self._extract_semantics_with_llm(content_analysis)
            
            # Step 3: Generate configuration recommendations
            logger.info("Generating configuration recommendations")
            recommendations = self._generate_config_recommendations(content_analysis, llm_analysis)
            
            generation_time = time.time() - start_time
            recommendations.generation_time = generation_time
            
            # Update performance statistics
            self.generation_stats["configurations_generated"] += 1
            self.generation_stats["total_generation_time"] += generation_time
            self.generation_stats["avg_generation_time"] = (
                self.generation_stats["total_generation_time"] 
                / self.generation_stats["configurations_generated"]
            )
            
            logger.info(f"Configuration generation completed in {generation_time:.2f}s")
            return recommendations
            
        except Exception as e:
            logger.error(f"Configuration generation failed for {file_path}: {e}")
            raise

    async def _extract_semantics_with_llm(self, content_analysis: UnifiedAnalysis) -> LLMExtraction:
        """
        Extract semantic insights using Azure OpenAI.
        
        Focus on high-level semantic understanding that statistical analysis cannot provide:
        - Domain characteristics and context
        - Semantic relationships and patterns
        - Processing complexity assessment
        - Content structure analysis
        """
        if not self.openai_client:
            logger.warning("LLM unavailable, using statistical fallback")
            return self._create_statistical_fallback_extraction(content_analysis)

        try:
            # Build LLM prompt with statistical context
            prompt = self._build_llm_prompt(content_analysis)
            
            # Call Azure OpenAI
            response = await self.openai_client.chat.completions.create(
                model=self.model_config.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a domain analysis expert. Analyze content characteristics and provide semantic insights for configuration optimization."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=self.hybrid_config.llm_max_tokens,
                temperature=self.hybrid_config.llm_temperature,
            )
            
            # Parse LLM response
            llm_content = response.choices[0].message.content
            extraction = self._parse_llm_response(llm_content, content_analysis)
            
            self.generation_stats["llm_extractions"] += 1
            logger.info("LLM semantic extraction completed successfully")
            
            return extraction
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._create_statistical_fallback_extraction(content_analysis)

    def _build_llm_prompt(self, content_analysis: UnifiedAnalysis) -> str:
        """Build comprehensive LLM prompt with statistical context"""
        # Extract key statistical insights to guide LLM
        top_concepts = list(content_analysis.concept_frequency.keys())[:5]
        top_entities = content_analysis.entity_candidates[:5]
        top_actions = content_analysis.action_patterns[:5]
        
        prompt = f"""
Analyze the following content characteristics and provide semantic insights:

STATISTICAL CONTEXT:
- Word count: {content_analysis.word_count}
- Vocabulary richness: {content_analysis.vocabulary_richness:.3f}
- Entropy score: {content_analysis.entropy_score:.2f}
- Complexity score: {content_analysis.complexity_score:.3f}
- Analysis quality: {content_analysis.analysis_quality}

KEY CONCEPTS: {', '.join(top_concepts)}
ENTITIES: {', '.join(top_entities)}
ACTIONS: {', '.join(top_actions)}

Please provide analysis in this JSON format:
{{
    "domain_characteristics": ["characteristic1", "characteristic2", ...],
    "key_concepts": ["concept1", "concept2", ...],
    "entity_types": ["type1", "type2", ...],
    "relationship_patterns": ["pattern1", "pattern2", ...],
    "processing_complexity": "low|medium|high",
    "content_structure": "structured|semi_structured|unstructured",
    "semantic_density": 0.0-1.0,
    "confidence_score": 0.0-1.0,
    "reasoning": "Brief explanation of analysis"
}}

Focus on semantic characteristics that statistical analysis cannot capture.
"""
        return prompt

    def _parse_llm_response(self, llm_content: str, content_analysis: UnifiedAnalysis) -> LLMExtraction:
        """Parse LLM response into structured extraction"""
        try:
            # Try to extract JSON from response
            json_start = llm_content.find("{")
            json_end = llm_content.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = llm_content[json_start:json_end]
                llm_data = json.loads(json_content)
                
                return LLMExtraction(
                    domain_characteristics=llm_data.get("domain_characteristics", []),
                    key_concepts=llm_data.get("key_concepts", []),
                    entity_types=llm_data.get("entity_types", []),
                    relationship_patterns=llm_data.get("relationship_patterns", []),
                    processing_complexity=llm_data.get("processing_complexity", "medium"),
                    content_structure=llm_data.get("content_structure", "semi_structured"),
                    semantic_density=float(llm_data.get("semantic_density", 0.5)),
                    confidence_score=float(llm_data.get("confidence_score", 0.7)),
                    reasoning=llm_data.get("reasoning", "LLM semantic analysis")
                )
            else:
                logger.warning("Could not extract JSON from LLM response")
                return self._create_statistical_fallback_extraction(content_analysis)
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._create_statistical_fallback_extraction(content_analysis)

    def _create_statistical_fallback_extraction(self, content_analysis: UnifiedAnalysis) -> LLMExtraction:
        """Create fallback extraction using only statistical analysis"""
        # Derive characteristics from statistical features
        complexity_level = "low"
        if content_analysis.complexity_score > 0.7:
            complexity_level = "high"
        elif content_analysis.complexity_score > 0.4:
            complexity_level = "medium"
            
        structure_type = "unstructured"
        if content_analysis.technical_density > 0.3:
            structure_type = "structured"
        elif content_analysis.technical_density > 0.1:
            structure_type = "semi_structured"
            
        return LLMExtraction(
            domain_characteristics=["statistical_analysis_based"],
            key_concepts=list(content_analysis.concept_frequency.keys())[:5],
            entity_types=["general_entities"],
            relationship_patterns=["statistical_patterns"],
            processing_complexity=complexity_level,
            content_structure=structure_type,
            semantic_density=min(1.0, content_analysis.entropy_score / 10.0),
            confidence_score=0.6,  # Lower confidence for statistical fallback
            reasoning="Statistical analysis fallback (LLM unavailable)"
        )

    def _generate_config_recommendations(
        self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction
    ) -> ConfigurationRecommendations:
        """
        Generate configuration recommendations by combining LLM insights with statistical analysis.
        
        This is the unique value-add: intelligent configuration optimization based on
        both semantic understanding (LLM) and statistical characteristics (unified analyzer).
        """
        # Processing configuration optimization
        optimal_chunk_size = self._calculate_optimal_chunk_size(content_analysis, llm_analysis) 
        entity_thresholds = self._calculate_entity_thresholds(content_analysis, llm_analysis)
        relationship_confidence = self._calculate_relationship_confidence(content_analysis, llm_analysis)
        processing_timeout = self._calculate_processing_timeout(content_analysis, llm_analysis)
        
        # ML configuration optimization
        vector_dimensions = self._calculate_vector_dimensions(content_analysis, llm_analysis)
        clustering_params = self._generate_clustering_parameters(content_analysis, llm_analysis)
        model_params = self._generate_model_parameters(content_analysis, llm_analysis)
        
        # Domain-specific tuning
        complexity_adjustments = self._generate_complexity_adjustments(content_analysis, llm_analysis)
        quality_thresholds = self._generate_quality_thresholds(content_analysis, llm_analysis)
        performance_targets = self._generate_performance_targets(content_analysis, llm_analysis)
        
        # Generation metadata
        generation_confidence = self._calculate_generation_confidence(content_analysis, llm_analysis)
        reasoning = self._generate_configuration_reasoning(content_analysis, llm_analysis)
        
        return ConfigurationRecommendations(
            # Processing configuration
            optimal_chunk_size=optimal_chunk_size,
            entity_extraction_thresholds=entity_thresholds,
            relationship_confidence_min=relationship_confidence,
            processing_timeout=processing_timeout,
            # ML configuration
            vector_dimensions=vector_dimensions,
            clustering_parameters=clustering_params,
            model_parameters=model_params,
            # Domain-specific tuning
            complexity_adjustments=complexity_adjustments,
            quality_thresholds=quality_thresholds,
            performance_targets=performance_targets,
            # Metadata
            generation_confidence=generation_confidence,
            reasoning=reasoning,
            source_analysis=str(content_analysis.source_file),
            generation_time=0.0,  # Will be set by caller
        )

    def _calculate_optimal_chunk_size(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> int:
        """Calculate optimal chunk size based on content and LLM characteristics"""
        base_size = self.hybrid_config.base_chunk_size
        
        # Adjust based on LLM complexity assessment
        complexity_multiplier = {
            "high": self.hybrid_config.chunk_size_multiplier_high_complexity,
            "medium": self.hybrid_config.chunk_size_multiplier_medium_complexity,
            "low": self.hybrid_config.chunk_size_multiplier_low_complexity,
        }.get(llm_analysis.processing_complexity, self.hybrid_config.chunk_size_multiplier_medium_complexity)
        
        # Adjust based on statistical complexity
        if content_analysis.complexity_score > 0.7:
            complexity_multiplier *= 0.8  # High complexity adjustment
        elif content_analysis.complexity_score < 0.3:
            complexity_multiplier *= 1.2  # Low complexity adjustment
            
        # Adjust based on vocabulary richness
        if content_analysis.vocabulary_richness > 0.5:
            complexity_multiplier *= 0.9  # High vocabulary richness adjustment
            
        optimal_size = int(base_size * complexity_multiplier)
        
        # Apply bounds
        return max(
            self.hybrid_config.chunk_size_min,
            min(self.hybrid_config.chunk_size_max, optimal_size)
        )

    def _calculate_entity_thresholds(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> Dict[str, float]:
        """Calculate entity extraction thresholds based on content density"""
        base_threshold = self.hybrid_config.base_entity_confidence_threshold
        
        # Adjust based on entity density from statistical analysis
        entity_density = len(content_analysis.entity_candidates) / max(1, content_analysis.word_count) * 100
        
        if entity_density > 15.0:  # High entity density threshold
            confidence_adjustment = 0.8  # High entity density multiplier
        elif entity_density < 5.0:  # Low entity density threshold
            confidence_adjustment = 1.2  # Low entity density multiplier
        else:
            confidence_adjustment = 1.0
            
        # Adjust based on LLM semantic density
        semantic_adjustment = 1.0 + (llm_analysis.semantic_density - 0.5) * 0.2  # Semantic density adjustment
        
        adjusted_threshold = base_threshold * confidence_adjustment * semantic_adjustment
        
        return {
            "entity_confidence": max(0.1, min(0.9, adjusted_threshold)),
            "relationship_confidence": max(0.2, min(0.8, adjusted_threshold * 0.9)),  # Relationship confidence multiplier
            "concept_confidence": max(0.1, min(0.9, adjusted_threshold * 0.8)),  # Concept confidence multiplier
        }

    def _calculate_relationship_confidence(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> float:
        """Calculate minimum relationship confidence based on content analysis"""
        base_confidence = self.hybrid_config.base_relationship_confidence_threshold
        
        # Adjust based on action pattern density
        action_density = len(content_analysis.action_patterns) / max(1, content_analysis.word_count) * 100
        
        if action_density > 10.0:  # High action density threshold
            confidence_adjustment = 1.1  # High action density multiplier
        else:
            confidence_adjustment = 0.9  # Low action density multiplier
            
        # Adjust based on LLM relationship patterns
        relationship_adjustment = 1.0 + len(llm_analysis.relationship_patterns) * 0.05  # Relationship pattern bonus
        
        final_confidence = base_confidence * confidence_adjustment * relationship_adjustment
        return max(0.1, min(0.9, final_confidence))

    def _calculate_processing_timeout(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> int:
        """Calculate processing timeout based on content complexity"""
        base_timeout = self.hybrid_config.timeout_base_seconds
        
        # Adjust based on word count
        size_multiplier = 1.0 + (content_analysis.word_count / 1000.0)  # Word count timeout divisor
        
        # Adjust based on complexity
        complexity_multiplier = {
            "high": self.hybrid_config.complexity_timeout_multiplier_high,
            "medium": self.hybrid_config.complexity_timeout_multiplier_medium,
            "low": self.hybrid_config.complexity_timeout_multiplier_low,
        }.get(llm_analysis.processing_complexity, self.hybrid_config.complexity_timeout_multiplier_medium)
        
        final_timeout = int(base_timeout * size_multiplier * complexity_multiplier)
        return max(10, min(300, final_timeout))  # Min 10s, max 300s

    def _calculate_vector_dimensions(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> int:
        """Calculate optimal vector dimensions for content"""
        base_dimensions = 1536  # Default vector dimensions
        
        # Adjust based on vocabulary richness
        if content_analysis.vocabulary_richness > 0.6:  # High vocabulary richness threshold
            return 2048  # High dimension vectors
        elif content_analysis.vocabulary_richness < 0.3:  # Low vocabulary richness threshold
            return 1024  # Low dimension vectors
        else:
            return base_dimensions

    def _generate_clustering_parameters(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> Dict[str, Any]:
        """Generate clustering parameters based on analysis"""
        # Use clustering results from unified analyzer
        clustering = content_analysis.clustering_results
        
        return {
            "n_clusters": min(8, max(2, len(clustering.get("top_features", {})) // 3)),  # Max clusters
            "max_iter": 300,  # Clustering max iterations
            "tol": 1e-4,  # Clustering tolerance
            "random_state": 42,  # Clustering random state
        }

    def _generate_model_parameters(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> Dict[str, Any]:
        """Generate ML model parameters based on analysis"""
        return {
            "learning_rate": 0.001 * (1.0 + content_analysis.complexity_score),  # Base learning rate
            "batch_size": max(16, int(content_analysis.word_count / 100)),  # Min batch size, batch size divisor
            "max_epochs": 100,  # Max training epochs
            "early_stopping_patience": 10,  # Early stopping patience
        }

    def _generate_complexity_adjustments(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> Dict[str, float]:
        """Generate complexity-based adjustments"""
        return {
            "chunk_overlap": self.hybrid_config.base_overlap_ratio * (1.0 + content_analysis.complexity_score),
            "entity_window": 5.0 * (1.0 + llm_analysis.semantic_density),  # Base entity window
            "relationship_distance": 3.0 * (1.0 + content_analysis.technical_density),  # Base relationship distance
        }

    def _generate_quality_thresholds(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> Dict[str, float]:
        """Generate quality thresholds based on content characteristics"""
        return {
            "min_confidence": max(0.1, content_analysis.complexity_score * 0.8),  # Confidence complexity multiplier
            "min_support": max(0.05, content_analysis.vocabulary_richness * 0.2),  # Support vocabulary multiplier
            "min_lift": 1.2 * (1.0 + llm_analysis.semantic_density),  # Base min lift
        }

    def _generate_performance_targets(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> Dict[str, float]:
        """Generate performance targets based on content complexity"""
        complexity_factor = content_analysis.complexity_score
        
        return {
            "max_processing_time": 30.0 * (1.0 + complexity_factor),  # Base max processing time
            "min_accuracy": max(0.7, 0.85 - complexity_factor * 0.1),  # Base min accuracy
            "max_memory_usage": 1024.0 * (1.0 + complexity_factor * 0.5),  # Base max memory (MB)
        }

    def _calculate_generation_confidence(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> float:
        """Calculate confidence in generated configuration"""
        # Base confidence from analysis quality
        base_confidence = 0.5
        
        # Boost from good statistical analysis
        if content_analysis.word_count > 500:
            base_confidence += 0.2
        if content_analysis.entropy_score > 3.0:
            base_confidence += 0.1
        if content_analysis.analysis_quality == "high_quality":
            base_confidence += 0.1
            
        # Boost from LLM confidence
        base_confidence += llm_analysis.confidence_score * 0.2
        
        return min(1.0, base_confidence)

    def _generate_configuration_reasoning(self, content_analysis: UnifiedAnalysis, llm_analysis: LLMExtraction) -> str:
        """Generate human-readable reasoning for configuration choices"""
        reasoning_parts = [
            f"Content analysis: {content_analysis.word_count} words, "
            f"{content_analysis.vocabulary_richness:.2f} vocabulary richness, "
            f"{content_analysis.complexity_score:.2f} complexity score",
            
            f"LLM insights: {llm_analysis.processing_complexity} complexity, "
            f"{llm_analysis.content_structure} structure, "
            f"{llm_analysis.semantic_density:.2f} semantic density",
            
            f"Configuration optimized for: {', '.join(llm_analysis.domain_characteristics[:3])}",
            
            llm_analysis.reasoning
        ]
        
        return " | ".join(reasoning_parts)

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get configuration generation performance statistics"""
        return {
            "configurations_generated": self.generation_stats["configurations_generated"],
            "llm_extractions": self.generation_stats["llm_extractions"],
            "avg_generation_time": self.generation_stats["avg_generation_time"],
            "total_generation_time": self.generation_stats["total_generation_time"],
            "llm_available": self.openai_client is not None,
            "content_analyzer_stats": self.content_analyzer.get_analysis_stats(),
        }


# Backward compatibility alias
HybridDomainAnalyzer = HybridConfigurationGenerator