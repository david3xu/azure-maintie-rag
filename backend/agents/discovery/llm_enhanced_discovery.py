"""
LLM-Enhanced Pattern Discovery

This module demonstrates how to use LLMs for semantic pattern discovery
as an alternative to pure statistical methods.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Import Azure OpenAI for LLM-powered discovery
from infra.azure_openai.azure_openai_client import AzureOpenAIClient
from .domain_pattern_engine import DiscoveredPattern, PatternType

logger = logging.getLogger(__name__)


@dataclass
class LLMDiscoveredPattern:
    """Pattern discovered using LLM semantic understanding"""
    pattern_text: str
    pattern_type: str
    semantic_category: str
    llm_confidence: float
    llm_reasoning: str
    contexts: List[str] = field(default_factory=list)
    statistical_support: Dict[str, float] = field(default_factory=dict)


class LLMEnhancedDiscovery:
    """
    Discovery system that uses LLMs for semantic understanding
    vs pure statistical analysis.
    
    Advantages:
    - True semantic understanding
    - Context-aware categorization  
    - Multi-language support via LLM
    - Domain adaptation without hardcoding
    
    Disadvantages:
    - Slower (seconds vs milliseconds)
    - More expensive (API calls)
    - Requires Azure OpenAI access
    - Less predictable results
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize Azure OpenAI client for semantic analysis
        try:
            self.openai_client = AzureOpenAI_Client(config.get("azure_openai_config", {}))
            self.llm_available = True
            self.logger.info("LLM-enhanced discovery initialized with Azure OpenAI")
        except Exception as e:
            self.logger.warning(f"LLM unavailable, falling back to statistical: {e}")
            self.llm_available = False
        
        # LLM discovery parameters
        self.max_texts_per_batch = config.get("max_texts_per_batch", 10)
        self.llm_temperature = config.get("llm_temperature", 0.1)  # Low for consistent analysis
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
    
    async def discover_patterns_with_llm(
        self, 
        texts: List[str], 
        domain_hint: Optional[str] = None
    ) -> List[LLMDiscoveredPattern]:
        """
        Discover patterns using LLM semantic understanding.
        
        Args:
            texts: Raw text data for analysis
            domain_hint: Optional domain context for LLM
            
        Returns:
            List of semantically discovered patterns
        """
        if not self.llm_available:
            raise ValueError("LLM not available - check Azure OpenAI configuration")
        
        self.logger.info(f"Starting LLM-enhanced pattern discovery on {len(texts)} texts")
        
        # Process texts in batches for LLM efficiency
        all_patterns = []
        
        for i in range(0, len(texts), self.max_texts_per_batch):
            batch = texts[i:i + self.max_texts_per_batch]
            
            # Semantic analysis via LLM
            batch_patterns = await self._analyze_batch_with_llm(batch, domain_hint)
            all_patterns.extend(batch_patterns)
            
            # Rate limiting to avoid API throttling
            if i + self.max_texts_per_batch < len(texts):
                await asyncio.sleep(1)  # 1 second between batches
        
        # Consolidate and rank patterns
        consolidated_patterns = await self._consolidate_llm_patterns(all_patterns)
        
        self.logger.info(f"LLM discovery complete: {len(consolidated_patterns)} semantic patterns")
        
        return consolidated_patterns
    
    async def _analyze_batch_with_llm(
        self, 
        texts: List[str], 
        domain_hint: Optional[str]
    ) -> List[LLMDiscoveredPattern]:
        """Analyze a batch of texts using LLM for semantic understanding"""
        
        # Construct LLM prompt for pattern discovery
        domain_context = f"in the {domain_hint} domain" if domain_hint else "across all domains"
        
        system_prompt = f"""You are an expert pattern analyst. Analyze the provided texts {domain_context} and identify:

1. Key entities (people, places, objects, concepts)
2. Important actions/processes  
3. Relationships between concepts
4. Domain-specific terminology
5. Semantic categories

For each pattern you identify, provide:
- The exact text/phrase
- Pattern type (entity/action/relationship/concept)
- Semantic category (what kind of entity/action it represents)
- Confidence score (0.0-1.0) based on semantic importance
- Brief reasoning for why this pattern is significant

Respond in JSON format with an array of patterns."""

        user_prompt = f"""Analyze these texts for semantic patterns:

{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts)])}

Return semantic patterns in JSON format:
{{
  "patterns": [
    {{
      "text": "pattern_text",
      "type": "entity|action|relationship|concept", 
      "category": "semantic_category",
      "confidence": 0.0-1.0,
      "reasoning": "why_this_pattern_matters"
    }}
  ]
}}"""

        try:
            # Call Azure OpenAI for semantic analysis
            response = await self.openai_client.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.llm_temperature,
                max_tokens=2000
            )
            
            # Parse LLM response
            try:
                llm_result = json.loads(response.choices[0].message.content)
                patterns = []
                
                for pattern_data in llm_result.get("patterns", []):
                    if pattern_data.get("confidence", 0) >= self.confidence_threshold:
                        pattern = LLMDiscoveredPattern(
                            pattern_text=pattern_data["text"],
                            pattern_type=pattern_data["type"], 
                            semantic_category=pattern_data["category"],
                            llm_confidence=pattern_data["confidence"],
                            llm_reasoning=pattern_data["reasoning"],
                            contexts=texts  # Associate with source texts
                        )
                        patterns.append(pattern)
                
                return patterns
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {e}")
                return []
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return []
    
    async def _consolidate_llm_patterns(
        self, 
        patterns: List[LLMDiscoveredPattern]
    ) -> List[LLMDiscoveredPattern]:
        """Consolidate similar patterns discovered by LLM"""
        
        # Group patterns by semantic similarity
        pattern_groups = {}
        
        for pattern in patterns:
            # Simple consolidation - in production would use embedding similarity
            key = (pattern.pattern_text.lower(), pattern.pattern_type)
            
            if key in pattern_groups:
                # Merge with existing pattern
                existing = pattern_groups[key]
                existing.contexts.extend(pattern.contexts)
                existing.llm_confidence = max(existing.llm_confidence, pattern.llm_confidence)
                existing.llm_reasoning += f"; {pattern.llm_reasoning}"
            else:
                pattern_groups[key] = pattern
        
        # Sort by LLM confidence
        consolidated = list(pattern_groups.values())
        consolidated.sort(key=lambda x: x.llm_confidence, reverse=True)
        
        return consolidated
    
    async def compare_statistical_vs_llm(
        self, 
        texts: List[str]
    ) -> Dict[str, Any]:
        """
        Compare statistical vs LLM discovery methods.
        
        Returns comparison analysis showing strengths/weaknesses of each approach.
        """
        results = {
            "statistical_method": {},
            "llm_method": {},
            "comparison": {}
        }
        
        # Statistical discovery (current method)
        import time
        start_time = time.time()
        
        # Simple statistical analysis (mimicking current system)
        statistical_patterns = []
        word_counts = {}
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        for word, count in word_counts.items():
            if count >= 2:  # Minimum frequency
                statistical_patterns.append({
                    "text": word,
                    "frequency": count,
                    "confidence": min(1.0, count / 10.0),
                    "method": "frequency_analysis"
                })
        
        statistical_time = time.time() - start_time
        
        results["statistical_method"] = {
            "patterns_found": len(statistical_patterns),
            "processing_time_seconds": statistical_time,
            "top_patterns": sorted(statistical_patterns, key=lambda x: x["confidence"], reverse=True)[:5],
            "advantages": [
                "extremely_fast",
                "no_api_costs", 
                "predictable_results",
                "works_offline"
            ],
            "disadvantages": [
                "no_semantic_understanding",
                "frequency_bias",
                "language_specific",
                "domain_assumptions_needed"
            ]
        }
        
        # LLM discovery (if available)
        if self.llm_available:
            start_time = time.time()
            
            try:
                llm_patterns = await self.discover_patterns_with_llm(texts[:5])  # Limit for demo
                llm_time = time.time() - start_time
                
                results["llm_method"] = {
                    "patterns_found": len(llm_patterns),
                    "processing_time_seconds": llm_time,
                    "top_patterns": [
                        {
                            "text": p.pattern_text,
                            "type": p.pattern_type,
                            "category": p.semantic_category,
                            "confidence": p.llm_confidence,
                            "reasoning": p.llm_reasoning
                        }
                        for p in llm_patterns[:5]
                    ],
                    "advantages": [
                        "semantic_understanding",
                        "context_aware",
                        "multi_language_capable",
                        "domain_adaptive"
                    ],
                    "disadvantages": [
                        "slower_processing",
                        "api_costs",
                        "requires_internet",
                        "less_predictable"
                    ]
                }
                
                # Comparison analysis
                results["comparison"] = {
                    "speed_ratio": f"Statistical {llm_time/statistical_time:.0f}x faster",
                    "pattern_overlap": self._calculate_pattern_overlap(statistical_patterns, llm_patterns),
                    "recommendation": self._generate_recommendation(statistical_time, llm_time, len(statistical_patterns), len(llm_patterns))
                }
                
            except Exception as e:
                results["llm_method"] = {"error": str(e)}
        else:
            results["llm_method"] = {"error": "LLM not available"}
        
        return results
    
    def _calculate_pattern_overlap(
        self, 
        statistical_patterns: List[Dict], 
        llm_patterns: List[LLMDiscoveredPattern]
    ) -> Dict[str, float]:
        """Calculate overlap between statistical and LLM discovered patterns"""
        
        stat_texts = set([p["text"].lower() for p in statistical_patterns])
        llm_texts = set([p.pattern_text.lower() for p in llm_patterns])
        
        if not stat_texts or not llm_texts:
            return {"overlap_percentage": 0.0}
        
        overlap = stat_texts & llm_texts
        
        return {
            "overlap_percentage": (len(overlap) / max(len(stat_texts), len(llm_texts))) * 100,
            "statistical_only": len(stat_texts - llm_texts),
            "llm_only": len(llm_texts - stat_texts),
            "common_patterns": list(overlap)
        }
    
    def _generate_recommendation(
        self, 
        stat_time: float, 
        llm_time: float, 
        stat_count: int, 
        llm_count: int
    ) -> str:
        """Generate recommendation for which method to use"""
        
        speed_ratio = llm_time / stat_time
        
        if speed_ratio > 100:  # LLM much slower
            return "Use statistical for real-time applications, LLM for accuracy"
        elif llm_count > stat_count * 1.5:  # LLM finds significantly more patterns
            return "LLM provides richer semantic understanding, worth the cost"
        else:
            return "Statistical method sufficient for this domain, save LLM costs"


# Factory function
async def create_llm_enhanced_discovery(config: Optional[Dict[str, Any]] = None) -> LLMEnhancedDiscovery:
    """Create LLM-enhanced discovery system"""
    default_config = {
        "max_texts_per_batch": 10,
        "llm_temperature": 0.1,
        "confidence_threshold": 0.7
    }
    
    if config:
        default_config.update(config)
    
    return LLMEnhancedDiscovery(default_config)


__all__ = [
    'LLMEnhancedDiscovery',
    'LLMDiscoveredPattern', 
    'create_llm_enhanced_discovery'
]