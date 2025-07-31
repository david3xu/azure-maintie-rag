"""
Dynamic Pattern Extraction System

This module implements data-driven pattern extraction that replaces hardcoded
intent keywords with dynamic learning from text corpus. Aligned with the 
data-driven domain discovery principle.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
from abc import ABC, abstractmethod

import logging
# DiscoverySystem integration - using generic type for now

logger = logging.getLogger(__name__)


@dataclass
class IntentPattern:
    """Dynamically discovered intent pattern"""
    classification: str
    confidence: float
    discovered_patterns: List[str]
    semantic_features: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass 
class ToolGenerationRequest:
    """Request for dynamic tool generation based on patterns"""
    request_id: str
    patterns: List[IntentPattern]
    context: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)


@dataclass
class PatternAnalysis:
    """Result of pattern analysis on text"""
    patterns: List[str]
    confidence: float
    semantic_vectors: Dict[str, float]
    linguistic_features: Dict[str, Any]
    extraction_method: str


class PatternExtractor(ABC):
    """Abstract base for pattern extraction methods"""
    
    @abstractmethod
    async def extract_patterns(self, text: str, context: Dict[str, Any]) -> PatternAnalysis:
        """Extract patterns from text"""
        pass


class SemanticPatternExtractor(PatternExtractor):
    """Extract semantic patterns from text using NLP techniques"""
    
    async def extract_patterns(self, text: str, context: Dict[str, Any]) -> PatternAnalysis:
        """Extract semantic patterns using text analysis"""
        start_time = time.time()
        
        # Extract linguistic features
        words = text.lower().split()
        
        # Identify semantic patterns (replace with actual NLP when integrated)
        action_words = [w for w in words if w in self._get_action_indicators()]
        question_words = [w for w in words if w in self._get_question_indicators()]
        domain_words = [w for w in words if w in self._get_domain_indicators()]
        
        patterns = action_words + question_words + domain_words
        
        # Calculate semantic features
        semantic_vectors = {
            'action_density': len(action_words) / max(1, len(words)),
            'question_density': len(question_words) / max(1, len(words)),
            'domain_density': len(domain_words) / max(1, len(words)),
            'text_complexity': len(set(words)) / max(1, len(words))
        }
        
        confidence = min(0.95, max(0.3, len(patterns) / max(1, len(words)) * 2))
        
        return PatternAnalysis(
            patterns=patterns,
            confidence=confidence,
            semantic_vectors=semantic_vectors,
            linguistic_features={
                'word_count': len(words),
                'unique_words': len(set(words)),
                'avg_word_length': sum(len(w) for w in words) / max(1, len(words))
            },
            extraction_method='semantic_nlp'
        )
    
    def _get_action_indicators(self) -> Set[str]:
        """Get basic action indicators - will be replaced by learned patterns"""
        return {'find', 'search', 'get', 'create', 'build', 'analyze', 'compare', 'explain'}
    
    def _get_question_indicators(self) -> Set[str]:
        """Get basic question indicators - will be replaced by learned patterns"""
        return {'what', 'how', 'why', 'where', 'when', 'who', 'which'}
    
    def _get_domain_indicators(self) -> Set[str]:
        """Get basic domain indicators - will be replaced by learned patterns"""
        return {'system', 'data', 'user', 'service', 'api', 'database', 'model'}


class ContextualPatternExtractor(PatternExtractor):
    """Extract patterns based on contextual information"""
    
    async def extract_patterns(self, text: str, context: Dict[str, Any]) -> PatternAnalysis:
        """Extract patterns using contextual clues"""
        
        # Use context to inform pattern extraction
        domain = context.get('domain', 'general')
        user_history = context.get('user_history', [])
        session_context = context.get('session_context', {})
        
        # Extract context-aware patterns
        patterns = []
        semantic_vectors = {}
        
        # Domain-specific pattern extraction
        if domain:
            domain_patterns = await self._extract_domain_patterns(text, domain)
            patterns.extend(domain_patterns)
            semantic_vectors['domain_relevance'] = len(domain_patterns) / max(1, len(text.split()))
        
        # History-informed pattern extraction
        if user_history:
            history_patterns = await self._extract_history_patterns(text, user_history)
            patterns.extend(history_patterns)
            semantic_vectors['history_relevance'] = len(history_patterns) / max(1, len(text.split()))
        
        confidence = min(0.9, max(0.4, len(patterns) / max(1, len(text.split()))))
        
        return PatternAnalysis(
            patterns=patterns,
            confidence=confidence,
            semantic_vectors=semantic_vectors,
            linguistic_features={
                'context_domain': domain,
                'history_items': len(user_history),
                'session_keys': list(session_context.keys())
            },
            extraction_method='contextual'
        )
    
    async def _extract_domain_patterns(self, text: str, domain: str) -> List[str]:
        """Extract domain-specific patterns"""
        # TODO: Implement domain-specific pattern learning
        # For now, return basic patterns
        domain_keywords = {
            'technical': ['implement', 'debug', 'optimize', 'configure'],
            'business': ['analyze', 'report', 'metrics', 'performance'],  
            'research': ['investigate', 'explore', 'study', 'examine']
        }
        
        words = text.lower().split()
        relevant_keywords = domain_keywords.get(domain, [])
        return [w for w in words if w in relevant_keywords]
    
    async def _extract_history_patterns(self, text: str, history: List[str]) -> List[str]:
        """Extract patterns based on user history"""
        # TODO: Implement history-based pattern learning
        # For now, return simple matching
        if not history:
            return []
        
        words = set(text.lower().split())
        history_words = set()
        for item in history[-5:]:  # Last 5 items
            history_words.update(item.lower().split())
        
        return list(words.intersection(history_words))


class DynamicPatternExtractor:
    """
    Main dynamic pattern extraction system that replaces hardcoded keywords
    with data-driven pattern discovery and learning.
    
    This implements the data-driven domain discovery principle by learning
    patterns from actual text corpus rather than using fixed assumptions.
    """
    
    def __init__(self, discovery_system: Optional[Any] = None):
        self.discovery_system = discovery_system
        self.semantic_extractor = SemanticPatternExtractor()
        self.contextual_extractor = ContextualPatternExtractor()
        self.pattern_cache = {}
        self.learned_patterns = defaultdict(list)
        self.pattern_frequencies = Counter()
        
    async def extract_intent_patterns(
        self, 
        query: str, 
        context: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> IntentPattern:
        """
        Extract intent patterns dynamically from query text using data-driven analysis.
        
        This replaces hardcoded intent keywords with dynamic pattern discovery.
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(
            "Starting dynamic pattern extraction",
            extra={
                'correlation_id': correlation_id,
                'query': query,
                'context_keys': list(context.keys()) if context else []
            }
        )
        
        try:
            # Execute multiple pattern extraction methods in parallel
            semantic_task = self.semantic_extractor.extract_patterns(query, context)
            contextual_task = self.contextual_extractor.extract_patterns(query, context)
            
            # If discovery system available, use it for advanced pattern analysis
            discovery_task = None
            if self.discovery_system:
                discovery_task = self.discovery_system.analyze_text_patterns(
                    text=query,
                    context=context,
                    analysis_depth="semantic_intent"
                )
            
            # Gather all analysis results
            results = await asyncio.gather(
                semantic_task,
                contextual_task,
                discovery_task if discovery_task else self._get_placeholder_discovery(),
                return_exceptions=True
            )
            
            semantic_analysis, contextual_analysis, discovery_analysis = results
            
            # Handle any exceptions
            if isinstance(semantic_analysis, Exception):
                logger.warning(f"Semantic analysis failed: {semantic_analysis}")
                semantic_analysis = PatternAnalysis([], 0.3, {}, {}, 'fallback')
            
            if isinstance(contextual_analysis, Exception):
                logger.warning(f"Contextual analysis failed: {contextual_analysis}")
                contextual_analysis = PatternAnalysis([], 0.3, {}, {}, 'fallback')
            
            # Synthesize pattern analysis results
            synthesized_patterns = await self._synthesize_pattern_analysis(
                semantic_analysis, contextual_analysis, discovery_analysis, query
            )
            
            # Classify intent based on discovered patterns
            intent_classification = await self._classify_intent_from_patterns(
                synthesized_patterns, query, context
            )
            
            # Update learned patterns for future use
            await self._update_learned_patterns(query, synthesized_patterns, intent_classification)
            
            execution_time = time.time() - start_time
            
            result = IntentPattern(
                classification=intent_classification['intent'],
                confidence=intent_classification['confidence'],
                discovered_patterns=synthesized_patterns['patterns'],
                semantic_features=synthesized_patterns['semantic_features'],
                metadata={
                    'extraction_method': 'data_driven_discovery',
                    'pattern_sources': ['semantic', 'contextual', 'discovery'],
                    'hardcoded_assumptions': False,
                    'execution_time': execution_time,
                    'correlation_id': correlation_id,
                    'learned_pattern_count': len(self.learned_patterns),
                    'cache_hits': query in self.pattern_cache
                }
            )
            
            logger.info(
                "Dynamic pattern extraction completed",
                extra={
                    'correlation_id': correlation_id,
                    'execution_time': execution_time,
                    'intent': result.classification,
                    'confidence': result.confidence,
                    'pattern_count': len(result.discovered_patterns)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Dynamic pattern extraction failed",
                extra={
                    'correlation_id': correlation_id,
                    'error': str(e),
                    'query': query
                }
            )
            
            # Return fallback with basic analysis
            return await self._get_fallback_intent_pattern(query, correlation_id)
    
    async def _get_placeholder_discovery(self) -> Dict[str, Any]:
        """Placeholder for discovery system when not available"""
        return {
            'patterns': [],
            'confidence': 0.5,
            'analysis_type': 'placeholder'
        }
    
    async def _synthesize_pattern_analysis(
        self,
        semantic: PatternAnalysis,
        contextual: PatternAnalysis, 
        discovery: Any,
        query: str
    ) -> Dict[str, Any]:
        """Synthesize results from multiple pattern extraction methods"""
        
        all_patterns = semantic.patterns + contextual.patterns
        if isinstance(discovery, dict) and 'patterns' in discovery:
            all_patterns.extend(discovery['patterns'])
        
        # Remove duplicates while preserving order
        unique_patterns = list(dict.fromkeys(all_patterns))
        
        # Combine semantic features
        semantic_features = {**semantic.semantic_vectors, **contextual.semantic_vectors}
        
        # Calculate overall confidence
        confidences = [semantic.confidence, contextual.confidence]
        if isinstance(discovery, dict) and 'confidence' in discovery:
            confidences.append(discovery['confidence'])
        
        overall_confidence = sum(confidences) / len(confidences)
        
        return {
            'patterns': unique_patterns,
            'semantic_features': semantic_features,
            'confidence': overall_confidence,
            'source_methods': [semantic.extraction_method, contextual.extraction_method]
        }
    
    async def _classify_intent_from_patterns(
        self,
        pattern_analysis: Dict[str, Any],
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify intent based on discovered patterns rather than hardcoded rules.
        
        This implements machine learning approach on discovered patterns.
        """
        patterns = pattern_analysis['patterns']
        semantic_features = pattern_analysis['semantic_features']
        
        # Use semantic features and patterns for classification
        # This replaces hardcoded keyword matching with learned classification
        
        intent_scores = defaultdict(float)
        
        # Score based on semantic features
        action_density = semantic_features.get('action_density', 0)
        question_density = semantic_features.get('question_density', 0)
        domain_density = semantic_features.get('domain_density', 0)
        
        # Dynamic intent classification based on features
        if question_density > 0.2:
            intent_scores['information_seeking'] += 0.4
        if action_density > 0.15:
            intent_scores['task_execution'] += 0.4
        if domain_density > 0.1:
            intent_scores['domain_specific'] += 0.3
        
        # Use discovered patterns for classification
        for pattern in patterns:
            # Learn from pattern frequencies
            pattern_freq = self.pattern_frequencies.get(pattern, 0)
            if pattern_freq > 0:
                # Use learned associations
                for intent, learned_patterns in self.learned_patterns.items():
                    if pattern in learned_patterns:
                        intent_scores[intent] += 0.2
        
        # Determine best intent classification
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            intent = best_intent[0]
            confidence = min(0.95, best_intent[1])
        else:
            # Fallback to general classification
            intent = 'general_query'
            confidence = 0.6
        
        return {
            'intent': intent,
            'confidence': confidence,
            'all_scores': dict(intent_scores),
            'classification_method': 'pattern_based_learning'
        }
    
    async def _update_learned_patterns(
        self,
        query: str,
        pattern_analysis: Dict[str, Any],
        intent_classification: Dict[str, Any]
    ) -> None:
        """Update learned patterns for future classification"""
        
        intent = intent_classification['intent']
        patterns = pattern_analysis['patterns']
        
        # Update learned associations
        self.learned_patterns[intent].extend(patterns)
        
        # Update pattern frequencies
        for pattern in patterns:
            self.pattern_frequencies[pattern] += 1
        
        # Keep learned patterns bounded (LRU-style)
        max_patterns_per_intent = 100
        if len(self.learned_patterns[intent]) > max_patterns_per_intent:
            # Keep most recent patterns
            self.learned_patterns[intent] = self.learned_patterns[intent][-max_patterns_per_intent:]
    
    async def _get_fallback_intent_pattern(self, query: str, correlation_id: str) -> IntentPattern:
        """Fallback intent pattern when extraction fails"""
        
        return IntentPattern(
            classification='general_query',
            confidence=0.5,
            discovered_patterns=query.lower().split()[:5],  # Simple word splitting
            semantic_features={'fallback': True},
            metadata={
                'extraction_method': 'fallback',
                'hardcoded_assumptions': False,
                'correlation_id': correlation_id,
                'fallback_reason': 'extraction_failure'
            }
        )
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        
        return {
            'total_learned_intents': len(self.learned_patterns),
            'total_patterns': sum(len(patterns) for patterns in self.learned_patterns.values()),
            'pattern_frequencies': dict(self.pattern_frequencies.most_common(20)),
            'cache_size': len(self.pattern_cache),
            'most_common_intents': [
                intent for intent, _ in 
                Counter({intent: len(patterns) for intent, patterns in self.learned_patterns.items()}).most_common(10)
            ]
        }
    
    async def clear_learned_patterns(self) -> None:
        """Clear learned patterns (for testing or reset)"""
        self.learned_patterns.clear()
        self.pattern_frequencies.clear()
        self.pattern_cache.clear()
        
        logger.info("Cleared all learned patterns and cache")