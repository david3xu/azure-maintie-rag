"""
Universal Query Analyzer
Replaces MaintenanceQueryAnalyzer with domain-agnostic query analysis
Works with any domain through dynamic LLM-powered pattern discovery
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import networkx as nx
from collections import defaultdict
from pathlib import Path
import json

from ..models.azure_rag_data_models import (
    UniversalQueryAnalysis, UniversalEnhancedQuery, QueryType, UniversalEntity
)
from ..azure_openai.text_processor import AzureOpenAITextProcessor
from ..azure_openai.extraction_client import OptimizedLLMExtractor
from ...config.settings import settings

logger = logging.getLogger(__name__)


class AzureSearchQueryAnalyzer:
    """Universal query analyzer that works with any domain through dynamic discovery"""

    def __init__(self, text_processor: Optional[AzureOpenAITextProcessor] = None, domain: str = "general"):
        """Initialize universal analyzer with dynamic domain discovery"""
        self.text_processor = text_processor
        self.domain = domain
        self.config = settings

        # Universal components (no domain assumptions)
        self.llm_extractor = OptimizedLLMExtractor(domain)
        self.knowledge_graph: Optional[nx.Graph] = None
        self.entity_vocabulary: Dict[str, Any] = {}

        # Dynamic discovery storage (replaces hardcoded patterns)
        self.discovered_concepts: Set[str] = set()
        self.discovered_entities: Set[str] = set()
        self.discovered_relations: Set[str] = set()
        self.query_patterns: Dict[str, List[str]] = {}

        # Universal query classification (no domain assumptions)
        self.universal_query_types = {
            QueryType.FACTUAL: ["what", "which", "who", "where", "when"],
            QueryType.PROCEDURAL: ["how", "steps", "process", "procedure", "guide"],
            QueryType.TROUBLESHOOTING: ["problem", "issue", "error", "fix", "solve"],
            QueryType.COMPARISON: ["compare", "difference", "vs", "versus", "better"],
            QueryType.EXPLANATION: ["why", "explain", "reason", "cause", "because"],
            QueryType.CLASSIFICATION: ["type", "category", "classify", "kind", "sort"]
        }

        # Cache for performance
        self._analysis_cache: Dict[str, UniversalQueryAnalysis] = {}
        self._concept_cache: Dict[str, List[str]] = {}

        # Load knowledge if text processor provided
        if self.text_processor:
            self._discover_domain_knowledge()

        logger.info(f"AzureSearchQueryAnalyzer initialized for domain: {domain}")

    def _discover_domain_knowledge(self):
        """Discover domain-specific knowledge from text content using LLM"""
        try:
            if not self.text_processor:
                logger.info("No text processor available for domain discovery")
                return

            if not hasattr(self.text_processor, 'documents') or not self.text_processor.documents:
                logger.info("No documents available in text processor for domain discovery")
                return

            documents = self.text_processor.documents
            if isinstance(documents, dict) and documents:
                sample_texts = []
                for doc_id, doc in list(documents.items())[:10]:
                    if hasattr(doc, 'text') and doc.text:
                        sample_texts.append(doc.text[:500])

                if sample_texts:
                    logger.info(f"Discovering domain knowledge from {len(sample_texts)} document samples")
                    # Use LLM to discover domain concepts
                    extraction_results = self.llm_extractor.extract_entities_and_relations(sample_texts)  # Sample for discovery

                    # Store discovered knowledge
                    self.discovered_entities.update(extraction_results.get('entities', []))
                    self.discovered_relations.update(extraction_results.get('relations', []))

                    # Build universal query patterns from discovered content
                    self._build_universal_patterns(sample_texts)  # Sample for pattern building

                    logger.info(f"Discovered {len(self.discovered_entities)} entity types, "
                              f"{len(self.discovered_relations)} relation types for domain: {self.domain}")
            else:
                logger.warning("No valid text content found for domain discovery")
        except Exception as e:
            logger.error(f"Domain discovery failed: {e}")
            self._load_universal_patterns()

    def _load_universal_patterns(self):
        """Load universal query patterns when domain discovery fails"""
        self.query_patterns = {
            "factual": ["what", "which", "who", "where", "when"],
            "procedural": ["how", "steps", "process", "procedure"],
            "troubleshooting": ["problem", "issue", "error", "fix"]
        }
        logger.info("Loaded universal query patterns")

    def _build_universal_patterns(self, text_samples: List[str]):
        """Build query patterns dynamically from text content"""
        try:
            # Sample text for pattern discovery
            sample_text = "\n".join(text_samples)[:2000]  # Limit for efficiency

            prompt = f"""
            Analyze this {self.domain} domain text and identify key concepts and terminology patterns.

            Text sample:
            {sample_text}

            Return 5-8 important concepts/terms that appear frequently in this domain.
            Format: JSON array of strings, lowercase with underscores.

            Example: ["medical_device", "patient_care", "diagnostic_procedure"]
            """

            try:
                response = self.llm_extractor.client.chat.completions.create(
                    model=self.llm_extractor.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.3
                )

                # Parse discovered concepts
                content = response.choices[0].message.content.strip()
                concepts = json.loads(content) if content.startswith('[') else []

                self.discovered_concepts.update(concepts)

                # Build simple patterns from discovered concepts
                self.query_patterns = {
                    "domain_concepts": list(self.discovered_concepts),
                    "entities": list(self.discovered_entities),
                    "relations": list(self.discovered_relations)
                }

            except Exception as e:
                logger.warning(f"LLM pattern discovery failed: {e}")
                # Fallback to simple keyword extraction
                self._extract_keywords_fallback(text_samples)

        except Exception as e:
            logger.warning(f"Pattern building failed: {e}")

    def _extract_keywords_fallback(self, text_samples: List[str]):
        """Fallback keyword extraction when LLM is unavailable"""
        all_text = " ".join(text_samples).lower()

        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
        word_freq = defaultdict(int)

        for word in words:
            if len(word) > 3:  # Skip very short words
                word_freq[word] += 1

        # Get top concepts
        top_concepts = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
        self.discovered_concepts.update(top_concepts)

        self.query_patterns = {"domain_concepts": top_concepts}

    def analyze_query_universal(self, query: str) -> UniversalQueryAnalysis:
        """Universal query analysis that works with any domain"""

        # Check cache first
        cache_key = f"{query.lower()}_{self.domain}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        logger.info(f"Analyzing query: {query}")

        try:
            # Step 1: Classify query type using universal patterns
            query_type, type_confidence = self._classify_query_universal(query)

            # Step 2: Extract entities using discovered domain knowledge
            entities_detected = self._extract_entities_universal(query)

            # Step 3: Detect domain concepts
            concepts_detected = self._detect_concepts_universal(query)

            # Step 4: Determine intent
            intent = self._determine_intent_universal(query, query_type)

            # Step 5: Assess complexity
            complexity = self._assess_complexity_universal(query)

            # Create analysis
            analysis = UniversalQueryAnalysis(
                query_text=query,
                query_type=query_type,
                confidence=type_confidence,
                entities_detected=entities_detected,
                concepts_detected=concepts_detected,
                intent=intent,
                complexity=complexity,
                metadata={
                    "domain": self.domain,
                    "analysis_method": "universal_llm_guided",
                    "discovered_concepts_used": len(self.discovered_concepts) > 0
                }
            )

            # Cache result
            self._analysis_cache[cache_key] = analysis

            logger.info(f"Query analysis complete: {query_type.value}, {len(entities_detected)} entities")
            return analysis

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return self._create_fallback_analysis(query)

    def _classify_query_universal(self, query: str) -> Tuple[QueryType, float]:
        """Universal query classification without domain assumptions"""
        query_lower = query.lower()

        # Score each query type
        type_scores = {}

        for query_type, keywords in self.universal_query_types.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1

            # Normalize score
            if keywords:
                type_scores[query_type] = score / len(keywords)

        # Find best match
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0], min(best_type[1] * 2, 1.0)  # Scale confidence

        # Fallback classification
        return QueryType.UNKNOWN, 0.5

    def _extract_entities_universal(self, query: str) -> List[str]:
        """Extract entities from query using discovered domain knowledge"""
        entities = []
        query_lower = query.lower()

        # Use discovered entities
        for entity in self.discovered_entities:
            entity_lower = entity.lower().replace('_', ' ')
            if entity_lower in query_lower:
                entities.append(entity)

        # Use discovered concepts as potential entities
        for concept in self.discovered_concepts:
            concept_lower = concept.lower().replace('_', ' ')
            if concept_lower in query_lower:
                entities.append(concept)

        return list(set(entities))  # Remove duplicates

    def _detect_concepts_universal(self, query: str) -> List[str]:
        """Detect domain concepts in query"""
        concepts = []
        query_lower = query.lower()

        # Check discovered patterns
        for pattern_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                pattern_lower = pattern.lower().replace('_', ' ')
                if pattern_lower in query_lower:
                    concepts.append(pattern)

        return list(set(concepts))

    def _determine_intent_universal(self, query: str, query_type: QueryType) -> str:
        """Determine query intent universally"""
        intents = {
            QueryType.FACTUAL: "information_seeking",
            QueryType.PROCEDURAL: "instruction_seeking",
            QueryType.TROUBLESHOOTING: "problem_solving",
            QueryType.COMPARISON: "evaluation",
            QueryType.EXPLANATION: "understanding",
            QueryType.CLASSIFICATION: "categorization"
        }

        return intents.get(query_type, "general_inquiry")

    def _assess_complexity_universal(self, query: str) -> str:
        """Assess query complexity universally"""
        words = len(query.split())

        if words <= 5:
            return "simple"
        elif words <= 12:
            return "medium"
        else:
            return "complex"

    def enhance_query_universal(self, query: str) -> UniversalEnhancedQuery:
        """Enhance query with expanded concepts and terms"""

        # Analyze query first
        analysis = self.analyze_query_universal(query)

        # Expand concepts based on discovered knowledge
        expanded_concepts = list(set(
            analysis.entities_detected +
            analysis.concepts_detected +
            self._get_related_concepts(query)
        ))

        # Create search terms
        search_terms = [query] + expanded_concepts

        # Create enhanced query
        enhanced_query = UniversalEnhancedQuery(
            original_query=query,
            expanded_concepts=expanded_concepts,
            related_entities=analysis.entities_detected,
            query_analysis=analysis,
            search_terms=search_terms,
            metadata={
                "domain": self.domain,
                "enhancement_method": "universal_discovery",
                "concepts_expanded": len(expanded_concepts)
            }
        )

        logger.info(f"Query enhanced: {len(expanded_concepts)} concepts added")
        return enhanced_query

    def _get_related_concepts(self, query: str) -> List[str]:
        """Get related concepts for query expansion"""
        related = []
        query_lower = query.lower()

        # Simple related concept discovery
        for concept in self.discovered_concepts:
            concept_words = concept.lower().split('_')
            for word in concept_words:
                if word in query_lower and concept not in related:
                    related.append(concept)
                    break

        return related[:5]  # Limit expansion

    def _create_fallback_analysis(self, query: str) -> UniversalQueryAnalysis:
        """Create fallback analysis when main analysis fails"""
        return UniversalQueryAnalysis(
            query_text=query,
            query_type=QueryType.UNKNOWN,
            confidence=0.3,
            entities_detected=[],
            concepts_detected=[],
            intent="general_inquiry",
            complexity="medium",
            metadata={
                "domain": self.domain,
                "analysis_method": "fallback",
                "note": "Analysis failed, using fallback"
            }
        )

    # Simplified methods for backward compatibility
    def analyze_query_simple(self, query: str) -> UniversalQueryAnalysis:
        """Simplified analysis method (backward compatibility)"""
        return self.analyze_query_universal(query)

    def enhance_query_simple(self, query: str) -> UniversalEnhancedQuery:
        """Simplified enhancement method (backward compatibility)"""
        return self.enhance_query_universal(query)

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered domain knowledge"""
        return {
            "domain": self.domain,
            "discovered_entities": len(self.discovered_entities),
            "discovered_relations": len(self.discovered_relations),
            "discovered_concepts": len(self.discovered_concepts),
            "query_patterns": len(self.query_patterns),
            "cached_analyses": len(self._analysis_cache)
        }


def create_universal_analyzer(text_processor: Optional[AzureOpenAITextProcessor] = None,
                            domain: str = "general") -> AzureSearchQueryAnalyzer:
    """Factory function to create universal query analyzer"""
    return AzureSearchQueryAnalyzer(text_processor, domain)


# Universal RAG - no backward compatibility needed


if __name__ == "__main__":
    # Example usage
    analyzer = AzureSearchQueryAnalyzer(domain="maintenance")

    query = "How do I fix a pump failure?"
    analysis = analyzer.analyze_query_universal(query)
    enhanced = analyzer.enhance_query_universal(query)

    print(f"Analysis: {analysis.to_dict()}")
    print(f"Enhanced: {enhanced.to_dict()}")