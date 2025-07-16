"""
Base LLM Knowledge Extractor
Provides core functionality for extracting knowledge from text using LLMs
"""

import logging
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMKnowledgeExtractor(ABC):
    """Base class for LLM-powered knowledge extraction"""

    def __init__(self, domain_name: str, cache_dir: Optional[Path] = None):
        """Initialize LLM knowledge extractor"""
        self.domain_name = domain_name
        self.cache_dir = cache_dir or Path("data/cache/llm_extraction")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Knowledge storage
        self.discovered_entities: Set[str] = set()
        self.discovered_relations: Set[str] = set()
        self.entity_patterns: Dict[str, List[str]] = {}
        self.relation_patterns: Dict[str, List[str]] = {}

        logger.info(f"LLMKnowledgeExtractor initialized for domain: {domain_name}")

    @abstractmethod
    def extract_entities_and_relations(self, texts: List[str]) -> Dict[str, Any]:
        """Extract entities and relations from texts using LLM"""
        pass

    @abstractmethod
    def generate_domain_schema(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate domain schema from extraction results"""
        pass

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key"""
        return self.cache_dir / f"{self.domain_name}_{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load results from cache if available"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save results to cache"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results cached to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

    def _validate_extraction_results(self, results: Dict[str, Any]) -> bool:
        """Validate extraction results structure"""
        required_keys = ['entities', 'relations', 'confidence_score']
        return all(key in results for key in required_keys)

    def _merge_extraction_results(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple extraction results"""
        merged_entities = set()
        merged_relations = set()
        total_confidence = 0.0

        for results in results_list:
            if self._validate_extraction_results(results):
                merged_entities.update(results['entities'])
                merged_relations.update(results['relations'])
                total_confidence += results.get('confidence_score', 0.0)

        avg_confidence = total_confidence / len(results_list) if results_list else 0.0

        return {
            'entities': list(merged_entities),
            'relations': list(merged_relations),
            'confidence_score': avg_confidence,
            'extraction_count': len(results_list),
            'timestamp': datetime.now().isoformat()
        }

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about extracted knowledge"""
        return {
            'domain': self.domain_name,
            'entities_discovered': len(self.discovered_entities),
            'relations_discovered': len(self.discovered_relations),
            'entity_patterns': len(self.entity_patterns),
            'relation_patterns': len(self.relation_patterns)
        }