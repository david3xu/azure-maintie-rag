"""
Optimized LLM Knowledge Extractor - performance and cost optimized
"""

import logging
import hashlib
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import random
from datetime import datetime
from openai import AzureOpenAI

from ..models.azure_rag_data_models import UniversalEntity, UniversalRelation
from ...config.settings import settings

logger = logging.getLogger(__name__)


class OptimizedLLMExtractor:
    """Performance and cost optimized LLM knowledge extractor"""

    def __init__(self, domain_name: str, cache_dir: Optional[Path] = None):
        super().__init__(domain_name, cache_dir)
        self.cache_dir = cache_dir or (settings.BASE_DIR / "data" / "cache" / "llm_extractions")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Performance settings
        self.max_texts_for_discovery = 100  # Sample size for entity/relation discovery
        self.batch_size = 10  # Process multiple texts per LLM call
        self.cache_enabled = True

        # Azure OpenAI client setup
        self.deployment_name = settings.openai_deployment_name
        self.client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
            azure_endpoint=settings.openai_api_base
        )

        logger.info(f"OptimizedLLMExtractor initialized with Azure deployment {self.deployment_name}")

    def extract_domain_knowledge(self, text_corpus: List[str]) -> Dict[str, Any]:
        """Optimized domain knowledge extraction with caching and sampling"""

        logger.info(f"Optimized extraction for {len(text_corpus)} texts")

        # Check cache first
        cache_key = self._generate_cache_key(text_corpus)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            logger.info("Using cached extraction results")
            return cached_result

        # Sample texts for entity/relation discovery (performance optimization)
        discovery_sample = self._sample_texts_for_discovery(text_corpus)
        logger.info(f"Using {len(discovery_sample)} texts for entity/relation discovery")

        # Phase 1: Discover entities and relationships (sampled)
        entities = self._discover_entities_optimized(discovery_sample)
        relationships = self._discover_relationships_optimized(discovery_sample, entities)

        # Phase 2: Extract triplets (all texts, but batched)
        triplets = self._extract_triplets_batched(text_corpus, entities, relationships)

        result = {
            "domain_name": self.domain_name,
            "entities": entities,
            "relationships": relationships,
            "triplets": triplets,
            "statistics": {
                "documents_processed": len(text_corpus),
                "discovery_sample_size": len(discovery_sample),
                "entities_discovered": len(entities),
                "relationships_discovered": len(relationships),
                "triplets_extracted": len(triplets)
            }
        }

        # Cache result
        self._save_to_cache(cache_key, result)

        return result

    def _sample_texts_for_discovery(self, text_corpus: List[str]) -> List[str]:
        """Sample texts for entity/relation discovery to improve performance"""

        if len(text_corpus) <= self.max_texts_for_discovery:
            return text_corpus

        # Stratified sampling: take texts from different parts of corpus
        sample_indices = []
        step = len(text_corpus) // self.max_texts_for_discovery

        for i in range(0, len(text_corpus), step):
            if len(sample_indices) < self.max_texts_for_discovery:
                sample_indices.append(i)

        # Add some random samples for diversity
        remaining_slots = self.max_texts_for_discovery - len(sample_indices)
        if remaining_slots > 0:
            available_indices = set(range(len(text_corpus))) - set(sample_indices)
            random_indices = random.sample(list(available_indices),
                                         min(remaining_slots, len(available_indices)))
            sample_indices.extend(random_indices)

        return [text_corpus[i] for i in sample_indices]

    def _discover_entities_optimized(self, text_sample: List[str]) -> List[str]:
        """Optimized entity discovery using multiple strategies"""

        # Strategy 1: Batch processing of multiple texts
        batched_texts = self._create_batches(text_sample, self.batch_size)
        all_entities = set()

        for batch in batched_texts[:5]:  # Limit to 5 batches for discovery
            batch_text = "\n---\n".join(batch)

            prompt = f"""
            Analyze the following {self.domain_name} text samples and identify entity types.

            Guidelines:
            1. Return 8-12 most important entity types
            2. Use lowercase with underscores (e.g., "system_component")
            3. Focus on concrete, identifiable objects/concepts
            4. Avoid overly specific or generic terms

            Text samples:
            {batch_text[:2000]}  # Limit context size

            Entity types (JSON array only, no explanation):
            """

            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,  # Shorter response
                    temperature=0.3   # More deterministic
                )

                batch_entities = self._parse_entity_response(response.choices[0].message.content)
                all_entities.update(batch_entities)

            except Exception as e:
                logger.warning(f"Entity discovery batch failed: {e}")
                continue

        # Return top entities by frequency/importance
        entity_list = list(all_entities)
        return entity_list[:15]  # Reasonable limit

    def _discover_relationships_optimized(self, text_sample: List[str], entities: List[str]) -> List[str]:
        """Optimized relationship discovery using entity context"""

        # Strategy: Use discovered entities to guide relationship discovery
        batched_texts = self._create_batches(text_sample, self.batch_size)
        all_relationships = set()

        # Sample entity types for context
        entity_context = entities[:10] if entities else ["component", "system", "process", "action"]

        for batch in batched_texts[:5]:  # Limit to 5 batches for discovery
            batch_text = "\n---\n".join(batch)

            prompt = f"""
            Analyze the following {self.domain_name} text samples and identify relationship types between entities.

            Known entity types: {', '.join(entity_context)}

            Guidelines:
            1. Return 6-10 most important relationship types
            2. Use lowercase with underscores (e.g., "requires", "part_of", "causes")
            3. Focus on meaningful connections between entities
            4. Avoid overly specific relationships

            Text samples:
            {batch_text[:2000]}  # Limit context size

            Relationship types (JSON array only, no explanation):
            """

            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,  # Shorter response
                    temperature=0.3   # More deterministic
                )

                batch_relationships = self._parse_relationship_response(response.choices[0].message.content)
                all_relationships.update(batch_relationships)

            except Exception as e:
                logger.warning(f"Relationship discovery batch failed: {e}")
                continue

        # Return top relationships
        relationship_list = list(all_relationships)
        return relationship_list[:12]  # Reasonable limit

    def _extract_triplets_batched(self, text_corpus: List[str], entities: List[str], relationships: List[str]) -> List[tuple]:
        """Extract triplets using batched processing"""

        all_triplets = []
        text_batches = self._create_batches(text_corpus, self.batch_size)

        # Process only a reasonable number of batches
        max_batches = min(50, len(text_batches))  # Limit processing

        for i, batch in enumerate(text_batches[:max_batches]):
            if i % 10 == 0:
                logger.info(f"Processing triplet batch {i+1}/{max_batches}")

            batch_text = "\n---\n".join(batch)

            prompt = f"""
            Extract knowledge triplets from the {self.domain_name} text below.

            Entity types: {', '.join(entities[:10])}  # Limit context
            Relationship types: {', '.join(relationships[:8])}

            Text:
            {batch_text[:1500]}  # Limit context size

            Return up to 10 triplets in format: [("entity1", "relation", "entity2"), ...]
            JSON array only:
            """

            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.4
                )

                batch_triplets = self._parse_triplet_response(response.choices[0].message.content)
                all_triplets.extend(batch_triplets)

            except Exception as e:
                logger.warning(f"Triplet extraction batch {i} failed: {e}")
                continue

        logger.info(f"Extracted {len(all_triplets)} triplets from {max_batches} batches")
        return all_triplets

    def _create_batches(self, texts: List[str], batch_size: int) -> List[List[str]]:
        """Create batches of texts for processing"""
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append(batch)
        return batches

    def _generate_cache_key(self, text_corpus: List[str]) -> str:
        """Generate cache key for corpus"""
        corpus_hash = hashlib.md5()
        for text in text_corpus[:100]:  # Sample for hash
            corpus_hash.update(text.encode('utf-8'))

        return f"{self.domain_name}_{corpus_hash.hexdigest()[:12]}"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached extraction results"""
        if not self.cache_enabled:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save extraction results to cache"""
        if not self.cache_enabled:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def _extract_entities(self, texts: List[str]) -> List[str]:
        """Extract entities from texts using LLM"""
        entities = set()

        # Use the optimized entity discovery
        discovered_entities = self._discover_entities_optimized(texts)
        entities.update(discovered_entities)

        return list(entities)

    def _extract_relationships(self, texts: List[str]) -> List[str]:
        """Extract relationships from texts using LLM"""
        relationships = set()

        # Use the optimized relationship discovery
        discovered_relationships = self._discover_relationships_optimized(texts, [])
        relationships.update(discovered_relationships)

        return list(relationships)

    def extract_entities_and_relations(self, texts: List[str]) -> Dict[str, Any]:
        """Extract entities and relations from texts using optimized LLM approach"""
        logger.info(f"Starting optimized extraction for {len(texts)} texts")

        # Use sampling strategy for discovery
        sample_texts = self._sample_texts_for_discovery(texts)
        logger.info(f"Using {len(sample_texts)} texts for knowledge discovery")

        # Extract entities and relationships
        entities = self._extract_entities(sample_texts)
        relationships = self._extract_relationships(sample_texts)

        # Store discovered knowledge
        self.discovered_entities.update(entities)
        self.discovered_relations.update(relationships)

        # Calculate confidence based on extraction quality
        confidence_score = min(0.9, len(entities) / 50 + len(relationships) / 30)

        result = {
            'entities': list(entities),
            'relations': list(relationships),
            'confidence_score': confidence_score,
            'texts_processed': len(sample_texts),
            'domain': self.domain_name,
            'extraction_method': 'optimized_llm'
        }

        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relations")
        return result

    def generate_domain_schema(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate domain schema from extraction results"""
        entities = extraction_results.get('entities', [])
        relations = extraction_results.get('relations', [])

        logger.info(f"Generating schema for {len(entities)} entities and {len(relations)} relations")

        # Create entity types with categorization
        entity_types = {}
        for entity in entities[:50]:  # Limit for schema
            entity_types[entity] = {
                'type': 'discovered_entity',
                'category': self._categorize_entity(entity),
                'confidence': 0.8,
                'source': 'llm_extraction'
            }

        # Create relation types
        relation_types = {}
        for relation in relations[:30]:  # Limit for schema
            relation_types[relation] = {
                'type': 'discovered_relation',
                'category': self._categorize_relation(relation),
                'confidence': 0.8,
                'source': 'llm_extraction'
            }

        schema = {
            'domain_name': self.domain_name,
            'version': '1.0.0',
            'entity_types': entity_types,
            'relation_types': relation_types,
            'extraction_metadata': {
                'total_entities': len(entities),
                'total_relations': len(relations),
                'confidence': extraction_results.get('confidence_score', 0.0),
                'generated_at': datetime.now().isoformat()
            },
            'processing_settings': {
                'max_entities_in_schema': 50,
                'max_relations_in_schema': 30,
                'confidence_threshold': 0.7
            }
        }

        logger.info(f"Generated schema with {len(entity_types)} entity types and {len(relation_types)} relation types")
        return schema

    def _categorize_entity(self, entity: str) -> str:
        """Categorize entity type based on content"""
        entity_lower = entity.lower()

        if any(word in entity_lower for word in ['component', 'system', 'unit', 'module']):
            return 'component'
        elif any(word in entity_lower for word in ['part', 'component', 'element']):
            return 'component'
        elif any(word in entity_lower for word in ['problem', 'issue', 'fault', 'error']):
            return 'issue'
        elif any(word in entity_lower for word in ['action', 'task', 'procedure']):
            return 'action'
        else:
            return 'general'

    def _categorize_relation(self, relation: str) -> str:
        """Categorize relation type based on content"""
        relation_lower = relation.lower()

        if any(word in relation_lower for word in ['requires', 'needs', 'depends']):
            return 'dependency'
        elif any(word in relation_lower for word in ['causes', 'leads', 'results']):
            return 'causality'
        elif any(word in relation_lower for word in ['part_of', 'contains', 'includes']):
            return 'composition'
        elif any(word in relation_lower for word in ['located', 'position', 'place']):
            return 'spatial'
        else:
            return 'general'

    def _parse_entity_response(self, response_content: str) -> List[str]:
        """Parse entity response from LLM"""
        try:
            # Try to parse as JSON array
            import re
            # Extract JSON array from response
            json_match = re.search(r'\[(.*?)\]', response_content, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                entities = json.loads(json_str)
                return [str(e).strip().lower().replace(' ', '_') for e in entities if e]

            # Fallback: parse line by line
            lines = response_content.strip().split('\n')
            entities = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Remove quotes and clean
                    entity = line.strip('"').strip("'").strip()
                    if entity:
                        entities.append(entity.lower().replace(' ', '_'))

            return entities[:15]  # Limit

        except Exception as e:
            logger.warning(f"Failed to parse entity response: {e}")
            return []

    def _parse_relationship_response(self, response_content: str) -> List[str]:
        """Parse relationship response from LLM"""
        try:
            # Try to parse as JSON array
            import re
            # Extract JSON array from response
            json_match = re.search(r'\[(.*?)\]', response_content, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                relationships = json.loads(json_str)
                return [str(r).strip().lower().replace(' ', '_') for r in relationships if r]

            # Fallback: parse line by line
            lines = response_content.strip().split('\n')
            relationships = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Remove quotes and clean
                    relationship = line.strip('"').strip("'").strip()
                    if relationship:
                        relationships.append(relationship.lower().replace(' ', '_'))

            return relationships[:12]  # Limit

        except Exception as e:
            logger.warning(f"Failed to parse relationship response: {e}")
            return []

    def _parse_triplet_response(self, response_content: str) -> List[tuple]:
        """Parse triplet response from LLM"""
        try:
            # Try to parse as JSON array of tuples
            import re
            # Extract JSON array from response
            json_match = re.search(r'\[(.*?)\]', response_content, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                triplets_raw = json.loads(json_str)
                triplets = []
                for item in triplets_raw:
                    if isinstance(item, list) and len(item) == 3:
                        triplets.append(tuple(str(x).strip() for x in item))
                return triplets[:20]  # Limit

            # Fallback: empty list
            return []

        except Exception as e:
            logger.warning(f"Failed to parse triplet response: {e}")
            return []