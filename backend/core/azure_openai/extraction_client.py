"""
Optimized LLM Knowledge Extractor - performance and cost optimized
"""

import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import random
from datetime import datetime
from openai import AzureOpenAI

from ..models.universal_rag_models import UniversalEntity, UniversalRelation
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings

logger = logging.getLogger(__name__)


class OptimizedLLMExtractor:
    """Performance and cost optimized LLM knowledge extractor"""

    def __init__(self, domain_name: str, cache_dir: Optional[Path] = None):
        # Remove the incorrect super().__init__ call
        self.domain_name = domain_name
        self.cache_dir = cache_dir or (settings.BASE_DIR / "data" / "cache" / "llm_extractions")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Performance settings
        self.max_texts_for_discovery = 100
        self.batch_size = 10
        self.cache_enabled = True

        # Azure OpenAI client setup
        self.deployment_name = settings.openai_deployment_name
        self.client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
            azure_endpoint=settings.openai_api_base
        )

        logger.info(f"OptimizedLLMExtractor initialized with Azure deployment {self.deployment_name}")
        # Knowledge Discovery State Management - Azure Enterprise Pattern
        self.discovered_entities: Set[str] = set()
        self.discovered_relations: Set[str] = set()

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

        max_discovery_batches = getattr(settings, 'max_discovery_batches', 20)
        for batch in batched_texts[:max_discovery_batches]:
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
        max_entity_types = getattr(settings, 'max_entity_types_discovery', 50)
        return entity_list[:max_entity_types]  # Reasonable limit

    def _discover_relationships_optimized(self, text_sample: List[str], entities: List[str]) -> List[str]:
        """Optimized relationship discovery using entity context"""

        # Strategy: Use discovered entities to guide relationship discovery
        batched_texts = self._create_batches(text_sample, self.batch_size)
        all_relationships = set()

        # Remove hardcoded fallback for entity context
        # entity_context = entities[:10] if entities else ["component", "system", "process", "action"]
        entity_context = entities[:10] if entities else []

        max_discovery_batches = getattr(settings, 'max_discovery_batches', 20)
        for batch in batched_texts[:max_discovery_batches]:
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
        max_relation_types = getattr(settings, 'max_relation_types_discovery', 30)
        return relationship_list[:max_relation_types]  # Reasonable limit

    def _parse_entity_response_fallback(self, response_content: str, attempt: int, max_retries: int) -> List[str]:
        """Fallback entity parsing using simpler approach"""
        try:
            # Simple word extraction fallback
            import re
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', response_content)
            # Filter common words and take reasonable entities
            filtered_words = [w.lower() for w in words if len(w) > 2 and w.lower() not in ['the', 'and', 'for', 'are', 'with']]
            return list(set(filtered_words))[:10]  # Return unique, limited set
        except Exception as e:
            logger.error(f"Fallback entity parsing also failed: {e}")
            return []

    def _parse_relationship_response_fallback(self, response_content: str, attempt: int, max_retries: int) -> List[str]:
        """Fallback relationship parsing using simpler approach"""
        try:
            # Look for verb-like patterns
            import re
            verbs = re.findall(r'\b[a-zA-Z_]+(?:es|ed|ing|s)?\b', response_content)
            # Filter to likely relationship words
            filtered_verbs = [v.lower() for v in verbs if len(v) > 2 and v.lower() not in ['the', 'and', 'for', 'are', 'with']]
            return list(set(filtered_verbs))[:8]  # Return unique, limited set
        except Exception as e:
            logger.error(f"Fallback relationship parsing also failed: {e}")
            return []

    def _parse_triplet_response_fallback(self, response_content: str, attempt: int, max_retries: int) -> List[tuple]:
        """Fallback triplet parsing using simpler approach"""
        try:
            # Simple pattern matching for triplet-like structures
            import re
            # Look for patterns like (entity1, relation, entity2)
            triplet_matches = re.findall(r'\([^)]+,[^)]+,[^)]+\)', response_content)
            triplets = []
            for match in triplet_matches[:10]:  # Limit results
                # Clean and split
                clean_match = match.strip('()').split(',')
                if len(clean_match) == 3:
                    triplets.append(tuple(item.strip().strip('"').strip("'") for item in clean_match))
            return triplets
        except Exception as e:
            logger.error(f"Fallback triplet parsing also failed: {e}")
            return []

    def _extract_triplets_batched(self, text_corpus: List[str], entities: List[str], relationships: List[str]) -> List[tuple]:
        """Extract triplets using batched processing"""

        all_triplets = []
        text_batches = self._create_batches(text_corpus, self.batch_size)

        # Process only a reasonable number of batches
        max_triplet_batches = getattr(settings, 'max_triplet_extraction_batches', 100)
        max_batches = min(max_triplet_batches, len(text_batches))

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
        """Extract relationships from texts using LLM with proper entity context"""
        relationships = set()

        # FIX: Ensure entity context is available for relationship discovery
        # First extract entities to provide context for relationships
        discovered_entities = self._discover_entities_optimized(texts)
        # Now use proper entity context for relationship discovery
        discovered_relationships = self._discover_relationships_optimized(texts, discovered_entities)
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

    # Refactor prompt generation to use universal, non-biased instructions
    def generate_universal_entity_prompt(self, texts: List[str]) -> str:
        return f"""
        Analyze the following text and extract all significant entities.
        Universal Instructions:
        1. Identify noun phrases, key concepts, and important terms
        2. Do not impose predetermined categories or types
        3. Let the entities emerge naturally from the text
        4. Focus on terms that carry semantic meaning
        5. Extract entities as they appear in the text
        Text samples:
        {chr(10).join(texts[:3])}
        Return entities as JSON array of strings (entity names only):
        """

    def generate_universal_relationship_prompt(self, texts: List[str], entities: List[str] = None) -> str:
        entity_context = ""
        if entities:
            entity_context = f"Focus on relationships involving these entities: {', '.join(entities[:10])}"
        return f"""
        Analyze the following text and identify relationships between entities.
        {entity_context}
        Universal Instructions:
        1. Identify relationships as they are expressed in the text
        2. Use simple, descriptive terms for relationship types
        3. Do not assume domain-specific relationship categories
        4. Focus on actual connections mentioned or implied in the text
        5. Use verbs or verb phrases that describe the connections
        Text samples:
        {chr(10).join(texts[:3])}
        Return relationship types as JSON array of strings:
        """

    # Remove any mention of hardcoded relationship types in docstrings, comments, or code
    def _categorize_relation(self, relation: str) -> str:
        """Universal relation categorization (no hardcoded types)"""
        # Instead of hardcoded categories, return the relation as-is or use a universal prompt
        return relation

    def _categorize_entity(self, entity: str) -> str:
        """Universal entity categorization (no hardcoded types)"""
        return entity

    def _parse_entity_response(self, response_content: str, attempt: int = 1, max_retries: int = 2) -> List[str]:
        """Parse entity response from LLM with retry logic"""
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
            logger.error(f"Failed to parse entity response (attempt {attempt}): {e}")
            if attempt < max_retries:
                logger.info(f"Retrying entity parsing, attempt {attempt + 1}")
                # Try alternative parsing approach
                return self._parse_entity_response_fallback(response_content, attempt + 1, max_retries)
            else:
                logger.warning("Entity parsing failed after all retries, using empty result")
                return []

    def _parse_relationship_response(self, response_content: str, attempt: int = 1, max_retries: int = 2) -> List[str]:
        """Parse relationship response from LLM with retry logic"""
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
            logger.error(f"Failed to parse relationship response (attempt {attempt}): {e}")
            if attempt < max_retries:
                logger.info(f"Retrying relationship parsing, attempt {attempt + 1}")
                return self._parse_relationship_response_fallback(response_content, attempt + 1, max_retries)
            else:
                logger.warning("Relationship parsing failed after all retries, using empty result")
                return []

    def _parse_triplet_response(self, response_content: str, attempt: int = 1, max_retries: int = 2) -> List[tuple]:
        """Parse triplet response from LLM with retry logic"""
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
            logger.error(f"Failed to parse triplet response (attempt {attempt}): {e}")
            if attempt < max_retries:
                logger.info(f"Retrying triplet parsing, attempt {attempt + 1}")
                return self._parse_triplet_response_fallback(response_content, attempt + 1, max_retries)
            else:
                logger.warning("Triplet parsing failed after all retries, using empty result")
                return []