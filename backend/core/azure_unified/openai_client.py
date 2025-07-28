"""
Unified Azure OpenAI Client
Consolidates all OpenAI functionality: extraction, completion, text processing, rate limiting
Replaces: knowledge_extractor.py, extraction_client.py, completion_service.py, text_processor.py, etc.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
from openai import AzureOpenAI

from .base_client import BaseAzureClient
from config.settings import settings, azure_settings
from ..models.universal_rag_models import UniversalEntity, UniversalRelation

logger = logging.getLogger(__name__)


class UnifiedAzureOpenAIClient(BaseAzureClient):
    """Unified client for all Azure OpenAI operations"""
    
    def _get_default_endpoint(self) -> str:
        return azure_settings.openai_api_base
        
    def _get_default_key(self) -> str:
        return azure_settings.openai_api_key
        
    def _initialize_client(self):
        """Initialize Azure OpenAI client"""
        self._client = AzureOpenAI(
            api_key=self.key,
            api_version=azure_settings.openai_api_version,
            azure_endpoint=self.endpoint
        )
        
        # Rate limiting
        self.rate_limiter = SimpleRateLimiter()
        
    # === KNOWLEDGE EXTRACTION ===
    
    async def extract_knowledge(self, texts: List[str], domain: str = "general") -> Dict[str, Any]:
        """Extract entities and relationships from texts"""
        self.ensure_initialized()
        
        try:
            all_entities = []
            all_relationships = []
            
            for i, text in enumerate(texts):
                result = await self._extract_from_single_text(text, domain)
                
                if result['success']:
                    all_entities.extend(result['entities'])
                    all_relationships.extend(result['relationships'])
                    
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
            
            # Deduplicate and assign IDs
            entities = self._deduplicate_entities(all_entities)
            relationships = self._deduplicate_relationships(all_relationships)
            
            return self.create_success_response('extract_knowledge', {
                'entities': entities,
                'relationships': relationships,
                'total_texts': len(texts),
                'entities_count': len(entities),
                'relationships_count': len(relationships)
            })
            
        except Exception as e:
            return self.handle_azure_error('extract_knowledge', e)
    
    async def _extract_from_single_text(self, text: str, domain: str) -> Dict[str, Any]:
        """Extract from single text using optimized prompt"""
        await self.rate_limiter.wait_if_needed()
        
        prompt = self._create_extraction_prompt(text, domain)
        
        try:
            response = self._client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            return self._parse_extraction_response(content)
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'entities': [], 'relationships': []}
    
    def _create_extraction_prompt(self, text: str, domain: str) -> str:
        """Create optimized extraction prompt"""
        return f'''Extract entities and relationships from this {domain} text.

Text: {text}

Return JSON format:
{{
  "entities": [
    {{"text": "entity_name", "type": "entity_type", "context": "surrounding_context"}}
  ],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "relation": "relationship_type", "context": "context"}}
  ]
}}

Focus on: equipment, components, actions, issues, locations for maintenance domain.
Be precise and context-aware.'''
    
    def _parse_extraction_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from extraction"""
        try:
            import json
            data = json.loads(content.strip())
            return {
                'success': True,
                'entities': data.get('entities', []),
                'relationships': data.get('relationships', [])
            }
        except:
            return {'success': False, 'entities': [], 'relationships': []}
    
    # === TEXT COMPLETION ===
    
    async def get_completion(self, prompt: str, model: str = "gpt-4", **kwargs) -> str:
        """Get text completion"""
        self.ensure_initialized()
        await self.rate_limiter.wait_if_needed()
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            return f"Error: {str(e)}"
    
    # === TEXT PROCESSING ===
    
    def process_text(self, text: str, operation: str = "clean") -> str:
        """Process text (cleaning, chunking, etc.)"""
        if operation == "clean":
            return self._clean_text(text)
        elif operation == "chunk":
            return self._chunk_text(text)
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        import re
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text.strip())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\\w\\s\\.,!?;:-]', '', text)
        return text
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    # === UTILITY METHODS ===
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities and assign unique IDs"""
        seen = set()
        unique_entities = []
        
        for i, entity in enumerate(entities):
            key = (entity['text'].lower().strip(), entity.get('type', 'unknown'))
            if key not in seen:
                seen.add(key)
                entity['entity_id'] = f"entity_{len(unique_entities)}"
                unique_entities.append(entity)
        
        return unique_entities
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships"""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            key = (rel['source'], rel['target'], rel.get('relation', 'related'))
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships


class SimpleRateLimiter:
    """Simple rate limiter for Azure OpenAI"""
    
    def __init__(self, requests_per_minute: int = 50):
        self.requests_per_minute = requests_per_minute
        self.last_request = 0
        
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        import time
        now = time.time()
        time_since_last = now - self.last_request
        min_interval = 60.0 / self.requests_per_minute
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request = time.time()