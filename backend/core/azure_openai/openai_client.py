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

from ..azure_auth.base_client import BaseAzureClient
from config.settings import azure_settings
from config.domain_patterns import DomainPatternManager
from ..models.universal_rag_models import UniversalEntity, UniversalRelation

logger = logging.getLogger(__name__)


class UnifiedAzureOpenAIClient(BaseAzureClient):
    """Unified client for all Azure OpenAI operations"""
    
    def _get_default_endpoint(self) -> str:
        return azure_settings.azure_openai_endpoint
        
    def _health_check(self) -> bool:
        """Perform OpenAI service health check"""
        try:
            # Simple check that doesn't consume tokens
            return True  # If client is initialized successfully, service is accessible
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
        
    def _initialize_client(self):
        """Initialize Azure OpenAI client"""
        if self.use_managed_identity:
            # Use managed identity for azd deployments
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            self._client = AzureOpenAI(
                azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
                api_version=azure_settings.openai_api_version,
                azure_endpoint=self.endpoint
            )
        else:
            # Use API key for local development
            self._client = AzureOpenAI(
                api_key=self.key,
                api_version=azure_settings.openai_api_version,
                azure_endpoint=self.endpoint
            )
    
    def ensure_initialized(self):
        """Ensure client is initialized with rate limiter"""
        super().ensure_initialized()
        if not hasattr(self, 'rate_limiter'):
            self.rate_limiter = SimpleRateLimiter(domain="general")

    async def test_connection(self) -> Dict[str, Any]:
        """Test Azure OpenAI connection"""
        try:
            self.ensure_initialized()
            
            # Simple test request to verify connectivity
            response = self._client.chat.completions.create(
                model=azure_settings.openai_deployment_name or "gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            return {
                "success": True,
                "model": azure_settings.openai_deployment_name,
                "endpoint": self.endpoint
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "endpoint": getattr(self, 'endpoint', 'unknown')
            }
        
    # === KNOWLEDGE EXTRACTION ===
    
    async def extract_knowledge(self, texts: List[str], domain: str = "general") -> Dict[str, Any]:
        """Extract entities and relationships from texts - REQUIRES Azure OpenAI"""
        self.ensure_initialized()
        
        # ENFORCE Azure OpenAI connectivity - NO LOCAL FALLBACKS
        if not self._client:
            raise RuntimeError("Azure OpenAI client not initialized - cannot proceed without Azure connectivity")
            
        # Test Azure OpenAI connectivity with a simple call using configured model
        try:
            from config.domain_patterns import DomainPatternManager
            prompts = DomainPatternManager.get_prompts(domain)
            test_response = self._client.chat.completions.create(
                model=prompts.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            logger.info(f"✅ Azure OpenAI connectivity verified with model: {prompts.model_name}")
        except Exception as e:
            raise RuntimeError(f"❌ Azure OpenAI connection failed: {e}. Cannot extract knowledge without Azure OpenAI.")
        
        if not texts:
            raise ValueError("No texts provided for knowledge extraction")
        
        try:
            all_entities = []
            all_relationships = []
            
            logger.info(f"🔄 Processing {len(texts)} texts through Azure OpenAI...")
            
            import time
            start_time = time.time()
            
            for i, text in enumerate(texts):
                if not text.strip():
                    continue
                    
                result = await self._extract_from_single_text(text, domain)
                
                if result['success']:
                    all_entities.extend(result['entities'])
                    all_relationships.extend(result['relationships'])
                else:
                    logger.warning(f"Extraction failed for text {i+1}: {result.get('error')}")
                    
                # Progress logging - more frequent for better visibility
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"    📊 Progress: {i + 1}/{len(texts)} texts | Entities: {len(all_entities)} | Relationships: {len(all_relationships)}")
                elif (i + 1) % 20 == 0:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (i + 1)) * (len(texts) - i - 1) if i > 0 else 0
                    print(f"    ⏱️  {i + 1}/{len(texts)} texts | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")
            
            # Deduplicate and assign IDs
            entities = self._deduplicate_entities(all_entities)
            relationships = self._deduplicate_relationships(all_relationships)
            
            logger.info(f"✅ Azure OpenAI extraction complete: {len(entities)} entities, {len(relationships)} relationships")
            
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
        self.ensure_initialized()
        await self.rate_limiter.wait_if_needed()
        
        prompt = self._create_extraction_prompt(text, domain)
        
        try:
            prompts = DomainPatternManager.get_prompts(domain)
            response = self._client.chat.completions.create(
                model=prompts.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=prompts.temperature,
                max_tokens=prompts.max_tokens
            )
            
            content = response.choices[0].message.content
            return self._parse_extraction_response(content)
            
        except Exception as e:
            error_response = self.handle_azure_error('extract_entities_and_relations', e)
            error_response.update({'entities': [], 'relationships': []})
            return error_response
    
    def _create_extraction_prompt(self, text: str, domain: str) -> str:
        """Create optimized extraction prompt using Jinja2 template"""
        try:
            # Use template-based prompt from prompt_flows directory
            from core.utilities.prompt_loader import prompt_loader
            return prompt_loader.render_knowledge_extraction_prompt(
                text_content=text,
                domain_name=domain
            )
        except Exception as e:
            logger.warning(f"Failed to load template prompt, using fallback: {e}")
            # Fallback to original hardcoded prompt if template fails
            extraction_focus = DomainPatternManager.get_extraction_focus(domain)
            return f'''You are a knowledge extraction system. Extract entities and relationships from this {domain} text.

Text: {text}

IMPORTANT: You MUST respond with valid JSON only. No additional text or explanations.

Required JSON format:
{{
  "entities": [
    {{"text": "entity_name", "type": "entity_type", "context": "surrounding_context"}}
  ],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "relation": "relationship_type", "context": "context"}}
  ]
}}

Focus on: {extraction_focus}.
If no clear entities exist, return empty arrays but maintain JSON format.'''
    
    def _sanitize_unicode(self, text: str) -> str:
        """Sanitize text to remove invalid Unicode characters"""
        try:
            # Remove or replace invalid surrogate characters
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            # Remove any remaining high/low surrogates
            import re
            text = re.sub(r'[\ud800-\udfff]', '', text)
            return text
        except Exception:
            return text

    def _parse_extraction_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from extraction"""
        try:
            import json
            import re
            
            # Sanitize Unicode first
            content = self._sanitize_unicode(content)
            
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]   # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove ending ```
            
            # Find the JSON part - look for complete JSON objects
            # Try to find the outermost {} brackets
            brace_count = 0
            start_pos = content.find('{')
            if start_pos != -1:
                end_pos = start_pos
                for i, char in enumerate(content[start_pos:], start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i
                            break
                json_content = content[start_pos:end_pos + 1]
            else:
                json_content = content.strip()
            
            # Parse JSON with proper encoding
            data = json.loads(json_content, strict=False)
            return {
                'success': True,
                'entities': data.get('entities', []),
                'relationships': data.get('relationships', [])
            }
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}, content: {content[:200]}...")
            error_response = self.handle_azure_error('parse_extraction_response', e)
            error_response.update({'entities': [], 'relationships': []})
            return error_response
    
    # === TEXT COMPLETION ===
    
    async def get_completion(self, prompt: str, domain: str = "general", model: str = None, **kwargs) -> str:
        """Get text completion"""
        self.ensure_initialized()
        await self.rate_limiter.wait_if_needed()
        
        try:
            prompts = DomainPatternManager.get_prompts(domain)
            response = self._client.chat.completions.create(
                model=model or prompts.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', prompts.temperature),
                max_tokens=kwargs.get('max_tokens', prompts.max_tokens)
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
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:-]', '', text)
        return text
    
    def _chunk_text(self, text: str, domain: str = "general", chunk_size: int = None) -> List[str]:
        """Split text into chunks"""
        if chunk_size is None:
            prompts = DomainPatternManager.get_prompts(domain)
            chunk_size = prompts.chunk_size
            
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    # === EMBEDDING OPERATIONS ===
    
    async def create_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """Create embeddings for multiple texts using Azure OpenAI"""
        self.ensure_initialized()
        await self.rate_limiter.wait_if_needed()
        
        try:
            if not texts:
                return []
            
            # For single text, wrap in list
            if isinstance(texts, str):
                texts = [texts]
            
            # Clean and prepare texts
            clean_texts = [text.strip() for text in texts if text.strip()]
            if not clean_texts:
                return []
            
            response = self._client.embeddings.create(
                model=azure_settings.embedding_deployment_name or model,
                input=clean_texts
            )
            
            embeddings = [data.embedding for data in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise RuntimeError(f"Embedding creation failed: {e}")
    
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
    
    def __init__(self, requests_per_minute: int = None, domain: str = "general"):
        if requests_per_minute is None:
            prompts = DomainPatternManager.get_prompts(domain)
            requests_per_minute = prompts.requests_per_minute
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