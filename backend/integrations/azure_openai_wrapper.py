"""Azure OpenAI integration for Universal RAG system."""

from typing import Dict, List, Any, Optional
import os
from openai import AzureOpenAI
import tiktoken
from datetime import datetime

# Import settings for configuration
from config.settings import settings


class AzureOpenAIClient:
    """Universal Azure OpenAI client that works with any domain."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Azure OpenAI client."""
        self.config = config or {}

        # Load from settings first, then config, then environment
        self.api_key = self.config.get('api_key') or settings.openai_api_key or os.getenv('AZURE_API_KEY')
        self.azure_endpoint = self.config.get('azure_endpoint') or settings.openai_api_base or os.getenv('AZURE_ENDPOINT', 'https://clu-project-foundry-instance.openai.azure.com/')
        self.deployment = self.config.get('deployment') or settings.openai_deployment_name or os.getenv('AZURE_DEPLOYMENT', 'gpt-4.1')
        self.api_version = self.config.get('api_version') or settings.openai_api_version or '2023-12-01-preview'

        if not self.api_key:
            raise ValueError("Azure API key is required")

        # Initialize client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4.1")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate completion from prompt - async version."""
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        start_time = datetime.now()

        try:
            import asyncio
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            duration = (datetime.now() - start_time).total_seconds()

            result = {
                'text': response.choices[0].message.content,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
                'duration_seconds': duration,
                'model': self.deployment,
                'success': True
            }

            return result

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            return {
                'text': '',
                'error': str(e),
                'duration_seconds': duration,
                'model': self.deployment,
                'success': False
            }

    async def process_documents(self, texts: list, domain: str) -> list:
        """Process documents using generate_completion pattern"""
        processed_docs = []
        for i, text in enumerate(texts):
            prompt = f"Process this text for domain '{domain}': {text[:500]}..."
            result = await self.generate_completion(prompt)
            processed_docs.append({
                "doc_id": f"{domain}_{i}",
                "processed_text": result,
                "original_text": text
            })
        return processed_docs

    async def extract_knowledge(self, text_input, domain: str) -> dict:
        """Extract knowledge using generate_completion pattern"""
        # Handle both single string and list of strings
        if isinstance(text_input, str):
            texts = [text_input]
        else:
            texts = text_input
            
        knowledge_results = []
        for text in texts:
            # Create a proper extraction prompt
            prompt = f"""Extract entities and relationships from this {domain} maintenance text: "{text}"

Return a JSON object with:
- entities: list of objects with "text", "entity_type", and "confidence" fields
- relationships: list of objects with "source_entity", "target_entity", "relation_type", and "confidence" fields

Example format:
{{
  "entities": [
    {{"text": "equipment_name", "entity_type": "equipment", "confidence": 0.9}},
    {{"text": "component_name", "entity_type": "component", "confidence": 0.8}}
  ],
  "relationships": [
    {{"source_entity": "equipment_name", "target_entity": "component_name", "relation_type": "has_component", "confidence": 0.85}}
  ]
}}"""

            result = await self.generate_completion(prompt, max_tokens=2000, temperature=0.1)
            
            # Parse JSON response
            entities = []
            relationships = []
            
            if result['success']:
                try:
                    import json
                    response_text = result['text'].strip()
                    
                    # Clean up JSON markers
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1]
                    if '```' in response_text:
                        response_text = response_text.split('```')[0]
                        
                    parsed = json.loads(response_text)
                    entities = parsed.get('entities', [])
                    relationships = parsed.get('relationships', [])
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed: {e}")
                    print(f"Response text: {result['text'][:200]}...")
                except Exception as e:
                    print(f"Extraction parsing error: {e}")
            
            knowledge_results.append({
                "entities": entities,
                "relationships": relationships,
                "source_text": text
            })
        
        # Return single result if single input, list if multiple
        if isinstance(text_input, str):
            return knowledge_results[0]
        else:
            return knowledge_results

    def extract_entities_and_relations(
        self,
        texts: List[str],
        domain: str,
        entity_types: List[str],
        relation_types: List[str]
    ) -> Dict[str, Any]:
        """Extract entities and relations from texts."""
        # Combine texts for batch processing
        combined_text = "\n\n".join(texts[:10])  # Limit to avoid token limits

        system_message = f"""You are an expert knowledge extractor for the {domain} domain.
Extract entities and relationships from the provided text.

Entity Types: {', '.join(entity_types)}
Relation Types: {', '.join(relation_types)}

Return a JSON object with:
- entities: list of {{id, name, type, confidence, context}}
- relations: list of {{id, source_entity_id, target_entity_id, type, confidence, context}}"""

        prompt = f"""Text to analyze:
{combined_text}

Extract entities and relations in JSON format:"""

        response = self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            max_tokens=2000,
            temperature=0.1
        )

        if response['success']:
            try:
                # Try to parse JSON from response
                import json

                # Extract JSON from response text
                text = response['text'].strip()
                if text.startswith('```json'):
                    text = text[7:]
                if text.endswith('```'):
                    text = text[:-3]

                extracted_data = json.loads(text)

                response['entities'] = extracted_data.get('entities', [])
                response['relations'] = extracted_data.get('relations', [])
                response['parse_success'] = True

            except Exception as e:
                response['entities'] = []
                response['relations'] = []
                response['parse_success'] = False
                response['parse_error'] = str(e)

        return response

    def generate_domain_schema(
        self,
        texts: List[str],
        domain: str
    ) -> Dict[str, Any]:
        """Generate domain schema from sample texts."""
        # Use first 20 texts as samples
        sample_texts = texts[:20]
        combined_text = "\n\n".join(sample_texts)

        system_message = f"""You are an expert domain analyst for the {domain} domain.
Analyze the provided texts and generate a comprehensive domain schema.

Return a JSON object with:
- entity_types: list of entity types found in this domain
- relation_types: list of relationship types found in this domain
- domain_vocabulary: key terms and concepts specific to this domain"""

        prompt = f"""Sample texts from {domain} domain:
{combined_text}

Generate domain schema in JSON format:"""

        response = self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            max_tokens=1500,
            temperature=0.2
        )

        if response['success']:
            try:
                import json

                text = response['text'].strip()
                if text.startswith('```json'):
                    text = text[7:]
                if text.endswith('```'):
                    text = text[:-3]

                schema_data = json.loads(text)

                response['entity_types'] = schema_data.get('entity_types', [])
                response['relation_types'] = schema_data.get('relation_types', [])
                response['domain_vocabulary'] = schema_data.get('domain_vocabulary', [])
                response['parse_success'] = True

            except Exception as e:
                response['entity_types'] = []
                response['relation_types'] = []
                response['domain_vocabulary'] = []
                response['parse_success'] = False
                response['parse_error'] = str(e)

        return response

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for API call."""
        # GPT-4 pricing (approximate)
        prompt_cost = prompt_tokens * 0.00003  # $0.03 per 1K tokens
        completion_cost = completion_tokens * 0.00006  # $0.06 per 1K tokens

        return prompt_cost + completion_cost

    def get_client_info(self) -> Dict[str, Any]:
        """Get client configuration info."""
        return {
            'azure_endpoint': self.azure_endpoint,
            'deployment': self.deployment,
            'api_version': self.api_version,
            'has_api_key': bool(self.api_key),
            'tokenizer': self.tokenizer.name
        }

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status for service validation"""
        try:
            # Simple test to check if API key and endpoint are configured
            if not self.api_key or self.api_key == "1234567890":
                return {
                    "status": "unhealthy",
                    "error": "API key not configured",
                    "service": "openai"
                }

            if not self.azure_endpoint:
                return {
                    "status": "unhealthy",
                    "error": "Azure endpoint not configured",
                    "service": "openai"
                }

            return {
                "status": "healthy",
                "service": "openai",
                "endpoint": self.azure_endpoint,
                "deployment": self.deployment
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service": "openai"
            }

    async def generate_response(self, query: str, documents: List[str], domain: str) -> str:
        """Generate response based on query and retrieved documents."""
        try:
            # Create context from documents
            context = "\n\n".join([f"Document {i+1}: {doc[:500]}..." for i, doc in enumerate(documents[:3])])

            # Create system message
            system_message = f"""You are a helpful assistant for the {domain} domain.
            Answer the user's question based on the provided documents.
            If the documents don't contain relevant information, say so politely.
            Keep your response concise and accurate."""

            # Create user message
            user_message = f"Question: {query}\n\nContext from documents:\n{context}"

            # Generate response
            result = await self.generate_completion(
                prompt=user_message,
                system_message=system_message,
                max_tokens=500,
                temperature=0.1
            )

            if result['success']:
                return result['text']
            else:
                return f"Sorry, I couldn't generate a response. Error: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"Error generating response: {str(e)}"