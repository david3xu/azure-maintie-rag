"""Azure OpenAI integration for Universal RAG system."""

from typing import Dict, List, Any, Optional
import os
from openai import AzureOpenAI
import tiktoken
from datetime import datetime


class AzureOpenAIClient:
    """Universal Azure OpenAI client that works with any domain."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Azure OpenAI client."""
        self.config = config or {}

        # Load from environment if not in config
        self.api_key = self.config.get('api_key') or os.getenv('AZURE_API_KEY')
        self.azure_endpoint = self.config.get('azure_endpoint') or os.getenv('AZURE_ENDPOINT', 'https://clu-project-foundry-instance.openai.azure.com/')
        self.deployment = self.config.get('deployment') or os.getenv('AZURE_DEPLOYMENT', 'gpt-4.1')
        self.api_version = self.config.get('api_version', '2023-12-01-preview')

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
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate completion from prompt."""
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        start_time = datetime.now()

        try:
            response = self.client.chat.completions.create(
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