"""
Universal LLM Interface
Replaces MaintenanceLLMInterface with domain-agnostic response generation
Works with any domain through dynamic prompt generation and universal models
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from openai import AzureOpenAI

from ..models.universal_rag_models import (
    UniversalSearchResult, UniversalRAGResponse, UniversalEnhancedQuery
)
from ...config.settings import settings

logger = logging.getLogger(__name__)


class AzureOpenAICompletionService:
    """Universal LLM interface that works with any domain through dynamic prompting"""

    def __init__(self, domain: str = "general", api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize universal LLM interface for any domain"""

        self.domain = domain
        self.api_key = api_key or settings.openai_api_key
        self.deployment_name = settings.openai_deployment_name
        self.api_base = settings.openai_api_base
        self.api_version = settings.openai_api_version
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature
        self.config = settings

        # Azure OpenAI client setup
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_base
        )

        # Universal context - works with any MD data from data/raw directory
        self.domain_contexts = {
            "general": "universal knowledge processing from markdown documents"
        }

        logger.info(f"AzureOpenAICompletionService initialized for domain: {domain}")

    def test_connection(self, prompt: str = "Hello") -> str:
        """Simple connectivity test method"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return f"Connection failed: {e}"

    def generate_universal_response(self,
                                  query: str,
                                  search_results: List[UniversalSearchResult],
                                  enhanced_query: Optional[UniversalEnhancedQuery] = None) -> UniversalRAGResponse:
        """Generate universal response that works with any domain"""

        try:
            logger.info(f"Generating universal response for domain: {self.domain}")

            # Step 1: Build dynamic domain context
            system_prompt = self._build_domain_system_prompt()

            # Step 2: Build context from search results
            context = self._build_context_from_results(search_results)

            # Step 3: Create domain-appropriate user prompt
            user_prompt = self._build_user_prompt(query, context, enhanced_query)

            # Step 4: Generate response using LLM
            llm_response = self._call_llm(system_prompt, user_prompt)

            # Step 5: Extract citations and build final response
            citations = self._extract_citations(llm_response, search_results)

            # Step 6: Generate domain-appropriate warnings/notes
            domain_notes = self._generate_domain_notes(query, enhanced_query)

            # Create universal RAG response
            rag_response = UniversalRAGResponse(
                query=query,
                answer=llm_response,
                confidence=self._estimate_confidence(llm_response, search_results),
                sources=search_results,
                entities_used=enhanced_query.related_entities if enhanced_query else [],
                processing_metadata={
                    "domain": self.domain,
                    "generation_method": "universal_llm",
                    "system_prompt_type": "dynamic_domain",
                    "context_sources": len(search_results),
                    "response_length": len(llm_response)
                },
                citations=citations,
                domain=self.domain
            )

            # Add domain notes if any
            if domain_notes:
                rag_response.answer += f"\n\n{domain_notes}"

            logger.info(f"Universal response generated successfully for {self.domain}")
            return rag_response

        except Exception as e:
            logger.error(f"Universal response generation failed: {e}")
            return self._create_fallback_response(query, str(e))

    def generate_response_fixed(self,
                               query: Union[str, UniversalEnhancedQuery],
                               search_results: List[UniversalSearchResult]) -> UniversalRAGResponse:
        """Fixed response generation method (backward compatibility)"""

        # Handle both string and enhanced query inputs
        if isinstance(query, str):
            query_text = query
            enhanced_query = None
        else:
            query_text = query.original_query
            enhanced_query = query

        return self.generate_universal_response(query_text, search_results, enhanced_query)

    def _build_domain_system_prompt(self) -> str:
        """Build dynamic system prompt based on domain"""

        domain_context = self.domain_contexts.get(self.domain, "universal knowledge from markdown documents")

        base_prompt = f"""You are an expert AI assistant specialized in {domain_context}.

Your role:
- Provide accurate, helpful, and comprehensive responses
- Use the provided context from markdown documents to support your answers
- Cite sources when referencing specific information
- Be clear about limitations or uncertainties
- Work with any type of markdown content from the data/raw directory"""

        # Universal guidance - works with any MD data
        domain_guidance = {
            "general": "\n- Provide clear and accurate information from the markdown documents\n- Include relevant context and explanations\n- Ensure comprehensive coverage of the topic\n- Cite specific markdown sources when available"
        }

        if self.domain in domain_guidance:
            base_prompt += domain_guidance[self.domain]

        return base_prompt

    def _build_context_from_results(self, search_results: List[UniversalSearchResult]) -> str:
        """Build context string from universal search results"""

        if not search_results:
            return "No specific context available."

        context_parts = []
        for i, result in enumerate(search_results[:5]):  # Limit to top 5 results
            context_parts.append(f"[Source {i+1}] {result.content}")

        return "\n\n".join(context_parts)

    def _build_user_prompt(self, query: str, context: str, enhanced_query: Optional[UniversalEnhancedQuery]) -> str:
        """Build dynamic user prompt"""

        prompt = f"""Based on the following context from markdown documents, please answer this question:

Question: {query}

Context:
{context}

"""

        # Add enhanced query information if available
        if enhanced_query and enhanced_query.expanded_concepts:
            prompt += f"Related concepts: {', '.join(enhanced_query.expanded_concepts[:5])}\n\n"

        prompt += f"""Please provide a comprehensive answer that:
1. Directly addresses the question
2. Uses the provided context from markdown documents appropriately
3. Includes relevant citations [Source X]
4. Is based on the available markdown data
5. Acknowledges any limitations in the available information"""

        return prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM with dynamic prompts"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _extract_citations(self, response: str, search_results: List[UniversalSearchResult]) -> List[str]:
        """Extract citations from response"""

        citations = []
        for i, result in enumerate(search_results):
            source_ref = f"[Source {i+1}]"
            if source_ref in response:
                citation = f"Source {i+1}: {result.metadata.get('title', result.doc_id)}"
                citations.append(citation)

        return citations

    def _generate_domain_notes(self, query: str, enhanced_query: Optional[UniversalEnhancedQuery]) -> str:
        """Generate domain-appropriate notes and warnings"""

        notes = []

        # Universal disclaimer - works with any MD data
        disclaimers = {
            "general": "ℹ️ Information Note: This response is based on markdown documents from the data/raw directory and should be verified for specific applications."
        }

        if self.domain in disclaimers:
            notes.append(disclaimers[self.domain])

        return "\n".join(notes)

    def _estimate_confidence(self, response: str, search_results: List[UniversalSearchResult]) -> float:
        """Estimate response confidence based on various factors"""

        confidence = 0.5  # Base confidence

        # Boost confidence based on available context
        if search_results:
            confidence += min(0.3, len(search_results) * 0.1)

        # Boost confidence based on response length (longer = more detailed)
        if len(response) > 200:
            confidence += 0.1

        # Boost confidence if citations are used
        citation_count = response.count("[Source")
        if citation_count > 0:
            confidence += min(0.2, citation_count * 0.05)

        return min(confidence, 0.95)  # Cap at 95%

    def _create_fallback_response(self, query: str, error_msg: str) -> UniversalRAGResponse:
        """Create fallback response when generation fails"""

        return UniversalRAGResponse(
            query=query,
            answer=f"I apologize, but I encountered an issue generating a response for your {self.domain} question. Please try rephrasing your question or contact support if the issue persists.",
            confidence=0.1,
            sources=[],
            entities_used=[],
            processing_metadata={
                "domain": self.domain,
                "generation_method": "fallback",
                "error": error_msg
            },
            citations=[],
            domain=self.domain
        )

    async def configure_domain_knowledge(self,
                                       entities: List[Any] = None,
                                       relations: List[Any] = None,
                                       discovered_types: Dict[str, Any] = None,
                                       domain_context: str = "general") -> Dict[str, Any]:
        """Configure LLM interface with discovered domain knowledge"""
        try:
            self.domain_entities = entities or []
            self.domain_relations = relations or []
            self.discovered_types = discovered_types or {}
            self.domain_context = domain_context

            if discovered_types:
                entity_types = discovered_types.get("entity_types", [])
                relation_types = discovered_types.get("relation_types", [])
                logger.info(f"Configured domain knowledge: {len(entity_types)} entity types, {len(relation_types)} relation types")

            return {
                "success": True,
                "configured_entities": len(self.domain_entities),
                "configured_relations": len(self.domain_relations),
                "domain": domain_context
            }

        except Exception as e:
            logger.error(f"Domain knowledge configuration failed: {e}")
            return {"success": False, "error": str(e)}


# Universal RAG - no backward compatibility needed


def create_universal_llm_interface(domain: str = "general") -> AzureOpenAICompletionService:
    """Factory function to create universal LLM interface"""
    return AzureOpenAICompletionService(domain)


if __name__ == "__main__":
    # Example usage
    llm = AzureOpenAICompletionService("general")

    # Test connection
    test_result = llm.test_connection()
    print(f"Connection test: {test_result}")

    # Sample response generation
    sample_results = [
        UniversalSearchResult(
            doc_id="doc1",
            content="System monitoring requires regular inspection",
            score=0.8,
            source="universal_search"
        )
    ]

    response = llm.generate_universal_response(
        "How do I monitor a system?",
        sample_results
    )
    print(f"Generated response: {response.answer[:100]}...")