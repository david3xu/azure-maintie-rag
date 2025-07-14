"""
LLM interface module for maintenance response generation
Integrates with OpenAI and other LLM providers for domain-aware response generation
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from openai import AzureOpenAI

from src.models.maintenance_models import SearchResult, EnhancedQuery, RAGResponse
from config.settings import settings

from config.prompt_templates import template_manager


logger = logging.getLogger(__name__)


class MaintenanceLLMInterface:
    """LLM interface specialized for maintenance domain responses (Azure OpenAI only)"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLM interface for Azure OpenAI only"""

        self.api_key = api_key or settings.openai_api_key
        self.deployment_name = settings.openai_deployment_name
        self.api_base = settings.openai_api_base
        self.api_version = settings.openai_api_version
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature
        self.config = settings
        self.template_manager = template_manager

        # Azure OpenAI client setup using new SDK
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_base
        )

        logger.info(f"MaintenanceLLMInterface initialized with Azure deployment {self.deployment_name}")

    # New method for simple test connection
    def test_connection(self, prompt: str = "Hello") -> str:
        """Simple connectivity test method"""
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()

    def generate_response(
        self,
        enhanced_query: EnhancedQuery,
        search_results: List[SearchResult],
        include_citations: bool = True,
        include_safety_warnings: bool = True
    ) -> Dict[str, Any]:
        """Generate maintenance response using Azure OpenAI"""
        start_time = time.time()
        try:
            prompt = self._build_maintenance_prompt(enhanced_query, search_results)
            response = self._call_openai(prompt)
            enhanced_response = self._enhance_response(
                response, enhanced_query, search_results,
                include_citations, include_safety_warnings
            )
            processing_time = time.time() - start_time
            return {
                "generated_response": enhanced_response["response"],
                "confidence_score": enhanced_response["confidence"],
                "sources": enhanced_response["sources"],
                "safety_warnings": enhanced_response["safety_warnings"],
                "citations": enhanced_response["citations"],
                "processing_time": processing_time,
                "model_used": self.deployment_name,
                "prompt_type": enhanced_response["prompt_type"]
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._create_fallback_response(enhanced_query, search_results)

    def _build_maintenance_prompt(
        self,
        enhanced_query: EnhancedQuery,
        search_results: List[SearchResult]
    ) -> str:
        """Build maintenance-specific prompt with domain context"""

        # Base maintenance prompt
        prompt_parts = [
            "You are a maintenance expert assistant helping with industrial equipment maintenance.",
            f"Query Type: {enhanced_query.analysis.query_type.value}",
            f"Equipment Category: {enhanced_query.equipment_category or 'General'}",
        ]

        # Add safety context if critical
        if enhanced_query.safety_critical:
            prompt_parts.extend([
                "",
                "⚠️ SAFETY CRITICAL EQUIPMENT DETECTED ⚠️",
                "Always prioritize safety in your response.",
                "Include relevant safety warnings and procedures.",
            ])

        # Add maintenance context
        if enhanced_query.maintenance_context:
            context = enhanced_query.maintenance_context
            urgency = context.get("task_urgency", "normal")

            if urgency == "high" or urgency == "critical":
                prompt_parts.append(f"URGENT: This is a {urgency} priority maintenance issue.")

        # Add query and context
        prompt_parts.extend([
            "",
            f"Original Query: {enhanced_query.analysis.original_query}",
            "",
            "Relevant Documentation:",
        ])

        # Add search results with relevance scores
        for i, result in enumerate(search_results[:5], 1):
            score_info = ""
            if hasattr(result, 'metadata') and result.metadata:
                if 'knowledge_graph_score' in result.metadata:
                    kg_score = result.metadata['knowledge_graph_score']
                    score_info = f" (Domain Relevance: {kg_score:.2f})"

            prompt_parts.append(f"{i}. {result.title}{score_info}")
            prompt_parts.append(f"   {result.content[:200]}...")

        # Response instructions
        prompt_parts.extend([
            "",
            "Instructions:",
            "1. Provide a comprehensive maintenance response",
            "2. Include step-by-step procedures when applicable",
            "3. Highlight any safety considerations",
            "4. Reference the provided documentation",
            "5. Use proper maintenance terminology",
        ])

        if enhanced_query.safety_critical:
            prompt_parts.append("6. MANDATORY: Include safety warnings for critical equipment")

        return "\n".join(prompt_parts)

    def _build_context(self, search_results: List[SearchResult]) -> str:
        """Build context from search results"""
        if not search_results:
            return "No specific maintenance documentation found."

        context_parts = []
        for i, result in enumerate(search_results[:5], 1):  # Use top 5 results
            context_part = f"""
Document {i} (Relevance: {result.score:.2f}):
Title: {result.title}
Content: {result.content}
Entities: {", ".join(result.entities) if result.entities else "None specified"}
---"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _call_openai(self, prompt: str) -> str:
        """Call Azure OpenAI API with new SDK client"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert maintenance engineer with 20+ years of experience in industrial equipment maintenance. Provide accurate, practical, and safety-focused maintenance guidance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.config.llm_top_p,
                frequency_penalty=self.config.llm_frequency_penalty,
                presence_penalty=self.config.llm_presence_penalty
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI: {e}")
            raise

    def _enhance_response(
        self,
        response: str,
        enhanced_query: EnhancedQuery,
        search_results: List[SearchResult],
        include_citations: bool,
        include_safety_warnings: bool
    ) -> Dict[str, Any]:
        """Enhance generated response with maintenance-specific features"""

        enhanced = {
            "response": response,
            "confidence": self._calculate_confidence(response, search_results),
            "sources": [],
            "citations": [],
            "safety_warnings": [],
            "prompt_type": enhanced_query.analysis.query_type.value
        }

        # Add citations
        if include_citations and search_results:
            citations = []
            sources = []
            for result in search_results[:3]:  # Top 3 sources
                citation = f"[{result.doc_id}] {result.title}"
                citations.append(citation)
                sources.append(result.doc_id)

            enhanced["citations"] = citations
            enhanced["sources"] = sources

            # Add citation section to response
            if citations:
                enhanced["response"] += f"\n\n**Sources:**\n" + "\n".join(f"- {citation}" for citation in citations)

        # Add safety warnings
        if include_safety_warnings:
            safety_warnings = self._extract_safety_warnings(enhanced_query, response)
            enhanced["safety_warnings"] = safety_warnings

            if safety_warnings:
                safety_section = "\n\n⚠️ **SAFETY WARNINGS:**\n" + "\n".join(f"- {warning}" for warning in safety_warnings)
                enhanced["response"] = safety_section + "\n\n" + enhanced["response"]

        # Add procedural enhancements
        if enhanced_query.analysis.query_type.value in ["troubleshooting", "procedural"]:
            enhanced["response"] = self._add_procedural_structure(enhanced["response"])

        return enhanced

    def _calculate_confidence(self, response: str, search_results: List[SearchResult]) -> float:
        """Calculate confidence score for the response"""
        confidence = 0.5  # Base confidence

        # Boost confidence based on search result quality
        if search_results:
            avg_score = sum(result.score for result in search_results[:3]) / min(3, len(search_results))
            confidence += avg_score * 0.3

        # Boost confidence based on response quality indicators
        quality_indicators = [
            len(response) > 100,  # Sufficient detail
            "step" in response.lower() or "procedure" in response.lower(),  # Procedural content
            any(word in response.lower() for word in ["safety", "caution", "warning"]),  # Safety awareness
            len(response.split('\n')) > 3  # Structured format
        ]

        confidence += sum(quality_indicators) * 0.05

        return min(confidence, 1.0)

    def _extract_safety_warnings(self, enhanced_query: EnhancedQuery, response: str) -> List[str]:
        """Extract and add safety warnings"""
        warnings = list(enhanced_query.safety_considerations)

        # Add response-specific warnings
        response_lower = response.lower()

        if "electrical" in response_lower or "power" in response_lower:
            warnings.append("Electrical hazard - ensure proper lockout/tagout procedures")

        if "pressure" in response_lower or "hydraulic" in response_lower:
            warnings.append("Pressure hazard - properly isolate and depressurize system")

        if "hot" in response_lower or "temperature" in response_lower:
            warnings.append("Temperature hazard - allow equipment to cool and use appropriate PPE")

        if "chemical" in response_lower or "fluid" in response_lower:
            warnings.append("Chemical hazard - review MSDS and use proper containment")

        # Remove duplicates while preserving order
        seen = set()
        unique_warnings = []
        for warning in warnings:
            if warning not in seen:
                seen.add(warning)
                unique_warnings.append(warning)

        return unique_warnings

    def _add_procedural_structure(self, response: str) -> str:
        """Add procedural structure to response"""
        # Check if response already has good structure
        if any(indicator in response for indicator in ["1.", "Step 1", "First", "•"]):
            return response

        # Add basic structure hints
        if "troubleshoot" in response.lower():
            return "To troubleshoot:\n\n" + response
        elif "procedure" in response.lower() or "steps" in response.lower():
            return "Follow these steps:\n\n" + response
        return response

    def _create_fallback_response(
        self,
        enhanced_query: EnhancedQuery,
        search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Create a fallback response when LLM generation fails"""
        logger.warning("Falling back to default response due to LLM error.")

        fallback_response = (
            "I apologize, but I was unable to generate a detailed response at this time. "
            "This might be due to an issue with the underlying language model or insufficient context. "
            "Please try rephrasing your query or checking the system logs for more details."
        )
        if enhanced_query.analysis.original_query:
            fallback_response += f"\n\nOriginal Query: {enhanced_query.analysis.original_query}"

        if search_results:
            fallback_response += "\n\nRelevant Documents Found:\n"
            for i, result in enumerate(search_results[:3], 1):
                fallback_response += f"- [{i}] {result.title} (Score: {result.score:.2f})\n"

        return {
            "generated_response": fallback_response,
            "confidence_score": 0.1,
            "sources": [s.doc_id for s in search_results[:3]],
            "safety_warnings": ["Could not generate a full response due to system error."],
            "citations": [],
            "processing_time": 0.0,
            "model_used": "fallback",
            "prompt_type": enhanced_query.analysis.query_type.value
        }

def create_llm_interface(api_key: Optional[str] = None, model: Optional[str] = None) -> MaintenanceLLMInterface:
    """Factory function to create and initialize LLM interface"""
    return MaintenanceLLMInterface(api_key=api_key, model=model)


if __name__ == "__main__":
    # Example usage
    from src.models.maintenance_models import QueryAnalysis, EnhancedQuery, QueryType

    # Create sample enhanced query
    analysis = QueryAnalysis(
        original_query="How to replace pump seal?",
        query_type=QueryType.PROCEDURAL,
        entities=["pump", "seal"],
        intent="replacement",
        complexity="medium"
    )

    enhanced_query = EnhancedQuery(
        analysis=analysis,
        expanded_concepts=["mechanical seal", "gasket", "O-ring"],
        related_entities=["bearing", "impeller"],
        domain_context={},
        structured_search="pump AND seal AND replacement",
        safety_considerations=["Lockout power", "Drain system"]
    )

    # Create sample search results
    search_results = [
        SearchResult(
            doc_id="MWO_001",
            title="Pump Seal Replacement Procedure",
            content="Standard procedure for replacing mechanical seals...",
            score=0.9,
            source="vector"
        )
    ]

    # Generate response
    llm = MaintenanceLLMInterface()
    response = llm.generate_response(enhanced_query, search_results)

    print("Generated Response:")
    print(response["generated_response"])
    print(f"\nConfidence: {response['confidence']:.2f}")
    print(f"Safety Warnings: {response['safety_warnings']}")
