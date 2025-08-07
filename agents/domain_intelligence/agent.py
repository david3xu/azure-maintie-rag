"""
Universal Domain Intelligence Agent - REAL Azure OpenAI with PydanticAI
=======================================================================

This agent discovers content characteristics WITHOUT hardcoded domain assumptions.
Follows PydanticAI best practices with proper dependency injection and tool usage.

Key Principles:
- ZERO predetermined domain categories (technical, legal, medical, etc.)
- Discovers characteristics from content analysis (vocabulary, patterns, structure)
- Uses centralized dependencies (no duplicate Azure clients)
- Atomic tools that don't orchestrate other agents
- Proper RunContext usage for dependency injection
"""

import asyncio
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agents.core.universal_deps import UniversalDeps, get_universal_deps
from agents.core.universal_models import (
    UniversalDomainAnalysis,
    UniversalDomainCharacteristics,
    UniversalProcessingConfiguration,
)
from agents.shared.query_tools import generate_analysis_query


class ContentCharacteristics(BaseModel):
    """Content characteristics discovered from analysis (not predetermined)."""

    vocabulary_complexity: float = Field(
        ge=0.0, le=1.0, description="Measured vocabulary complexity"
    )
    concept_density: float = Field(
        ge=0.0, le=1.0, description="Density of concepts per content unit"
    )
    structural_patterns: List[str] = Field(
        default_factory=list, description="Discovered structural patterns"
    )
    entity_indicators: List[str] = Field(
        default_factory=list, description="Potential entity types found"
    )
    relationship_indicators: List[str] = Field(
        default_factory=list, description="Potential relationship types"
    )
    content_signature: str = Field(
        description="Unique signature based on measured properties"
    )


# Create the Domain Intelligence Agent with proper PydanticAI patterns
domain_intelligence_agent = Agent[UniversalDeps, UniversalDomainAnalysis](
    "azure_openai:gpt-4o",  # Use Azure OpenAI instead of OpenAI API
    deps_type=UniversalDeps,
    output_type=UniversalDomainAnalysis,
    system_prompt="""You are the Universal Domain Intelligence Agent.

Your role is to discover content characteristics WITHOUT making domain assumptions.

CRITICAL RULES:
- NEVER assume domain types (technical, legal, medical, business, etc.)
- DISCOVER characteristics through vocabulary analysis, pattern recognition, and structure analysis
- MEASURE properties like complexity, density, and relationships
- ADAPT processing parameters based on discovered characteristics
- GENERATE universal configurations that work for ANY content type

You analyze content and discover:
1. Vocabulary characteristics (complexity, specialization, diversity)
2. Structural patterns (formatting, organization, relationships)
3. Concept density and distribution
4. Entity and relationship indicators (discovered, not assumed)
5. Processing requirements based on measured properties

Always base recommendations on measured characteristics, not predetermined categories.""",
)


@domain_intelligence_agent.tool
async def analyze_content_characteristics(
    ctx: RunContext[UniversalDeps], content: str, detailed_analysis: bool = True
) -> ContentCharacteristics:
    """
    Analyze content to discover characteristics without domain assumptions.

    This tool performs atomic content analysis and discovery.
    """
    # Use analysis query tool for configuration
    analysis_config = await generate_analysis_query(
        ctx, content, analysis_type="characteristics"
    )

    # Statistical analysis (domain-agnostic)
    words = content.split()
    sentences = content.split(".")
    unique_words = set(word.lower().strip('.,!?;:"()[]') for word in words)

    # Measure vocabulary complexity (not domain classification)
    vocab_complexity = min(len(unique_words) / max(len(words), 1), 1.0)

    # Measure concept density
    potential_concepts = [word for word in unique_words if len(word) > 6]
    concept_density = min(len(potential_concepts) / max(len(words), 1) * 10, 1.0)

    # Discover structural patterns (not hardcoded types)
    structural_patterns = []
    if "```" in content or "def " in content or "class " in content:
        structural_patterns.append("code_blocks")
    if content.count("\n- ") > 5 or content.count("1. ") > 3:
        structural_patterns.append("list_structures")
    if content.count("\n#") > 2 or content.count("##") > 1:
        structural_patterns.append("hierarchical_headers")
    if "|" in content and content.count("|") > 10:
        structural_patterns.append("tabular_data")

    # Discover entity indicators (not predetermined entity types)
    entity_indicators = []
    if any(word.isupper() and len(word) > 2 for word in words[:50]):
        entity_indicators.append("acronym_rich")
    if (
        sum(1 for word in words if word and word[0].isupper()) / max(len(words), 1)
        > 0.15
    ):
        entity_indicators.append("proper_noun_rich")
    if any(char.isdigit() for char in content[:500]):
        entity_indicators.append("numeric_data")

    # Discover relationship indicators
    relationship_indicators = []
    relationship_words = [
        "related",
        "connects",
        "linked",
        "associated",
        "depends",
        "causes",
        "leads",
        "results",
    ]
    for word in relationship_words:
        if word in content.lower():
            relationship_indicators.append(f"explicit_{word}")

    # Generate content signature based on measured properties
    signature_components = [
        f"vc{vocab_complexity:.2f}",  # vocabulary complexity
        f"cd{concept_density:.2f}",  # concept density
        f"sp{len(structural_patterns)}",  # structural patterns count
        f"ei{len(entity_indicators)}",  # entity indicators count
        f"ri{len(relationship_indicators)}",  # relationship indicators count
    ]
    content_signature = "_".join(signature_components)

    return ContentCharacteristics(
        vocabulary_complexity=vocab_complexity,
        concept_density=concept_density,
        structural_patterns=structural_patterns,
        entity_indicators=entity_indicators,
        relationship_indicators=relationship_indicators,
        content_signature=content_signature,
    )


@domain_intelligence_agent.tool
async def generate_processing_configuration(
    ctx: RunContext[UniversalDeps],
    characteristics: ContentCharacteristics,
    processing_type: str = "extraction",
) -> UniversalProcessingConfiguration:
    """
    Generate adaptive processing configuration based on discovered characteristics.

    This tool creates configurations dynamically, not from hardcoded domain rules.
    """
    config_manager = ctx.deps.config_manager
    base_config = config_manager.get_processing_config()

    # Adapt chunk size based on measured complexity
    chunk_size = base_config.chunk_size
    if characteristics.vocabulary_complexity > 0.7:
        chunk_size = int(chunk_size * 1.3)  # Larger chunks for complex vocabulary
    if "code_blocks" in characteristics.structural_patterns:
        chunk_size = int(chunk_size * 1.5)  # Even larger for code
    if characteristics.concept_density > 0.8:
        chunk_size = int(chunk_size * 1.2)  # Larger for concept-dense content

    # Adapt overlap based on structural patterns
    overlap = base_config.overlap
    if "hierarchical_headers" in characteristics.structural_patterns:
        overlap = int(overlap * 0.8)  # Less overlap for well-structured content
    if characteristics.concept_density > 0.7:
        overlap = int(overlap * 1.4)  # More overlap for concept-rich content

    # Adapt confidence thresholds based on content characteristics
    confidence_threshold = base_config.confidence_threshold
    if "proper_noun_rich" in characteristics.entity_indicators:
        confidence_threshold *= 0.9  # Lower threshold for entity-rich content
    if characteristics.vocabulary_complexity < 0.3:
        confidence_threshold *= 1.1  # Higher threshold for simple content

    # Set max entities/relationships based on discovered indicators
    max_entities = min(
        len(characteristics.entity_indicators) * 5 + 10, base_config.max_entities
    )
    max_relationships = min(
        len(characteristics.relationship_indicators) * 3 + 5,
        base_config.max_relationships,
    )

    return UniversalProcessingConfiguration(
        chunk_size=chunk_size,
        overlap=overlap,
        confidence_threshold=confidence_threshold,
        max_entities=max_entities,
        max_relationships=max_relationships,
        processing_strategy="adaptive",
        content_signature=characteristics.content_signature,
        discovered_patterns=characteristics.structural_patterns,
    )


@domain_intelligence_agent.tool
async def validate_azure_services(ctx: RunContext[UniversalDeps]) -> Dict[str, bool]:
    """
    Validate required Azure services are available for domain intelligence.

    This tool checks service availability without duplicating clients.
    """
    # Use centralized dependencies (no duplicate Azure clients)
    if not ctx.deps._initialized:
        await ctx.deps.initialize_all_services()

    required_services = ["openai"]  # Domain Intelligence only requires OpenAI
    optional_services = ["monitoring", "storage"]

    service_status = {
        "required_available": all(
            ctx.deps.is_service_available(service) for service in required_services
        ),
        "optional_available": {
            service: ctx.deps.is_service_available(service)
            for service in optional_services
        },
        "total_services": len(ctx.deps.get_available_services()),
    }

    return service_status


# Factory function for proper agent initialization
async def create_domain_intelligence_agent() -> (
    Agent[UniversalDeps, UniversalDomainAnalysis]
):
    """
    Create Domain Intelligence Agent with initialized dependencies.

    Follows PydanticAI best practices for agent creation and dependency injection.
    """
    deps = await get_universal_deps()

    # Validate required services
    if not deps.is_service_available("openai"):
        raise RuntimeError("Domain Intelligence Agent requires Azure OpenAI service")

    return domain_intelligence_agent


# Main execution function for testing
async def run_domain_analysis(
    content: str, detailed: bool = True
) -> UniversalDomainAnalysis:
    """
    Run domain analysis with proper PydanticAI patterns.

    This function demonstrates proper agent usage with dependency injection.
    """
    deps = await get_universal_deps()
    agent = await create_domain_intelligence_agent()

    result = await agent.run(
        f"Analyze the following content and discover its characteristics without domain assumptions:\n\n{content}",
        deps=deps,
    )

    return result.output


if __name__ == "__main__":
    # Test the agent with sample content
    sample_content = """
    Python is a programming language that emphasizes readability and simplicity.
    Functions are defined using the def keyword, and classes use the class keyword.
    Popular frameworks include Django for web development and NumPy for data science.
    
    Example code:
    ```python
    def hello_world():
        print("Hello, World!")
    ```
    """

    async def test_agent():
        try:
            result = await run_domain_analysis(sample_content)
            print("Domain Analysis Result:")
            print(f"Vocabulary Complexity: {result.vocabulary_complexity}")
            print(f"Concept Density: {result.concept_density}")
            print(f"Discovered Patterns: {result.discovered_patterns}")
            print(f"Content Signature: {result.content_signature}")
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(test_agent())
