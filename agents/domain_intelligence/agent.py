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

# from agents.shared.query_tools import generate_analysis_query  # Temporarily disabled


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
import os

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables from .env file
load_dotenv()

# Use environment-based model configuration for Azure OpenAI
# PydanticAI will use OPENAI_API_KEY and OPENAI_BASE_URL from environment
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("AZURE_OPENAI_ENDPOINT", ""))

# Use centralized Azure PydanticAI provider
from agents.core.azure_pydantic_provider import get_azure_openai_model

domain_intelligence_agent = Agent[UniversalDeps, UniversalDomainAnalysis](
    get_azure_openai_model(),
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
    # Simple analysis configuration without complex dependencies
    analysis_config = {
        "analysis_type": "characteristics",
        "max_patterns": 10,
        "confidence_threshold": 0.7,
    }

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
    base_config = await config_manager.get_processing_config("universal")

    # Adapt chunk size based on measured complexity
    optimal_chunk_size = base_config.optimal_chunk_size
    if characteristics.vocabulary_complexity > 0.7:
        optimal_chunk_size = int(
            optimal_chunk_size * 1.3
        )  # Larger chunks for complex vocabulary
    if "code_blocks" in characteristics.structural_patterns:
        optimal_chunk_size = int(optimal_chunk_size * 1.5)  # Even larger for code
    if characteristics.concept_density > 0.8:
        optimal_chunk_size = int(
            optimal_chunk_size * 1.2
        )  # Larger for concept-dense content

    # Adapt overlap based on structural patterns
    chunk_overlap_ratio = base_config.chunk_overlap_ratio
    if "hierarchical_headers" in characteristics.structural_patterns:
        chunk_overlap_ratio *= 0.8  # Less overlap for well-structured content
    if characteristics.concept_density > 0.7:
        chunk_overlap_ratio = min(
            chunk_overlap_ratio * 1.4, 0.5
        )  # More overlap but cap at 0.5

    # Adapt confidence thresholds based on content characteristics
    entity_confidence_threshold = base_config.entity_confidence_threshold
    if "proper_noun_rich" in characteristics.entity_indicators:
        entity_confidence_threshold *= 0.9  # Lower threshold for entity-rich content
    if characteristics.vocabulary_complexity < 0.3:
        entity_confidence_threshold *= 1.1  # Higher threshold for simple content
    entity_confidence_threshold = max(
        0.5, min(1.0, entity_confidence_threshold)
    )  # Keep in valid range

    # Adapt relationship density based on discovered indicators
    relationship_density = base_config.relationship_density
    if len(characteristics.relationship_indicators) > 3:
        relationship_density *= 1.2  # More relationships expected
    relationship_density = min(1.0, relationship_density)  # Keep in valid range

    # Adapt search weights based on content complexity
    vector_search_weight = base_config.vector_search_weight
    graph_search_weight = base_config.graph_search_weight

    if characteristics.vocabulary_complexity > 0.7:
        # High complexity - favor graph search for concept relationships
        graph_search_weight = min(1.0, graph_search_weight * 1.2)
        vector_search_weight = max(0.0, 1.0 - graph_search_weight)

    # Determine processing complexity
    if (
        characteristics.vocabulary_complexity > 0.8
        or characteristics.concept_density > 0.8
    ):
        processing_complexity = "high"
    elif (
        characteristics.vocabulary_complexity < 0.3
        and characteristics.concept_density < 0.3
    ):
        processing_complexity = "low"
    else:
        processing_complexity = "medium"

    return UniversalProcessingConfiguration(
        optimal_chunk_size=optimal_chunk_size,
        chunk_overlap_ratio=chunk_overlap_ratio,
        entity_confidence_threshold=entity_confidence_threshold,
        relationship_density=relationship_density,
        vector_search_weight=vector_search_weight,
        graph_search_weight=graph_search_weight,
        expected_extraction_quality=base_config.expected_extraction_quality,
        processing_complexity=processing_complexity,
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
