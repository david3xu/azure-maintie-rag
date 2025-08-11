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
from pydantic_ai import Agent, RunContext, PromptedOutput

from agents.core.universal_deps import UniversalDeps, get_universal_deps
from agents.core.universal_models import (
    UniversalDomainAnalysis,
    UniversalDomainCharacteristics,
    UniversalProcessingConfiguration,
)

# from agents.shared.query_tools import generate_analysis_query  # Temporarily disabled


class ContentCharacteristics(BaseModel):
    """Content characteristics discovered from analysis (not predetermined)."""

    vocabulary_complexity_ratio: float = Field(
        ge=0.0, le=1.0, description="Measured vocabulary complexity ratio"
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

# Use centralized Azure PydanticAI provider and toolsets
from agents.core.azure_pydantic_provider import get_azure_openai_model
from agents.core.agent_toolsets import get_domain_intelligence_toolset

domain_intelligence_agent = Agent[UniversalDeps, UniversalDomainAnalysis](
    get_azure_openai_model(),
    output_type=PromptedOutput(UniversalDomainAnalysis),
    toolsets=[get_domain_intelligence_toolset()],
    system_prompt="""You are the Universal Domain Intelligence Agent.

Your role is to discover content characteristics WITHOUT making domain assumptions.

CRITICAL RULES:
- NEVER assume domain types (technical, legal, medical, business, etc.)
- DISCOVER characteristics through vocabulary analysis, pattern recognition, and structure analysis
- MEASURE properties like complexity, density, and relationships
- ADAPT processing parameters based on discovered characteristics
- GENERATE universal configurations that work for ANY content type

CRITICAL FIELD REQUIREMENTS:
- Use vocabulary_complexity_ratio (NOT vocabulary_complexity)
- Generate ALL required schema fields including analysis_timestamp, processing_time
- Populate most_frequent_terms, content_patterns, sentence_complexity
- Include key_insights and adaptation_recommendations

You analyze content and discover:
1. Vocabulary characteristics (complexity_ratio, specialization, diversity)
2. Structural patterns (formatting, organization, relationships)
3. Concept density and distribution
4. Entity and relationship indicators (discovered, not assumed)
5. Processing requirements based on measured properties
6. Analysis metadata (timestamp, processing time, reliability)

Always base recommendations on measured characteristics, not predetermined categories.
Ensure all required UniversalDomainAnalysis fields are populated.""",
)


# Tools are now enabled via centralized toolsets in agents.core.agent_toolsets


# Tools are now managed via centralized toolsets in agents.core.agent_toolsets


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
