"""
🎯 PHASE 1: PydanticAI-Compliant Agent Implementation
Agent 1 Domain Intelligence with Proper Toolset Co-Location

This implements the official PydanticAI toolset pattern following the documentation.
Replaces the scattered @domain_agent.tool approach with proper FunctionToolset.
"""

import os
from typing import Dict, List, Optional
from pydantic_ai import Agent

# Import models from centralized data models (NEW CONSOLIDATED MODELS)
from agents.core.data_models import (
    # Legacy compatibility
    DomainIntelligenceDeps as DomainDeps,
    DomainDetectionResult,
    AnalysisResult as AvailableDomainsResult,
    AnalysisResult as DomainAnalysisResult,
    # Enhanced models deleted - use basic DomainAnalysisContract instead
    ConsolidatedExtractionConfiguration,
    ConfigurationResolver,
    PydanticAIContextualModel,
)

from agents.domain_intelligence.toolsets import domain_intelligence_toolset

# Clean configuration imports (CODING_STANDARDS compliant)
from config.centralized_config import get_model_config


# Backward compatibility
class AgentConfig:
    def __init__(self):
        model_config = get_model_config()
        self.default_openai_api_version = model_config.api_version
        self.default_model_deployment = model_config.deployment_name


get_agent_config = lambda: AgentConfig()


def get_azure_openai_model():
    """Get Azure OpenAI model using environment variable deployment name"""
    try:
        # Get centralized configuration
        agent_config = get_agent_config()

        # Azure OpenAI configuration from environment
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION", agent_config.default_openai_api_version
        )
        deployment_name = os.getenv(
            "OPENAI_MODEL_DEPLOYMENT", agent_config.default_model_deployment
        )

        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")

        # Use the deployment name from environment variables
        # This matches your Azure AI Foundry deployment: gpt-4.1
        model_name = f"openai:{deployment_name}"

        print(f"Using Azure OpenAI model: {model_name}")
        return model_name

    except Exception as e:
        print(f"Error getting Azure OpenAI model: {e}")
        raise


def create_domain_intelligence_agent() -> Agent:
    """
    🎯 CORE INNOVATION: PydanticAI-compliant Agent 1 with proper toolset co-location

    Following official PydanticAI documentation patterns:
    - FunctionToolset for tool organization
    - Proper tool co-location
    - Structured output types
    - Clear agent boundaries
    """

    model_name = get_azure_openai_model()

    # Create agent with proper toolset pattern
    agent = Agent(
        model_name,
        deps_type=DomainDeps,
        toolsets=[domain_intelligence_toolset],  # ✅ PydanticAI compliant toolset
        system_prompt="""You are the Domain Intelligence Agent specializing in zero-config pattern discovery.
        
        Core capabilities:
        - Discover domains from filesystem subdirectories (data/raw/Programming-Language → programming_language)
        - Generate 100% learned extraction configurations with zero hardcoded critical values
        - Analyze corpus statistics for data-driven parameter learning
        - Extract semantic patterns using hybrid LLM + statistical methods
        - Validate configuration quality and optimization
        
        Key principles:
        - All critical parameters (entity_threshold, chunk_size, classification_rules, response_sla) MUST be learned from data
        - Only acceptable hardcoded values are non-critical defaults (cache_ttl, batch_size, etc.)
        - Always provide structured responses using your tools
        - Base all decisions on actual corpus analysis, not assumptions
        
        Always use your available tools to provide structured, data-driven responses.""",
    )

    return agent


# Lazy initialization to avoid import-time Azure connection requirements
_domain_intelligence_agent = None


def get_domain_intelligence_agent():
    """Get domain intelligence agent with lazy initialization"""
    global _domain_intelligence_agent
    if _domain_intelligence_agent is None:
        _domain_intelligence_agent = create_domain_intelligence_agent()
    return _domain_intelligence_agent


# Module-level lazy access
def domain_intelligence_agent():
    return get_domain_intelligence_agent()


def domain_agent():
    return get_domain_intelligence_agent()


# For direct access (but lazy)
domain_intelligence_agent = get_domain_intelligence_agent
domain_agent = get_domain_intelligence_agent

# Alias for imports that expect get_domain_agent
get_domain_agent = get_domain_intelligence_agent
