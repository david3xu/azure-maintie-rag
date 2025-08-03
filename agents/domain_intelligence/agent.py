"""
🎯 PHASE 1: PydanticAI-Compliant Agent Implementation
Agent 1 Domain Intelligence with Proper Toolset Co-Location

This implements the official PydanticAI toolset pattern following the documentation.
Replaces the scattered @domain_agent.tool approach with proper FunctionToolset.
"""

import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents.models.domain_models import DomainDeps
from agents.domain_intelligence.toolsets import domain_intelligence_toolset


# Required models for backward compatibility
class DomainDetectionResult(BaseModel):
    """Result of domain detection from query"""
    domain: str = Field(description="Detected domain name")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    matched_patterns: List[str] = Field(description="Patterns that matched the query")
    reasoning: str = Field(description="Explanation of domain detection")
    discovered_entities: List[str] = Field(description="Entity types discovered for this domain")
    ml_config: Optional[Dict] = Field(description="ML configuration for the domain", default=None)


class AvailableDomainsResult(BaseModel):
    """Result of domain discovery"""
    domains: List[str] = Field(description="List of discovered domain names")
    source: str = Field(description="Source of domain discovery (filesystem, cache, etc.)")
    total_patterns: int = Field(description="Total patterns available across all domains")


class DomainAnalysisResult(BaseModel):
    """Result of complete domain analysis"""
    domain: str = Field(description="Domain name")
    classification: Dict = Field(description="Domain classification details")
    patterns_extracted: int = Field(description="Number of patterns extracted")
    config_generated: bool = Field(description="Whether configuration was generated")
    confidence: float = Field(description="Overall confidence score")


def get_azure_openai_model():
    """Get Azure OpenAI model using environment variable deployment name"""
    try:
        # Azure OpenAI configuration from environment
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        deployment_name = os.getenv("OPENAI_MODEL_DEPLOYMENT", "gpt-4.1")
        
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
        
        Always use your available tools to provide structured, data-driven responses."""
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