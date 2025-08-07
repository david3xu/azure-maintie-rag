"""
Azure Universal RAG Multi-Agent System - PydanticAI Architecture
===============================================================

Production-grade multi-agent system following PydanticAI best practices:
- Proper agent delegation with ctx.deps and ctx.usage
- Centralized dependency management via UniversalDeps
- Universal content processing without domain assumptions
- Real Azure service integration (OpenAI, Cosmos DB, Cognitive Search, ML)

PydanticAI Usage:
    from agents import UniversalOrchestrator
    from agents.domain_intelligence.agent import run_domain_analysis

    # Universal content analysis
    analysis = await run_domain_analysis(content)
    print(f"Content signature: {analysis.content_signature}")

    # Multi-agent orchestration
    orchestrator = UniversalOrchestrator()
    result = await orchestrator.process_knowledge_extraction_workflow(content)
"""

from .core.universal_deps import UniversalDeps, get_universal_deps

# Core universal components  
from .core.universal_models import (
    UniversalDomainAnalysis,
    UniversalDomainCharacteristics, 
    UniversalProcessingConfiguration,
    SearchResult,
    ExtractedEntity,
    ExtractedRelationship,
)

# PydanticAI Agents (proper architecture)
from .domain_intelligence.agent import (
    create_domain_intelligence_agent,
    domain_intelligence_agent,
    run_domain_analysis,
)
from .knowledge_extraction.agent import (
    create_knowledge_extraction_agent,
    knowledge_extraction_agent,
    run_knowledge_extraction,
)

# Multi-agent orchestration
from .orchestrator import UniversalOrchestrator, UniversalWorkflowResult

# Shared utilities (utility functions called by agent tools)
from .shared.query_tools import (
    generate_analysis_query,
    generate_gremlin_query,
    generate_search_query,
    orchestrate_query_workflow,
)
from .universal_search.agent import (
    create_universal_search_agent,
    run_universal_search,
    universal_search_agent,
)

__version__ = "3.0.0-pydantic-ai"

__all__ = [
    # Core dependencies
    "UniversalDeps",
    "get_universal_deps",
    # PydanticAI Agents
    "domain_intelligence_agent",
    "knowledge_extraction_agent",
    "universal_search_agent",
    "run_domain_analysis",
    "run_knowledge_extraction",
    "run_universal_search",
    # Factory functions
    "create_domain_intelligence_agent",
    "create_knowledge_extraction_agent",
    "create_universal_search_agent",
    # Multi-agent orchestration
    "UniversalOrchestrator",
    "UniversalWorkflowResult",
    # Universal data models
    "UniversalDomainAnalysis",
    "UniversalDomainCharacteristics",
    "UniversalProcessingConfiguration",
    "SearchResult",
    "ExtractedEntity",
    "ExtractedRelationship",
    # Utility functions
    "generate_gremlin_query",
    "generate_search_query",
    "generate_analysis_query",
    "orchestrate_query_workflow",
]
