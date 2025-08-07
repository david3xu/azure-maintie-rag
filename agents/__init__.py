"""
Universal RAG Agent System - Zero Hardcoded Domain Knowledge
===========================================================

Truly universal multi-agent system that adapts to ANY content type through
data-driven discovery rather than predetermined assumptions.

Universal Usage:
    from @agents import run_universal_analysis
    
    # Works with ANY content type
    analysis = await run_universal_analysis("/path/to/any/content")
    print(f"Discovered signature: {analysis.domain_signature}")
"""

from .core.universal_models import *
# Now using the fixed agent.py (broken version backed up)
from .domain_intelligence.agent import (
    run_universal_domain_analysis,
    UniversalDomainDeps,
    UniversalDomainAnalysis
)
from .orchestrator import UniversalOrchestrator

# Import other agents (to be updated for universal integration)
# Temporarily commented out due to PydanticAI API key issues
# from .knowledge_extraction.agent import agent as extraction_agent
# from .universal_search.agent import agent as search_agent

__version__ = "2.0.0-universal-rag"

__all__ = [
    # Universal orchestrator
    "UniversalOrchestrator",
    
    # Universal domain intelligence (working version)
    "run_universal_domain_analysis", 
    "UniversalDomainDeps",
    "UniversalDomainAnalysis",
    
    # Universal data models
    "UniversalDomainCharacteristics",
    "UniversalProcessingConfiguration",
    "UniversalOrchestrationResult",
    "AgentHandoffData",
]