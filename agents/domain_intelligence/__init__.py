"""Universal Domain Intelligence Agent - Zero Hardcoded Domain Knowledge"""

# Now using the working agent.py (broken version backed up as agent_broken_pydanticai.py)
from .agent import (
    run_domain_analysis,
    create_domain_intelligence_agent,
    domain_intelligence_agent,
)

# Import universal models from core
from agents.core.universal_models import (
    UniversalDomainAnalysis,
    UniversalDomainCharacteristics,
    UniversalProcessingConfiguration,
)

__all__ = [
    "run_domain_analysis",
    "create_domain_intelligence_agent",
    "domain_intelligence_agent",
    "UniversalDomainAnalysis",
    "UniversalDomainCharacteristics",
    "UniversalProcessingConfiguration",
]
