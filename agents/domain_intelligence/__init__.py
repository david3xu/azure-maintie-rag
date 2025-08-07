"""Universal Domain Intelligence Agent - Zero Hardcoded Domain Knowledge"""

# Now using the working agent.py (broken version backed up as agent_broken_pydanticai.py)
from .agent import (
    run_universal_domain_analysis,
    UniversalDomainDeps, 
    UniversalDomainAnalysis
)

# Import characteristics and config from agent
from .agent import (
    WorkingDomainCharacteristics as UniversalDomainCharacteristics,
    WorkingProcessingConfiguration as UniversalProcessingConfiguration
)

__all__ = [
    "run_universal_domain_analysis", 
    "UniversalDomainDeps",
    "UniversalDomainAnalysis",
    "UniversalDomainCharacteristics",
    "UniversalProcessingConfiguration"
]