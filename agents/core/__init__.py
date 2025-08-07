"""Universal core utilities and models for zero-hardcoded-values architecture"""

from .constants import (
    AzureConstants,
    ProcessingConstants,
    ConfidenceConstants,
)
from .universal_models import (
    UniversalDomainCharacteristics,
    UniversalProcessingConfiguration, 
    UniversalDomainAnalysis,
    UniversalDomainDeps,
    UniversalOrchestrationResult,
    AgentHandoffData,
)

__all__ = [
    "UniversalDomainCharacteristics",
    "UniversalProcessingConfiguration",
    "UniversalDomainAnalysis",
    "UniversalDomainDeps",
    "UniversalOrchestrationResult",
    "AgentHandoffData",
]
