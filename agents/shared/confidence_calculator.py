"""
Confidence Calculation Utilities - Shared Infrastructure

PydanticAI-enhanced confidence scoring functions following PydanticAI best practices:
- Tool-like functions with clear confidence scoring schemas
- Type-safe Pydantic models for confidence calculations
- Reusable across Knowledge Extraction and Domain Intelligence agents

Extracted from unified_extraction_processor.py to enable cross-agent confidence scoring.
Used by Knowledge Extraction for entity/relationship confidence and Domain Intelligence
for quality assessment scoring.
"""

from typing import Any, Dict

from agents.core.constants import StubConstants

# Import consolidated data models when needed by actual implementations
from agents.core.data_models import (
    ConfidenceMethod,
    ConfidenceScore,
)


# Stub functions - these should be properly implemented
def calculate_adaptive_confidence(data: Dict[str, Any]) -> float:
    """Stub for adaptive confidence calculation"""
    return StubConstants.STUB_ADAPTIVE_CONFIDENCE


# Export the models for use by other modules
__all__ = [
    "calculate_adaptive_confidence",
    "ConfidenceScore",
    "ConfidenceMethod",
]
