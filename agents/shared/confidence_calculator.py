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

from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator
import statistics
import math
from enum import Enum

# Import consolidated data models
from agents.core.data_models import (
    ConfidenceScore,
    # AggregatedConfidence, EntityConfidenceFactors deleted - using simple calculations
    RelationshipConfidenceFactors,
    ConfidenceMethod,
)


# Stub functions - these should be properly implemented
def calculate_adaptive_confidence(data: Dict[str, Any]) -> float:
    """Stub for adaptive confidence calculation"""
    return 0.8


# ConfidenceScore and other models imported from data_models.py above
