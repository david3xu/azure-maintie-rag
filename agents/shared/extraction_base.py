"""
Extraction Base Patterns - Shared Infrastructure

PydanticAI-enhanced base extraction patterns following PydanticAI best practices:
- Tool-like base classes with clear extraction interfaces
- Type-safe Pydantic models for extraction results
- Reusable extraction patterns across Knowledge Extraction and Domain Intelligence agents

Extracted common patterns from unified_extraction_processor.py to enable
consistent extraction interfaces across agent boundaries.
"""

from typing import Dict, List, Any, Optional, Union, Protocol, runtime_checkable
from pydantic import BaseModel, Field, validator
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExtractionType(Enum):
    """Extraction operation types"""

    ENTITY = "entity"
    RELATIONSHIP = "relationship"


class ExtractionStatus(Enum):
    """Extraction status values"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# Models consolidated to agents.core.data_models
