"""Knowledge Extraction Agent"""

from .agent import (
    knowledge_extraction_agent,
    create_knowledge_extraction_agent,
    run_knowledge_extraction,
    ExtractionResult,
)

# Import universal models from core
from agents.core.universal_models import ExtractedEntity, ExtractedRelationship

__all__ = [
    "knowledge_extraction_agent",
    "create_knowledge_extraction_agent",
    "run_knowledge_extraction",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedRelationship",
]
