"""Knowledge Extraction Agent"""

# Import universal models from core
from agents.core.universal_models import ExtractedEntity, ExtractedRelationship

from .agent import (
    ExtractionResult,
    create_knowledge_extraction_agent,
    knowledge_extraction_agent,
    run_knowledge_extraction,
)

__all__ = [
    "knowledge_extraction_agent",
    "create_knowledge_extraction_agent",
    "run_knowledge_extraction",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedRelationship",
]
