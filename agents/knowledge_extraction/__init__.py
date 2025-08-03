"""
Knowledge Extraction Agent - Enterprise Entity and Relationship Extraction

This agent provides advanced knowledge extraction capabilities including:
- Multi-strategy entity extraction (pattern-based, NLP-based, hybrid)
- Advanced relationship extraction and validation
- Knowledge graph construction and optimization
- Quality assessment and validation
- PydanticAI Tools: Enterprise integration for PydanticAI agents

Key Features:
- Multiple extraction strategies for enhanced accuracy
- Confidence scoring and validation frameworks
- Performance optimization and caching
- Domain-aware extraction adaptation
- Enterprise-grade monitoring and observability

Competitive Advantages:
- Multi-strategy extraction approach
- Advanced confidence scoring
- Real-time quality assessment
- Enterprise-grade scalability
"""

# Import processors
from .processors.entity_processor import EntityProcessor
from .processors.relationship_processor import RelationshipProcessor
from .processors.validation_processor import ValidationProcessor

# Import PydanticAI tools
from .pydantic_tools import (
    extract_entities_tool,
    extract_relationships_tool,
    build_knowledge_graph_tool,
    validate_extraction_quality_tool
)

__all__ = [
    # Processors
    "EntityProcessor",
    "RelationshipProcessor", 
    "ValidationProcessor",
    
    # PydanticAI Tools
    "extract_entities_tool",
    "extract_relationships_tool",
    "build_knowledge_graph_tool",
    "validate_extraction_quality_tool"
]