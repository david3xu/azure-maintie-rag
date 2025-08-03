"""
Knowledge Extraction Agent - Following Target Architecture

This agent provides advanced knowledge extraction capabilities with:
- Lazy initialization pattern
- FunctionToolset integration  
- Azure OpenAI integration with environment variables
- Multi-strategy entity and relationship extraction
- Quality validation and knowledge graph construction

Target Architecture Implementation:
- PydanticAI-compliant toolset pattern
- Self-contained with agent co-location
- No import-time side effects
"""

# Import main agent functions (lazy initialization)
from .agent import (
    get_knowledge_extraction_agent,
    extract_knowledge_from_document,
    extract_knowledge_from_documents,
    test_knowledge_extraction_agent,
)

# Import toolset for direct access if needed
from .toolsets import (
    KnowledgeExtractionToolset,
    KnowledgeExtractionDeps,
    knowledge_extraction_toolset,
)

__all__ = [
    # Main agent interfaces
    "get_knowledge_extraction_agent",
    "extract_knowledge_from_document", 
    "extract_knowledge_from_documents",
    "test_knowledge_extraction_agent",
    # Toolset classes
    "KnowledgeExtractionToolset",
    "KnowledgeExtractionDeps", 
    "knowledge_extraction_toolset",
]
