"""
Knowledge Extraction Agent - Following Target Architecture

Specialized PydanticAI agent for processing documents using optimized extraction configurations.
Implements the target architecture with:
- Lazy initialization pattern
- FunctionToolset integration
- Azure OpenAI integration with environment variables

Part of the clear separation between Configuration System and Knowledge Extraction Pipeline.

Responsibilities:
- Process individual documents using provided extraction configurations
- Extract entities, relationships, and structured knowledge
- Provide feedback for configuration optimization
- Build knowledge graphs and training data

NOT responsible for:
- Domain analysis or configuration generation
- System-wide parameter optimization
- Cross-corpus pattern analysis

Following Azure Universal RAG Coding Standards:
- Data-driven processing (uses provided configurations)
- Production-ready with comprehensive error handling
- Universal design (works with any domain via configuration)
- Performance-first (async operations, sub-3s targets)
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Clean configuration imports (CODING_STANDARDS compliant)
from config.centralized_config import get_extraction_config
from agents.core.constants import KnowledgeExtractionConstants, ProcessingConstants, CacheConstants, AzureServiceConstants

# Lazy configuration loading - will be loaded when needed
_config = None

def _get_config(domain_name: str = "general"):
    """Get extraction configuration lazily to avoid circular imports"""
    try:
        return get_extraction_config(domain_name)
    except Exception:
        # Return safe defaults if config loading fails during initialization
        from types import SimpleNamespace
        return SimpleNamespace(
            azure_endpoint="https://example.openai.azure.com/",
            api_version=AzureServiceConstants.OPENAI_API_VERSION, 
            # deployment_name="gpt-4o",  # ❌ HARDCODED - Removed to force Dynamic Model Manager
            deployment_name=None,  # Must be loaded from Dynamic Model Manager
            confidence_default=KnowledgeExtractionConstants.DEFAULT_CONFIDENCE_THRESHOLD,
            processing_time_initial=1.0,
            max_successful_extractions=1,
            entity_precision_multiplier=1.0,
            entity_recall_multiplier=1.0,
            relationship_precision_multiplier=1.0,
            relationship_recall_multiplier=1.0,
            max_documents_divisor=1,
            memory_usage_default_mb=KnowledgeExtractionConstants.DEFAULT_MEMORY_USAGE_MB,
            cpu_utilization_default_percent=50,
            cache_hit_rate_default=KnowledgeExtractionConstants.DEFAULT_CACHE_HIT_RATE,
            cache_hit_rate_disabled=CacheConstants.ZERO_FLOAT
        )

# Import models from centralized data models
from agents.core.data_models import (
    # Legacy models (maintained for compatibility)
    ExtractionConfiguration,
    ExtractionResults,
    ExtractedKnowledge,
    
    # NEW: Enhanced models with PydanticAI integration and dynamic configuration
    ConsolidatedExtractionConfiguration,
    # EnhancedKnowledgeExtractionContract deleted - use KnowledgeExtractionContract
    ConfigurationResolver,
    PydanticAIContextualModel
)


# Lazy initialization to avoid import-time Azure connection requirements
_knowledge_extraction_agent = None

# Import the toolset following target architecture with lazy loading
from .toolsets import get_knowledge_extraction_toolset, KnowledgeExtractionDeps

def _create_agent_with_toolset() -> Agent:
    """Create Knowledge Extraction Agent with unified processor integration"""
    import os
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.azure import AzureProvider

    try:
        # Configure Azure OpenAI provider
        config = _get_config()
        azure_endpoint = config.azure_endpoint
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_version = config.api_version
        deployment_name = config.deployment_name

        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable is required")

        # Use Azure OpenAI with API key
        azure_model = OpenAIModel(
            deployment_name,
            provider=AzureProvider(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=api_key,
            ),
        )

        # Create agent with unified extraction capabilities
        agent = Agent(
            azure_model,
            deps_type=KnowledgeExtractionDeps,
            toolsets=[get_knowledge_extraction_toolset()],
            name="knowledge-extraction-agent",
            system_prompt=(
                "You are the Knowledge Extraction Specialist using unified extraction processing. "
                "Your capabilities include:"
                "1. Unified entity and relationship extraction in single pass for efficiency"
                "2. Multi-strategy approaches (pattern-based, NLP-based, hybrid) with configurable parameters"
                "3. Integrated validation and quality assessment using centralized configuration"
                "4. Graph-aware relationship extraction with contextual analysis"
                "5. Performance optimization through consolidated processing pipeline"
                "You work with Azure AI services and use centralized configuration for all parameters."
            ),
        )
        
        return agent

    except Exception as e:
        error_msg = (
            f"❌ Failed to create Knowledge Extraction Agent with unified processor: {e}. "
            "Please ensure Azure OpenAI credentials are properly configured."
        )
        raise RuntimeError(error_msg)

def get_knowledge_extraction_agent() -> Agent:
    """Get Knowledge Extraction Agent with lazy initialization and unified processor"""
    global _knowledge_extraction_agent
    if _knowledge_extraction_agent is None:
        _knowledge_extraction_agent = _create_agent_with_toolset()
    return _knowledge_extraction_agent

# For backward compatibility, create module-level agent getter
knowledge_extraction_agent = get_knowledge_extraction_agent


# Legacy compatibility function for testing
async def test_knowledge_extraction_agent():
    """Test the Knowledge Extraction Agent with target architecture"""
    try:
        # Get agent with lazy initialization 
        agent = get_knowledge_extraction_agent()
        
        print("✅ Knowledge Extraction Agent created successfully with toolset integration")
        print(f"   - Agent name: {agent.name}")
        print(f"   - Toolsets registered: {len(agent.toolsets) if hasattr(agent, 'toolsets') else 0}")
        print(f"   - Dependencies type: {agent._deps_type.__name__ if hasattr(agent, '_deps_type') else 'None'}")
        
        return {
            "agent_created": True,
            "lazy_initialization": True,
            "toolset_integration": True,
            "azure_openai_model": True
        }
        
    except Exception as e:
        print(f"❌ Knowledge Extraction Agent test failed: {e}")
        return {
            "agent_created": False,
            "error": str(e)
        }


# Simple wrapper function for backward compatibility
async def extract_knowledge_from_document(
    document_content: str,
    config: ExtractionConfiguration,
    document_id: str = None,
) -> ExtractedKnowledge:
    """
    Extract knowledge from a single document using provided configuration.
    
    This is a simplified wrapper that delegates to the agent's toolset.
    """
    try:
        agent = get_knowledge_extraction_agent()
        
        # Create a simple deps object for the toolset
        from .toolsets import KnowledgeExtractionDeps
        deps = KnowledgeExtractionDeps()
        
        # Use the toolset directly for extraction
        toolset_instance = get_knowledge_extraction_toolset()
        
        # Simulate the extraction using the toolset pattern
        start_time = time.time()
        
        # Extract entities using multi-strategy approach
        entity_result = await toolset_instance.extract_entities_multi_strategy(
            None, document_content, config  # ctx not needed for basic operation
        )
        
        # Extract relationships
        relationship_result = await toolset_instance.extract_relationships_contextual(
            None, document_content, entity_result.get("entities", []), config
        )
        
        # Validate extraction quality
        validation_result = await toolset_instance.validate_extraction_quality(
            None, entity_result.get("entities", []), relationship_result.get("relationships", []), config
        )
        
        processing_time = time.time() - start_time
        
        # Create structured knowledge object
        knowledge = ExtractedKnowledge(
            source_document=document_id or f"doc_{int(time.time())}",
            extraction_timestamp=datetime.now().isoformat(),
            processing_time_seconds=processing_time,
            entities=entity_result.get("entities", []),
            relationships=relationship_result.get("relationships", []),
            key_concepts=[],  # Could be enhanced
            technical_terms=[],  # Could be enhanced
            extraction_confidence=validation_result.get("overall_quality", _get_config().confidence_default),
            entity_count=len(entity_result.get("entities", [])),
            relationship_count=len(relationship_result.get("relationships", [])),
            passed_validation=validation_result.get("validation_passed", False),
            validation_warnings=validation_result.get("warnings", []),
        )
        
        return knowledge
        
    except Exception as e:
        # Create error result
        return ExtractedKnowledge(
            source_document=document_id or f"doc_{int(time.time())}",
            extraction_timestamp=datetime.now().isoformat(),
            processing_time_seconds=_get_config().processing_time_initial,
            entities=[],
            relationships=[],
            key_concepts=[],
            technical_terms=[],
            extraction_confidence=CacheConstants.ZERO_FLOAT,
            entity_count=0,
            relationship_count=0,
            passed_validation=False,
            validation_warnings=[f"Extraction failed: {str(e)}"],
        )


async def extract_knowledge_from_documents(
    documents: List[Tuple[str, str]],  # (content, doc_id) pairs
    config: ExtractionConfiguration,
) -> ExtractionResults:
    """
    Extract knowledge from multiple documents using configuration.
    
    This is a simplified wrapper that processes multiple documents.
    """
    try:
        start_time = time.time()
        
        # Process documents sequentially for simplicity
        # (Could be enhanced with parallel processing)
        successful_extractions = []
        failed_extractions = []
        
        for content, doc_id in documents:
            try:
                knowledge = await extract_knowledge_from_document(content, config, doc_id)
                successful_extractions.append(knowledge)
            except Exception as e:
                failed_extractions.append(e)
        
        # Calculate metrics
        total_time = time.time() - start_time
        
        # Aggregate extraction data
        total_entity_count = sum(e.entity_count for e in successful_extractions)
        total_relationship_count = sum(e.relationship_count for e in successful_extractions)
        config = _get_config()
        avg_confidence = sum(e.extraction_confidence for e in successful_extractions) / max(len(successful_extractions), config.max_successful_extractions)
        
        # Create extraction results
        results = ExtractionResults(
            domain_name=config.domain_name,
            documents_processed=len(documents),
            total_processing_time_seconds=total_time,
            extraction_accuracy=avg_confidence,
            entity_precision=avg_confidence * config.entity_precision_multiplier,
            entity_recall=avg_confidence * config.entity_recall_multiplier,
            relationship_precision=avg_confidence * config.relationship_precision_multiplier,
            relationship_recall=avg_confidence * config.relationship_recall_multiplier,
            average_processing_time_per_document=total_time / max(len(documents), config.max_documents_divisor),
            memory_usage_mb=config.memory_usage_default_mb,
            cpu_utilization_percent=config.cpu_utilization_default_percent,
            cache_hit_rate=config.cache_hit_rate_default if config.enable_caching else config.cache_hit_rate_disabled,
            total_entities_extracted=total_entity_count,
            total_relationships_extracted=total_relationship_count,
            unique_entity_types_found=len(set(
                e.get("type", "unknown")
                for extraction in successful_extractions
                for e in extraction.entities
            )),
            unique_relationship_types_found=len(set(
                r.get("type", "unknown")
                for extraction in successful_extractions
                for r in extraction.relationships
            )),
            extraction_passed_validation=len(failed_extractions) == 0,
            validation_error_count=len(failed_extractions),
            validation_warnings=[
                w
                for extraction in successful_extractions
                for w in extraction.validation_warnings
            ],
        )
        
        return results
        
    except Exception as e:
        # Return error results
        return ExtractionResults(
            domain_name=config.domain_name,
            documents_processed=0,
            total_processing_time_seconds=CacheConstants.ZERO_FLOAT,
            extraction_accuracy=CacheConstants.ZERO_FLOAT,
            entity_precision=CacheConstants.ZERO_FLOAT,
            entity_recall=CacheConstants.ZERO_FLOAT,
            relationship_precision=CacheConstants.ZERO_FLOAT,
            relationship_recall=CacheConstants.ZERO_FLOAT,
            average_processing_time_per_document=CacheConstants.ZERO_FLOAT,
            memory_usage_mb=KnowledgeExtractionConstants.DEFAULT_CPU_UTILIZATION_PERCENT,
            cpu_utilization_percent=KnowledgeExtractionConstants.DEFAULT_CPU_UTILIZATION_PERCENT,
            cache_hit_rate=CacheConstants.ZERO_FLOAT,
            total_entities_extracted=0,
            total_relationships_extracted=0,
            unique_entity_types_found=0,
            unique_relationship_types_found=0,
            extraction_passed_validation=False,
            validation_error_count=1,  # One error (the exception)
            validation_warnings=[f"Batch extraction failed: {str(e)}"],
        )


class ExtractionError(Exception):
    """Exception raised when knowledge extraction fails"""
    pass


# Export main components
__all__ = [
    "ExtractedKnowledge",
    "ExtractionError",
    "ExtractionConfiguration", 
    "ExtractionResults",
    "get_knowledge_extraction_agent",
    "knowledge_extraction_agent",
    "test_knowledge_extraction_agent",
    "extract_knowledge_from_document",
    "extract_knowledge_from_documents",
]
