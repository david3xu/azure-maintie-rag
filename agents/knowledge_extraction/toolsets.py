"""
🎯 Knowledge Extraction Agent Toolset - Following Target Architecture

This implements the PydanticAI-compliant toolset pattern as specified in:
/docs/implementation/AGENT_BOUNDARY_FIXES_IMPLEMENTATION.md

Target Structure:
agents/knowledge_extraction/toolsets.py  # Extraction-specific Toolset classes

Replaces tools/extraction_tools.py with proper agent co-location.
Uses unified extraction processor for consolidated entity and relationship extraction.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

# Import constants for zero-hardcoded-values compliance
from agents.core.constants import CacheConstants

# Import centralized configuration
from config.centralized_config import (
    get_quality_assessment_config,
    get_confidence_calculation_config
)
from agents.core.constants import KnowledgeExtractionConstants

# Import models from centralized data models
from agents.core.data_models import (
    ExtractionConfiguration,
    AzureServicesDeps,  # CORRECTED: For real Azure service access
    KnowledgeExtractionDeps,
    ExtractedKnowledge
)


class KnowledgeExtractionToolset(FunctionToolset):
    """
    🎯 PydanticAI-Compliant Knowledge Extraction Toolset
    
    Following AGENT_BOUNDARY_FIXES_IMPLEMENTATION.md target architecture:
    - Extraction-specific Toolset class in knowledge_extraction/toolsets.py
    - Replaces scattered @extraction_agent.tool decorators
    - Self-contained with agent co-location
    - Uses unified extraction processor for consolidated processing
    """

    def __init__(self):
        super().__init__()
        
        # Initialize unified extraction processor  
        from .processors.unified_extraction_processor import UnifiedExtractionProcessor
        self.unified_processor = UnifiedExtractionProcessor()
        
        # Register unified knowledge extraction tools
        self.add_function(self.extract_knowledge_unified, name='extract_knowledge_unified')
        self.add_function(self.extract_entities_multi_strategy, name='extract_entities_multi_strategy')
        self.add_function(self.extract_relationships_contextual, name='extract_relationships_contextual') 
        self.add_function(self.validate_extraction_quality, name='validate_extraction_quality')
        self.add_function(self.generate_knowledge_graph, name='generate_knowledge_graph')

    async def extract_knowledge_unified(
        self, ctx: RunContext[KnowledgeExtractionDeps], 
        document_content: str,
        config: ExtractionConfiguration,
        extraction_method: str = "hybrid"
    ) -> Dict[str, Any]:
        """Unified knowledge extraction using consolidated processor"""
        try:
            # Use the unified processor for complete extraction
            result = await self.unified_processor.extract_knowledge_complete(
                document_content, config, extraction_method
            )
            
            return {
                "entities": result.entities,
                "relationships": result.relationships,
                "entity_confidence_distribution": result.entity_confidence_distribution,
                "relationship_confidence_distribution": result.relationship_confidence_distribution,
                "processing_time": result.processing_time,
                "total_entities": result.total_entities,
                "total_relationships": result.total_relationships,
                "average_entity_confidence": result.average_entity_confidence,
                "average_relationship_confidence": result.average_relationship_confidence,
                "validation_passed": result.validation_passed,
                "validation_warnings": result.validation_warnings,
                "extraction_method": result.extraction_method,
                "graph_density": result.graph_density,
                "connected_components": result.connected_components,
                "coverage_percentage": result.coverage_percentage
            }
            
        except Exception as e:
            raise RuntimeError(f"Unified knowledge extraction failed: {str(e)}")

    async def extract_entities_multi_strategy(
        self, ctx: RunContext[KnowledgeExtractionDeps], 
        document_content: str, 
        config: ExtractionConfiguration,
        extraction_method: str = "hybrid"
    ) -> Dict[str, Any]:
        """Multi-strategy entity extraction using unified processor"""
        try:
            # Use unified processor for entity extraction only
            entities = await self.unified_processor._extract_entities_unified(
                document_content, config, extraction_method
            )
            
            # Convert to dict format expected by toolset
            entity_dicts = [self.unified_processor._entity_to_dict(e) for e in entities]
            
            return {
                "entities": entity_dicts,
                "total_entities": len(entity_dicts),
                "extraction_method": extraction_method,
                "confidence_score": sum(e.confidence for e in entities) / len(entities) if entities else CacheConstants.ZERO_FLOAT
            }
            
        except Exception as e:
            raise RuntimeError(f"Entity extraction failed: {str(e)}")

    async def extract_relationships_contextual(
        self, ctx: RunContext[KnowledgeExtractionDeps], 
        document_content: str, 
        entities: List[Dict[str, Any]], 
        config: ExtractionConfiguration,
        extraction_method: str = "hybrid"
    ) -> Dict[str, Any]:
        """Advanced contextual relationship extraction using unified processor"""
        try:
            if not entities:
                return {
                    "relationships": [],
                    "processing_time": CacheConstants.ZERO_FLOAT,
                    "relationship_types": [],
                    "confidence_score": CacheConstants.ZERO_FLOAT
                }
            
            # Convert entity dicts to EntityMatch objects for unified processor
            from .processors.unified_extraction_processor import EntityMatch
            entity_matches = []
            for e in entities:
                entity_matches.append(EntityMatch(
                    text=e.get("name", e.get("text", "")),
                    entity_type=e.get("type", "unknown"),
                    start_position=e.get("start_position", 0),
                    end_position=e.get("end_position", 0),
                    confidence=e.get("confidence", CacheConstants.ZERO_FLOAT),
                    extraction_method=e.get("extraction_method", "unknown"),
                    context=e.get("context", "")
                ))
            
            # Use unified processor for relationship extraction
            relationships = await self.unified_processor._extract_relationships_unified(
                document_content, entity_matches, config, extraction_method
            )
            
            # Convert to dict format expected by toolset
            relationship_dicts = [self.unified_processor._relationship_to_dict(r) for r in relationships]
            
            return {
                "relationships": relationship_dicts,
                "total_relationships": len(relationship_dicts),
                "relationship_types": list(set(r.relation_type for r in relationships)),
                "extraction_method": extraction_method,
                "confidence_score": sum(r.confidence for r in relationships) / len(relationships) if relationships else CacheConstants.ZERO_FLOAT
            }
            
        except Exception as e:
            return {
                "relationships": [],
                "processing_time": CacheConstants.ZERO_FLOAT,
                "relationship_types": [],
                "confidence_score": CacheConstants.ZERO_FLOAT,
                "error": str(e)
            }

    async def validate_extraction_quality(
        self, ctx: RunContext[KnowledgeExtractionDeps], 
        entities: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]],
        config: ExtractionConfiguration = None
    ) -> Dict[str, Any]:
        """Validate extraction quality using unified processor's validation system"""
        try:
            if config is None:
                config = ExtractionConfiguration()  # Use default config
            
            # Use the unified processor's validator
            validation_result = self.unified_processor.validator.validate_extraction(
                entities, relationships,
                config.entity_confidence_threshold,
                config.relationship_confidence_threshold
            )
            
            return {
                "validation_passed": validation_result.is_valid,
                "entity_count": validation_result.entity_count,
                "relationship_count": validation_result.relationship_count,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "overall_quality": CacheConstants.MAX_CONFIDENCE if validation_result.is_valid else CacheConstants.ZERO_FLOAT
            }
            
        except Exception as e:
            return {
                "validation_passed": False,
                "entity_count": 0,
                "relationship_count": 0,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "overall_quality": CacheConstants.ZERO_FLOAT
            }

    async def generate_knowledge_graph(
        self, ctx: RunContext[KnowledgeExtractionDeps], 
        entities: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]],
        config: ExtractionConfiguration = None
    ) -> Dict[str, Any]:
        """Generate knowledge graph from entities and relationships using unified processor"""
        try:
            if config is None:
                config = ExtractionConfiguration()
            
            # Calculate graph metrics using the unified processor
            from .processors.unified_extraction_processor import EntityMatch, RelationshipMatch
            
            # Convert to unified processor format for graph metrics calculation
            entity_matches = []
            for e in entities:
                entity_matches.append(EntityMatch(
                    text=e.get("name", e.get("text", "")),
                    entity_type=e.get("type", "unknown"),
                    start_position=e.get("start_position", 0),
                    end_position=e.get("end_position", 0),
                    confidence=e.get("confidence", CacheConstants.ZERO_FLOAT),
                    extraction_method=e.get("extraction_method", "unknown"),
                    context=e.get("context", "")
                ))
            
            relationship_matches = []
            for r in relationships:
                relationship_matches.append(RelationshipMatch(
                    source_entity=r.get("source", ""),
                    relation_type=r.get("relation", r.get("type", "related")),
                    target_entity=r.get("target", ""),
                    confidence=r.get("confidence", CacheConstants.ZERO_FLOAT),
                    extraction_method=r.get("extraction_method", "unknown"),
                    start_position=r.get("start_position", 0),
                    end_position=r.get("end_position", 0),
                    context=r.get("context", "")
                ))
            
            # Calculate graph metrics
            graph_metrics = self.unified_processor._calculate_graph_metrics(
                relationship_matches, entity_matches
            )
            
            # Prepare knowledge graph structure
            knowledge_graph = {
                "domain": config.domain_name,
                "timestamp": asyncio.get_event_loop().time(),
                "nodes": self._format_entities_as_nodes(entities),
                "edges": self._format_relationships_as_edges(relationships),
                "metadata": {
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                    "graph_density": graph_metrics["density"],
                    "connected_components": graph_metrics["components"],
                    "extraction_config": config.domain_name
                }
            }
            
            return {
                "knowledge_graph": knowledge_graph,
                "nodes_count": len(entities),
                "edges_count": len(relationships),
                "graph_density": graph_metrics["density"],
                "connected_components": graph_metrics["components"],
                "storage_ready": True
            }
            
        except Exception as e:
            return {
                "knowledge_graph": None,
                "nodes_count": 0,
                "edges_count": 0,
                "graph_density": CacheConstants.ZERO_FLOAT,
                "connected_components": 0,
                "storage_ready": False,
                "error": str(e)
            }

    # Helper methods for knowledge graph formatting
    def _format_entities_as_nodes(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format entities as knowledge graph nodes"""
        nodes = []
        
        for entity in entities:
            nodes.append({
                "id": entity.get("name", "").replace(" ", "_"),
                "label": entity.get("name", ""),
                "type": entity.get("type", "concept"),
                "confidence": entity.get("confidence", CacheConstants.ZERO_FLOAT),
                "properties": {
                    "extraction_method": entity.get("extraction_method", "unknown"),
                    "context": entity.get("context", "")
                }
            })
        
        return nodes

    def _format_relationships_as_edges(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format relationships as knowledge graph edges"""
        edges = []
        
        for rel in relationships:
            edges.append({
                "source": rel.get("source", "").replace(" ", "_"),
                "target": rel.get("target", "").replace(" ", "_"),
                "label": rel.get("relation", rel.get("type", "related")),
                "confidence": rel.get("confidence", CacheConstants.ZERO_FLOAT),
                "properties": {
                    "extraction_method": rel.get("extraction_method", "unknown")
                }
            })
        
        return edges


# Create the toolset instance lazily to avoid circular imports
_knowledge_extraction_toolset = None

def get_knowledge_extraction_toolset() -> KnowledgeExtractionToolset:
    """Get the Knowledge Extraction toolset with lazy initialization"""
    global _knowledge_extraction_toolset
    if _knowledge_extraction_toolset is None:
        _knowledge_extraction_toolset = KnowledgeExtractionToolset()
    return _knowledge_extraction_toolset

# For backward compatibility - this will be created on first access
def __getattr__(name):
    """Module-level lazy loading"""
    if name == "knowledge_extraction_toolset":
        return get_knowledge_extraction_toolset()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")