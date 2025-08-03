"""
🎯 Knowledge Extraction Agent Toolset - Following Target Architecture

This implements the PydanticAI-compliant toolset pattern as specified in:
/docs/implementation/AGENT_BOUNDARY_FIXES_IMPLEMENTATION.md

Target Structure:
agents/knowledge_extraction/toolsets.py  # Extraction-specific Toolset classes

Replaces tools/extraction_tools.py with proper agent co-location.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

# Import models and dependencies (avoid circular imports)
# Define simplified models locally to avoid circular dependencies
class ExtractionConfiguration(BaseModel):
    """Simplified extraction configuration model"""
    domain_name: str = "general"
    entity_confidence_threshold: float = 0.7
    relationship_confidence_threshold: float = 0.65
    expected_entity_types: List[str] = []
    technical_vocabulary: List[str] = []
    key_concepts: List[str] = []
    minimum_quality_score: float = 0.6
    enable_caching: bool = True


class KnowledgeExtractionDeps(BaseModel):
    """Knowledge Extraction Agent dependencies following target architecture"""
    azure_services: Optional[any] = None
    cache_manager: Optional[any] = None
    entity_extractor: Optional[any] = None
    relationship_extractor: Optional[any] = None
    knowledge_graph_builder: Optional[any] = None
    vector_embedding_generator: Optional[any] = None
    
    class Config:
        arbitrary_types_allowed = True


class ExtractedKnowledge(BaseModel):
    """Structured knowledge extracted from a document"""
    
    source_document: str
    extraction_timestamp: str
    processing_time_seconds: float
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    key_concepts: List[str]
    technical_terms: List[str]
    extraction_confidence: float
    entity_count: int
    relationship_count: int
    passed_validation: bool
    validation_warnings: List[str]


class KnowledgeExtractionToolset(FunctionToolset):
    """
    🎯 PydanticAI-Compliant Knowledge Extraction Toolset
    
    Following AGENT_BOUNDARY_FIXES_IMPLEMENTATION.md target architecture:
    - Extraction-specific Toolset class in knowledge_extraction/toolsets.py
    - Replaces scattered @extraction_agent.tool decorators
    - Self-contained with agent co-location
    """

    def __init__(self):
        super().__init__()
        
        # Register core knowledge extraction tools
        self.add_function(self.extract_entities_multi_strategy, name='extract_entities_multi_strategy')
        self.add_function(self.extract_relationships_contextual, name='extract_relationships_contextual')
        self.add_function(self.validate_extraction_quality, name='validate_extraction_quality')
        self.add_function(self.store_knowledge_graph, name='store_knowledge_graph')

    async def extract_entities_multi_strategy(
        self, ctx: RunContext[KnowledgeExtractionDeps], 
        document_content: str, 
        config: ExtractionConfiguration
    ) -> Dict[str, Any]:
        """Multi-strategy entity extraction with Azure integration"""
        try:
            start_time = time.time()
            
            # Strategy 1: Rule-based extraction for structured patterns
            rule_based_entities = await self._extract_entities_rule_based(document_content, config)
            
            # Strategy 2: Azure Cognitive Services for standard entity types
            azure_entities = await self._extract_entities_azure_cognitive(document_content, config)
            
            # Strategy 3: LLM-powered extraction for domain-specific entities
            llm_entities = await self._extract_entities_llm(document_content, config)
            
            # Fusion and deduplication
            entities = await self._fuse_entity_results(
                rule_based_entities, azure_entities, llm_entities, config
            )
            
            processing_time = time.time() - start_time
            
            return {
                "entities": entities,
                "processing_time": processing_time,
                "strategies_used": ["rule_based", "azure_cognitive", "llm"],
                "total_entities": len(entities),
                "confidence_score": self._calculate_entity_confidence(entities)
            }
            
        except Exception as e:
            # Fallback extraction
            return await self._fallback_entity_extraction(document_content, config, str(e))

    async def extract_relationships_contextual(
        self, ctx: RunContext[KnowledgeExtractionDeps], 
        document_content: str, 
        entities: List[Dict[str, Any]], 
        config: ExtractionConfiguration
    ) -> Dict[str, Any]:
        """Advanced contextual relationship extraction"""
        try:
            start_time = time.time()
            
            if not entities:
                return {
                    "relationships": [],
                    "processing_time": 0.0,
                    "relationship_types": [],
                    "confidence_score": 0.0
                }
            
            # Context-aware relationship extraction
            relationships = []
            
            # Find co-occurrence relationships
            cooccurrence_rels = await self._extract_cooccurrence_relationships(
                document_content, entities, config
            )
            relationships.extend(cooccurrence_rels)
            
            # Extract syntactic dependency relationships
            dependency_rels = await self._extract_dependency_relationships(
                document_content, entities, config
            )
            relationships.extend(dependency_rels)
            
            # Extract semantic relationships using LLM
            semantic_rels = await self._extract_semantic_relationships(
                document_content, entities, config
            )
            relationships.extend(semantic_rels)
            
            # Filter and validate relationships
            validated_relationships = await self._validate_relationships(
                relationships, config
            )
            
            processing_time = time.time() - start_time
            
            return {
                "relationships": validated_relationships,
                "processing_time": processing_time,
                "relationship_types": list(set(r.get("type", "related") for r in validated_relationships)),
                "confidence_score": self._calculate_relationship_confidence(validated_relationships)
            }
            
        except Exception as e:
            return {
                "relationships": [],
                "processing_time": 0.0,
                "relationship_types": [],
                "confidence_score": 0.0,
                "error": str(e)
            }

    async def validate_extraction_quality(
        self, ctx: RunContext[KnowledgeExtractionDeps], 
        entities: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]], 
        config: ExtractionConfiguration
    ) -> Dict[str, Any]:
        """Comprehensive quality validation framework"""
        try:
            validation_results = {
                "overall_quality": 0.0,
                "entity_quality": 0.0,
                "relationship_quality": 0.0,
                "coverage_score": 0.0,
                "consistency_score": 0.0,
                "validation_passed": False,
                "warnings": [],
                "recommendations": []
            }
            
            # Entity quality assessment
            entity_quality = await self._assess_entity_quality(entities, config)
            validation_results["entity_quality"] = entity_quality
            
            # Relationship quality assessment
            relationship_quality = await self._assess_relationship_quality(relationships, config)
            validation_results["relationship_quality"] = relationship_quality
            
            # Coverage assessment
            coverage_score = await self._assess_extraction_coverage(entities, relationships, config)
            validation_results["coverage_score"] = coverage_score
            
            # Consistency assessment
            consistency_score = await self._assess_extraction_consistency(entities, relationships, config)
            validation_results["consistency_score"] = consistency_score
            
            # Overall quality calculation
            overall_quality = (
                entity_quality * 0.4 +
                relationship_quality * 0.3 +
                coverage_score * 0.2 +
                consistency_score * 0.1
            )
            validation_results["overall_quality"] = overall_quality
            
            # Validation pass/fail determination
            validation_results["validation_passed"] = overall_quality >= config.minimum_quality_score
            
            # Generate warnings and recommendations
            if entity_quality < 0.7:
                validation_results["warnings"].append("Entity quality below threshold")
                validation_results["recommendations"].append("Consider adjusting entity confidence threshold")
            
            if relationship_quality < 0.6:
                validation_results["warnings"].append("Relationship quality below threshold")
                validation_results["recommendations"].append("Review relationship extraction patterns")
            
            if coverage_score < 0.5:
                validation_results["warnings"].append("Low extraction coverage")
                validation_results["recommendations"].append("Expand extraction strategies")
            
            return validation_results
            
        except Exception as e:
            return {
                "overall_quality": 0.0,
                "entity_quality": 0.0,
                "relationship_quality": 0.0,
                "coverage_score": 0.0,
                "consistency_score": 0.0,
                "validation_passed": False,
                "warnings": [f"Validation error: {str(e)}"],
                "recommendations": ["Review extraction configuration"]
            }

    async def store_knowledge_graph(
        self, ctx: RunContext[KnowledgeExtractionDeps], 
        entities: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]], 
        config: ExtractionConfiguration
    ) -> Dict[str, Any]:
        """Store validated knowledge graph in Azure Cosmos DB"""
        try:
            start_time = time.time()
            
            # Prepare knowledge graph structure
            knowledge_graph = {
                "domain": config.domain_name,
                "timestamp": time.time(),
                "nodes": self._format_entities_as_nodes(entities),
                "edges": self._format_relationships_as_edges(relationships),
                "metadata": {
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                    "extraction_config": config.domain_name
                }
            }
            
            # TODO: Implement actual Azure Cosmos DB storage
            # For now, return mock storage results to establish the pattern
            storage_results = {
                "storage_successful": True,
                "graph_id": f"graph_{config.domain_name}_{int(time.time())}",
                "nodes_stored": len(knowledge_graph["nodes"]),
                "edges_stored": len(knowledge_graph["edges"]),
                "storage_time": time.time() - start_time,
                "cosmos_db_endpoint": "mock://cosmos.db",
                "collection_name": f"knowledge_graphs_{config.domain_name}"
            }
            
            return storage_results
            
        except Exception as e:
            return {
                "storage_successful": False,
                "graph_id": None,
                "nodes_stored": 0,
                "edges_stored": 0,
                "storage_time": 0.0,
                "error": str(e)
            }

    # Helper methods for multi-strategy entity extraction
    async def _extract_entities_rule_based(
        self, content: str, config: ExtractionConfiguration
    ) -> List[Dict[str, Any]]:
        """Rule-based entity extraction for structured patterns"""
        entities = []
        
        # Extract entities from technical vocabulary
        for term in config.technical_vocabulary:
            if term.lower() in content.lower():
                entities.append({
                    "name": term,
                    "type": "technical_term",
                    "confidence": 0.85,
                    "extraction_method": "rule_based",
                    "context": self._extract_context(content, term)
                })
        
        return entities

    async def _extract_entities_azure_cognitive(
        self, content: str, config: ExtractionConfiguration
    ) -> List[Dict[str, Any]]:
        """Azure Cognitive Services entity extraction"""
        # TODO: Implement actual Azure Cognitive Services integration
        # For now, return mock entities to establish the pattern
        mock_entities = [
            {
                "name": "system",
                "type": "technology",
                "confidence": 0.92,
                "extraction_method": "azure_cognitive",
                "context": "technology system"
            },
            {
                "name": "process",
                "type": "procedure",
                "confidence": 0.88,
                "extraction_method": "azure_cognitive", 
                "context": "operational process"
            }
        ]
        
        return mock_entities

    async def _extract_entities_llm(
        self, content: str, config: ExtractionConfiguration
    ) -> List[Dict[str, Any]]:
        """LLM-powered domain-specific entity extraction"""
        # TODO: Implement actual LLM entity extraction using Azure OpenAI
        # For now, return mock entities to establish the pattern
        mock_entities = [
            {
                "name": "component",
                "type": "system_component", 
                "confidence": 0.79,
                "extraction_method": "llm",
                "context": "system component"
            }
        ]
        
        return mock_entities

    async def _fuse_entity_results(
        self, rule_entities: List, azure_entities: List, llm_entities: List, config: ExtractionConfiguration
    ) -> List[Dict[str, Any]]:
        """Fuse and deduplicate entity results from multiple strategies"""
        all_entities = rule_entities + azure_entities + llm_entities
        
        # Simple deduplication by name (could be enhanced with similarity matching)
        seen_names = set()
        fused_entities = []
        
        for entity in all_entities:
            name = entity.get("name", "").lower()
            if name not in seen_names and entity.get("confidence", 0.0) >= config.entity_confidence_threshold:
                seen_names.add(name)
                fused_entities.append(entity)
        
        return fused_entities

    # Helper methods for relationship extraction
    async def _extract_cooccurrence_relationships(
        self, content: str, entities: List[Dict[str, Any]], config: ExtractionConfiguration
    ) -> List[Dict[str, Any]]:
        """Extract relationships based on entity co-occurrence"""
        relationships = []
        
        # Find entities that appear in the same sentence/context
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if self._entities_cooccur(content, entity1["name"], entity2["name"]):
                    relationships.append({
                        "source": entity1["name"],
                        "target": entity2["name"],
                        "type": "co_occurs_with",
                        "confidence": 0.6,
                        "extraction_method": "cooccurrence"
                    })
        
        return relationships

    async def _extract_dependency_relationships(
        self, content: str, entities: List[Dict[str, Any]], config: ExtractionConfiguration
    ) -> List[Dict[str, Any]]:
        """Extract syntactic dependency relationships"""
        # TODO: Implement actual dependency parsing
        # For now, return mock relationships
        return [
            {
                "source": "system",
                "target": "component",
                "type": "contains",
                "confidence": 0.8,
                "extraction_method": "dependency_parsing"
            }
        ]

    async def _extract_semantic_relationships(
        self, content: str, entities: List[Dict[str, Any]], config: ExtractionConfiguration
    ) -> List[Dict[str, Any]]:
        """Extract semantic relationships using LLM"""
        # TODO: Implement actual LLM relationship extraction
        # For now, return mock relationships
        return [
            {
                "source": "process",
                "target": "procedure",
                "type": "implements",
                "confidence": 0.75,
                "extraction_method": "llm_semantic"
            }
        ]

    async def _validate_relationships(
        self, relationships: List[Dict[str, Any]], config: ExtractionConfiguration
    ) -> List[Dict[str, Any]]:
        """Filter and validate relationships"""
        validated = []
        
        for rel in relationships:
            if rel.get("confidence", 0.0) >= config.relationship_confidence_threshold:
                validated.append(rel)
        
        return validated

    # Helper methods for quality assessment
    async def _assess_entity_quality(
        self, entities: List[Dict[str, Any]], config: ExtractionConfiguration
    ) -> float:
        """Assess entity quality"""
        if not entities:
            return 0.0
        
        total_confidence = sum(e.get("confidence", 0.0) for e in entities)
        return total_confidence / len(entities)

    async def _assess_relationship_quality(
        self, relationships: List[Dict[str, Any]], config: ExtractionConfiguration
    ) -> float:
        """Assess relationship quality"""
        if not relationships:
            return 0.0
        
        total_confidence = sum(r.get("confidence", 0.0) for r in relationships)
        return total_confidence / len(relationships)

    async def _assess_extraction_coverage(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]], config: ExtractionConfiguration
    ) -> float:
        """Assess extraction coverage"""
        # Coverage based on expected entity types found
        found_types = set(e.get("type", "unknown") for e in entities)
        expected_types = set(config.expected_entity_types)
        
        if not expected_types:
            return 1.0
        
        coverage = len(found_types.intersection(expected_types)) / len(expected_types)
        return coverage

    async def _assess_extraction_consistency(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]], config: ExtractionConfiguration
    ) -> float:
        """Assess extraction consistency"""
        # Simple consistency check - relationships should reference existing entities
        if not relationships or not entities:
            return 1.0
        
        entity_names = set(e.get("name", "") for e in entities)
        valid_relationships = 0
        
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            if source in entity_names and target in entity_names:
                valid_relationships += 1
        
        return valid_relationships / len(relationships) if relationships else 1.0

    # Storage helper methods
    def _format_entities_as_nodes(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format entities as knowledge graph nodes"""
        nodes = []
        
        for entity in entities:
            nodes.append({
                "id": entity.get("name", "").replace(" ", "_"),
                "label": entity.get("name", ""),
                "type": entity.get("type", "concept"),
                "confidence": entity.get("confidence", 0.0),
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
                "label": rel.get("type", "related"),
                "confidence": rel.get("confidence", 0.0),
                "properties": {
                    "extraction_method": rel.get("extraction_method", "unknown")
                }
            })
        
        return edges

    # Utility methods
    def _extract_context(self, content: str, term: str, window_size: int = 50) -> str:
        """Extract context around a term"""
        try:
            index = content.lower().find(term.lower())
            if index != -1:
                start = max(0, index - window_size)
                end = min(len(content), index + len(term) + window_size)
                return content[start:end].strip()
        except:
            pass
        return ""

    def _entities_cooccur(self, content: str, entity1: str, entity2: str, window_size: int = 100) -> bool:
        """Check if two entities co-occur within a window"""
        try:
            content_lower = content.lower()
            pos1 = content_lower.find(entity1.lower())
            pos2 = content_lower.find(entity2.lower())
            
            if pos1 != -1 and pos2 != -1:
                return abs(pos1 - pos2) <= window_size
        except:
            pass
        return False

    def _calculate_entity_confidence(self, entities: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence for entities"""
        if not entities:
            return 0.0
        
        total_confidence = sum(e.get("confidence", 0.0) for e in entities)
        return total_confidence / len(entities)

    def _calculate_relationship_confidence(self, relationships: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence for relationships"""
        if not relationships:
            return 0.0
        
        total_confidence = sum(r.get("confidence", 0.0) for r in relationships)
        return total_confidence / len(relationships)

    async def _fallback_entity_extraction(
        self, content: str, config: ExtractionConfiguration, error: str
    ) -> Dict[str, Any]:
        """Fallback entity extraction when primary methods fail"""
        # Basic extraction from technical vocabulary
        basic_entities = []
        
        for term in config.technical_vocabulary[:5]:  # Limit to avoid overload
            if term.lower() in content.lower():
                basic_entities.append({
                    "name": term,
                    "type": "technical_term",
                    "confidence": 0.5,  # Low confidence for fallback
                    "extraction_method": "fallback"
                })
        
        return {
            "entities": basic_entities,
            "processing_time": 0.1,
            "strategies_used": ["fallback"],
            "total_entities": len(basic_entities),
            "confidence_score": 0.5,
            "error": error
        }


# Create the main toolset instance following target architecture
knowledge_extraction_toolset = KnowledgeExtractionToolset()