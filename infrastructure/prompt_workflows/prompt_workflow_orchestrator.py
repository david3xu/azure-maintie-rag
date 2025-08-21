#!/usr/bin/env python3
"""
Prompt Workflow Orchestrator - Universal RAG Integration
======================================================

Bridges dynamic prompt generation with Azure Prompt Flow execution.
Provides unified interface for both generated templates and static workflows.
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Removed UniversalPromptGenerator - Agent 1 provides template variables directly


class PromptWorkflowOrchestrator:
    """
    Orchestrates prompt workflows with dynamic template generation and Azure Prompt Flow integration.

    Supports both:
    1. Dynamic template generation based on domain intelligence
    2. Static Azure Prompt Flow workflow execution
    """

    def __init__(self, template_directory: str = None, domain_analyzer=None):
        """Initialize orchestrator with dependency injection"""
        base_directory = str(Path(__file__).parent)
        self.template_directory = template_directory or str(
            Path(base_directory) / "templates"
        )

        # Removed prompt_generator - Agent 1 provides template variables directly
        # Templates are used directly from template_directory

    @classmethod
    async def create_with_domain_intelligence(cls, template_directory: str = None):
        """Create orchestrator with direct template usage (no separate generator)"""
        # No prompt generator needed - Agent 1 provides template variables directly
        instance = cls(template_directory=template_directory)
        return instance

    async def prepare_workflow_templates(
        self,
        data_directory: str,
        content_config: Optional[Dict[str, Any]] = None,
        use_generated: bool = True,
    ) -> Dict[str, str]:
        """
        Prepare templates for workflow execution.

        Args:
            data_directory: Directory containing content for domain analysis
            content_config: Optional pre-analyzed content configuration
            use_generated: Whether to generate new templates or use existing ones

        Returns:
            Dictionary mapping template types to file paths
        """
        # SIMPLIFIED: Agent 1 provides template variables directly to universal templates
        # No template generation needed - use universal templates with Agent 1 variables

        print("🔧 Using universal templates with Agent 1 variables...")

        # Return unified template path only (no separate entity/relation templates)
        universal_templates = {
            "unified_knowledge_extraction": str(
                Path(self.template_directory) / "universal_knowledge_extraction.jinja2"
            ),
        }

        return universal_templates

    async def execute_extraction_workflow(
        self,
        texts: List[str],
        content_config: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.7,
        max_entities: int = 50,
        max_relationships: int = 40,
        force_chunking: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute complete extraction workflow with FORCED chunking for large texts.

        Args:
            texts: List of text content to process
            content_config: Optional content-specific configuration (includes chunking params)
            confidence_threshold: Minimum confidence for extractions
            max_entities: Maximum entities to extract (total across all chunks)
            max_relationships: Maximum relationships to extract (total across all chunks)
            force_chunking: FORCE chunking when text exceeds chunk_size (recommended: True)

        Returns:
            Complete extraction results with entities, relationships, chunking metadata
        """
        print(f"🚀 Executing extraction workflow on {len(texts)} documents...")

        # Step 1: Template preparation with chunking configuration
        template_config = content_config or {}

        # FORCE chunking parameters (always set, regardless of force_chunking flag)
        # Scale chunk size with content size (universal approach)
        total_content_size = sum(len(text) for text in texts)
        adaptive_chunk_size = min(max(1200, total_content_size // 20), 3000)  # Scale from 1.2KB to 3KB
        
        template_config.setdefault(
            "chunk_size", adaptive_chunk_size
        )  # Adaptive sizing based on content volume
        template_config.setdefault(
            "chunk_overlap_ratio", 0.3
        )  # Higher overlap for better context
        print(
            f"🔧 FORCED chunking: {template_config['chunk_size']} chars, {template_config['chunk_overlap_ratio']*100:.0f}% overlap"
        )

        # Step 2: Unified Knowledge Extraction with Chunking (preserves contextual relationships)
        print("🧠 Extracting entities and relationships with intelligent chunking...")
        unified_results = await self._extract_unified_knowledge(
            texts=texts,
            config=template_config,
            confidence_threshold=confidence_threshold,
            max_entities=max_entities,
            max_relationships=max_relationships,
            force_chunking=force_chunking,
        )

        # Extract results from unified extraction
        entity_results = {"entities": unified_results.get("entities", [])}
        relationship_results = {
            "relationships": unified_results.get("relationships", [])
        }

        # Step 3: Knowledge graph construction
        print("🕸️  Building knowledge graph...")
        knowledge_graph = await self._build_knowledge_graph(
            entities=entity_results.get("entities", []),
            relationships=relationship_results.get("relationships", []),
            confidence_threshold=confidence_threshold,
        )

        # Step 4: Quality assessment
        print("📊 Assessing extraction quality...")
        quality_metrics = await self._assess_quality(
            entities=knowledge_graph.get("entities", []),
            relationships=knowledge_graph.get("relationships", []),
            original_texts=texts,
        )

        # Compile results with chunking metadata
        workflow_results = {
            "workflow_metadata": {
                "processed_documents": len(texts),
                "total_entities": len(knowledge_graph.get("entities", [])),
                "total_relationships": len(knowledge_graph.get("relationships", [])),
                "overall_confidence": quality_metrics.get("overall_confidence", 0.0),
                "extraction_strategy": unified_results.get(
                    "extraction_method", "unified_contextual_extraction"
                ),
                "chunking_forced": force_chunking,
                "deduplication_applied": unified_results.get(
                    "deduplication_applied", False
                ),
            },
            "entities": knowledge_graph.get("entities", []),
            "relationships": knowledge_graph.get("relationships", []),
            "knowledge_graph": knowledge_graph,
            "quality_metrics": quality_metrics,
            "content_config": template_config,
            "chunking_metadata": unified_results.get("chunk_metadata", []),
        }

        print(f"✅ Workflow completed successfully!")
        print(
            f"   📊 Extracted: {len(workflow_results['entities'])} entities, {len(workflow_results['relationships'])} relationships"
        )

        # Log chunking details (always shown since chunking is forced)
        if workflow_results.get("chunking_metadata"):
            total_chunks = len(workflow_results["chunking_metadata"])
            chunked_texts = sum(
                1
                for m in workflow_results["chunking_metadata"]
                if m.get("chunking_used", True)
            )
            print(
                f"   🔧 FORCED Chunking: {total_chunks} chunks from {chunked_texts} documents"
            )

        if unified_results.get("deduplication_applied"):
            print(f"   🔗 Deduplication applied across chunks")
        print(
            f"   🎯 Quality score: {quality_metrics.get('overall_confidence', 0.0):.2f}"
        )

        return workflow_results

    async def _extract_unified_knowledge(
        self,
        texts: List[str],
        config: Dict[str, Any],
        confidence_threshold: float,
        max_entities: int,
        max_relationships: int = 20,
        force_chunking: bool = True,
    ) -> Dict[str, Any]:
        """UNIFIED knowledge extraction with FORCED chunking for large texts"""
        from jinja2 import Environment, FileSystemLoader

        from agents.core.semantic_chunker import SemanticChunker
        from agents.core.universal_deps import get_universal_deps

        # Get chunking configuration from config
        chunk_size = config.get("chunk_size", 1000)
        chunk_overlap_ratio = config.get("chunk_overlap_ratio", 0.2)

        try:
            all_entities = []
            all_relationships = []
            chunk_metadata = []

            # Process each text (usually one file per call)
            for text_idx, text in enumerate(texts):
                text_entities = []
                text_relationships = []

                # FORCE chunking for ANY text exceeding chunk_size (no disable option)
                if len(text) > chunk_size:
                    logger.info(
                        f"📄 Text {text_idx+1}: {len(text)} chars - chunking required"
                    )

                    # Initialize semantic chunker
                    chunker = SemanticChunker()
                    chunked_content = chunker.chunk_text(
                        text=text,
                        chunk_size=chunk_size,
                        overlap_ratio=chunk_overlap_ratio,
                        preserve_sentences=True,
                        preserve_code_blocks=True,
                    )

                    logger.info(f"   📊 Created {chunked_content.total_chunks} chunks")
                    logger.info(
                        f"   📏 Average chunk size: {chunked_content.average_chunk_size:.0f} chars"
                    )

                    # Process each chunk
                    for chunk_idx, chunk in enumerate(chunked_content.chunks):
                        chunk_entities, chunk_relationships = (
                            await self._process_single_chunk(
                                chunk=chunk,
                                chunk_id=f"{text_idx}_{chunk_idx}",
                                config=config,
                                confidence_threshold=confidence_threshold,
                                max_entities_per_chunk=max_entities
                                // max(1, chunked_content.total_chunks),
                                max_relationships_per_chunk=max_relationships
                                // max(1, chunked_content.total_chunks),
                            )
                        )

                        text_entities.extend(chunk_entities)
                        text_relationships.extend(chunk_relationships)

                        chunk_metadata.append(
                            {
                                "text_index": text_idx,
                                "chunk_index": chunk_idx,
                                "chunk_size": len(chunk),
                                "entities_found": len(chunk_entities),
                                "relationships_found": len(chunk_relationships),
                                "metadata": chunked_content.metadata[
                                    chunk_idx
                                ].__dict__,
                            }
                        )
                else:
                    logger.info(
                        f"📄 Text {text_idx+1}: {len(text)} chars - processing as single unit"
                    )

                    # Process as single unit (existing logic)
                    text_entities, text_relationships = (
                        await self._process_single_chunk(
                            chunk=text,
                            chunk_id=f"{text_idx}_0",
                            config=config,
                            confidence_threshold=confidence_threshold,
                            max_entities_per_chunk=max_entities,
                            max_relationships_per_chunk=max_relationships,
                        )
                    )

                    chunk_metadata.append(
                        {
                            "text_index": text_idx,
                            "chunk_index": 0,
                            "chunk_size": len(text),
                            "entities_found": len(text_entities),
                            "relationships_found": len(text_relationships),
                            "chunking_used": False,
                        }
                    )

                all_entities.extend(text_entities)
                all_relationships.extend(text_relationships)

            # Deduplicate entities and relationships across chunks
            deduplicated_entities, deduplicated_relationships = (
                self._deduplicate_extractions(
                    all_entities, all_relationships, confidence_threshold
                )
            )

            logger.info(
                f"🔗 Deduplication: {len(all_entities)} → {len(deduplicated_entities)} entities"
            )
            logger.info(
                f"🔗 Deduplication: {len(all_relationships)} → {len(deduplicated_relationships)} relationships"
            )

            return {
                "entities": deduplicated_entities[:max_entities],  # Final limit
                "relationships": deduplicated_relationships[
                    :max_relationships
                ],  # Final limit
                "extraction_method": (
                    "forced_chunked_unified"
                    if any(m.get("chunking_used", True) for m in chunk_metadata)
                    else "forced_single"
                ),
                "chunk_metadata": chunk_metadata,
                "deduplication_applied": len(all_entities) != len(deduplicated_entities)
                or len(all_relationships) != len(deduplicated_relationships),
            }

        except Exception as e:
            logger.error(f"❌ FORCED chunked unified knowledge extraction failed: {e}")
            # FAIL FAST - No fake success patterns, no fallbacks
            raise RuntimeError(
                f"Knowledge extraction with forced chunking failed: {e}. No fallback available - fix the underlying issue."
            ) from e

    async def _process_single_chunk(
        self,
        chunk: str,
        chunk_id: str,
        config: Dict[str, Any],
        confidence_threshold: float,
        max_entities_per_chunk: int,
        max_relationships_per_chunk: int,
    ) -> tuple:
        """Process a single chunk of text for entity and relationship extraction"""
        from jinja2 import Environment, FileSystemLoader

        from agents.core.universal_deps import get_universal_deps

        try:
            # Load unified template (preserves entity-relationship context)
            env = Environment(loader=FileSystemLoader(self.template_directory))
            template = env.get_template("universal_knowledge_extraction.jinja2")

            # Render unified template with chunk-specific configuration
            template_config = {
                "texts": [chunk],  # Single chunk as text list
                "entity_confidence_threshold": confidence_threshold,
                "relationship_confidence_threshold": confidence_threshold,
                "max_entities": max_entities_per_chunk,
                "max_relationships": max_relationships_per_chunk,
                "chunk_id": chunk_id,
                **config,
            }

            prompt = template.render(**template_config)

            # Real LLM integration for unified knowledge extraction
            deps = await get_universal_deps()
            if deps.is_service_available("openai"):
                openai_client = deps.openai_client

                # System message for unified extraction
                system_msg = f"You are an expert knowledge analyst. Extract entities AND their contextual relationships together from chunk {chunk_id}. Return JSON with both entities and relationships arrays."
                full_prompt = f"{system_msg}\n\n{prompt}"

                response = await openai_client.get_completion(
                    full_prompt, max_tokens=3000, temperature=0.1
                )

                # Parse unified LLM response
                try:
                    import json

                    response_text = (
                        response if isinstance(response, str) else str(response)
                    )

                    # Extract JSON from response if wrapped in text
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end]

                    knowledge = json.loads(response_text)
                    entities = knowledge.get("entities", [])[:max_entities_per_chunk]
                    relationships = knowledge.get("relationships", [])[
                        :max_relationships_per_chunk
                    ]

                    # Add chunk metadata to each entity and relationship
                    for entity in entities:
                        entity["chunk_id"] = chunk_id
                        entity["chunk_source"] = True

                    for relationship in relationships:
                        relationship["chunk_id"] = chunk_id
                        relationship["chunk_source"] = True

                    return entities, relationships

                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parsing failed for chunk {chunk_id}: {e}")
                    # FAIL FAST - JSON parsing must work, no fake success
                    raise RuntimeError(
                        f"Azure OpenAI returned malformed JSON for chunk {chunk_id}: {e}. No fallback available - check LLM response quality."
                    ) from e

            # FAIL FAST - OpenAI service is required for extraction
            raise RuntimeError(
                f"Azure OpenAI service not available for unified extraction of chunk {chunk_id}. No fallback available - check service configuration."
            )

        except Exception as e:
            logger.error(
                f"❌ Unified knowledge extraction failed for chunk {chunk_id}: {e}"
            )
            # FAIL FAST - No fake success, no empty results
            raise RuntimeError(
                f"Chunk {chunk_id} extraction failed: {e}. No fallback available - fix the underlying issue."
            ) from e

    # REMOVED: Legacy _extract_entities method - using unified extraction only

    # REMOVED: Legacy _extract_relationships method - using unified extraction only

    def _deduplicate_extractions(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        confidence_threshold: float,
    ) -> tuple:
        """
        Deduplicate entities and relationships across chunks using text similarity and confidence.

        Deduplication strategy:
        1. Entities: Group by normalized text, keep highest confidence
        2. Relationships: Group by normalized (source, relation, target), keep highest confidence
        3. Handle variations in entity naming across chunks
        """
        import re
        from collections import defaultdict

        # Helper function to normalize text for comparison
        def normalize_text(text: str) -> str:
            if not text:
                return ""
            # Convert to lowercase, remove extra whitespace, normalize punctuation
            normalized = re.sub(r"\s+", " ", text.lower().strip())
            normalized = re.sub(r"[^\w\s]", "", normalized)
            return normalized

        # Deduplicate entities
        entity_groups = defaultdict(list)
        for entity in entities:
            entity_text = entity.get("text", "")
            normalized_text = normalize_text(entity_text)
            if normalized_text:  # Only process non-empty entities
                entity_groups[normalized_text].append(entity)

        deduplicated_entities = []
        entity_id_mapping = {}  # Map old entity text to new deduplicated entity

        for normalized_text, group in entity_groups.items():
            # Choose entity with highest confidence, or first if no confidence
            best_entity = max(group, key=lambda e: e.get("confidence", 0.0))

            # If multiple entities in group, merge properties
            if len(group) > 1:
                # Combine chunk sources
                chunk_sources = set()
                for entity in group:
                    if entity.get("chunk_id"):
                        chunk_sources.add(entity["chunk_id"])

                best_entity["chunk_sources"] = list(chunk_sources)
                best_entity["deduplicated_from"] = len(group)

                # Map all original entity texts to the best one
                for entity in group:
                    original_text = entity.get("text", "")
                    entity_id_mapping[original_text] = best_entity.get("text", "")

            deduplicated_entities.append(best_entity)

        # Deduplicate relationships using entity mapping
        relationship_groups = defaultdict(list)

        for relationship in relationships:
            source = relationship.get("source_entity", relationship.get("source", ""))
            target = relationship.get("target_entity", relationship.get("target", ""))
            relation_type = relationship.get(
                "relationship_type", relationship.get("relation", "")
            )

            # Map entity names to deduplicated versions
            mapped_source = entity_id_mapping.get(source, source)
            mapped_target = entity_id_mapping.get(target, target)

            # Create normalized relationship key
            normalized_key = (
                normalize_text(mapped_source),
                normalize_text(relation_type),
                normalize_text(mapped_target),
            )

            if all(normalized_key):  # Only process complete relationships
                # Update relationship with mapped entity names
                relationship["source_entity"] = mapped_source
                relationship["target_entity"] = mapped_target
                relationship_groups[normalized_key].append(relationship)

        deduplicated_relationships = []
        for normalized_key, group in relationship_groups.items():
            # Choose relationship with highest confidence
            best_relationship = max(group, key=lambda r: r.get("confidence", 0.0))

            # If multiple relationships in group, merge chunk sources
            if len(group) > 1:
                chunk_sources = set()
                for relationship in group:
                    if relationship.get("chunk_id"):
                        chunk_sources.add(relationship["chunk_id"])

                best_relationship["chunk_sources"] = list(chunk_sources)
                best_relationship["deduplicated_from"] = len(group)

            deduplicated_relationships.append(best_relationship)

        # Filter by confidence threshold after deduplication
        final_entities = [
            e
            for e in deduplicated_entities
            if e.get("confidence", 0.0) >= confidence_threshold
        ]
        final_relationships = [
            r
            for r in deduplicated_relationships
            if r.get("confidence", 0.0) >= confidence_threshold
        ]

        return final_entities, final_relationships

    async def _build_knowledge_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        confidence_threshold: float,
    ) -> Dict[str, Any]:
        """Build knowledge graph from extracted entities and relationships"""
        # Filter by confidence
        filtered_entities = [
            e for e in entities if e.get("confidence", 0) >= confidence_threshold
        ]
        filtered_relationships = [
            r for r in relationships if r.get("confidence", 0) >= confidence_threshold
        ]

        return {
            "entities": filtered_entities,
            "relationships": filtered_relationships,
            "graph_metadata": {
                "total_nodes": len(filtered_entities),
                "total_edges": len(filtered_relationships),
                "confidence_threshold": confidence_threshold,
            },
        }

    async def _assess_quality(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        original_texts: List[str],
    ) -> Dict[str, Any]:
        """Assess extraction quality"""
        # TODO: DELETED WORD-SPLITTING QUALITY ASSESSMENT - IMPLEMENT LLM-BASED
        total_chars = sum(len(text) for text in original_texts)
        entity_density = len(entities) / max(
            total_chars / 50, 1
        )  # Approximate char-to-word ratio
        relationship_density = len(relationships) / max(len(entities), 1)

        # Calculate overall confidence
        entity_confidences = [e.get("confidence", 0.0) for e in entities]
        relationship_confidences = [r.get("confidence", 0.0) for r in relationships]

        all_confidences = entity_confidences + relationship_confidences
        overall_confidence = (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        )

        return {
            "overall_confidence": overall_confidence,
            "entity_density": entity_density,
            "relationship_density": relationship_density,
            "quality_score": overall_confidence,  # Use continuous score based on measured confidence
            "total_extractions": len(entities) + len(relationships),
        }

    async def cleanup_generated_templates(self, max_age_hours: int = 24):
        """No cleanup needed - using universal templates only"""
        print(
            "ℹ️  No generated templates to clean - using universal templates with Agent 1 variables"
        )

    def list_available_templates(self) -> Dict[str, List[str]]:
        """List universal templates - unified approach only"""
        templates = {
            "universal": [
                str(
                    Path(self.template_directory)
                    / "universal_knowledge_extraction.jinja2"
                ),  # Unified extraction only
            ]
        }
        return templates


async def main():
    """Demo: Orchestrate prompt workflow with universal templates"""
    # Create orchestrator with proper domain intelligence injection
    orchestrator = await PromptWorkflowOrchestrator.create_with_domain_intelligence()

    # Sample content
    sample_texts = [
        "Machine learning algorithms require training data to learn patterns and make predictions.",
        "Neural networks consist of interconnected nodes that process information through weighted connections.",
        "Deep learning uses multiple layers to extract hierarchical features from complex data.",
    ]

    print("🌍 Universal Prompt Workflow Orchestrator Demo")
    print("==============================================")

    # Execute workflow
    results = await orchestrator.execute_extraction_workflow(
        texts=sample_texts, confidence_threshold=0.7
    )

    print(f"\n📊 Workflow Results:")
    print(f"   Entities: {len(results['entities'])}")
    print(f"   Relationships: {len(results['relationships'])}")
    print(f"   Quality: {results['quality_metrics']['overall_confidence']:.2f}")

    return results
