"""
Universal Knowledge Extractor
Integrates all universal components to extract knowledge from pure text files
Works with any domain - no annotations, no schema files, no hardcoded types
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import networkx as nx
from collections import defaultdict

from ..models.universal_rag_models import (
    UniversalEntity, UniversalRelation, UniversalDocument
)
from .text_processor import AzureOpenAITextProcessor
from ..azure_ml.classification_service import (
    UniversalEntityClassifier, UniversalRelationClassifier
)
from .extraction_client import OptimizedLLMExtractor
from .azure_text_analytics_service import AzureTextAnalyticsService
from .azure_ml_quality_service import AzureMLQualityAssessment
from .azure_monitoring_service import AzureKnowledgeMonitor
from .azure_rate_limiter import AzureOpenAIRateLimiter
from ...config.settings import settings

logger = logging.getLogger(__name__)


class AzureOpenAIKnowledgeExtractor:
    """
    Universal Knowledge Extractor for any domain

    Key Features:
    - Works with pure text files (no annotations required)
    - Dynamic entity/relation type discovery
    - No schema files or configuration needed
    - Domain-agnostic knowledge extraction
    - Automatic quality assessment
    - Unified processing pipeline
    """

    def __init__(self, domain_name: str = "general"):
        """Initialize universal knowledge extractor with enterprise services"""
        self.domain_name = domain_name

        # Initialize universal components
        self.text_processor = AzureOpenAITextProcessor(domain_name)
        self.entity_classifier = UniversalEntityClassifier()
        self.relation_classifier = UniversalRelationClassifier()
        self.llm_extractor = OptimizedLLMExtractor(domain_name)

        # Initialize enterprise services
        self.text_analytics = AzureTextAnalyticsService()
        self.quality_assessor = AzureMLQualityAssessment(domain_name)
        self.monitor = AzureKnowledgeMonitor()
        self.rate_limiter = AzureOpenAIRateLimiter()

        # Enterprise extraction configuration
        self.extraction_config = self._load_extraction_config()

        # Knowledge containers
        self.entities: Dict[str, UniversalEntity] = {}
        self.relations: List[UniversalRelation] = []
        self.documents: Dict[str, UniversalDocument] = {}
        self.knowledge_graph: Optional[nx.Graph] = None

        # Dynamic type discovery
        self.discovered_entity_types: Set[str] = set()
        self.discovered_relation_types: Set[str] = set()
        self.type_frequencies: Dict[str, int] = defaultdict(int)

        # Extraction statistics
        self.extraction_stats = {
            "total_texts_processed": 0,
            "total_entities_extracted": 0,
            "total_relations_extracted": 0,
            "unique_entity_types": 0,
            "unique_relation_types": 0,
            "processing_time": 0.0
        }

        logger.info(f"AzureOpenAIKnowledgeExtractor initialized for domain: {domain_name}")

    def _load_extraction_config(self) -> Dict[str, Any]:
        """Load domain-specific extraction configuration"""
        return {
            "quality_tier": settings.extraction_quality_tier,
            "confidence_threshold": settings.extraction_confidence_threshold,
            "max_entities_per_document": settings.max_entities_per_document,
            "batch_size": settings.extraction_batch_size,
            "enable_preprocessing": settings.enable_text_analytics_preprocessing
        }

    async def extract_knowledge_from_texts(
        self,
        texts: List[str],
        text_sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract knowledge from raw text files

        Args:
            texts: List of text content to process
            text_sources: Optional list of source filenames/identifiers

        Returns:
            Dictionary with extraction results and statistics
        """
        logger.info(f"Starting universal knowledge extraction from {len(texts)} texts...")
        start_time = datetime.now()

        if text_sources and len(text_sources) != len(texts):
            text_sources = [f"text_{i}" for i in range(len(texts))]
        elif not text_sources:
            text_sources = [f"text_{i}" for i in range(len(texts))]

        try:
            # Start performance tracking
            self.monitor.start_performance_tracking()

            # Step 1: Enterprise text preprocessing
            if self.extraction_config["enable_preprocessing"]:
                logger.info("Step 1: Enterprise text preprocessing with Azure Text Analytics...")
                await self._preprocess_texts_with_analytics(texts)
            else:
                logger.info("Step 1: Converting texts to universal documents...")
                await self._create_universal_documents(texts, text_sources)

            # Step 2: Extract entities and relations using LLM with rate limiting
            logger.info("Step 2: Extracting entities and relations with LLM (rate limited)...")
            await self._extract_entities_and_relations_with_rate_limiting()

            # Step 3: Classify and normalize extracted knowledge
            logger.info("Step 3: Classifying and normalizing knowledge...")
            await self._classify_and_normalize()

            # Step 4: Build knowledge graph
            logger.info("Step 4: Building knowledge graph...")
            await self._build_universal_knowledge_graph()

            # Step 5: Enterprise quality assessment and monitoring
            logger.info("Step 5: Enterprise quality assessment and monitoring...")
            validation_results = await self._assess_extraction_quality_enterprise()

            # Track performance metrics
            performance_metrics = self.monitor.end_performance_tracking()
            await self.monitor.track_azure_openai_usage(
                performance_metrics["tokens_used"],
                performance_metrics["api_calls"],
                performance_metrics["duration_ms"]
            )

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_extraction_stats(processing_time)

            # Prepare results
            results = {
                "success": True,
                "domain": self.domain_name,
                "extraction_stats": self.extraction_stats,
                "discovered_types": {
                    "entity_types": list(self.discovered_entity_types),
                    "relation_types": list(self.discovered_relation_types)
                },
                "knowledge_summary": {
                    "total_entities": len(self.entities),
                    "total_relations": len(self.relations),
                    "total_documents": len(self.documents),
                    "graph_nodes": self.knowledge_graph.number_of_nodes() if self.knowledge_graph else 0,
                    "graph_edges": self.knowledge_graph.number_of_edges() if self.knowledge_graph else 0
                },
                "validation_results": validation_results,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Universal knowledge extraction completed successfully")
            logger.info(f"Extracted: {len(self.entities)} entities, {len(self.relations)} relations")
            logger.info(f"Discovered: {len(self.discovered_entity_types)} entity types, {len(self.discovered_relation_types)} relation types")

            return results

        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "domain": self.domain_name,
                "timestamp": datetime.now().isoformat()
            }

    async def _create_universal_documents(self, texts: List[str], sources: List[str]) -> None:
        """Create universal documents from raw texts"""
        for i, (text, source) in enumerate(zip(texts, sources)):
            doc_id = f"{self.domain_name}_{i}"

            # Clean and preprocess text
            cleaned_text = self._clean_text(text)

            # Create universal document
            document = UniversalDocument(
                doc_id=doc_id,
                text=cleaned_text,
                title=f"Document {i+1}: {source[:50]}..." if len(source) > 50 else source,
                metadata={
                    "source": source,
                    "original_length": len(text),
                    "cleaned_length": len(cleaned_text),
                    "domain": self.domain_name,
                    "processed_at": datetime.now().isoformat()
                }
            )

            self.documents[doc_id] = document

    async def _extract_entities_and_relations(self) -> None:
        """Extract entities and relations using LLM extractor"""
        all_texts = [doc.text for doc in self.documents.values()]

        # Use optimized LLM extractor for batch processing
        extraction_results = self.llm_extractor.extract_entities_and_relations(all_texts)

        # Process extracted entities
        for entity_data in extraction_results.get("entities", []):
            # Handle both string and dictionary formats
            if isinstance(entity_data, str):
                # OptimizedLLMExtractor returns strings
                entity = UniversalEntity(
                    entity_id=f"entity_{len(self.entities)}",
                    text=entity_data,
                    entity_type=entity_data,  # Use the string as both text and type
                    confidence=0.8,  # Default confidence for LLM-discovered entities
                    context="",
                    metadata={
                        "extraction_method": "llm",
                        "domain": self.domain_name,
                        "extracted_at": datetime.now().isoformat()
                    }
                )
            else:
                # Handle dictionary format (for backward compatibility)
                entity = UniversalEntity(
                    entity_id=f"entity_{len(self.entities)}",
                    text=entity_data.get("text", ""),
                    entity_type=entity_data.get("type", "unknown"),
                    confidence=entity_data.get("confidence", 0.5),
                    context=entity_data.get("context", ""),
                    metadata={
                        "extraction_method": "llm",
                        "domain": self.domain_name,
                        "extracted_at": datetime.now().isoformat()
                    }
                )
            self.entities[entity.entity_id] = entity
            self.discovered_entity_types.add(entity.entity_type)

        # Process extracted relations
        for relation_data in extraction_results.get("relations", []):
            # Handle both string and dictionary formats
            if isinstance(relation_data, str):
                # OptimizedLLMExtractor returns strings
                relation = UniversalRelation(
                    relation_id=f"relation_{len(self.relations)}",
                    source_entity_id="",  # Will be populated later during graph construction
                    target_entity_id="",  # Will be populated later during graph construction
                    relation_type=relation_data,  # Use the string as relation type
                    confidence=0.8,  # Default confidence for LLM-discovered relations
                    context="",
                    metadata={
                        "extraction_method": "llm",
                        "domain": self.domain_name,
                        "extracted_at": datetime.now().isoformat()
                    }
                )
            else:
                # Handle dictionary format (for backward compatibility)
                relation = UniversalRelation(
                    relation_id=f"relation_{len(self.relations)}",
                    source_entity_id=relation_data.get("source_entity", ""),
                    target_entity_id=relation_data.get("target_entity", ""),
                    relation_type=relation_data.get("type", "unknown"),
                    confidence=relation_data.get("confidence", 0.5),
                    context=relation_data.get("context", ""),
                    metadata={
                        "extraction_method": "llm",
                        "domain": self.domain_name,
                        "extracted_at": datetime.now().isoformat()
                    }
                )
            self.relations.append(relation)
            self.discovered_relation_types.add(relation.relation_type)

    async def _classify_and_normalize(self) -> None:
        """Classify and normalize extracted entities and relations"""
        # Classify entities
        for entity_id, entity in self.entities.items():
            classification_result = self.entity_classifier.classify_entity(
                entity.text, entity.context or ""
            )

            # Update entity with classification results
            if classification_result.confidence > entity.confidence:
                entity.entity_type = classification_result.entity_type
                entity.confidence = classification_result.confidence
                entity.metadata.update({
                    "classification_method": "universal_classifier",
                    "classification_confidence": classification_result.confidence,
                    "classification_category": classification_result.category
                })
                self.discovered_entity_types.add(entity.entity_type)

        # Classify relations
        for relation in self.relations:
            source_entity = self.entities.get(relation.source_entity_id)
            target_entity = self.entities.get(relation.target_entity_id)

            if source_entity and target_entity:
                classification_result = self.relation_classifier.classify_relation(
                    relation.context or "",
                    source_entity.text,
                    target_entity.text
                )

                # Update relation with classification results
                if classification_result.confidence > relation.confidence:
                    relation.relation_type = classification_result.entity_type  # classifier returns entity_type for relations too
                    relation.confidence = classification_result.confidence
                    relation.metadata.update({
                        "classification_method": "universal_classifier",
                        "classification_confidence": classification_result.confidence,
                        "classification_category": classification_result.category
                    })
                    self.discovered_relation_types.add(relation.relation_type)

    async def _build_universal_knowledge_graph(self) -> None:
        """Build knowledge graph from entities and relations"""
        self.knowledge_graph = nx.Graph()

        # Add entity nodes
        for entity_id, entity in self.entities.items():
            self.knowledge_graph.add_node(
                entity_id,
                text=entity.text,
                entity_type=entity.entity_type,
                confidence=entity.confidence,
                metadata=entity.metadata
            )

        # Add relation edges
        for relation in self.relations:
            if (relation.source_entity_id in self.knowledge_graph.nodes and
                relation.target_entity_id in self.knowledge_graph.nodes):

                self.knowledge_graph.add_edge(
                    relation.source_entity_id,
                    relation.target_entity_id,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence,
                    context=relation.context,
                    metadata=relation.metadata
                )

    async def _validate_extracted_knowledge(self) -> Dict[str, Any]:
        """Validate and analyze extracted knowledge"""
        validation_results = {
            "entity_validation": {
                "total_entities": len(self.entities),
                "high_confidence_entities": len([e for e in self.entities.values() if e.confidence > 0.8]),
                "medium_confidence_entities": len([e for e in self.entities.values() if 0.5 <= e.confidence <= 0.8]),
                "low_confidence_entities": len([e for e in self.entities.values() if e.confidence < 0.5]),
                "entity_type_distribution": self._get_type_distribution("entity")
            },
            "relation_validation": {
                "total_relations": len(self.relations),
                "high_confidence_relations": len([r for r in self.relations if r.confidence > 0.8]),
                "medium_confidence_relations": len([r for r in self.relations if 0.5 <= r.confidence <= 0.8]),
                "low_confidence_relations": len([r for r in self.relations if r.confidence < 0.5]),
                "relation_type_distribution": self._get_type_distribution("relation")
            },
            "graph_validation": {
                "nodes": self.knowledge_graph.number_of_nodes() if self.knowledge_graph else 0,
                "edges": self.knowledge_graph.number_of_edges() if self.knowledge_graph else 0,
                "connected_components": nx.number_connected_components(self.knowledge_graph) if self.knowledge_graph else 0,
                "density": nx.density(self.knowledge_graph) if self.knowledge_graph else 0.0
            },
            "quality_assessment": self._assess_extraction_quality()
        }

        return validation_results

    def _get_type_distribution(self, item_type: str) -> Dict[str, int]:
        """Get distribution of entity or relation types"""
        if item_type == "entity":
            types = [entity.entity_type for entity in self.entities.values()]
        else:
            types = [relation.relation_type for relation in self.relations]

        distribution = defaultdict(int)
        for type_name in types:
            distribution[type_name] += 1

        return dict(distribution)

    def _assess_extraction_quality(self) -> Dict[str, Any]:
        """Assess overall quality of extraction"""
        total_items = len(self.entities) + len(self.relations)
        if total_items == 0:
            return {"overall_quality": "poor", "confidence_score": 0.0, "issues": ["No entities or relations extracted"]}

        # Calculate average confidence
        entity_confidences = [e.confidence for e in self.entities.values()]
        relation_confidences = [r.confidence for r in self.relations]
        all_confidences = entity_confidences + relation_confidences

        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        # Assess quality
        issues = []
        if len(self.entities) < 3:
            issues.append("Very few entities extracted")
        if len(self.relations) < 2:
            issues.append("Very few relations extracted")
        if avg_confidence < 0.5:
            issues.append("Low average confidence scores")
        if len(self.discovered_entity_types) < 2:
            issues.append("Limited entity type diversity")

        if avg_confidence > 0.8 and len(issues) == 0:
            quality = "excellent"
        elif avg_confidence > 0.6 and len(issues) <= 1:
            quality = "good"
        elif avg_confidence > 0.4 and len(issues) <= 2:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "overall_quality": quality,
            "confidence_score": avg_confidence,
            "issues": issues,
            "recommendations": self._get_quality_recommendations(quality, issues)
        }

    def _get_quality_recommendations(self, quality: str, issues: List[str]) -> List[str]:
        """Get recommendations for improving extraction quality"""
        recommendations = []

        if "Very few entities extracted" in issues:
            recommendations.append("Consider adding more descriptive text or using different extraction parameters")
        if "Very few relations extracted" in issues:
            recommendations.append("Try providing text with more explicit relationships between concepts")
        if "Low average confidence scores" in issues:
            recommendations.append("Review and clean input text for better clarity")
        if "Limited entity type diversity" in issues:
            recommendations.append("Ensure input text covers diverse concepts and topics")

        if quality == "excellent":
            recommendations.append("Excellent extraction quality - ready for production use")
        elif quality == "good":
            recommendations.append("Good extraction quality - minor improvements possible")

        return recommendations

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        import re

        # Basic text cleaning
        cleaned = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        cleaned = re.sub(r'[^\w\s\-.,;:!?()]', '', cleaned)  # Remove special chars except basic punctuation
        cleaned = cleaned.strip()

        return cleaned

    def _update_extraction_stats(self, processing_time: float) -> None:
        """Update extraction statistics"""
        self.extraction_stats.update({
            "total_texts_processed": len(self.documents),
            "total_entities_extracted": len(self.entities),
            "total_relations_extracted": len(self.relations),
            "unique_entity_types": len(self.discovered_entity_types),
            "unique_relation_types": len(self.discovered_relation_types),
            "processing_time": processing_time
        })

    async def _preprocess_texts_with_analytics(self, texts: List[str]) -> None:
        """Enterprise text preprocessing with Azure Text Analytics"""
        try:
            # Preprocess texts with Azure Text Analytics
            preprocessing_results = await self.text_analytics.preprocess_for_extraction(texts)

            # Create enhanced documents with preprocessing metadata
            for i, (text, preprocess_data) in enumerate(zip(texts, preprocessing_results.get("enhanced_texts", texts))):
                doc_id = f"{self.domain_name}_{i}"

                # Validate text quality
                quality_validation = await self.text_analytics.validate_text_quality(text)

                # Create enhanced universal document
                document = UniversalDocument(
                    doc_id=doc_id,
                    text=text,
                    title=f"Enhanced Document {i+1}",
                    metadata={
                        "preprocessing_confidence": preprocessing_results.get("processing_confidence", 0.5),
                        "language_metadata": preprocessing_results.get("language_metadata", [{}])[i] if i < len(preprocessing_results.get("language_metadata", [])) else {},
                        "quality_validation": quality_validation,
                        "domain": self.domain_name,
                        "processed_at": datetime.now().isoformat()
                    }
                )

                self.documents[doc_id] = document

        except Exception as e:
            logger.error(f"Enterprise preprocessing failed: {e}")
            # Fallback to basic document creation
            await self._create_universal_documents(texts, [f"text_{i}" for i in range(len(texts))])

    async def _extract_entities_and_relations_with_rate_limiting(self) -> None:
        """Extract entities and relations with enterprise rate limiting"""
        all_texts = [doc.text for doc in self.documents.values()]

        # Estimate tokens for rate limiting
        estimated_tokens = sum(len(text.split()) * 1.3 for text in all_texts)

        # Execute extraction with rate limiting
        extraction_results = await self.rate_limiter.execute_with_rate_limiting(
            extraction_function=lambda: self.llm_extractor.extract_entities_and_relations(all_texts),
            estimated_tokens=estimated_tokens,
            priority="standard"
        )

        # Process extracted entities and relations
        await self._process_extraction_results(extraction_results)

    async def _process_extraction_results(self, extraction_results: Dict[str, Any]) -> None:
        """Process extraction results with enterprise enhancements"""
        # Process extracted entities
        for entity_data in extraction_results.get("entities", []):
            if isinstance(entity_data, str):
                entity = UniversalEntity(
                    entity_id=f"entity_{len(self.entities)}",
                    text=entity_data,
                    entity_type=entity_data,
                    confidence=0.8,
                    context="",
                    metadata={
                        "extraction_method": "enterprise_llm",
                        "domain": self.domain_name,
                        "extracted_at": datetime.now().isoformat()
                    }
                )
                self.entities[entity.entity_id] = entity
                self.discovered_entity_types.add(entity.entity_type)
                self.type_frequencies[entity.entity_type] += 1

        # Process extracted relations
        for relation_data in extraction_results.get("relations", []):
            if isinstance(relation_data, dict):
                relation = UniversalRelation(
                    relation_id=f"relation_{len(self.relations)}",
                    source=relation_data.get("source", ""),
                    target=relation_data.get("target", ""),
                    relation_type=relation_data.get("type", "related_to"),
                    confidence=relation_data.get("confidence", 0.8),
                    metadata={
                        "extraction_method": "enterprise_llm",
                        "domain": self.domain_name,
                        "extracted_at": datetime.now().isoformat()
                    }
                )
                self.relations.append(relation)
                self.discovered_relation_types.add(relation.relation_type)

    async def _assess_extraction_quality_enterprise(self) -> Dict[str, Any]:
        """Enterprise quality assessment using Azure ML models"""

        # Prepare extraction context for ML assessment
        extraction_context = {
            "domain": self.domain_name,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entity_types": list(self.discovered_entity_types),
            "relation_types": list(self.discovered_relation_types),
            "documents_processed": len(self.documents),
            "extraction_config": self.extraction_config
        }

        try:
            # Use Azure ML for quality assessment
            quality_results = await self.quality_assessor.assess_extraction_quality(
                extraction_context,
                self.entities,
                self.relations
            )

            # Track quality metrics in Azure Monitor
            await self.monitor.track_extraction_quality(quality_results)

            return quality_results

        except Exception as e:
            logger.error(f"Enterprise quality assessment failed: {e}")
            # Fallback to basic quality assessment
            return await self._validate_extracted_knowledge()

    def get_extracted_knowledge(self) -> Dict[str, Any]:
        """Get all extracted knowledge in a structured format"""
        return {
            "entities": {entity_id: entity.to_dict() for entity_id, entity in self.entities.items()},
            "relations": [relation.to_dict() for relation in self.relations],
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "discovered_types": {
                "entity_types": list(self.discovered_entity_types),
                "relation_types": list(self.discovered_relation_types)
            },
            "statistics": self.extraction_stats
        }

    async def save_extracted_knowledge(self, output_path: Optional[Path] = None) -> Path:
        """Save extracted knowledge to file"""
        if not output_path:
            output_path = settings.processed_data_dir / f"universal_knowledge_{self.domain_name}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        knowledge_data = self.get_extracted_knowledge()

        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Extracted knowledge saved to: {output_path}")
        return output_path


# Convenience function for direct usage
async def extract_knowledge_from_text_files(
    text_files: List[Path],
    domain_name: str = "general"
) -> Dict[str, Any]:
    """
    Convenience function to extract knowledge from text files

    Args:
        text_files: List of text file paths to process
        domain_name: Domain name for the extraction

    Returns:
        Extraction results
    """
    extractor = AzureOpenAIKnowledgeExtractor(domain_name)

    # Read text files
    texts = []
    sources = []
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            texts.append(text)
            sources.append(file_path.name)
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")

    if not texts:
        return {"success": False, "error": "No valid text files found"}

    return await extractor.extract_knowledge_from_texts(texts, sources)


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        # Test with sample texts
        sample_texts = [
            "The system consists of multiple components working together to achieve the desired outcome.",
            "Regular monitoring includes checking performance metrics, replacing worn parts, and monitoring system health.",
            "Component failure can cause reduced efficiency and performance issues."
        ]

        extractor = AzureOpenAIKnowledgeExtractor("general")
        results = await extractor.extract_knowledge_from_texts(sample_texts)

        print("Extraction Results:")
        print(f"Success: {results['success']}")
        print(f"Entities: {results['knowledge_summary']['total_entities']}")
        print(f"Relations: {results['knowledge_summary']['total_relations']}")
        print(f"Entity Types: {results['discovered_types']['entity_types']}")
        print(f"Relation Types: {results['discovered_types']['relation_types']}")

    asyncio.run(main())