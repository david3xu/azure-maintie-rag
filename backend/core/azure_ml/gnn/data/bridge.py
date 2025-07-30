"""
Extraction-to-GNN Data Bridge Component.

This module provides the critical bridge between knowledge extraction output
and GNN training input, solving the data format mismatch identified in the
design analysis.

Key Features:
- Converts JSON extraction output to standardized graph format
- Validates graph quality before training
- Handles various extraction formats (prompt flow, direct extraction, etc.)
- Provides quality assessment and filtering

Created as part of GNN Training Stage Design Analysis remediation plan.
Location: /docs/workflows/GNN_Training_Stage_Design_Analysis.md
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import numpy as np

from core.models.gnn_data_models import (
    StandardizedGraphData,
    StandardizedEntity, 
    StandardizedRelation,
    GraphQualityMetrics
)

logger = logging.getLogger(__name__)


class ExtractionToGNNBridge:
    """
    Converts knowledge extraction output to GNN training format.
    
    This class solves the critical data format mismatch between:
    1. Knowledge extraction JSON output (entities/relations lists)
    2. GNN training expected input (standardized graph format)
    
    Supports multiple extraction formats and provides quality validation.
    """
    
    def __init__(self):
        """Initialize the data bridge."""
        self.supported_formats = [
            "prompt_flow_v1",      # Prompt flow extraction format
            "direct_extraction_v1", # Direct LLM extraction format
            "clean_extraction_v1"   # Clean knowledge extraction format
        ]
        
    def convert_extraction_to_gnn_data(
        self, 
        extraction_file: Union[str, Path],
        domain: Optional[str] = None,
        format_type: Optional[str] = None
    ) -> StandardizedGraphData:
        """
        Convert knowledge extraction file to standardized GNN format.
        
        Args:
            extraction_file: Path to extraction JSON file
            domain: Domain name (inferred from file if not provided)
            format_type: Extraction format type (auto-detected if not provided)
            
        Returns:
            StandardizedGraphData ready for GNN training
            
        Raises:
            ValueError: If file format is unsupported or data is invalid
            FileNotFoundError: If extraction file doesn't exist
            
        Example:
            >>> bridge = ExtractionToGNNBridge()
            >>> graph_data = bridge.convert_extraction_to_gnn_data(
            ...     "backend/data/extraction_outputs/clean_knowledge_extraction.json",
            ...     domain="maintenance"
            ... )
            >>> print(f"Converted {len(graph_data.entities)} entities")
        """
        file_path = Path(extraction_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Extraction file not found: {file_path}")
        
        logger.info(f"Converting extraction file: {file_path}")
        
        # Load extraction data
        with open(file_path, 'r', encoding='utf-8') as f:
            extraction_data = json.load(f)
        
        # Auto-detect format if not provided
        if format_type is None:
            format_type = self._detect_format(extraction_data)
            logger.info(f"Auto-detected format: {format_type}")
        
        # Infer domain from filename if not provided
        if domain is None:
            domain = self._infer_domain_from_filename(file_path)
            logger.info(f"Inferred domain: {domain}")
        
        # Convert based on format
        if format_type == "prompt_flow_v1":
            return self._convert_prompt_flow_format(extraction_data, domain, file_path)
        elif format_type == "direct_extraction_v1":
            return self._convert_direct_extraction_format(extraction_data, domain, file_path)
        elif format_type == "clean_extraction_v1":
            return self._convert_clean_extraction_format(extraction_data, domain, file_path)
        else:
            raise ValueError(f"Unsupported extraction format: {format_type}")
    
    def _detect_format(self, data: Dict[str, Any]) -> str:
        """
        Auto-detect extraction format from data structure.
        
        Args:
            data: Loaded extraction JSON data
            
        Returns:
            Detected format type
        """
        # Check for prompt flow format (has flow metadata)
        if "flow_metadata" in data or "prompt_flow" in str(data):
            return "prompt_flow_v1"
        
        # Check for clean extraction format (has entities and relations at root)
        if "entities" in data and "relations" in data:
            # Check if entities have the clean extraction structure
            if data["entities"] and "entity_id" in data["entities"][0]:
                return "clean_extraction_v1"
        
        # Default to direct extraction
        return "direct_extraction_v1"
    
    def _infer_domain_from_filename(self, file_path: Path) -> str:
        """
        Infer domain from extraction filename.
        
        Args:
            file_path: Path to extraction file
            
        Returns:
            Inferred domain name
        """
        filename = file_path.stem.lower()
        
        # Common domain patterns in filenames
        domain_patterns = {
            "maintenance": ["maintenance", "maint"],
            "medical": ["medical", "health", "clinical"],
            "legal": ["legal", "law", "contract"],
            "financial": ["financial", "finance", "banking"],
            "technical": ["technical", "tech", "engineering"]
        }
        
        for domain, patterns in domain_patterns.items():
            if any(pattern in filename for pattern in patterns):
                return domain
        
        # Default domain
        return "general"
    
    def _convert_clean_extraction_format(
        self, 
        data: Dict[str, Any], 
        domain: str,
        file_path: Path
    ) -> StandardizedGraphData:
        """
        Convert clean extraction format to standardized format.
        
        This handles the current format from clean_knowledge_extraction_prompt_flow.
        
        Args:
            data: Clean extraction data
            domain: Domain name
            file_path: Source file path
            
        Returns:
            StandardizedGraphData
        """
        entities = []
        relations = []
        
        # Convert entities
        for entity_data in data.get("entities", []):
            try:
                entity = StandardizedEntity.from_extraction_dict(entity_data)
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Skipping invalid entity {entity_data.get('entity_id', 'unknown')}: {e}")
        
        # Convert relations  
        for relation_data in data.get("relations", []):
            try:
                relation = StandardizedRelation.from_extraction_dict(relation_data)
                relations.append(relation)
            except Exception as e:
                logger.warning(f"Skipping invalid relation {relation_data.get('relation_id', 'unknown')}: {e}")
        
        # Extract metadata
        extraction_timestamp = None
        extraction_method = "unknown"
        
        if entities and entities[0].extraction_metadata:
            metadata = entities[0].extraction_metadata
            if "extracted_at" in metadata:
                try:
                    extraction_timestamp = datetime.fromisoformat(metadata["extracted_at"])
                except:
                    pass
            extraction_method = metadata.get("extraction_method", "unknown")
        
        graph_data = StandardizedGraphData(
            entities=entities,
            relations=relations,
            domain=domain,
            extraction_timestamp=extraction_timestamp,
            extraction_method=extraction_method,
            source_documents=[str(file_path)]
        )
        
        logger.info(f"Converted clean extraction: {len(entities)} entities, {len(relations)} relations")
        return graph_data
    
    def _convert_prompt_flow_format(
        self, 
        data: Dict[str, Any], 
        domain: str,
        file_path: Path
    ) -> StandardizedGraphData:
        """
        Convert prompt flow extraction format to standardized format.
        
        Args:
            data: Prompt flow extraction data
            domain: Domain name
            file_path: Source file path
            
        Returns:
            StandardizedGraphData
        """
        # Prompt flow typically has nested structure
        # Extract entities and relations from flow output
        
        entities = []
        relations = []
        
        # Handle different prompt flow output structures
        if "outputs" in data:
            output_data = data["outputs"]
        elif "results" in data:
            output_data = data["results"]
        else:
            output_data = data
        
        # Extract entities
        entity_data = output_data.get("entities", [])
        for i, entity in enumerate(entity_data):
            if isinstance(entity, str):
                # Simple string entities
                std_entity = StandardizedEntity(
                    entity_id=f"entity_{i}",
                    text=entity,
                    entity_type="unknown",
                    confidence=0.8,  # Default confidence for prompt flow
                    extraction_metadata={
                        "extraction_method": "prompt_flow",
                        "domain": domain,
                        "extracted_at": datetime.now().isoformat()
                    }
                )
            else:
                # Structured entity objects
                std_entity = StandardizedEntity.from_extraction_dict(entity)
            
            entities.append(std_entity)
        
        # Extract relations
        relation_data = output_data.get("relations", [])
        for i, relation in enumerate(relation_data):
            if isinstance(relation, dict):
                std_relation = StandardizedRelation.from_extraction_dict(relation)
            else:
                # Handle other relation formats
                continue
            
            relations.append(std_relation)
        
        graph_data = StandardizedGraphData(
            entities=entities,
            relations=relations,
            domain=domain,
            extraction_timestamp=datetime.now(),
            extraction_method="prompt_flow",
            source_documents=[str(file_path)]
        )
        
        logger.info(f"Converted prompt flow: {len(entities)} entities, {len(relations)} relations")
        return graph_data
    
    def _convert_direct_extraction_format(
        self, 
        data: Dict[str, Any], 
        domain: str,
        file_path: Path
    ) -> StandardizedGraphData:
        """
        Convert direct LLM extraction format to standardized format.
        
        Args:
            data: Direct extraction data
            domain: Domain name  
            file_path: Source file path
            
        Returns:
            StandardizedGraphData
        """
        # Direct extraction may have various formats
        # Attempt to extract entities and relations
        
        entities = []
        relations = []
        
        # Try different data structures
        if "entities" in data:
            for entity_data in data["entities"]:
                entity = StandardizedEntity.from_extraction_dict(entity_data)
                entities.append(entity)
        
        if "relations" in data:
            for relation_data in data["relations"]:
                relation = StandardizedRelation.from_extraction_dict(relation_data)
                relations.append(relation)
        
        graph_data = StandardizedGraphData(
            entities=entities,
            relations=relations,
            domain=domain,
            extraction_timestamp=datetime.now(),
            extraction_method="direct_extraction",
            source_documents=[str(file_path)]
        )
        
        logger.info(f"Converted direct extraction: {len(entities)} entities, {len(relations)} relations")
        return graph_data
    
    def validate_graph_quality(self, graph_data: StandardizedGraphData) -> Tuple[bool, List[str]]:
        """
        Validate graph quality for GNN training readiness.
        
        Args:
            graph_data: Standardized graph data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
            
        Example:
            >>> bridge = ExtractionToGNNBridge()
            >>> is_valid, issues = bridge.validate_graph_quality(graph_data)
            >>> if not is_valid:
            ...     print(f"Quality issues: {issues}")
        """
        if not graph_data.quality_metrics:
            return False, ["Quality metrics not computed"]
        
        is_valid = graph_data.quality_metrics.is_training_ready()
        issues = graph_data.quality_metrics.get_quality_issues()
        
        if is_valid:
            logger.info("Graph validation passed - ready for GNN training")
        else:
            logger.warning(f"Graph validation failed: {issues}")
        
        return is_valid, issues
    
    def filter_low_quality_data(
        self, 
        graph_data: StandardizedGraphData,
        min_confidence: float = 0.5
    ) -> StandardizedGraphData:
        """
        Filter out low-quality entities and relations.
        
        Args:
            graph_data: Input graph data
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered graph data
        """
        # Filter entities
        filtered_entities = [
            entity for entity in graph_data.entities
            if entity.confidence >= min_confidence
        ]
        
        # Filter relations and ensure both entities exist
        valid_entity_ids = {entity.entity_id for entity in filtered_entities}
        filtered_relations = [
            relation for relation in graph_data.relations
            if (relation.confidence >= min_confidence and
                relation.source_entity in valid_entity_ids and
                relation.target_entity in valid_entity_ids)
        ]
        
        # Create filtered graph
        filtered_graph = StandardizedGraphData(
            entities=filtered_entities,
            relations=filtered_relations,
            domain=graph_data.domain,
            extraction_timestamp=graph_data.extraction_timestamp,
            extraction_method=graph_data.extraction_method,
            source_documents=graph_data.source_documents
        )
        
        logger.info(f"Filtered graph: {len(filtered_entities)} entities "
                   f"({len(graph_data.entities) - len(filtered_entities)} removed), "
                   f"{len(filtered_relations)} relations "
                   f"({len(graph_data.relations) - len(filtered_relations)} removed)")
        
        return filtered_graph
    
    def preview_conversion(
        self, 
        extraction_file: Union[str, Path],
        max_entities: int = 5,
        max_relations: int = 5
    ) -> Dict[str, Any]:
        """
        Preview conversion without full processing (for debugging).
        
        Args:
            extraction_file: Path to extraction file
            max_entities: Maximum entities to show in preview
            max_relations: Maximum relations to show in preview
            
        Returns:
            Preview information
        """
        try:
            graph_data = self.convert_extraction_to_gnn_data(extraction_file)
            
            preview = {
                "success": True,
                "domain": graph_data.domain,
                "total_entities": len(graph_data.entities),
                "total_relations": len(graph_data.relations),
                "sample_entities": [
                    {
                        "entity_id": e.entity_id,
                        "text": e.text,
                        "entity_type": e.entity_type,
                        "confidence": e.confidence
                    }
                    for e in graph_data.entities[:max_entities]
                ],
                "sample_relations": [
                    {
                        "relation_id": r.relation_id,
                        "source": r.source_entity,
                        "target": r.target_entity,
                        "relation_type": r.relation_type,
                        "confidence": r.confidence
                    }
                    for r in graph_data.relations[:max_relations]
                ],
                "quality_metrics": graph_data.quality_metrics.__dict__ if graph_data.quality_metrics else None
            }
            
            return preview
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }


class GraphDataValidator:
    """
    Specialized validator for graph data quality assessment.
    
    Provides detailed analysis of graph structure and quality metrics
    to ensure successful GNN training.
    """
    
    def __init__(self):
        """Initialize validator with default thresholds."""
        self.min_entities = 10
        self.min_relations = 5
        self.min_confidence = 0.5
        self.max_isolated_ratio = 0.3  # Max 30% isolated entities
    
    def comprehensive_validation(
        self, 
        graph_data: StandardizedGraphData
    ) -> Dict[str, Any]:
        """
        Perform comprehensive graph validation.
        
        Args:
            graph_data: Graph data to validate
            
        Returns:
            Detailed validation report
        """
        report = {
            "overall_valid": False,
            "validation_timestamp": datetime.now().isoformat(),
            "checks": {},
            "recommendations": [],
            "graph_statistics": {}
        }
        
        # Basic counts
        report["checks"]["entity_count"] = {
            "passed": len(graph_data.entities) >= self.min_entities,
            "value": len(graph_data.entities),
            "threshold": self.min_entities
        }
        
        report["checks"]["relation_count"] = {
            "passed": len(graph_data.relations) >= self.min_relations,
            "value": len(graph_data.relations),
            "threshold": self.min_relations
        }
        
        # Confidence checks
        entity_confidences = [e.confidence for e in graph_data.entities]
        avg_entity_confidence = np.mean(entity_confidences) if entity_confidences else 0.0
        
        report["checks"]["entity_confidence"] = {
            "passed": avg_entity_confidence >= self.min_confidence,
            "value": avg_entity_confidence,
            "threshold": self.min_confidence
        }
        
        # Connectivity check
        isolated_count = graph_data.quality_metrics.isolated_entities if graph_data.quality_metrics else 0
        isolated_ratio = isolated_count / len(graph_data.entities) if graph_data.entities else 1.0
        
        report["checks"]["connectivity"] = {
            "passed": isolated_ratio <= self.max_isolated_ratio,
            "value": isolated_ratio,
            "threshold": self.max_isolated_ratio
        }
        
        # Overall validation
        report["overall_valid"] = all(check["passed"] for check in report["checks"].values())
        
        # Generate recommendations
        if not report["checks"]["entity_count"]["passed"]:
            report["recommendations"].append(f"Increase entity extraction - need {self.min_entities - len(graph_data.entities)} more entities")
        
        if not report["checks"]["relation_count"]["passed"]:
            report["recommendations"].append(f"Increase relation extraction - need {self.min_relations - len(graph_data.relations)} more relations")
        
        if not report["checks"]["entity_confidence"]["passed"]:
            report["recommendations"].append("Improve entity extraction confidence or lower confidence threshold")
        
        if not report["checks"]["connectivity"]["passed"]:
            report["recommendations"].append("Improve relation extraction to reduce isolated entities")
        
        return report