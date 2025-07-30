"""
Standardized data models for GNN training pipeline.

This module provides unified data structures for converting knowledge extraction
output to GNN training format, ensuring type safety and consistency across
the pipeline.

Created as part of GNN Training Stage Design Analysis remediation plan.
Location: /docs/workflows/GNN_Training_Stage_Design_Analysis.md
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import numpy as np
from pathlib import Path

from ...config.domain_patterns import DomainPatternManager


@dataclass
class StandardizedEntity:
    """
    Standardized entity representation for GNN training.
    
    Converts various entity formats from knowledge extraction into
    a consistent structure suitable for graph neural network processing.
    """
    entity_id: str
    text: str
    entity_type: str
    confidence: float
    
    # Optional semantic features
    embeddings: Optional[List[float]] = None
    semantic_features: Optional[Dict[str, Any]] = None
    
    # Metadata from extraction
    context: Optional[str] = None
    extraction_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity data after initialization."""
        if not self.entity_id:
            raise ValueError("entity_id cannot be empty")
        if not self.text:
            raise ValueError("text cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "entity_id": self.entity_id,
            "text": self.text,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "embeddings": self.embeddings,
            "semantic_features": self.semantic_features,
            "context": self.context,
            "extraction_metadata": self.extraction_metadata
        }
    
    @classmethod
    def from_extraction_dict(cls, data: Dict[str, Any]) -> "StandardizedEntity":
        """
        Create StandardizedEntity from knowledge extraction output.
        
        Args:
            data: Entity dictionary from extraction JSON
            
        Returns:
            StandardizedEntity instance
            
        Example:
            >>> entity_data = {
            ...     "entity_id": "entity_0",
            ...     "text": "location",
            ...     "entity_type": "location", 
            ...     "confidence": 0.8,
            ...     "metadata": {...}
            ... }
            >>> entity = StandardizedEntity.from_extraction_dict(entity_data)
        """
        return cls(
            entity_id=data.get("entity_id", ""),
            text=data.get("text", ""),
            entity_type=data.get("entity_type", "unknown"),
            confidence=float(data.get("confidence", 1.0)),
            context=data.get("context"),
            extraction_metadata=data.get("metadata", {})
        )


@dataclass  
class StandardizedRelation:
    """
    Standardized relation representation for GNN training.
    
    Represents relationships between entities in a format suitable
    for graph neural network edge construction.
    """
    relation_id: str
    source_entity: str  # entity_id
    target_entity: str  # entity_id
    relation_type: str
    confidence: float
    
    # Optional semantic features
    embeddings: Optional[List[float]] = None
    semantic_features: Optional[Dict[str, Any]] = None
    
    # Metadata from extraction
    context: Optional[str] = None
    extraction_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate relation data after initialization."""
        if not self.relation_id:
            raise ValueError("relation_id cannot be empty")
        if not self.source_entity:
            raise ValueError("source_entity cannot be empty")
        if not self.target_entity:
            raise ValueError("target_entity cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary representation."""
        return {
            "relation_id": self.relation_id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "embeddings": self.embeddings,
            "semantic_features": self.semantic_features,
            "context": self.context,
            "extraction_metadata": self.extraction_metadata
        }
    
    @classmethod
    def from_extraction_dict(cls, data: Dict[str, Any]) -> "StandardizedRelation":
        """
        Create StandardizedRelation from knowledge extraction output.
        
        Args:
            data: Relation dictionary from extraction JSON
            
        Returns:
            StandardizedRelation instance
        """
        return cls(
            relation_id=data.get("relation_id", ""),
            source_entity=data.get("source_entity", ""),
            target_entity=data.get("target_entity", ""),
            relation_type=data.get("relation_type", "unknown"),
            confidence=float(data.get("confidence", 1.0)),
            context=data.get("context"),
            extraction_metadata=data.get("metadata", {})
        )


@dataclass
class GraphQualityMetrics:
    """Quality metrics for extracted knowledge graphs."""
    total_entities: int
    total_relations: int
    unique_entity_types: int
    unique_relation_types: int
    avg_entity_confidence: float
    avg_relation_confidence: float
    connected_components: int
    isolated_entities: int
    
    # Quality flags
    has_low_confidence_entities: bool = False
    has_isolated_entities: bool = False
    has_missing_relations: bool = False
    
    def is_training_ready(self, domain: str = "general") -> bool:
        """
        Determine if graph quality is sufficient for GNN training.
        
        Args:
            domain: Domain for retrieving validation thresholds
        
        Returns:
            True if graph meets minimum quality thresholds
        """
        training_patterns = DomainPatternManager.get_training(domain)
        min_entities = training_patterns.min_entities_threshold
        min_relations = training_patterns.min_relations_threshold
        min_avg_confidence = training_patterns.min_avg_confidence
        
        return (
            self.total_entities >= min_entities and
            self.total_relations >= min_relations and
            self.avg_entity_confidence >= min_avg_confidence and
            self.avg_relation_confidence >= min_avg_confidence and
            not self.has_missing_relations
        )
    
    def get_quality_issues(self) -> List[str]:
        """Get list of quality issues preventing training."""
        issues = []
        
        if self.total_entities < 10:
            issues.append(f"Insufficient entities: {self.total_entities} < 10")
        if self.total_relations < 5:
            issues.append(f"Insufficient relations: {self.total_relations} < 5")
        if self.avg_entity_confidence < 0.5:
            issues.append(f"Low entity confidence: {self.avg_entity_confidence:.2f} < 0.5")
        if self.avg_relation_confidence < 0.5:
            issues.append(f"Low relation confidence: {self.avg_relation_confidence:.2f} < 0.5")
        if self.has_isolated_entities:
            issues.append(f"Isolated entities detected: {self.isolated_entities}")
        if self.has_missing_relations:
            issues.append("Missing or malformed relations detected")
            
        return issues


@dataclass
class StandardizedGraphData:
    """
    Complete standardized graph representation for GNN training.
    
    This is the unified format that bridges knowledge extraction output
    with GNN training input, ensuring consistency and type safety.
    """
    entities: List[StandardizedEntity]
    relations: List[StandardizedRelation]
    domain: str
    
    # Extraction metadata
    extraction_timestamp: Optional[datetime] = None
    extraction_method: Optional[str] = None
    source_documents: Optional[List[str]] = field(default_factory=list)
    
    # Quality metrics
    quality_metrics: Optional[GraphQualityMetrics] = None
    
    def __post_init__(self):
        """Validate graph data and compute quality metrics."""
        if not self.entities:
            raise ValueError("Graph must contain at least one entity")
        if not self.domain:
            raise ValueError("Domain cannot be empty")
        
        # Compute quality metrics if not provided
        if self.quality_metrics is None:
            self.quality_metrics = self._compute_quality_metrics()
    
    def _compute_quality_metrics(self) -> GraphQualityMetrics:
        """Compute quality metrics for the graph."""
        entity_types = set(e.entity_type for e in self.entities)
        relation_types = set(r.relation_type for r in self.relations)
        
        entity_confidences = [e.confidence for e in self.entities]
        relation_confidences = [r.confidence for r in self.relations] if self.relations else [0.0]
        
        # Find isolated entities (entities not in any relation)
        entities_in_relations = set()
        for rel in self.relations:
            entities_in_relations.add(rel.source_entity)
            entities_in_relations.add(rel.target_entity)
        
        isolated_entities = len([e for e in self.entities if e.entity_id not in entities_in_relations])
        
        return GraphQualityMetrics(
            total_entities=len(self.entities),
            total_relations=len(self.relations),
            unique_entity_types=len(entity_types),
            unique_relation_types=len(relation_types),
            avg_entity_confidence=np.mean(entity_confidences) if entity_confidences else 0.0,
            avg_relation_confidence=np.mean(relation_confidences) if relation_confidences else 0.0,
            connected_components=1,  # Simplified - would need graph analysis for exact count
            isolated_entities=isolated_entities,
            has_low_confidence_entities=any(c < 0.5 for c in entity_confidences),
            has_isolated_entities=isolated_entities > 0,
            has_missing_relations=len(self.relations) == 0
        )
    
    def get_entity_by_id(self, entity_id: str) -> Optional[StandardizedEntity]:
        """Get entity by ID."""
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        return None
    
    def get_relations_for_entity(self, entity_id: str) -> List[StandardizedRelation]:
        """Get all relations involving a specific entity."""
        return [
            rel for rel in self.relations 
            if rel.source_entity == entity_id or rel.target_entity == entity_id
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "domain": self.domain,
            "extraction_timestamp": self.extraction_timestamp.isoformat() if self.extraction_timestamp else None,
            "extraction_method": self.extraction_method,
            "source_documents": self.source_documents,
            "quality_metrics": self.quality_metrics.__dict__ if self.quality_metrics else None
        }
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save standardized graph to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "StandardizedGraphData":
        """Load standardized graph from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardizedGraphData":
        """Create StandardizedGraphData from dictionary."""
        entities = [StandardizedEntity.from_extraction_dict(e) for e in data.get("entities", [])]
        relations = [StandardizedRelation.from_extraction_dict(r) for r in data.get("relations", [])]
        
        return cls(
            entities=entities,
            relations=relations,
            domain=data.get("domain", "unknown"),
            extraction_timestamp=datetime.fromisoformat(data["extraction_timestamp"]) if data.get("extraction_timestamp") else None,
            extraction_method=data.get("extraction_method"),
            source_documents=data.get("source_documents", [])
        )


@dataclass
class GNNTrainingConfig:
    """Configuration for GNN training pipeline."""
    domain: str = "general"
    
    # Feature engineering options
    use_semantic_embeddings: bool = True
    normalize_features: bool = True
    
    def __post_init__(self):
        """Initialize training patterns from domain configuration."""
        self.training_patterns = DomainPatternManager.get_training(self.domain)
    
    @property
    def model_type(self) -> str:
        return self.training_patterns.model_type
    
    @property
    def hidden_dim(self) -> int:
        return self.training_patterns.hidden_dim
    
    @property
    def num_layers(self) -> int:
        return self.training_patterns.num_layers
    
    @property
    def dropout(self) -> float:
        return self.training_patterns.dropout
    
    @property
    def learning_rate(self) -> float:
        return self.training_patterns.learning_rate
    
    @property
    def weight_decay(self) -> float:
        return self.training_patterns.weight_decay
    
    @property
    def batch_size(self) -> int:
        return self.training_patterns.batch_size
    
    @property
    def epochs(self) -> int:
        return self.training_patterns.epochs
    
    @property
    def patience(self) -> int:
        return self.training_patterns.patience
    
    @property
    def embedding_dim(self) -> int:
        return self.training_patterns.embedding_dim
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_type": self.model_type,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "use_semantic_embeddings": self.use_semantic_embeddings,
            "embedding_dim": self.embedding_dim,
            "normalize_features": self.normalize_features
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GNNTrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingResult:
    """Result of GNN training process."""
    success: bool
    model_path: Optional[str] = None
    training_metrics: Optional[Dict[str, float]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    training_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    # Model metadata
    model_version: Optional[str] = None
    domain: Optional[str] = None
    config_used: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "model_path": self.model_path,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "training_time_seconds": self.training_time_seconds,
            "error_message": self.error_message,
            "model_version": self.model_version,
            "domain": self.domain,
            "config_used": self.config_used
        }