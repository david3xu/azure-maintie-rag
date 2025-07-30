"""
Domain-Specific Pattern Configuration
Centralized configuration for query analysis patterns, field schemas, and domain-specific terminology
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import os


class DomainType(Enum):
    """Supported domain types"""
    MAINTENANCE = "maintenance"
    GENERAL = "general"


@dataclass
class QueryPatterns:
    """Query analysis patterns for a domain"""
    issue_terms: List[str]
    action_terms: List[str]
    enhancement_keywords: List[str]
    domain_indicators: List[str]


@dataclass
class IndexSchema:
    """Index schema configuration for a domain"""
    name: str
    fields: List[Dict[str, Any]]
    vector_fields: List[str] = None


@dataclass
class MetadataPatterns:
    """Metadata patterns for document and entity types"""
    document_types: Dict[str, str]
    entity_types: Dict[str, str]
    relationship_types: Dict[str, str]
    title_patterns: Dict[str, str]
    
    # Default Fallback Values
    default_entity_type: str = "entity"
    default_relation_type: str = "relates_to"
    default_confidence: float = 1.0


@dataclass
class NamingPatterns:
    """Naming patterns for Azure resources"""
    graph_name: str
    index_name: str
    container_name: str


@dataclass
class PromptPatterns:
    """Prompt templates and focus areas for LLM operations"""
    extraction_focus: str
    completion_context: str
    query_enhancement: str
    
    # Model Configuration  
    model_name: str = os.getenv("OPENAI_MODEL_DEPLOYMENT", "gpt-4o")
    temperature: float = 0.1
    max_tokens: int = 2000
    requests_per_minute: int = 50
    chunk_size: int = 1000


@dataclass
class PyTorchGeometricPatterns:
    """PyTorch Geometric configuration patterns for graph construction"""
    # Feature Dimensions
    node_feature_dim: int = 64
    edge_feature_dim: int = 32
    
    # Entity Types (in order of label encoding)
    entity_types: List[str] = None
    relationship_types: List[str] = None
    
    # Feature Engineering Configuration
    text_length_normalization: int = 50
    word_count_normalization: int = 10
    context_length_normalization: int = 100
    source_target_length_normalization: int = 30
    
    # Domain-specific keywords
    equipment_keywords: List[str] = None
    issue_keywords: List[str] = None
    maintenance_keywords: List[str] = None
    
    # Progress reporting
    entity_progress_interval: int = 100
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ['component', 'equipment', 'issue', 'location', 'action', 
                               'procedure', 'symptom', 'solution', 'condition', 'unknown']
        if self.relationship_types is None:
            self.relationship_types = ['procedure', 'has_issue', 'symptom', 'part_of', 'exhibits', 
                                     'has_component', 'requires', 'located_at', 'action_on', 
                                     'component_of', 'location', 'issue_with', 'solution', 
                                     'causes', 'unknown']
        if self.equipment_keywords is None:
            self.equipment_keywords = ['pump', 'valve', 'motor']
        if self.issue_keywords is None:
            self.issue_keywords = ['leak', 'failure', 'break']
        if self.maintenance_keywords is None:
            self.maintenance_keywords = ['maintenance', 'repair', 'failure', 'issue']


@dataclass
class TrainingPatterns:
    """Training and ML configuration patterns"""
    trigger_threshold: int
    training_frequency: str
    model_retention_days: int
    deployment_tier: str
    batch_size: int
    learning_rate: float
    train_ratio: float
    validation_ratio: float
    
    # GNN Model Configuration
    model_type: str = "gcn"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 20
    embedding_dim: int = 768
    
    # Validation Thresholds
    min_entities_threshold: int = 10
    min_relations_threshold: int = 5
    min_avg_confidence: float = 0.5
    
    # Timeout Configuration
    query_timeout: int = 30
    max_wait_time: int = 3600
    check_interval: int = 60


# === DOMAIN CONFIGURATIONS ===

MAINTENANCE_PATTERNS = QueryPatterns(
    issue_terms=[
        'broken', 'not working', 'failed', 'leaking', 'damaged', 'malfunction',
        'error', 'fault', 'defective', 'worn', 'corroded', 'overheating',
        'noise', 'vibration', 'stuck', 'blocked', 'clogged'
    ],
    action_terms=[
        'repair', 'replace', 'fix', 'check', 'maintenance', 'service', 'inspect',
        'clean', 'adjust', 'calibrate', 'lubricate', 'tighten', 'reset',
        'troubleshoot', 'diagnose', 'test', 'monitor'
    ],
    enhancement_keywords=[
        'equipment', 'troubleshooting', 'repair', 'diagnostic', 'procedure'
    ],
    domain_indicators=[
        'equipment', 'machinery', 'component', 'system', 'device', 'tool',
        'parts', 'assembly', 'unit', 'installation'
    ]
)

GENERAL_PATTERNS = QueryPatterns(
    issue_terms=[
        'problem', 'issue', 'error', 'wrong', 'incorrect', 'missing'
    ],
    action_terms=[
        'help', 'solve', 'find', 'search', 'get', 'need', 'want'
    ],
    enhancement_keywords=[
        'information', 'details', 'explanation', 'guide'
    ],
    domain_indicators=[
        'question', 'how', 'what', 'where', 'when', 'why'
    ]
)

# === METADATA PATTERNS ===

MAINTENANCE_METADATA = MetadataPatterns(
    document_types={
        'structured': 'maintenance_record',
        'chunk': 'maintenance_chunk', 
        'document': 'document'
    },
    entity_types={
        'structured': 'maintenance_issue',
        'chunk': 'maintenance_content',
        'document': 'document'
    },
    relationship_types={
        'sequential': 'follows',
        'related': 'relates_to',
        'causes': 'causes',
        'fixes': 'fixes'
    },
    title_patterns={
        'structured': 'Maintenance Issue {i}',
        'chunk': 'Maintenance Chunk {i}',
        'document': '{filename}'
    }
)

GENERAL_METADATA = MetadataPatterns(
    document_types={
        'structured': 'record',
        'chunk': 'content_chunk',
        'document': 'document'
    },
    entity_types={
        'structured': 'entity',
        'chunk': 'content',
        'document': 'document'
    },
    relationship_types={
        'sequential': 'follows',
        'related': 'relates_to'
    },
    title_patterns={
        'structured': 'Record {i}',
        'chunk': 'Content Chunk {i}',
        'document': '{filename}'
    }
)

# === NAMING PATTERNS ===

MAINTENANCE_NAMING = NamingPatterns(
    graph_name="maintenance-graph-{domain}",
    index_name="{index_base}-{domain}",
    container_name="{container_base}-{domain}"
)

GENERAL_NAMING = NamingPatterns(
    graph_name="general-graph-{domain}",
    index_name="{index_base}-{domain}",
    container_name="{container_base}-{domain}"
)

# === PROMPT PATTERNS ===

MAINTENANCE_PROMPTS = PromptPatterns(
    extraction_focus="equipment, components, actions, issues, locations, procedures, symptoms, solutions",
    completion_context="maintenance and equipment management",
    query_enhancement="technical procedures, troubleshooting, and equipment maintenance"
)

GENERAL_PROMPTS = PromptPatterns(
    extraction_focus="entities, concepts, actions, relationships, key terms",
    completion_context="general information processing",
    query_enhancement="comprehensive information retrieval"
)

# === PYTORCH GEOMETRIC PATTERNS ===

MAINTENANCE_PYTORCH_GEOMETRIC = PyTorchGeometricPatterns(
    node_feature_dim=64,
    edge_feature_dim=32,
    entity_types=['component', 'equipment', 'issue', 'location', 'action', 
                  'procedure', 'symptom', 'solution', 'condition', 'unknown'],
    relationship_types=['procedure', 'has_issue', 'symptom', 'part_of', 'exhibits', 
                       'has_component', 'requires', 'located_at', 'action_on', 
                       'component_of', 'location', 'issue_with', 'solution', 
                       'causes', 'unknown'],
    text_length_normalization=50,
    word_count_normalization=10,
    context_length_normalization=100,
    source_target_length_normalization=30,
    equipment_keywords=['pump', 'valve', 'motor'],
    issue_keywords=['leak', 'failure', 'break'],
    maintenance_keywords=['maintenance', 'repair', 'failure', 'issue'],
    entity_progress_interval=100
)

GENERAL_PYTORCH_GEOMETRIC = PyTorchGeometricPatterns(
    node_feature_dim=64,
    edge_feature_dim=32,
    entity_types=['person', 'organization', 'location', 'concept', 'event', 
                  'product', 'service', 'document', 'process', 'unknown'],
    relationship_types=['works_for', 'located_in', 'part_of', 'related_to', 'causes',
                       'contains', 'produces', 'uses', 'follows', 'unknown'],
    text_length_normalization=50,
    word_count_normalization=10,
    context_length_normalization=100,
    source_target_length_normalization=30,
    equipment_keywords=[],
    issue_keywords=['problem', 'issue', 'error'],
    maintenance_keywords=[],
    entity_progress_interval=100
)

# === TRAINING PATTERNS ===

MAINTENANCE_TRAINING = TrainingPatterns(
    trigger_threshold=100,  # New entities/relations before retraining
    training_frequency="daily",
    model_retention_days=30,
    deployment_tier="standard",
    batch_size=32,
    learning_rate=0.001,
    train_ratio=0.8,
    validation_ratio=0.2
)

GENERAL_TRAINING = TrainingPatterns(
    trigger_threshold=200,  # Higher threshold for general domain
    training_frequency="weekly", 
    model_retention_days=14,
    deployment_tier="basic",
    batch_size=64,
    learning_rate=0.01,
    train_ratio=0.7,
    validation_ratio=0.3
)

# === INDEX SCHEMAS ===

MAINTENANCE_INDEX_SCHEMA = IndexSchema(
    name="maintenance_documents",
    fields=[
        {"name": "id", "type": "Edm.String", "key": True},
        {"name": "content", "type": "Edm.String", "searchable": True},
        {"name": "title", "type": "Edm.String", "searchable": True},
        {"name": "equipment_type", "type": "Edm.String", "filterable": True},
        {"name": "issue_category", "type": "Edm.String", "filterable": True},
        {"name": "action_type", "type": "Edm.String", "filterable": True},
        {"name": "domain", "type": "Edm.String", "filterable": True},
        {"name": "metadata", "type": "Edm.String"},
        {"name": "confidence_score", "type": "Edm.Double", "sortable": True}
    ],
    vector_fields=["content_vector", "title_vector"]
)

GENERAL_INDEX_SCHEMA = IndexSchema(
    name="general_documents",
    fields=[
        {"name": "id", "type": "Edm.String", "key": True},
        {"name": "content", "type": "Edm.String", "searchable": True},
        {"name": "title", "type": "Edm.String", "searchable": True},
        {"name": "category", "type": "Edm.String", "filterable": True},
        {"name": "domain", "type": "Edm.String", "filterable": True},
        {"name": "metadata", "type": "Edm.String"}
    ]
)

# === DOMAIN REGISTRY ===

DOMAIN_CONFIGURATIONS = {
    DomainType.MAINTENANCE: {
        'patterns': MAINTENANCE_PATTERNS,
        'schema': MAINTENANCE_INDEX_SCHEMA,
        'metadata': MAINTENANCE_METADATA,
        'naming': MAINTENANCE_NAMING,
        'prompts': MAINTENANCE_PROMPTS,
        'training': MAINTENANCE_TRAINING,
        'pytorch_geometric': MAINTENANCE_PYTORCH_GEOMETRIC
    },
    DomainType.GENERAL: {
        'patterns': GENERAL_PATTERNS,
        'schema': GENERAL_INDEX_SCHEMA,
        'metadata': GENERAL_METADATA,
        'naming': GENERAL_NAMING,
        'prompts': GENERAL_PROMPTS,
        'training': GENERAL_TRAINING,
        'pytorch_geometric': GENERAL_PYTORCH_GEOMETRIC
    }
}


class DomainPatternManager:
    """Manager for domain-specific patterns and configurations"""
    
    @staticmethod
    def get_patterns(domain: str) -> QueryPatterns:
        """Get query patterns for a domain"""
        domain_type = DomainType(domain.lower()) if domain.lower() in [d.value for d in DomainType] else DomainType.GENERAL
        return DOMAIN_CONFIGURATIONS[domain_type]['patterns']
    
    @staticmethod
    def get_schema(domain: str) -> IndexSchema:
        """Get index schema for a domain"""
        domain_type = DomainType(domain.lower()) if domain.lower() in [d.value for d in DomainType] else DomainType.GENERAL
        return DOMAIN_CONFIGURATIONS[domain_type]['schema']
    
    @staticmethod
    def get_metadata(domain: str) -> MetadataPatterns:
        """Get metadata patterns for a domain"""
        domain_type = DomainType(domain.lower()) if domain.lower() in [d.value for d in DomainType] else DomainType.GENERAL
        return DOMAIN_CONFIGURATIONS[domain_type]['metadata']
    
    @staticmethod
    def detect_domain(query: str) -> str:
        """Detect domain from query content"""
        query_lower = query.lower()
        
        # Check maintenance indicators
        maintenance_indicators = MAINTENANCE_PATTERNS.domain_indicators
        if any(indicator in query_lower for indicator in maintenance_indicators):
            return DomainType.MAINTENANCE.value
        
        return DomainType.GENERAL.value
    
    @staticmethod
    def enhance_query(query: str, domain: str = None) -> Dict[str, Any]:
        """Enhanced query analysis using domain patterns"""
        if domain is None:
            domain = DomainPatternManager.detect_domain(query)
        
        patterns = DomainPatternManager.get_patterns(domain)
        query_lower = query.lower()
        words = query_lower.split()
        
        analysis = {
            'original_query': query,
            'detected_domain': domain,
            'word_count': len(words),
            'issues_found': [term for term in patterns.issue_terms if term in query_lower],
            'actions_found': [term for term in patterns.action_terms if term in query_lower],
            'domain_indicators': [term for term in patterns.domain_indicators if term in query_lower]
        }
        
        # Enhanced query construction
        enhanced_query = query
        if analysis['issues_found'] or analysis['actions_found']:
            # Add domain-specific enhancement keywords
            enhancement_words = []
            if analysis['issues_found']:
                enhancement_words.extend(patterns.enhancement_keywords[:2])  # Add first 2 keywords
            
            if enhancement_words:
                enhanced_query += " " + " ".join(enhancement_words)
        
        analysis['enhanced_query'] = enhanced_query
        analysis['confidence'] = len(analysis['issues_found']) + len(analysis['actions_found']) + len(analysis['domain_indicators'])
        
        return analysis
    
    @staticmethod
    def get_document_type(domain: str, content_type: str) -> str:
        """Get document type for a domain and content type"""
        metadata = DomainPatternManager.get_metadata(domain)
        return metadata.document_types.get(content_type, 'document')
    
    @staticmethod
    def get_entity_type(domain: str, content_type: str) -> str:
        """Get entity type for a domain and content type"""
        metadata = DomainPatternManager.get_metadata(domain)
        return metadata.entity_types.get(content_type, 'entity')
    
    @staticmethod
    def get_relationship_type(domain: str, relationship: str) -> str:
        """Get relationship type for a domain"""
        metadata = DomainPatternManager.get_metadata(domain)
        return metadata.relationship_types.get(relationship, 'related')
    
    @staticmethod
    def get_title_pattern(domain: str, content_type: str) -> str:
        """Get title pattern for a domain and content type"""
        metadata = DomainPatternManager.get_metadata(domain)
        return metadata.title_patterns.get(content_type, '{filename}')
    
    @staticmethod
    def get_naming(domain: str) -> NamingPatterns:
        """Get naming patterns for a domain"""
        domain_type = DomainType(domain.lower()) if domain.lower() in [d.value for d in DomainType] else DomainType.GENERAL
        return DOMAIN_CONFIGURATIONS[domain_type]['naming']
    
    @staticmethod
    def get_graph_name(domain: str) -> str:
        """Get graph name for a domain"""
        naming = DomainPatternManager.get_naming(domain)
        return naming.graph_name.format(domain=domain)
    
    @staticmethod
    def get_index_name(domain: str, index_base: str) -> str:
        """Get index name for a domain"""
        naming = DomainPatternManager.get_naming(domain)
        return naming.index_name.format(index_base=index_base, domain=domain)
    
    @staticmethod
    def get_container_name(domain: str, container_base: str) -> str:
        """Get container name for a domain"""
        naming = DomainPatternManager.get_naming(domain)
        return naming.container_name.format(container_base=container_base, domain=domain)
    
    @staticmethod
    def get_prompts(domain: str) -> PromptPatterns:
        """Get prompt patterns for a domain"""
        domain_type = DomainType(domain.lower()) if domain.lower() in [d.value for d in DomainType] else DomainType.GENERAL
        return DOMAIN_CONFIGURATIONS[domain_type]['prompts']
    
    @staticmethod
    def get_extraction_focus(domain: str) -> str:
        """Get extraction focus for a domain"""
        prompts = DomainPatternManager.get_prompts(domain)
        return prompts.extraction_focus
    
    @staticmethod
    def get_training(domain: str) -> TrainingPatterns:
        """Get training patterns for a domain"""
        domain_type = DomainType(domain.lower()) if domain.lower() in [d.value for d in DomainType] else DomainType.GENERAL
        return DOMAIN_CONFIGURATIONS[domain_type]['training']
    
    @staticmethod
    def get_training_schedule(domain: str) -> Dict[str, Any]:
        """Get training schedule configuration for a domain"""
        training = DomainPatternManager.get_training(domain)
        return {
            "trigger_threshold": training.trigger_threshold,
            "training_frequency": training.training_frequency,
            "model_retention_days": training.model_retention_days,
            "deployment_tier": training.deployment_tier
        }
    
    @staticmethod
    def get_train_ratio(domain: str) -> float:
        """Get training ratio for a domain"""
        training = DomainPatternManager.get_training(domain)
        return training.train_ratio
    
    @staticmethod
    def get_pytorch_geometric(domain: str) -> PyTorchGeometricPatterns:
        """Get PyTorch Geometric patterns for a domain"""
        domain_type = DomainType(domain.lower()) if domain.lower() in [d.value for d in DomainType] else DomainType.GENERAL
        return DOMAIN_CONFIGURATIONS[domain_type]['pytorch_geometric']


# === CONFIGURATION VALIDATION ===

def validate_domain_config(domain_type: DomainType) -> bool:
    """Validate domain configuration completeness"""
    try:
        config = DOMAIN_CONFIGURATIONS[domain_type]
        patterns = config['patterns']
        schema = config['schema']
        
        # Check patterns
        assert len(patterns.issue_terms) > 0, f"No issue terms for {domain_type}"
        assert len(patterns.action_terms) > 0, f"No action terms for {domain_type}"
        
        # Check schema
        assert len(schema.fields) > 0, f"No schema fields for {domain_type}"
        assert any(field['name'] == 'id' for field in schema.fields), f"No ID field for {domain_type}"
        
        return True
        
    except Exception as e:
        print(f"Domain configuration validation failed for {domain_type}: {e}")
        return False


def validate_all_configs() -> bool:
    """Validate all domain configurations"""
    return all(validate_domain_config(domain_type) for domain_type in DomainType)


if __name__ == "__main__":
    # Self-validation
    if validate_all_configs():
        print("✅ All domain configurations are valid")
    else:
        print("❌ Domain configuration validation failed")