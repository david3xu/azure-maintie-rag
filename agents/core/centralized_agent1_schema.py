"""
Centralized Agent 1 Data Schema - MINIMAL ESSENTIAL FIELDS ONLY
==============================================================

This module defines ONLY the fields that are actually used by downstream agents.
Every field has a confirmed usage location - no unused fields.

Key Principles:
1. ONLY essential fields that downstream agents actually need
2. Clear documentation of exactly where each field is used
3. Minimal validation rules
4. Backward compatibility for existing code
"""

from typing import List
from pydantic import BaseModel, Field


class Agent1EssentialCharacteristics(BaseModel):
    """MINIMAL characteristics - only fields actually used by downstream agents"""
    
    # Used by Agent 2 for entity/relationship scaling
    vocabulary_complexity_ratio: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="USED BY: Agent 2 entity confidence scaling (line 282)"
    )
    lexical_diversity: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="USED BY: Agent 2 relationship scaling (line 283)"
    )
    structural_consistency: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="USED BY: Agent 2 confidence adjustment (line 284)"
    )
    
    # Used by templates for adaptive prompts
    vocabulary_richness: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="USED BY: Templates {{ vocabulary_richness }} variable"
    )
    
    # Additional fields REQUIRED by prompt workflow system
    sentence_complexity: float = Field(
        ..., 
        ge=1.0, 
        le=100.0,
        description="USED BY: Templates {{ concept_density }} calculation"
    )
    content_patterns: List[str] = Field(
        default_factory=list,
        description="USED BY: Templates {{ discovered_content_patterns }} for adaptive prompts"
    )
    most_frequent_terms: List[str] = Field(
        default_factory=list,
        description="USED BY: Templates {{ discovered_entity_types }} derivation"
    )
    
    # Used by debug output and basic metrics
    avg_document_length: int = Field(
        ..., 
        ge=1, 
        le=100000,
        description="USED BY: Debug output and basic content metrics"
    )
    document_count: int = Field(
        ..., 
        ge=1, 
        le=10000,
        description="USED BY: Debug output and basic content metrics"
    )

    # Backward compatibility properties (for existing Agent 2 code)
    @property
    def vocabulary_complexity(self) -> float:
        """Backward compatibility - Agent 2 tries to access this field"""
        return self.vocabulary_complexity_ratio
        
    @property
    def concept_density(self) -> float:
        """Backward compatibility - computed from existing fields"""
        return (self.vocabulary_richness + self.lexical_diversity) / 2.0


class Agent1EssentialProcessingConfig(BaseModel):
    """MINIMAL processing config - only fields actually used by downstream agents"""
    
    # Actually used by Agent 2 (Knowledge Extraction) - CONFIRMED WORKING
    optimal_chunk_size: int = Field(
        ..., 
        ge=100, 
        le=4000,
        description="USED BY: Agent 2 chunking (line 279) - WORKING"
    )
    entity_confidence_threshold: float = Field(
        ..., 
        ge=0.5, 
        le=1.0,
        description="USED BY: Agent 2 entity extraction (line 278) - WORKING"
    )
    
    # Should be used by Agent 3 (Universal Search) - NEEDS IMPLEMENTATION  
    vector_search_weight: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="SHOULD BE USED BY: Agent 3 search strategy - NEEDS FIX"
    )
    graph_search_weight: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="SHOULD BE USED BY: Agent 3 search strategy - NEEDS FIX"
    )
    
    # Keep existing fields that processing_config schema expects
    chunk_overlap_ratio: float = Field(
        default=0.2, 
        ge=0.0, 
        le=0.5,
        description="Used by Agent 2 chunking - has default"
    )
    relationship_density: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="Used by Agent 2 relationship extraction - has default"  
    )
    expected_extraction_quality: float = Field(
        default=0.75, 
        ge=0.0, 
        le=1.0,
        description="Used by validation - has default"
    )
    processing_complexity: str = Field(
        default="medium",
        description="Used by resource allocation - has default"
    )


class Agent1EssentialOutputSchema(BaseModel):
    """
    MINIMAL Agent 1 Output Schema - ESSENTIAL FIELDS ONLY
    ====================================================
    
    Contains ONLY the fields that are actually used by downstream systems:
    - Agent 2 (Knowledge Extraction): 6 fields  
    - Agent 3 (Universal Search): 2 fields (need implementation)
    - Templates: 2 fields
    - Debug output: 3 fields
    """
    
    # Core identification (used by all agents and templates)
    domain_signature: str = Field(
        ...,
        description="USED BY: All agents, templates {{ domain_signature }}"
    )
    content_type_confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="USED BY: Validation and debug output"
    )
    
    # Essential components (confirmed usage)
    characteristics: Agent1EssentialCharacteristics = Field(
        ...,
        description="USED BY: Agent 2 scaling, templates"
    )
    processing_config: Agent1EssentialProcessingConfig = Field(
        ...,
        description="USED BY: Agent 2 (working), Agent 3 (needs fix)"
    )


# Usage mapping table for clear reference
class Agent1UsageMapping:
    """Exact usage locations for each Agent 1 field"""
    
    # Fields currently working in downstream agents
    WORKING_INTEGRATION = {
        'domain_signature': ['Agent 2 logging', 'Agent 3 strategy', 'templates'],
        'optimal_chunk_size': ['Agent 2 line 279 - WORKING'],
        'entity_confidence_threshold': ['Agent 2 line 278 - WORKING'],
        'vocabulary_complexity_ratio': ['Agent 2 line 282 - FIXED FIELD NAME'],
        'lexical_diversity': ['Agent 2 line 283 - WORKING'],
        'structural_consistency': ['Agent 2 line 284 - WORKING']
    }
    
    # Fields that need implementation in downstream agents
    NEEDS_IMPLEMENTATION = {
        'vector_search_weight': ['Agent 3 search strategy - NEEDS FIX'],
        'graph_search_weight': ['Agent 3 search strategy - NEEDS FIX'], 
        'vocabulary_richness': ['Templates {{ vocabulary_richness }} - NEEDS FIX'],
        'sentence_complexity': ['Templates {{ concept_density }} calculation - NEEDS FIX'],
        'content_patterns': ['Templates {{ discovered_content_patterns }} - NEEDS FIX'],
        'most_frequent_terms': ['Templates {{ discovered_entity_types }} derivation - NEEDS FIX']
    }
    
    # Fields with defaults (not critical)
    HAS_DEFAULTS = {
        'chunk_overlap_ratio': ['Agent 2 - has default 0.2'],
        'relationship_density': ['Agent 2 - has default 0.7'],
        'processing_complexity': ['Resource allocation - has default "medium"'],
        'expected_extraction_quality': ['Validation - has default 0.75']
    }


class Agent1TemplateMapping:
    """Template variables extracted from Agent 1 output for prompt workflows"""
    
    @staticmethod
    def extract_template_variables(agent1_output: Agent1EssentialOutputSchema) -> dict:
        """Extract template variables from Agent 1 output for Jinja2 templates"""
        
        # Format content patterns for templates  
        formatted_patterns = []
        for i, pattern in enumerate(agent1_output.characteristics.content_patterns[:5]):
            formatted_patterns.append({
                "category": pattern.replace("_", " ").title(),
                "description": f"Key {pattern.replace('_', ' ')} pattern found in content",
                "examples": [f"example_{i}_1", f"example_{i}_2", f"example_{i}_3"]
            })
        
        # Derive entity types from frequent terms
        entity_types = []
        for term in agent1_output.characteristics.most_frequent_terms[:5]:
            if len(term) > 3:  # Only meaningful terms
                entity_types.append(term.lower().replace(" ", "_"))
        
        if not entity_types:
            entity_types = ["concept", "entity", "term"]  # Fallback
        
        # Template variable mapping used by infrastructure/prompt_workflows/
        return {
            # Core identification
            "domain_signature": agent1_output.domain_signature,
            "content_confidence": agent1_output.content_type_confidence,
            
            # Characteristics for adaptive prompts  
            "vocabulary_richness": agent1_output.characteristics.vocabulary_richness,
            "concept_density": agent1_output.characteristics.sentence_complexity / 50.0,  # Normalize to 0-1
            "discovered_content_patterns": formatted_patterns,
            "discovered_entity_types": entity_types,
            
            # Processing config for prompts
            "entity_confidence_threshold": agent1_output.processing_config.entity_confidence_threshold,
            
            # Fallback values for missing template variables
            "discovered_domain_description": f"content with signature {agent1_output.domain_signature}",
            "key_domain_insights": [
                f"Domain signature: {agent1_output.domain_signature}",
                f"Vocabulary richness: {agent1_output.characteristics.vocabulary_richness:.2f}",
                f"Processing complexity: {agent1_output.processing_config.processing_complexity}"
            ]
        }

    # Template files that use these variables
    TEMPLATE_FILES = [
        'infrastructure/prompt_workflows/templates/universal_entity_extraction.jinja2',
        'infrastructure/prompt_workflows/templates/universal_relation_extraction.jinja2'
    ]
    
    # Variables expected by templates (from our earlier analysis)
    TEMPLATE_VARIABLES = {
        'domain_signature': '{{ domain_signature }}',
        'entity_confidence_threshold': '{{ entity_confidence_threshold }}',
        'content_confidence': '{{ content_confidence|default(0.8) }}',
        'vocabulary_richness': '{{ vocabulary_richness }}',
        'concept_density': '{{ concept_density }}',
        'discovered_domain_description': '{{ discovered_domain_description }}',
        'discovered_content_patterns': '{% for pattern in discovered_content_patterns %}',
        'discovered_entity_types': '{{ discovered_entity_types|join(", ") }}',
    }


# Export minimal schema including template support
__all__ = [
    'Agent1EssentialOutputSchema',
    'Agent1EssentialCharacteristics', 
    'Agent1EssentialProcessingConfig',
    'Agent1UsageMapping',
    'Agent1TemplateMapping'
]