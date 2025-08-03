# Agent 1: Data-Driven Configuration Schema Design

**Date**: August 3, 2025  
**Purpose**: Define comprehensive schema for Agent 1 to generate ALL configuration values from raw text data  
**Requirement**: Zero hardcoded values in Phase 1 implementation

## Overview

Agent 1 (Domain Intelligence Agent) must be **100% self-contained** and generate ALL configuration parameters from statistical analysis of raw text data. This document defines the required schema enhancements to eliminate all hardcoded values before Phase 1 implementation.

## Current State vs Required State

### ❌ Current Hardcoded Values in Config System

```python
# config/models.py - Lines 142-149 (HARDCODED)
recommended_chunk_size: int = Field(default=1000)        # ❌ Hardcoded
recommended_overlap: int = Field(default=200)            # ❌ Hardcoded
extraction_confidence_threshold: float = Field(default=0.7)  # ❌ Hardcoded
expected_response_time: float = Field(default=3.0)       # ❌ Hardcoded
cache_enabled: bool = Field(default=True)                # ❌ Hardcoded
cache_ttl: int = Field(default=3600)                     # ❌ Hardcoded

# config/models.py - Lines 370-387 (HARDCODED LOGIC)
def _determine_entity_type_from_content(self, entity: str) -> str:
    if any(term in entity for term in ["api", "endpoint"]):  # ❌ Hardcoded patterns
        return "api_interface"
    elif any(term in entity for term in ["function", "method"]):  # ❌ Hardcoded patterns
        return "code_element"
```

### ✅ Required Agent 1 Data-Driven Schema

```python
# Required: 100% Data-Driven Configuration Generation
class DataDrivenConfigurationSchema(BaseModel):
    """Schema for Agent 1 to generate ALL config values from statistical analysis"""
    
    # Learned from statistical analysis
    optimal_chunk_size: int = Field(..., description="Learned from document size distribution")
    optimal_overlap: int = Field(..., description="Learned from content coherence analysis")
    
    # Learned from validation performance
    confidence_threshold: float = Field(..., description="Learned from precision/recall curves")
    performance_sla: float = Field(..., description="Learned from processing time distribution")
    
    # Learned from content patterns
    entity_classification_rules: Dict[str, List[str]] = Field(..., description="Learned from frequency analysis")
    relationship_patterns: List[str] = Field(..., description="Learned from dependency parsing")
    
    # Learned from operational data
    cache_strategy: Dict[str, Any] = Field(..., description="Learned from access patterns")
    optimization_parameters: Dict[str, float] = Field(..., description="Learned from performance data")
```

## Required Schema Components

### 1. Document Analysis Schema

```python
class DocumentAnalysisResult(BaseModel):
    """Results from statistical analysis of corpus documents"""
    
    # Document structure analysis
    avg_document_length: int
    document_size_distribution: Dict[str, float]  # percentiles
    optimal_chunk_boundaries: List[int]  # learned from content coherence
    
    # Content complexity analysis
    vocabulary_size: int
    avg_sentence_length: float
    concept_density: float  # concepts per 100 words
    technical_term_ratio: float
    
    # Processing performance data
    processing_time_by_size: Dict[int, float]  # document_size -> avg_time
    memory_usage_by_size: Dict[int, float]     # document_size -> avg_memory
    
    @computed_field
    @property
    def optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size from document analysis"""
        # Find chunk size that optimizes processing speed vs coherence
        sizes = list(self.processing_time_by_size.keys())
        times = list(self.processing_time_by_size.values())
        
        # Use mathematical optimization to find sweet spot
        return self._optimize_chunk_size(sizes, times, self.document_size_distribution)
    
    @computed_field
    @property  
    def optimal_overlap(self) -> int:
        """Calculate optimal overlap from content coherence analysis"""
        # Analyze sentence boundaries and concept continuity
        return max(50, int(self.avg_sentence_length * 2))  # At least 2 sentences
```

### 2. Performance Learning Schema

```python
class PerformanceLearningResult(BaseModel):
    """Results from performance analytics on historical data"""
    
    # Response time analysis
    response_time_distribution: Dict[str, float]  # percentiles
    sla_compliance_rate: float
    performance_bottlenecks: List[str]
    
    # Confidence threshold optimization
    precision_by_threshold: Dict[float, float]  # threshold -> precision
    recall_by_threshold: Dict[float, float]     # threshold -> recall
    f1_by_threshold: Dict[float, float]         # threshold -> f1_score
    
    # Cache effectiveness analysis
    cache_hit_rates: Dict[str, float]  # operation -> hit_rate
    cache_latency_reduction: Dict[str, float]  # operation -> latency_reduction
    
    @computed_field
    @property
    def optimal_confidence_threshold(self) -> float:
        """Find confidence threshold that maximizes F1 score"""
        if not self.f1_by_threshold:
            return 0.7  # Fallback only if no data
        
        max_f1_threshold = max(self.f1_by_threshold.items(), key=lambda x: x[1])[0]
        return max_f1_threshold
    
    @computed_field
    @property
    def learned_sla_target(self) -> float:
        """Learn SLA target from 95th percentile of historical performance"""
        if not self.response_time_distribution:
            return 3.0  # Fallback only if no data
        
        return self.response_time_distribution.get("p95", 3.0)
```

### 3. Content Pattern Learning Schema

```python
class ContentPatternLearningResult(BaseModel):
    """Results from content pattern analysis for entity classification"""
    
    # Entity pattern discovery
    token_frequency_analysis: Dict[str, int]
    entity_clustering_results: Dict[str, List[str]]  # cluster_name -> entities
    pattern_confidence_scores: Dict[str, float]      # pattern -> confidence
    
    # Relationship pattern discovery
    dependency_patterns: List[Dict[str, str]]  # syntactic patterns
    cooccurrence_matrix: Dict[str, Dict[str, float]]  # word -> word -> strength
    relationship_types: List[str]  # discovered relationship types
    
    # Classification rule generation
    @computed_field
    @property
    def learned_entity_classification_rules(self) -> Dict[str, List[str]]:
        """Generate entity classification rules from pattern analysis"""
        rules = {}
        
        for cluster_name, entities in self.entity_clustering_results.items():
            # Extract common patterns from entities in each cluster
            patterns = self._extract_common_patterns(entities)
            rules[cluster_name] = patterns
        
        return rules
    
    def _extract_common_patterns(self, entities: List[str]) -> List[str]:
        """Extract common patterns from entity list using statistical analysis"""
        # Analyze word co-occurrence and create pattern rules
        patterns = []
        for entity in entities:
            words = entity.lower().split()
            for word in words:
                if self.token_frequency_analysis.get(word, 0) > 10:  # Frequent terms
                    patterns.append(word)
        
        return list(set(patterns))  # Remove duplicates
```

### 4. Unified Agent 1 Configuration Generator

```python
class Agent1ConfigurationGenerator:
    """Agent 1: Self-contained configuration generator from raw text data"""
    
    async def generate_complete_configuration(
        self, 
        raw_data_path: str
    ) -> CompleteDataDrivenConfiguration:
        """Generate 100% data-driven configuration from raw text analysis"""
        
        # Step 1: Analyze document structure and performance
        doc_analysis = await self._analyze_document_corpus(raw_data_path)
        
        # Step 2: Learn performance patterns from historical data
        perf_learning = await self._analyze_performance_patterns(raw_data_path)
        
        # Step 3: Discover content patterns and entity types
        pattern_learning = await self._analyze_content_patterns(raw_data_path)
        
        # Step 4: Generate unified configuration
        return CompleteDataDrivenConfiguration(
            # Document processing parameters (learned)
            chunk_size=doc_analysis.optimal_chunk_size,
            chunk_overlap=doc_analysis.optimal_overlap,
            
            # Quality thresholds (learned)
            entity_confidence_threshold=perf_learning.optimal_confidence_threshold,
            relationship_confidence_threshold=perf_learning.optimal_confidence_threshold * 0.9,
            
            # Performance parameters (learned)
            response_time_sla=perf_learning.learned_sla_target,
            cache_ttl=self._calculate_optimal_cache_ttl(perf_learning.cache_hit_rates),
            
            # Entity classification (learned)
            entity_classification_rules=pattern_learning.learned_entity_classification_rules,
            relationship_types=pattern_learning.relationship_types,
            
            # Processing optimization (learned)
            parallel_processing_threshold=doc_analysis.optimal_chunk_size * 2,
            memory_optimization_level=self._calculate_memory_optimization(doc_analysis),
            
            # Metadata
            generated_timestamp=datetime.now(),
            source_analysis=f"Statistical analysis of {len(doc_analysis.processing_time_by_size)} documents",
            learning_confidence=self._calculate_learning_confidence(doc_analysis, perf_learning, pattern_learning)
        )
```

## Integration with Phase 1 Plan

### Required Updates to Phase 1 Implementation

**Before starting Phase 1 tool co-location, we must:**

1. **Update config/models.py** - Remove all hardcoded defaults, replace with data-driven generation
2. **Implement Agent1ConfigurationGenerator** - Create self-contained config generation from raw data
3. **Update Phase 1 toolsets** - Use Agent 1 generated configuration instead of hardcoded values
4. **Add validation** - Ensure Phase 1 toolsets receive 100% data-driven configuration

### Updated Phase 1 Tool Co-Location with Data-Driven Config

```python
# agents/domain_intelligence/toolsets.py - UPDATED for 100% data-driven
class DomainIntelligenceToolset(Toolset):
    
    @tool
    async def generate_complete_domain_config(
        self, 
        ctx: RunContext[DomainDeps], 
        raw_data_path: str
    ) -> CompleteDataDrivenConfiguration:
        """Generate 100% data-driven configuration from raw text analysis"""
        
        # Use Agent 1 configuration generator (NO hardcoded values)
        config_generator = Agent1ConfigurationGenerator()
        complete_config = await config_generator.generate_complete_configuration(raw_data_path)
        
        # Validate that NO hardcoded values exist
        validation_result = self._validate_zero_hardcoded_values(complete_config)
        if not validation_result.is_valid:
            raise ValueError(f"Hardcoded values detected: {validation_result.violations}")
        
        return complete_config
    
    @tool  
    async def create_extraction_config(
        self, 
        ctx: RunContext[DomainDeps], 
        complete_config: CompleteDataDrivenConfiguration
    ) -> ExtractionConfiguration:
        """Create extraction config using 100% learned parameters"""
        
        return ExtractionConfiguration(
            # ✅ ALL VALUES LEARNED FROM DATA
            entity_confidence_threshold=complete_config.entity_confidence_threshold,
            relationship_confidence_threshold=complete_config.relationship_confidence_threshold,
            chunk_size=complete_config.chunk_size,
            chunk_overlap=complete_config.chunk_overlap,
            entity_classification_rules=complete_config.entity_classification_rules,
            relationship_types=complete_config.relationship_types,
            response_time_sla=complete_config.response_time_sla,
            cache_ttl=complete_config.cache_ttl
        )
```

## Implementation Priority

### Critical Path for Phase 1

1. **BEFORE Phase 1 Day 1**: Implement complete Agent 1 data-driven schema
2. **Phase 1 Day 1**: Update toolsets to use Agent 1 generated configuration  
3. **Phase 1 Validation**: Verify zero hardcoded values in all toolsets

### Validation Requirements

```python
class HardcodedValueValidator:
    """Validate that NO hardcoded values exist in configuration"""
    
    def validate_configuration(self, config: Any) -> ValidationResult:
        """Ensure configuration contains zero hardcoded values"""
        
        violations = []
        
        # Check for hardcoded numerical values
        if hasattr(config, 'chunk_size') and config.chunk_size == 1000:
            violations.append("chunk_size appears to be hardcoded default (1000)")
        
        if hasattr(config, 'confidence_threshold') and config.confidence_threshold == 0.7:
            violations.append("confidence_threshold appears to be hardcoded default (0.7)")
        
        # Check for hardcoded string patterns
        if hasattr(config, 'entity_classification_rules'):
            for rule_type, patterns in config.entity_classification_rules.items():
                if "api" in patterns and "endpoint" in patterns:
                    violations.append(f"Entity rule '{rule_type}' contains hardcoded patterns")
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            confidence=1.0 if len(violations) == 0 else 0.0
        )
```

## Conclusion

**Yes, we absolutely need to design and implement this comprehensive Agent 1 schema BEFORE Phase 1 implementation.** The current config system still contains hardcoded values that violate the "Data-Driven Everything" principle.

**Recommended Action:**
1. Implement the complete Agent 1 data-driven schema outlined above
2. Update the existing config system to be 100% data-driven
3. Then proceed with Phase 1 tool co-location using the fully data-driven configuration

This ensures Phase 1 toolsets will be truly "self-contained" as Agent 1 will generate ALL configuration values from raw text data analysis.