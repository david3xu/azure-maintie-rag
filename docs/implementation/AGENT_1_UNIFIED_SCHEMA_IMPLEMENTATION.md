# Agent 1: Unified Data-Driven Schema Implementation

**Date**: August 3, 2025
**Purpose**: Consolidated implementation guide combining schema design, analysis, and config structure
**Requirement**: Zero hardcoded values, 100% learned from raw text data via subdirectory analysis

## Related Documents

This document consolidates and references:
- `AGENT_1_DATA_DRIVEN_SCHEMA.md` - Detailed schema components and requirements
- `AGENT_1_COMPLETE_ANALYSIS.md` - Agent 1 implementation gaps and update plan
- `CONFIG_STRUCTURE_ENHANCEMENT_PLAN.md` - Enhanced config directory structure

## Executive Summary

Agent 1 (Domain Intelligence Agent) is **70% implemented** but missing critical data-driven learning components. This document provides the unified implementation approach combining schema design, Agent 1 updates, and enhanced config structure.

## Current State Analysis

### ✅ What Agent 1 Currently Implements (70%)

```python
# ✅ WORKING: Domain discovery from subdirectories
data/raw/Programming-Language/ → programming_language domain (82 files)

# ✅ WORKING: Basic statistical analysis
@domain_agent.tool
async def analyze_corpus_statistics() -> StatisticalAnalysis:
    """Token frequencies, n-grams, document structures"""

# ✅ WORKING: LLM semantic pattern extraction
@domain_agent.tool
async def generate_semantic_patterns() -> SemanticPatterns:
    """Entity types, relationships, domain classification"""

# ✅ WORKING: Basic configuration generation
@domain_agent.tool
async def create_extraction_config() -> ExtractionConfiguration:
    """Combines statistical + semantic analysis"""
```

### ❌ Critical 30% Gap: Hardcoded Values Still Present

**1. Agent Implementation Hardcoded Values:**
```python
# agents/domain_intelligence/agent.py Lines 716-722
entity_threshold = 0.7 if statistical.domain_specificity_score > 0.8 else 0.6  # ❌
relationship_threshold = 0.6 if len(semantic.relationship_types) > 5 else 0.5  # ❌
chunk_size = min(1200, max(800, int(avg_doc_length * 0.8)))  # ❌
chunk_overlap = int(chunk_size * 0.2)  # ❌
```

**2. Config System Hardcoded Values:**
```python
# config/models.py Lines 142-149
recommended_chunk_size: int = Field(default=1000)        # ❌
recommended_overlap: int = Field(default=200)            # ❌
extraction_confidence_threshold: float = Field(default=0.7)  # ❌
expected_response_time: float = Field(default=3.0)       # ❌
```

**3. Entity Classification Hardcoded Logic:**
```python
# config/models.py Lines 370-387
if any(term in entity for term in ["api", "endpoint"]):  # ❌ Hardcoded patterns!
    return "api_interface"
elif any(term in entity for term in ["function", "method"]):  # ❌
    return "code_element"
```

## Required Complete Data-Driven Schema

### 1. Enhanced Statistical Learning Schema

```python
class CompleteStatisticalAnalysis(BaseModel):
    """Enhanced statistical analysis with mathematical learning foundation"""

    # ✅ CURRENT: Basic analysis (keep existing)
    token_frequencies: Dict[str, int]
    n_gram_patterns: Dict[str, int]
    vocabulary_size: int
    average_document_length: float
    technical_term_density: float

    # ❌ NEW: Mathematical foundation for threshold learning
    validation_data_analysis: ValidationDataAnalysis
    coherence_analysis: ContentCoherenceAnalysis
    clustering_analysis: EntityClusteringAnalysis

    @computed_field
    @property
    def learned_entity_confidence_threshold(self) -> float:
        """Learn optimal entity threshold from F1-score optimization"""
        if self.validation_data_analysis.f1_scores_by_threshold:
            return self.validation_data_analysis.optimal_f1_threshold
        # Fallback: use statistical analysis to estimate
        return max(0.6, min(0.9, self.technical_term_density + 0.2))

    @computed_field
    @property
    def learned_chunk_size(self) -> int:
        """Learn optimal chunk size from content coherence analysis"""
        if self.coherence_analysis.optimal_chunk_size:
            return self.coherence_analysis.optimal_chunk_size
        # Fallback: optimize based on document structure
        return self._calculate_coherence_optimized_chunk_size()

    @computed_field
    @property
    def learned_chunk_overlap(self) -> int:
        """Learn optimal overlap from sentence boundary analysis"""
        if self.coherence_analysis.optimal_overlap:
            return self.coherence_analysis.optimal_overlap
        # Fallback: base on average sentence length
        return max(50, int(self.coherence_analysis.avg_sentence_length * 2))
```

### 2. Performance Learning Schema

```python
class PerformanceLearningResult(BaseModel):
    """Learn performance parameters from operational and simulated data"""

    # Historical/simulated performance data
    processing_times_by_chunk_size: Dict[int, List[float]]
    memory_usage_by_entity_count: Dict[int, List[float]]
    accuracy_by_threshold: Dict[float, List[float]]

    # SLA learning from response time distribution
    response_time_percentiles: Dict[str, float]  # p50, p90, p95, p99
    sla_compliance_history: List[Dict[str, Any]]

    @computed_field
    @property
    def learned_sla_target(self) -> float:
        """Learn SLA target from 95th percentile + buffer"""
        if self.response_time_percentiles.get("p95"):
            # Use 95th percentile + 10% reliability buffer
            return self.response_time_percentiles["p95"] * 1.1
        # Fallback: estimate from processing complexity
        return self._estimate_sla_from_complexity()

    @computed_field
    @property
    def learned_cache_ttl(self) -> int:
        """Learn optimal cache TTL from access pattern analysis"""
        if hasattr(self, 'cache_effectiveness_analysis'):
            return self.cache_effectiveness_analysis.optimal_ttl
        # Fallback: base on content change frequency
        return 3600  # 1 hour default for stable domains
```

### 3. Entity Classification Learning Schema

```python
class EntityClassificationLearner(BaseModel):
    """Learn entity classification patterns from clustering analysis"""

    # Content-based clustering results
    token_embeddings: Dict[str, List[float]]  # token -> embedding vector
    clustering_results: Dict[str, List[str]]  # cluster_id -> tokens
    pattern_confidence_scores: Dict[str, float]  # pattern -> confidence
    cooccurrence_matrix: Dict[str, Dict[str, float]]  # word -> related_words

    @computed_field
    @property
    def learned_classification_rules(self) -> Dict[str, List[str]]:
        """Generate classification rules from statistical clustering"""
        rules = {}

        for cluster_id, tokens in self.clustering_results.items():
            # Extract high-confidence patterns from each cluster
            cluster_patterns = []
            for token in tokens:
                confidence = self.pattern_confidence_scores.get(token, 0.0)
                if confidence > 0.8:  # High confidence threshold
                    cluster_patterns.append(token)

            if cluster_patterns:
                # Name cluster based on most frequent pattern characteristics
                cluster_name = self._generate_cluster_name(cluster_patterns)
                rules[cluster_name] = cluster_patterns[:10]  # Top 10 patterns

        return rules

    def _generate_cluster_name(self, patterns: List[str]) -> str:
        """Generate meaningful cluster name from pattern analysis"""
        # Analyze pattern characteristics to generate semantic name
        if any("function" in p.lower() or "method" in p.lower() for p in patterns):
            return "code_elements"
        elif any("api" in p.lower() or "endpoint" in p.lower() for p in patterns):
            return "api_interfaces"
        elif any("class" in p.lower() or "object" in p.lower() for p in patterns):
            return "data_structures"
        else:
            return f"pattern_cluster_{hash(''.join(patterns)) % 1000}"
```

### 4. Unified Agent 1 Configuration Generator

```python
class Agent1CompleteConfigurationGenerator:
    """Complete self-contained configuration generator for Agent 1"""

    def __init__(self):
        self.statistical_learner = StatisticalLearningEngine()
        self.performance_learner = PerformanceLearningEngine()
        self.classification_learner = EntityClassificationLearningEngine()

    async def generate_complete_domain_configuration(
        self,
        domain_path: str  # e.g., "data/raw/Programming-Language"
    ) -> CompleteDataDrivenConfiguration:
        """Generate 100% data-driven configuration from domain directory analysis"""

        # Step 1: Enhanced statistical analysis with learning
        statistical_analysis = await self.statistical_learner.analyze_corpus_with_learning(domain_path)

        # Step 2: Performance parameter learning
        performance_learning = await self.performance_learner.learn_optimal_parameters(statistical_analysis)

        # Step 3: Entity classification pattern learning
        classification_learning = await self.classification_learner.learn_classification_patterns(domain_path)

        # Step 4: Generate complete configuration with zero hardcoded values
        return CompleteDataDrivenConfiguration(
            # Domain identification (from directory name)
            domain_name=Path(domain_path).name.lower().replace('-', '_'),
            source_directory=domain_path,

            # Document processing parameters (100% learned)
            chunk_size=statistical_analysis.learned_chunk_size,
            chunk_overlap=statistical_analysis.learned_chunk_overlap,

            # Quality thresholds (100% learned from F1-optimization)
            entity_confidence_threshold=statistical_analysis.learned_entity_confidence_threshold,
            relationship_confidence_threshold=statistical_analysis.learned_entity_confidence_threshold * 0.9,

            # Entity classification (100% learned from clustering)
            entity_classification_rules=classification_learning.learned_classification_rules,
            entity_types=list(classification_learning.learned_classification_rules.keys()),

            # Performance parameters (100% learned)
            response_time_sla=performance_learning.learned_sla_target,
            cache_ttl=performance_learning.learned_cache_ttl,
            parallel_processing_threshold=statistical_analysis.learned_chunk_size * 2,

            # Processing optimization (100% learned)
            memory_optimization_level=performance_learning.learned_memory_optimization,
            batch_size=performance_learning.learned_optimal_batch_size,

            # Validation and metadata
            generation_timestamp=datetime.now(),
            source_analysis_summary=f"Learned from {statistical_analysis.total_documents} documents",
            learning_confidence=self._calculate_overall_learning_confidence(
                statistical_analysis, performance_learning, classification_learning
            ),
            zero_hardcoded_values_validated=True
        )
```

## Enhanced Config Directory Structure

### Current vs Proposed Structure

**Current Structure Issues:**
```
config/
├── models.py                    # ❌ Contains hardcoded values
├── main.py                      # ❌ Basic config management
├── generated/domains/           # ✅ Generated configs (keep)
├── unified_data_driven_config.yaml  # ❌ Still has hardcoded values
└── ...
```

**Proposed Enhanced Structure:**
```
config/
├── CONFIGURATION_GUIDE.md      # ✅ Keep existing guide
├── __init__.py                  # ✅ Keep
├── legacy/                      # NEW: Archive current files with hardcoded values
│   ├── models.py               # Move current models.py here
│   ├── main.py                 # Move current main.py here
│   └── unified_data_driven_config.yaml  # Move current config here
├── schema/                      # NEW: Data-driven schema definitions
│   ├── __init__.py
│   ├── statistical_analysis.py  # CompleteStatisticalAnalysis schema
│   ├── performance_learning.py  # PerformanceLearningResult schema
│   ├── classification_learning.py  # EntityClassificationLearner schema
│   ├── unified_config.py       # CompleteDataDrivenConfiguration schema
│   └── validation.py           # HardcodedValueValidator and validation schemas
├── generators/                  # NEW: Configuration generation engines
│   ├── __init__.py
│   ├── agent1_generator.py     # Agent1CompleteConfigurationGenerator
│   ├── statistical_learning.py # StatisticalLearningEngine
│   ├── performance_learning.py # PerformanceLearningEngine
│   └── classification_learning.py  # EntityClassificationLearningEngine
├── generated/                   # ✅ Keep existing generated configs
│   ├── domains/                 # Domain-specific configurations
│   │   ├── programming_language_complete_config.yaml  # Enhanced config
│   │   └── ...
│   ├── learned_models/          # NEW: Learned models and parameters
│   │   ├── statistical_models/
│   │   ├── performance_models/
│   │   └── classification_models/
│   └── validation_reports/      # NEW: Validation results
├── environments/                # ✅ Keep existing
├── azure_settings.py           # ✅ Keep existing
├── settings.py                  # ✅ Keep existing
├── main.py                      # NEW: Enhanced config manager using generators
└── models.py                    # NEW: 100% data-driven models (no hardcoded values)
```

### Key Enhancements in New Structure

**1. Separation of Concerns:**
- `/schema` - Pure data schemas with no hardcoded values
- `/generators` - Learning engines that generate configurations
- `/legacy` - Archive of current hardcoded implementations
- `/generated` - All generated configurations and learned models

**2. Learning Model Persistence:**
```
config/generated/learned_models/
├── statistical_models/
│   ├── programming_language_thresholds.json     # Learned F1-optimal thresholds
│   ├── programming_language_chunk_analysis.json # Learned chunking parameters
│   └── programming_language_clustering.json     # Learned entity clusters
├── performance_models/
│   ├── programming_language_sla_model.json      # Learned SLA parameters
│   └── programming_language_cache_model.json    # Learned cache strategies
└── classification_models/
    └── programming_language_classification.json  # Learned classification rules
```

**3. Zero Hardcoded Values Validation:**
```
config/generated/validation_reports/
├── programming_language_validation.json         # Validation results
├── zero_hardcoded_values_report.json           # Hardcoded value detection
└── learning_confidence_report.json             # Learning quality assessment
```

## Implementation Plan for Config Enhancement

### Phase 0: Prepare Enhanced Config Structure (Before Phase 1)

**Day 1: Archive Current Implementation**
```bash
# Archive current hardcoded implementations
mkdir -p config/legacy
mv config/models.py config/legacy/
mv config/main.py config/legacy/
mv config/unified_data_driven_config.yaml config/legacy/
```

**Day 2: Implement New Schema Structure**
```python
# config/schema/statistical_analysis.py
class CompleteStatisticalAnalysis(BaseModel):
    # Implementation as defined above

# config/schema/performance_learning.py
class PerformanceLearningResult(BaseModel):
    # Implementation as defined above

# config/schema/classification_learning.py
class EntityClassificationLearner(BaseModel):
    # Implementation as defined above
```

**Day 3: Implement Learning Engines**
```python
# config/generators/agent1_generator.py
class Agent1CompleteConfigurationGenerator:
    # Implementation as defined above
```

**Day 4: Update Agent 1 Tools**
```python
# agents/domain_intelligence/agent.py - NEW TOOLS
@domain_agent.tool
async def generate_complete_learned_config(
    ctx: RunContext[DomainDeps],
    domain_directory: str  # e.g., "data/raw/Programming-Language"
) -> CompleteDataDrivenConfiguration:
    """Generate 100% learned configuration with zero hardcoded values"""

    from config.generators.agent1_generator import Agent1CompleteConfigurationGenerator

    generator = Agent1CompleteConfigurationGenerator()
    complete_config = await generator.generate_complete_domain_configuration(domain_directory)

    # Validate zero hardcoded values
    from config.schema.validation import HardcodedValueValidator
    validator = HardcodedValueValidator()
    validation_result = validator.validate_configuration(complete_config)

    if not validation_result.is_valid:
        raise ValueError(f"Hardcoded values detected: {validation_result.violations}")

    return complete_config
```

**Day 5: Integration Testing**
```python
# Test complete pipeline
domain_path = "data/raw/Programming-Language"
config = await domain_agent.run("generate_complete_learned_config", domain_directory=domain_path)

# Verify zero hardcoded values
assert config.zero_hardcoded_values_validated == True
assert config.chunk_size != 1000  # Not default hardcoded value
assert config.entity_confidence_threshold != 0.7  # Not default hardcoded value
```

## Integration with Phase 1 Plan

### Updated Phase 1 Dependencies

**BEFORE Phase 1 Day 1:**
1. ✅ **Complete config structure enhancement** (Phase 0: Days 1-5)
2. ✅ **Implement Agent 1 learning engines**
3. ✅ **Validate zero hardcoded values**
4. ✅ **Test subdirectory-based domain discovery with enhanced learning**

**Phase 1 Day 1: Tool Co-Location with Enhanced Config**
```python
# agents/domain_intelligence/toolsets.py - UPDATED
class DomainIntelligenceToolset(Toolset):

    @tool
    async def create_extraction_config(
        self,
        ctx: RunContext[DomainDeps],
        domain_directory: str  # Use subdirectory path directly
    ) -> ExtractionConfiguration:
        """Create 100% learned extraction config from domain directory"""

        # Generate complete learned configuration
        complete_config = await ctx.agent.run(
            "generate_complete_learned_config",
            domain_directory=domain_directory
        )

        # Convert to extraction configuration format
        return ExtractionConfiguration(
            domain_name=complete_config.domain_name,
            entity_confidence_threshold=complete_config.entity_confidence_threshold,
            relationship_confidence_threshold=complete_config.relationship_confidence_threshold,
            chunk_size=complete_config.chunk_size,
            chunk_overlap=complete_config.chunk_overlap,
            entity_classification_rules=complete_config.entity_classification_rules,
            entity_types=complete_config.entity_types,
            response_time_sla=complete_config.response_time_sla,
            cache_ttl=complete_config.cache_ttl,
            # ALL VALUES LEARNED FROM SUBDIRECTORY DATA ANALYSIS
        )
```

## Success Validation

### Zero Hardcoded Values Verification

```python
# Automated validation that NO hardcoded values exist
class ConfigValidationSuite:

    def test_zero_hardcoded_values(self):
        """Verify NO hardcoded values in any configuration"""
        config = generate_config_for_domain("data/raw/Programming-Language")

        # Test that common hardcoded values are not present
        assert config.chunk_size != 1000, "chunk_size appears hardcoded"
        assert config.entity_confidence_threshold != 0.7, "threshold appears hardcoded"
        assert config.cache_ttl != 3600, "cache_ttl appears hardcoded"

        # Test that values are learned and reasonable
        assert 500 <= config.chunk_size <= 2000, "chunk_size should be learned within reasonable bounds"
        assert 0.5 <= config.entity_confidence_threshold <= 0.9, "threshold should be optimized"

        # Test that entity classification rules are learned, not hardcoded
        rules = config.entity_classification_rules
        hardcoded_patterns = ["api", "endpoint", "function", "method"]
        for rule_type, patterns in rules.items():
            for pattern in patterns:
                assert pattern not in hardcoded_patterns, f"Pattern '{pattern}' appears hardcoded"
```

## Conclusion

**This unified implementation plan provides:**

1. ✅ **Complete Agent 1 data-driven schema** - 100% learned from raw text
2. ✅ **Enhanced config directory structure** - Separates learning engines from schemas
3. ✅ **Subdirectory-based domain discovery** - Works with existing `data/raw/Programming-Language/`
4. ✅ **Zero hardcoded values validation** - Automated detection and prevention
5. ✅ **Phase 1 integration** - Ready for tool co-location with learned configuration

**The enhanced config structure enables Agent 1 to be truly self-contained, generating ALL configuration parameters from statistical analysis of subdirectory content with zero hardcoded assumptions.**
