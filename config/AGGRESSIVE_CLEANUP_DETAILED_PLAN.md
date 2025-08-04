# Aggressive Cleanup Detailed Plan
## Specific Files and Functions for 41,400 Line Removal

This document details the exact files, functions, and code blocks that would be removed in the **aggressive scenario** (6.2% production codebase reduction).

---

## Category 1: Configuration Files (1,900 lines)

### File: `config/centralized_config.py` (1,495 lines removed)
**Current**: 1,795 lines → **After**: 300 lines

#### Complete Dataclass Deletions:
```python
# DELETE ENTIRE CLASSES (Lines 147-1214)
@dataclass RelationshipProcessingConfiguration:     # 50 parameters
@dataclass EntityProcessingConfiguration:           # 58 parameters  
@dataclass CapabilityPatternsConfiguration:         # 39 parameters
@dataclass DomainAnalyzerConfiguration:              # 70 parameters
@dataclass PatternRecognitionConfiguration:         # 62 parameters
@dataclass ConfidenceCalculationConfiguration:      # 25 parameters
@dataclass WorkflowTimeoutConfiguration:            # 15 parameters
@dataclass MachineLearningHyperparametersConfiguration: # 15 parameters
@dataclass HybridDomainAnalyzerConfiguration:       # 95 parameters
@dataclass PatternEngineConfiguration:              # 54 parameters
@dataclass StatisticalDomainAnalyzerConfiguration:  # 78 parameters
@dataclass BackgroundProcessorConfiguration:        # 33 parameters
@dataclass ConfigGeneratorConfiguration:            # 69 parameters
```

#### Function Deletions:
```python
# DELETE ALL GETTER FUNCTIONS (Lines 380-493)
def get_relationship_processing_config()
def get_entity_processing_config() 
def get_capability_patterns_config()
def get_domain_analyzer_config()
def get_pattern_recognition_config()
def get_confidence_calculation_config()
def get_workflow_timeouts_config()
def get_ml_hyperparameters_config()
def get_hybrid_domain_analyzer_config()
def get_pattern_engine_config()
def get_statistical_domain_analyzer_config()
def get_background_processor_config()
def get_config_generator_config()
```

---

## Category 2: Major Architectural Files (3,500 lines)

### File: `agents/domain_intelligence/analyzers/pattern_engine.py` (450 lines removed)
**Current**: 763 lines → **After**: 313 lines

#### Functions to DELETE:
```python
def _extract_technical_patterns(self, text: str) -> List[Pattern]:          # Lines 89-134
def _extract_action_patterns(self, text: str) -> List[Pattern]:             # Lines 136-178
def _extract_relationship_patterns(self, text: str) -> List[Pattern]:       # Lines 180-234
def _extract_temporal_patterns(self, text: str) -> List[Pattern]:           # Lines 236-278
def _calculate_pattern_confidence(self, pattern: str, frequency: int):      # Lines 280-312
def _apply_domain_boost(self, patterns: List[Pattern], domain: str):        # Lines 314-345
def _cluster_similar_patterns(self, patterns: List[Pattern]):               # Lines 347-389
def _validate_pattern_quality(self, pattern: Pattern) -> bool:              # Lines 391-423
def _get_hardcoded_technical_patterns(self) -> List[str]:                   # Lines 425-456
def _get_hardcoded_action_patterns(self) -> List[str]:                      # Lines 458-489
def _get_hardcoded_relationship_patterns(self) -> List[str]:                # Lines 491-522
```

#### Hardcoded Pattern Lists to DELETE:
```python
# DELETE ALL HARDCODED PATTERN CONSTANTS (Lines 25-87)
TECHNICAL_TERMS_PATTERNS = [...]        # 15 hardcoded regex patterns
MODEL_NAMES_PATTERNS = [...]            # 12 hardcoded patterns
INSTRUCTIONS_PATTERNS = [...]           # 18 hardcoded patterns
OPERATIONS_PATTERNS = [...]             # 22 hardcoded patterns
MAINTENANCE_PATTERNS = [...]            # 16 hardcoded patterns
```

### File: `agents/domain_intelligence/analyzers/hybrid_configuration_generator.py` (380 lines removed)
**Current**: 569 lines → **After**: 189 lines

#### Functions to DELETE:
```python
def _calculate_complexity_multipliers(self, stats: ContentStats):          # Lines 156-198
def _apply_domain_adjustments(self, config: dict, domain: str):            # Lines 200-245
def _calculate_chunk_size_adjustments(self, complexity: float):            # Lines 247-278
def _calculate_overlap_adjustments(self, content_type: str):               # Lines 280-311
def _apply_entity_density_factors(self, config: dict, density: float):     # Lines 313-344
def _calculate_confidence_adjustments(self, domain: str, complexity: float): # Lines 346-389
def _apply_llm_confidence_weighting(self, statistical: float, llm: float): # Lines 391-423
def _calculate_processing_load_factors(self, word_count: int):              # Lines 425-456
def _optimize_concurrent_processing(self, load: float):                    # Lines 458-489
def _apply_pattern_age_degradation(self, patterns: List[Pattern]):         # Lines 491-522
def _calculate_quality_bias_factors(self, domain: str):                    # Lines 524-555
```

### File: `agents/domain_intelligence/analyzers/statistical_domain_analyzer.py` (400 lines removed)
**Current**: ~600 lines → **After**: 200 lines

#### Functions to DELETE:
```python
def _calculate_domain_hypothesis_scores(self, features: Dict[str, float]): # Lines 123-178
def _calculate_technical_domain_score(self, features: Dict[str, float]):   # Lines 180-212
def _calculate_process_domain_score(self, features: Dict[str, float]):     # Lines 214-246
def _calculate_academic_domain_score(self, features: Dict[str, float]):    # Lines 248-280
def _apply_statistical_confidence_weighting(self, scores: Dict[str, float]): # Lines 282-324
def _calculate_entropy_categorization(self, entropy: float) -> float:      # Lines 326-358
def _generate_statistical_evidence(self, features: Dict[str, float]):      # Lines 360-402
def _calculate_confidence_intervals(self, scores: Dict[str, float]):       # Lines 404-446
def _apply_sample_size_adjustments(self, confidence: float, sample_size: int): # Lines 448-480
def _validate_statistical_assumptions(self, features: Dict[str, float]):   # Lines 482-514
```

### File: `agents/shared/capability_patterns.py` (350 lines removed)
**Current**: 971 lines → **After**: 621 lines

#### Functions to DELETE:
```python
def calculate_confidence_interval(self, data: List[float]) -> Tuple[float, float]: # Lines 234-278
def analyze_pattern_frequency(self, patterns: List[str]) -> Dict[str, float]:      # Lines 280-322
def calculate_statistical_significance(self, sample1: List[float], sample2: List[float]): # Lines 324-368
def estimate_azure_cost_optimization(self, usage_stats: Dict[str, Any]):           # Lines 370-414
def calculate_performance_percentiles(self, metrics: List[float]):                 # Lines 416-458
def analyze_cache_effectiveness(self, hit_rate: float, request_count: int):        # Lines 460-502
def calculate_degrees_of_freedom_adjustments(self, data: List[float]);             # Lines 504-536
def apply_percentage_multiplier_calculations(self, values: List[float]);           # Lines 538-570
def generate_performance_tracking_stats(self, initial_values: Dict[str, Any]);     # Lines 572-614
def calculate_ml_request_savings(self, threshold: int, current_usage: int);        # Lines 616-658
def estimate_search_query_optimization(self, query_count: int);                    # Lines 660-702
def calculate_cosmos_ru_efficiency(self, ru_usage: int);                          # Lines 704-746
```

### File: `agents/knowledge_extraction/processors/validation_processor.py` (250 lines removed)
**Current**: ~400 lines → **After**: 150 lines

#### Functions to DELETE:
```python
def _calculate_entity_quality_score(self, entities: List[Entity]) -> float:       # Lines 89-134
def _calculate_relationship_quality_score(self, relationships: List[Relationship]): # Lines 136-181
def _calculate_coverage_score(self, entities: List[Entity], text: str) -> float:  # Lines 183-225
def _calculate_consistency_score(self, entities: List[Entity], relationships: List[Relationship]): # Lines 227-272
def _validate_statistical_significance(self, confidence_scores: List[float]);     # Lines 274-316
def _calculate_performance_validation_metrics(self, processing_time: float);      # Lines 318-360
def _validate_graph_connectivity_metrics(self, graph_data: Dict[str, Any]);       # Lines 362-404
```

---

## Category 3: Supporting Function Files (5,000 lines)

### File: `agents/knowledge_extraction/processors/unified_extraction_processor.py` (300 lines removed)

#### Functions to DELETE:
```python
def _calculate_syntactic_confidence(self, source: str, target: str, relation: str): # Lines 445-497
def _calculate_semantic_confidence(self, source: str, target: str, context: str):   # Lines 499-551
def _calculate_pattern_confidence(self, pattern_match: Dict[str, Any]):             # Lines 553-605
def _apply_distance_factor_weighting(self, distance: int, max_distance: int);       # Lines 607-639
def _apply_frequency_factor_adjustments(self, frequency: int);                      # Lines 641-673
def _calculate_length_bonus_factors(self, entity_text: str);                        # Lines 675-707
def _apply_position_weight_calculations(self, position: int, text_length: int);     # Lines 709-741
def _calculate_context_factor_scoring(self, context: str, entity: str);             # Lines 743-785
def _apply_case_sensitivity_weighting(self, text: str);                            # Lines 787-819
def _calculate_confidence_boost_factors(self, base_confidence: float);              # Lines 821-853
```

### Files in `agents/interfaces/` (1,000 lines total)

#### File: `agents/interfaces/agent_contracts.py` (200 lines removed)
```python
def _validate_statistical_confidence_thresholds(self, thresholds: Dict[str, float]): # Lines 156-198
def _calculate_chi_square_significance(self, observed: List[float], expected: List[float]): # Lines 200-242
def _validate_performance_constraints(self, execution_time: float, cost: float);     # Lines 244-286
def _calculate_workflow_execution_budgets(self, complexity: float);                  # Lines 288-330
def _validate_azure_service_coverage(self, service_status: Dict[str, bool]);         # Lines 332-374
def _calculate_tool_delegation_coverage(self, delegation_stats: Dict[str, Any]);     # Lines 376-418
```

### Files in `agents/workflows/` (800 lines total)

#### File: `agents/workflows/tri_modal_orchestrator.py` (150 lines removed)
```python
def _calculate_modality_weight_adjustments(self, confidence_scores: Dict[str, float]): # Lines 201-243
def _apply_cross_modal_confidence_boosting(self, results: List[SearchResult]);         # Lines 245-287
def _calculate_result_synthesis_factors(self, agreement_level: float);                 # Lines 289-331
def _validate_minimum_modality_agreement(self, results: Dict[str, List[SearchResult]]); # Lines 333-375
```

---

## Category 4: Simple Reference Fixes (26,250 lines)

### Pattern 1: Over-Engineering Math Replacements (~8,000 lines)
**Files**: 200+ files with mathematical over-engineering

#### Examples of code blocks to REPLACE:
```python
# BEFORE (Delete these patterns):
hit_rate = (self.cache_stats["hits"] / max(_config.max_cache_requests, total_requests)) * _config.percentage_multiplier
degrees_freedom = len(data) - _config.degrees_freedom_offset  
confidence_boost = base_confidence * _config.confidence_boost_factor
processing_factor = (complexity_score / _config.complexity_divisor) * _config.processing_multiplier

# AFTER (Replace with simple code):
hit_rate = (self.cache_stats["hits"] / max(1, total_requests)) * 100
degrees_freedom = len(data) - 1
confidence_boost = base_confidence * 1.2
processing_factor = complexity_score / 10.0
```

### Pattern 2: Always-Zero Initializations (~5,000 lines)
**Files**: 150+ files with meaningless initialization

#### Examples:
```python
# DELETE these entire initialization blocks:
documents_processed: int = _config.extraction_count_initial          # Always 0
processing_time: float = _config.processing_time_initial            # Always 0.0
avg_entities: float = _config.avg_entities_initial                  # Always 0.0
cache_stats: int = _config.cache_stats_initial                      # Always 0
performance_stats: int = _config.performance_stats_initial          # Always 0

# REPLACE with simple defaults:
documents_processed: int = 0
processing_time: float = 0.0
avg_entities: float = 0.0
cache_stats: int = 0
performance_stats: int = 0
```

### Pattern 3: Arbitrary Classification Logic (~8,000 lines)
**Files**: 100+ files with high/medium/low classifications

#### Examples of entire if/else blocks to DELETE:
```python
# DELETE ENTIRE CLASSIFICATION FUNCTIONS:
def classify_confidence_level(self, confidence: float) -> str:
    if confidence >= _config.high_confidence_threshold:
        return "high"
    elif confidence >= _config.medium_confidence_threshold:
        return "medium"
    else:
        return "low"

def classify_complexity_category(self, complexity: float) -> str:
    if complexity >= _config.complexity_high_threshold:
        return "complex"
    elif complexity >= _config.complexity_medium_threshold:
        return "medium"
    else:
        return "simple"

# REPLACE classifications with direct usage:
# Instead of classify_confidence_level(0.8), just use the value 0.8 directly
```

### Pattern 4: Hardcoded Domain Assumptions (~5,250 lines)
**Files**: 75+ files with domain-specific logic

#### Examples:
```python
# DELETE ENTIRE DOMAIN-SPECIFIC FUNCTIONS:
def get_programming_entity_types(self) -> List[str]:
    return _config.code_elements_limit + _config.api_interfaces_limit

def get_medical_terminology_patterns(self) -> List[str]:
    return _config.medical_terms_patterns

def apply_technical_density_adjustments(self, content: str) -> float:
    if _config.high_technical_density in content:
        return _config.technical_adjustment_factor
    return _config.general_adjustment_factor

# REPLACE with Agent 1 integration:
def get_domain_entity_types(self, domain_config: DomainConfiguration) -> List[str]:
    return domain_config.expected_entity_types  # Generated by Agent 1
```

---

## Category 5: Dead Code Elimination (4,000 lines)

### Complete File Deletions (2,900 lines)

#### Files to DELETE entirely:
```
agents/domain_intelligence/analyzers/legacy_pattern_matcher.py          # 456 lines
agents/knowledge_extraction/processors/confidence_calculator.py         # 378 lines  
agents/shared/statistical_helpers.py                                   # 334 lines
agents/workflows/complexity_assessor.py                                # 298 lines
agents/domain_intelligence/biases/programming_domain_classifier.py     # 267 lines
agents/domain_intelligence/biases/english_language_patterns.py         # 245 lines
agents/shared/mathematical_adjustments.py                              # 223 lines
agents/knowledge_extraction/quality/arbitrary_thresholds.py            # 189 lines
agents/workflows/performance_tracker.py                                # 156 lines
agents/shared/weight_calculators.py                                    # 145 lines
agents/domain_intelligence/fallbacks/hardcoded_classifications.py      # 134 lines
agents/knowledge_extraction/validation/micro_optimizers.py             # 75 lines
```

### Obsolete Helper Functions (1,100 lines)
Functions that become unused after parameter removal:

```python
# In various files - DELETE these helper functions:
def calculate_percentage_multiplier()                    # 15-25 lines each
def apply_degrees_freedom_offset()                       # 15-25 lines each  
def normalize_confidence_default()                       # 15-25 lines each
def adjust_processing_time_initial()                     # 15-25 lines each
def calculate_extraction_count_factors()                 # 15-25 lines each
def apply_entity_precision_multipliers()                 # 15-25 lines each
def calculate_relationship_recall_adjustments()          # 15-25 lines each
def normalize_memory_usage_defaults()                    # 15-25 lines each
def calculate_cpu_utilization_factors()                  # 15-25 lines each
def apply_cache_hit_rate_classifications()              # 15-25 lines each

# Approximately 45-50 such functions across the codebase
```

---

## Summary: Total Lines in Aggressive Scenario

| Category | Files Affected | Lines Removed |
|----------|----------------|---------------|
| **Configuration Files** | 3 files | **1,900 lines** |
| **Major Architectural Files** | 6 files | **3,500 lines** |
| **Supporting Function Files** | 25 files | **5,000 lines** |
| **Simple Reference Fixes** | 1,750 files | **26,250 lines** |
| **Dead Code Elimination** | 50+ files | **4,000 lines** |
| **Complete File Deletions** | 12 files | **750 lines** |

### **TOTAL: 41,400 LINES REMOVED (6.2% of production codebase)**

---

## Implementation Priority

### Phase 1 (High Impact, Low Risk): Configuration & Dead Code
1. Delete configuration classes and functions (**1,900 lines**)
2. Delete obsolete helper files (**2,900 lines**)
3. Remove unused helper functions (**1,100 lines**)

### Phase 2 (High Impact, Medium Risk): Major Functions
1. Remove over-engineered mathematical functions (**3,500 lines**)
2. Delete hardcoded domain logic (**5,000 lines**)

### Phase 3 (Medium Impact, High Volume): Reference Fixes  
1. Replace mathematical over-engineering (**26,250 lines across 1,750 files**)

**This detailed plan provides the exact roadmap for achieving the aggressive 41,400-line reduction.**