# Hardcoded Values Audit - Systematic Analysis

## Executive Summary
Found hundreds of hardcoded numeric values across the agent codebase. Need systematic categorization into:
1. **Critical thresholds** (should be learned)
2. **Legitimate constants** (acceptable)
3. **Infrastructure limits** (configuration)

## Priority 1: Domain Intelligence Agent Critical Issues

### File: `/agents/domain_intelligence/toolsets.py`

#### ❌ HIGH PRIORITY - Should Be Learned:

**Line 187**: Semantic Pattern Confidence
```python
"confidence_score": 0.8,  # Should be calculated from statistical distribution
```

**Lines 288-293**: Vocabulary Diversity Thresholds
```python
if vocabulary_diversity > 0.7:  # Hardcoded threshold - should be learned percentile
    base_threshold = 0.8       # Hardcoded decision value
elif vocabulary_diversity > 0.3:  # Hardcoded threshold 
    base_threshold = 0.7       # Hardcoded decision value
else:
    base_threshold = 0.6       # Hardcoded fallback
```

**Lines 309-313**: Document Length Processing Ratios
```python
return min(1500, int(avg_doc_length * 0.4))  # Hardcoded ratio & limit
return min(1200, int(avg_doc_length * 0.6))  # Hardcoded ratio & limit  
return min(800, max(400, int(avg_doc_length * 0.8)))  # Multiple hardcoded values
```

**Lines 362+**: SLA Estimation Thresholds
```python
if complexity_score > 1.5:  # Hardcoded complexity threshold
    return 5.0              # Hardcoded SLA target
elif complexity_score > 0.8:  # Hardcoded threshold
    return 3.5              # Hardcoded SLA
else:
    return 2.5              # Hardcoded fallback
```

#### ✅ ACCEPTABLE - Mathematical Relationships:
```python
relationship_confidence_threshold=entity_threshold * 0.85,  # Ratio relationship
chunk_overlap=max(50, int(chunk_size * 0.15)),  # Percentage relationship
```

## Priority 2: Knowledge Extraction Agent Issues

### File: `/agents/knowledge_extraction/toolsets.py`

#### ❌ CRITICAL - Default Configuration Values:
```python
entity_confidence_threshold: float = 0.7          # Should come from Agent 1
relationship_confidence_threshold: float = 0.65   # Should come from Agent 1  
minimum_quality_score: float = 0.6               # Should be learned
```

#### ❌ HIGH PRIORITY - Quality Assessment Weights:
```python
entity_quality * 0.4 +           # Hardcoded weight
relationship_quality * 0.3 +     # Hardcoded weight
coverage_score * 0.2 +           # Hardcoded weight
consistency_score * 0.1          # Hardcoded weight
```

```python
if entity_quality < 0.7:         # Hardcoded threshold
if relationship_quality < 0.6:   # Hardcoded threshold
if coverage_score < 0.5:         # Hardcoded threshold
```

#### ❌ HIGH PRIORITY - Processing Parameters:
```python
"entity_confidence_threshold": 0.85,    # Mock optimization values
"relationship_confidence_threshold": 0.8,
"precision_improvement": 0.15,
"recall_improvement": 0.12
```

## Priority 3: Processor Files (Lower Priority)

### Pattern Found: Confidence Calculation Formulas
Multiple processor files contain hardcoded confidence calculation weights:

```python
# entity_processor.py
length_factor * 0.3 + position_factor * 0.2 + context_factor * 0.3 + case_factor * 0.2

# relationship_processor.py  
base_confidence * 0.4 + distance_factor * 0.3 + context_factor * 0.3
distance_factor * 0.7 + frequency_factor * 0.3
```

## Root Cause Analysis

### Why This Is Hard to Fix:

1. **Scattered Logic**: Thresholds embedded in decision trees throughout codebase
2. **No Centralized Schema**: Missing clear separation of learnable vs. constant values
3. **Missing Statistical Foundation**: No methods to learn thresholds from data distributions
4. **Circular Dependencies**: Some hardcoded values depend on other hardcoded values

## Proposed Solution Architecture

### 1. Create Centralized Constants Schema
```python
# agents/models/constants.py
class LearnableThresholds(BaseModel):
    """Values that should be learned from data"""
    vocabulary_diversity_percentiles: Dict[int, float]  # 25th, 50th, 75th, 95th
    complexity_score_distribution: Dict[str, float]     # mean, std, percentiles
    quality_score_weights: Dict[str, float]             # entity, relationship, coverage
    confidence_calculation_coefficients: Dict[str, float]

class SystemConstants(BaseModel):
    """Legitimate constants that should not be learned"""
    cache_ttl_seconds: int = 3600
    max_concurrent_chunks: int = 5
    mathematical_ratios: Dict[str, float] = {
        "relationship_to_entity_ratio": 0.85,
        "chunk_overlap_percentage": 0.15
    }
```

### 2. Statistical Learning Methods
```python
# agents/domain_intelligence/statistical_learner.py
class StatisticalThresholdLearner:
    async def learn_vocabulary_thresholds(self, all_corpora: List[Path]) -> Dict[int, float]:
        """Learn vocabulary diversity percentiles from actual data"""
        
    async def learn_quality_weights(self, validation_results: List[Dict]) -> Dict[str, float]:
        """Learn optimal quality assessment weights from performance data"""
        
    async def learn_confidence_coefficients(self, ground_truth: List[Dict]) -> Dict[str, float]:
        """Learn confidence calculation coefficients from validated extractions"""
```

### 3. Updated Agent 1 Output Schema
```python
class EnhancedExtractionConfiguration(ExtractionConfiguration):
    """Extended configuration with learned statistical parameters"""
    learned_thresholds: LearnableThresholds
    system_constants: SystemConstants
    statistical_confidence: float  # Confidence in the learned parameters
    data_source_metrics: Dict[str, Any]  # What data was used for learning
```

## Implementation Strategy

### Phase 1: Immediate Fixes (High Priority Issues)
1. Replace hardcoded vocabulary diversity thresholds in domain intelligence
2. Remove default configuration values from knowledge extraction  
3. Replace hardcoded quality assessment weights

### Phase 2: Statistical Learning Infrastructure
1. Implement statistical threshold learning methods
2. Create centralized constants management
3. Add data-driven confidence calculation

### Phase 3: Validation & Optimization  
1. Performance testing with learned vs. hardcoded values
2. Continuous learning from production data
3. Automated threshold optimization

## Next Steps

1. **TodoWrite**: Track systematic replacement of critical hardcoded values
2. **Focus**: Start with Domain Intelligence Agent vocabulary diversity logic
3. **Validate**: Test each replacement against real data
4. **Document**: Clear schema for what should be learned vs. constant

This systematic approach ensures we only replace values that should genuinely be data-driven while preserving legitimate constants and mathematical relationships.