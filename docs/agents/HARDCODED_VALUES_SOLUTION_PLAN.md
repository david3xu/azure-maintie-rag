# 🎯 Hardcoded Values Solution Plan

## Executive Summary

**Problem Scale**: 1,206 hardcoded values across 47 files forming interconnected dependency clusters
**Root Cause**: No centralized constants system + cascading hardcoded decision trees
**Solution**: Systematic cluster-based replacement with centralized learnable parameter system

## 🔍 Critical Findings from Graph Analysis

### Dependency Clusters Identified:

#### **🟥 Cluster 1: Entity Detection Chain** (Highest Priority)
```
agents/domain_intelligence/toolsets.py:160-171
    ↓ Hard-coded entities: 'Azure', 'Python', 'API', 'ML', 'programming', 'language'
    ↓ Propagates to:
agents/domain_intelligence/config_generator.py:264-278  
    ↓ Hard-coded relationship verbs: ['connect', 'contain', 'use', 'create', 'part', 'depend']
    ↓ Cached by:
agents/domain_intelligence/background_processor.py
    ↓ Affects all downstream agents
```

#### **🟨 Cluster 2: Threshold Decision Trees** (High Priority)
```
agents/domain_intelligence/toolsets.py:288-293
    ↓ vocabulary_diversity > 0.7 → base_threshold = 0.8
    ↓ vocabulary_diversity > 0.3 → base_threshold = 0.7  
    ↓ else → base_threshold = 0.6
    ↓ Cascades to:
agents/domain_intelligence/statistical_domain_analyzer.py:517-610
    ↓ entropy < 2.0 → return 0.2
    ↓ entropy < 4.0 → return 0.5
    ↓ entropy < 6.0 → return 0.8
    ↓ Affects:
agents/domain_intelligence/hybrid_domain_analyzer.py:416+
```

#### **🟨 Cluster 3: Confidence Calculation Formulas** (Medium Priority)
```
215 occurrences across knowledge_extraction/processors/
    ↓ entity_quality * 0.4 + relationship_quality * 0.3 + coverage_score * 0.2
    ↓ base_confidence * 0.6 + distance_factor * 0.3 + context_factor * 0.1
```

#### **🟩 Cluster 4: Infrastructure Constants** (Low Priority - Acceptable)
```
cache_ttl_seconds=3600, max_concurrent_chunks=5, etc.
```

## 🏗️ Solution Architecture

### Phase 1: Centralized Constants System

#### **1.1 Create Constants Schema**
```python
# agents/constants/learned_parameters.py
from pydantic import BaseModel
from typing import Dict, List, Optional
from pathlib import Path

class EntityPatterns(BaseModel):
    """Learned entity patterns from corpus analysis"""
    high_frequency_terms: List[str]  # Learned from statistical analysis
    domain_specific_entities: Dict[str, List[str]]  # Per-domain learned entities
    entity_confidence_distribution: Dict[str, float]  # Percentiles: 25th, 50th, 75th, 95th
    
class ThresholdDistributions(BaseModel):
    """Statistical distributions for threshold calculation"""
    vocabulary_diversity_percentiles: Dict[int, float]  # 25, 50, 75, 95
    entropy_score_distribution: Dict[str, float]  # mean, std, percentiles
    complexity_score_ranges: Dict[str, tuple]  # (min, max) for low/medium/high
    
class ConfidenceFormulas(BaseModel):
    """Learned coefficients for confidence calculations"""
    quality_assessment_weights: Dict[str, float]  # entity, relationship, coverage weights
    confidence_calculation_coefficients: Dict[str, float]  # distance, context, frequency weights
    threshold_adjustment_factors: Dict[str, float]  # domain-specific adjustments

class LearnedParameters(BaseModel):
    """Complete learned parameter set"""
    entity_patterns: EntityPatterns
    threshold_distributions: ThresholdDistributions
    confidence_formulas: ConfidenceFormulas
    learning_metadata: Dict[str, Any]  # Source data, confidence, timestamp
    
class SystemConstants(BaseModel):
    """Legitimate constants that should never be learned"""
    infrastructure_limits: Dict[str, int] = {
        "max_concurrent_chunks": 5,
        "cache_ttl_seconds": 3600,
        "thread_pool_workers": 4
    }
    mathematical_constants: Dict[str, float] = {
        "relationship_to_entity_ratio": 0.85,  # Mathematical relationship
        "chunk_overlap_percentage": 0.15       # Processing requirement
    }
    performance_targets: Dict[str, float] = {
        "target_response_time": 3.0,  # SLA requirement
        "background_processing_timeout": 300.0  # Infrastructure limit
    }
```

#### **1.2 Statistical Learning Engine**
```python
# agents/constants/statistical_learner.py
class StatisticalParameterLearner:
    """Learn parameters from corpus data instead of hardcoding"""
    
    async def learn_entity_patterns(self, corpus_paths: List[Path]) -> EntityPatterns:
        """Learn entity patterns from actual corpus data"""
        # 1. Analyze all available corpora
        # 2. Calculate term frequency distributions
        # 3. Identify high-frequency domain-specific terms
        # 4. Calculate confidence distributions from real data
        
    async def learn_threshold_distributions(self, corpus_paths: List[Path]) -> ThresholdDistributions:
        """Learn optimal thresholds from statistical analysis"""
        # 1. Calculate vocabulary diversity across all corpora
        # 2. Determine percentile-based thresholds (not hardcoded 0.7, 0.3)
        # 3. Learn entropy score distributions from actual content
        # 4. Map complexity scores to performance data
        
    async def learn_confidence_formulas(self, validation_data: List[Dict]) -> ConfidenceFormulas:
        """Learn optimal confidence calculation coefficients"""
        # 1. Use validation data with known correct extractions
        # 2. Optimize weights for quality assessment formulas
        # 3. Learn distance/context/frequency coefficients
        # 4. Calculate domain-specific adjustment factors
```

### Phase 2: Systematic Cluster Replacement

#### **2.1 Replace Entity Detection Chain (Cluster 1)**

**Before (toolsets.py:160-171)**:
```python
# ❌ HARDCODED - Domain assumptions
if 'Azure' in line:
    entities.append('Azure')
if 'Python' in line or 'python' in line:
    entities.append('Python')
# ... more hardcoded entities
```

**After**:
```python
# ✅ DATA-DRIVEN - Learn from corpus
async def extract_entities_data_driven(self, content: str, learned_params: LearnedParameters):
    entities = []
    for term in learned_params.entity_patterns.high_frequency_terms:
        if term.lower() in content.lower():
            # Use learned confidence distribution
            confidence = self._calculate_learned_confidence(term, content, learned_params)
            if confidence >= learned_params.threshold_distributions.entity_confidence_threshold:
                entities.append({
                    "name": term,
                    "confidence": confidence,
                    "method": "statistical_frequency_analysis"
                })
    return entities
```

#### **2.2 Replace Threshold Decision Trees (Cluster 2)**

**Before (toolsets.py:288-293)**:
```python
# ❌ HARDCODED - Arbitrary thresholds
if vocabulary_diversity > 0.7:
    base_threshold = 0.8
elif vocabulary_diversity > 0.3:
    base_threshold = 0.7
else:
    base_threshold = 0.6
```

**After**:
```python
# ✅ DATA-DRIVEN - Use learned percentiles
def calculate_threshold_from_distribution(self, vocabulary_diversity: float, learned_params: LearnedParameters):
    percentiles = learned_params.threshold_distributions.vocabulary_diversity_percentiles
    
    if vocabulary_diversity >= percentiles[75]:  # 75th percentile (was 0.7)
        return self._interpolate_threshold(vocabulary_diversity, percentiles, "high")
    elif vocabulary_diversity >= percentiles[25]:  # 25th percentile (was 0.3)
        return self._interpolate_threshold(vocabulary_diversity, percentiles, "medium")
    else:
        return self._interpolate_threshold(vocabulary_diversity, percentiles, "low")
```

#### **2.3 Replace Confidence Formulas (Cluster 3)**

**Before (knowledge_extraction processors)**:
```python
# ❌ HARDCODED - Arbitrary weights
overall_quality = (
    entity_quality * 0.4 +
    relationship_quality * 0.3 +
    coverage_score * 0.2 +
    consistency_score * 0.1
)
```

**After**:
```python
# ✅ DATA-DRIVEN - Learned optimal weights
def calculate_quality_score(self, metrics: Dict, learned_params: LearnedParameters):
    weights = learned_params.confidence_formulas.quality_assessment_weights
    
    return (
        metrics["entity_quality"] * weights["entity_weight"] +
        metrics["relationship_quality"] * weights["relationship_weight"] +
        metrics["coverage_score"] * weights["coverage_weight"] +
        metrics["consistency_score"] * weights["consistency_weight"]
    )
```

### Phase 3: Integration & Validation

#### **3.1 Updated Agent 1 Output Schema**
```python
# agents/models/domain_models.py - Enhanced
class EnhancedExtractionConfiguration(ExtractionConfiguration):
    """Agent 1 output with learned parameters"""
    learned_parameters: LearnedParameters
    system_constants: SystemConstants
    learning_confidence: float  # How confident we are in the learned params
    corpus_sources: List[str]   # What data was used for learning
    generation_method: str = "statistical_parameter_learning"
```

#### **3.2 Initialization Flow**
```python
# agents/domain_intelligence/toolsets.py
async def create_fully_learned_extraction_config(self, ctx, corpus_path: str):
    # 1. Learn parameters from corpus using statistical methods
    learner = StatisticalParameterLearner()
    learned_params = await learner.learn_all_parameters(corpus_path)
    
    # 2. Create configuration with learned parameters (no hardcoded values)
    config = EnhancedExtractionConfiguration(
        domain_name=Path(corpus_path).name,
        learned_parameters=learned_params,
        system_constants=SystemConstants(),  # Legitimate constants only
        learning_confidence=learned_params.learning_metadata["confidence"]
    )
    
    return config
```

## 📋 Implementation Plan

### **Sprint 1: Foundation (Week 1)**
- [ ] Create `agents/constants/` package with schema definitions
- [ ] Implement `StatisticalParameterLearner` class
- [ ] Create enhanced `ExtractionConfiguration` model
- [ ] Build parameter validation framework

### **Sprint 2: Entity Detection Cluster (Week 2)**
- [ ] Replace hardcoded entity lists in `toolsets.py:160-171`
- [ ] Update `config_generator.py:264-278` relationship verb detection
- [ ] Fix `background_processor.py` to use learned parameters
- [ ] Validate entity detection quality vs. original

### **Sprint 3: Threshold Clusters (Week 3)**
- [ ] Replace vocabulary diversity thresholds in `toolsets.py:288-293`
- [ ] Update entropy-based thresholds in `statistical_domain_analyzer.py`
- [ ] Fix chunk size calculation in `hybrid_domain_analyzer.py`
- [ ] Performance test learned vs. hardcoded thresholds

### **Sprint 4: Confidence Formulas (Week 4)**
- [ ] Replace quality assessment weights in knowledge extraction processors
- [ ] Update confidence calculation coefficients
- [ ] Implement learned formula validation
- [ ] End-to-end integration testing

### **Sprint 5: Validation & Optimization (Week 5)**
- [ ] Comprehensive testing with real data
- [ ] Performance benchmarking (learned vs. hardcoded)
- [ ] Documentation update
- [ ] Production deployment preparation

## 🎯 Success Metrics

### **Quality Metrics**:
- **Entity Detection Accuracy**: ≥95% (vs. current hardcoded baseline)
- **Threshold Optimization**: ≥10% improvement in precision/recall
- **Confidence Calibration**: ≤5% deviation from actual performance

### **Architecture Metrics**:
- **Hardcoded Value Reduction**: From 1,206 to <100 (legitimate constants only)
- **Dependency Coupling**: Eliminate cross-file hardcoded dependencies
- **Maintainability**: Single source of truth for all learnable parameters

### **Performance Metrics**:
- **Response Time**: Maintain <3s SLA
- **Learning Time**: <30s to learn parameters from new corpus
- **Memory Usage**: <20% increase vs. hardcoded approach

## 🚨 Risk Mitigation

### **High Risk: Learning Quality**
- **Risk**: Learned parameters perform worse than hardcoded values
- **Mitigation**: A/B testing framework, gradual rollout, fallback to hardcoded

### **Medium Risk: Performance Impact**
- **Risk**: Statistical learning adds latency
- **Mitigation**: Background learning, parameter caching, lazy loading

### **Low Risk: Integration Complexity**
- **Risk**: Breaking existing functionality during replacement
- **Mitigation**: Comprehensive test suite, feature flags, phased deployment

## 🎉 Expected Outcomes

1. **Zero Critical Hardcoded Values**: All entity lists, thresholds, and formulas learned from data
2. **Maintainable Architecture**: Single `LearnedParameters` schema for all agents
3. **Domain Agnostic**: System works with any domain without code changes
4. **Production Ready**: Robust error handling, validation, and monitoring
5. **Future Proof**: Easy to add new learnable parameters without code changes

This systematic approach addresses the **root cause** (interconnected hardcoded clusters) rather than symptom-by-symptom fixes, ensuring a scalable and maintainable solution.