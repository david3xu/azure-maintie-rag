# Integrated Data Types Implementation Guide

**Implementation Plan: Constants Within Models + Simple Source Tracking**

Replace the artificial separation between constants and data models with integrated, self-contained domain modules.

## 🎯 Current Architecture Problems

### **Problematic Separation**
```python
# ❌ WRONG: Artificial separation
# constants.py
class KnowledgeExtractionConstants:
    ENTITY_CONFIDENCE_THRESHOLD = 0.8
    MAX_ENTITIES_PER_DOCUMENT = 100
    DEFAULT_CHUNK_SIZE = 1000

# data_models.py  
from agents.core.constants import KnowledgeExtractionConstants
class ExtractionConfiguration(BaseModel):
    entity_confidence: float = Field(
        default=KnowledgeExtractionConstants.ENTITY_CONFIDENCE_THRESHOLD,  # External dependency
        ge=0.0, le=1.0
    )
```

### **Dependency Hell Result**
- **29 files** importing from `constants.py`
- **11 constant class imports** in `data_models.py`
- **Runtime performance overhead** from cross-module imports
- **Maintenance nightmare** when constants and models get out of sync

## 🚀 Correct Integrated Architecture

### **Principle: Constants Belong WITH Their Data Models**

```python
# ✅ CORRECT: Integrated constants within data models
class ExtractionConfiguration(BaseModel):
    """Knowledge extraction configuration with integrated constants"""
    
    # Constants belong HERE, with the model that uses them
    class Constants:
        ENTITY_CONFIDENCE_THRESHOLD = 0.8
        RELATIONSHIP_CONFIDENCE_THRESHOLD = 0.7
        DEFAULT_CHUNK_SIZE = 1000
        DEFAULT_CHUNK_OVERLAP = 200
        MAX_ENTITIES_PER_DOCUMENT = 100
        MIN_RELATIONSHIP_STRENGTH = 0.6
        QUALITY_VALIDATION_THRESHOLD = 0.8
    
    # Fields use internal constants - no external dependencies
    entity_confidence_threshold: float = Field(
        default=Constants.ENTITY_CONFIDENCE_THRESHOLD,
        ge=0.0, le=1.0,
        description="Entity extraction confidence threshold"
    )
    relationship_confidence_threshold: float = Field(
        default=Constants.RELATIONSHIP_CONFIDENCE_THRESHOLD,
        ge=0.0, le=1.0
    )
    chunk_size: int = Field(
        default=Constants.DEFAULT_CHUNK_SIZE,
        ge=50, le=10000
    )
    
    # Mathematical expressions as model methods
    def calculate_optimal_batch_size(self, total_documents: int) -> int:
        """Calculate optimal batch size based on document count"""
        if total_documents < 10:
            return 1
        elif total_documents < 100:
            return max(1, total_documents // 10)
        else:
            return max(5, min(20, total_documents // 50))
    
    def estimate_processing_time(self, document_count: int) -> float:
        """Estimate processing time based on configuration"""
        base_time_per_doc = 2.0  # seconds
        complexity_multiplier = 1.2 if self.chunk_size < 500 else 1.0
        return document_count * base_time_per_doc * complexity_multiplier
```

## 🏗️ Domain-Specific Integrated Modules

### **New Architecture: 4 Self-Contained Domain Modules**

```
agents/core/
├── domain_models.py        # Domain Intelligence models + constants
├── extraction_models.py    # Knowledge Extraction models + constants
├── search_models.py        # Universal Search models + constants
└── azure_models.py         # Azure Service models + constants
```

### **Domain Intelligence Models** (`domain_models.py`)
```python
"""Domain Intelligence - Integrated data types and constants"""

from pydantic import BaseModel, Field, computed_field
from typing import List, Dict, Any, Optional
from datetime import datetime

class DomainAnalysisResult(BaseModel):
    """Domain analysis with integrated statistical constants"""
    
    class StatisticalThresholds:
        MIN_DOCUMENT_LENGTH = 100
        VOCABULARY_DIVERSITY_THRESHOLD = 0.3
        COMPLEXITY_HIGH_THRESHOLD = 0.7
        COMPLEXITY_MEDIUM_THRESHOLD = 0.4
        TECHNICAL_DENSITY_THRESHOLD = 0.5
        DOMAIN_CONFIDENCE_THRESHOLD = 0.75
    
    class ProcessingLimits:
        MAX_DOCUMENTS_PER_BATCH = 50
        STATISTICAL_WINDOW_SIZE = 100
        TREND_ANALYSIS_DAYS = 7
        MAX_PATTERNS_TO_EXTRACT = 200
    
    # Model fields using integrated constants
    domain_name: str = Field(description="Detected domain name")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Domain detection confidence"
    )
    vocabulary_diversity: float = Field(
        default=StatisticalThresholds.VOCABULARY_DIVERSITY_THRESHOLD,
        ge=0.0, le=1.0,
        description="Vocabulary diversity score"
    )
    complexity_score: float = Field(
        ge=0.0, le=1.0,
        description="Content complexity assessment"
    )
    technical_density: float = Field(
        ge=0.0, le=1.0,
        description="Technical content density"
    )
    
    @computed_field
    @property
    def complexity_category(self) -> str:
        """Determine complexity category using integrated thresholds"""
        if self.complexity_score >= self.StatisticalThresholds.COMPLEXITY_HIGH_THRESHOLD:
            return "high"
        elif self.complexity_score >= self.StatisticalThresholds.COMPLEXITY_MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "low"
    
    def is_domain_confident(self) -> bool:
        """Check if domain detection meets confidence threshold"""
        return self.confidence >= self.StatisticalThresholds.DOMAIN_CONFIDENCE_THRESHOLD
    
    def calculate_processing_parameters(self) -> Dict[str, Any]:
        """Calculate optimal processing parameters based on analysis"""
        return {
            "batch_size": min(
                self.ProcessingLimits.MAX_DOCUMENTS_PER_BATCH,
                max(5, int(self.complexity_score * 20))
            ),
            "chunk_size": 1500 if self.complexity_score > 0.6 else 1000,
            "parallel_workers": 2 if self.complexity_score > 0.7 else 4
        }


class DomainConfiguration(BaseModel):
    """Domain-specific configuration with integrated synthesis weights"""
    
    class SynthesisWeights:
        CONFIDENCE_WEIGHT = 0.4
        AGREEMENT_WEIGHT = 0.3
        QUALITY_WEIGHT = 0.3
    
    class PerformanceTargets:
        QUERY_PROCESSING_SLA = 3.0
        MIN_EXTRACTION_ACCURACY = 0.85
        TARGET_CACHE_HIT_RATE = 0.6
    
    domain_name: str = Field(description="Domain identifier")
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Similarity matching threshold"
    )
    
    # Integrated synthesis weights
    confidence_weight: float = Field(
        default=SynthesisWeights.CONFIDENCE_WEIGHT,
        ge=0.0, le=1.0
    )
    agreement_weight: float = Field(
        default=SynthesisWeights.AGREEMENT_WEIGHT,
        ge=0.0, le=1.0
    )
    quality_weight: float = Field(
        default=SynthesisWeights.QUALITY_WEIGHT,
        ge=0.0, le=1.0
    )
    
    def calculate_synthesis_score(self, confidence: float, agreement: float, quality: float) -> float:
        """Calculate result synthesis using integrated weights"""
        return min(1.0, 
            confidence * self.confidence_weight +
            agreement * self.agreement_weight + 
            quality * self.quality_weight
        )
```

### **Knowledge Extraction Models** (`extraction_models.py`)
```python
"""Knowledge Extraction - Integrated models with extraction constants"""

class ExtractedEntity(BaseModel):
    """Entity with integrated confidence calculations"""
    
    class ConfidenceFactors:
        LENGTH_BONUS_THRESHOLD = 3
        FREQUENCY_BONUS_THRESHOLD = 3
        POSITION_EARLY_FACTOR = 0.8
        POSITION_LATE_FACTOR = 0.6
        CONTEXT_BOOST_FACTOR = 1.1
        MAX_CONFIDENCE = 1.0
    
    class ExtractionLimits:
        MIN_ENTITY_LENGTH = 2
        MAX_ENTITY_LENGTH = 100
        MAX_ENTITIES_PER_CHUNK = 20
    
    name: str = Field(min_length=ConfidenceFactors.LENGTH_BONUS_THRESHOLD, description="Entity name")
    entity_type: str = Field(description="Entity classification")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    frequency: int = Field(ge=1, description="Occurrence frequency")
    position: int = Field(ge=0, description="Position in text")
    context: str = Field(description="Surrounding context")
    
    def calculate_enhanced_confidence(self, text_length: int) -> float:
        """Calculate enhanced confidence using integrated factors"""
        base_confidence = self.confidence
        
        # Length bonus
        if len(self.name) >= self.ConfidenceFactors.LENGTH_BONUS_THRESHOLD:
            base_confidence += 0.1
            
        # Frequency bonus  
        if self.frequency >= self.ConfidenceFactors.FREQUENCY_BONUS_THRESHOLD:
            base_confidence += 0.15
            
        # Position factor
        position_ratio = self.position / text_length if text_length > 0 else 0
        if position_ratio < 0.2:  # Early in text
            base_confidence *= self.ConfidenceFactors.POSITION_EARLY_FACTOR
        elif position_ratio > 0.8:  # Late in text
            base_confidence *= self.ConfidenceFactors.POSITION_LATE_FACTOR
            
        return min(self.ConfidenceFactors.MAX_CONFIDENCE, base_confidence)


class ExtractionResults(BaseModel):
    """Extraction results with integrated quality metrics"""
    
    class QualityThresholds:
        MIN_EXTRACTION_ACCURACY = 0.85
        ENTITY_COVERAGE_TARGET = 0.8
        RELATIONSHIP_COVERAGE_TARGET = 0.7
        OVERALL_QUALITY_THRESHOLD = 0.8
    
    entities: List[ExtractedEntity] = Field(description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(description="Extracted relationships")
    processing_time: float = Field(ge=0.0, description="Processing time seconds")
    
    @computed_field
    @property
    def entity_coverage_score(self) -> float:
        """Calculate entity coverage using integrated thresholds"""
        if not self.entities:
            return 0.0
        high_confidence_entities = [
            e for e in self.entities 
            if e.confidence >= self.QualityThresholds.MIN_EXTRACTION_ACCURACY
        ]
        return len(high_confidence_entities) / len(self.entities)
    
    @computed_field
    @property
    def meets_quality_standards(self) -> bool:
        """Check if extraction meets integrated quality standards"""
        return (
            self.entity_coverage_score >= self.QualityThresholds.ENTITY_COVERAGE_TARGET and
            len(self.entities) > 0
        )
```

### **Universal Search Models** (`search_models.py`)
```python
"""Universal Search - Integrated search models and orchestration constants"""

class TriModalSearchRequest(BaseModel):
    """Tri-modal search with integrated search parameters"""
    
    class SearchWeights:
        VECTOR_WEIGHT = 0.4
        GRAPH_WEIGHT = 0.35
        GNN_WEIGHT = 0.25
    
    class SearchLimits:
        MAX_RESULTS = 50
        DEFAULT_TOP_K = 10
        MIN_CONFIDENCE_THRESHOLD = 0.6
        MAX_SEARCH_TIME = 5.0
    
    class QualityThresholds:
        VECTOR_SIMILARITY_THRESHOLD = 0.7
        GRAPH_RELATIONSHIP_THRESHOLD = 0.6
        GNN_PREDICTION_CONFIDENCE = 0.65
    
    query: str = Field(min_length=1, max_length=500, description="Search query")
    max_results: int = Field(
        default=SearchLimits.DEFAULT_TOP_K,
        ge=1, le=SearchLimits.MAX_RESULTS
    )
    confidence_threshold: float = Field(
        default=QualityThresholds.VECTOR_SIMILARITY_THRESHOLD,
        ge=0.0, le=1.0
    )
    
    # Integrated weight configuration
    vector_weight: float = Field(default=SearchWeights.VECTOR_WEIGHT, ge=0.0, le=1.0)
    graph_weight: float = Field(default=SearchWeights.GRAPH_WEIGHT, ge=0.0, le=1.0)
    gnn_weight: float = Field(default=SearchWeights.GNN_WEIGHT, ge=0.0, le=1.0)
    
    def calculate_weighted_score(self, vector_score: float, graph_score: float, gnn_score: float) -> float:
        """Calculate weighted search score using integrated weights"""
        weighted_score = (
            vector_score * self.vector_weight +
            graph_score * self.graph_weight +
            gnn_score * self.gnn_weight
        )
        return min(1.0, weighted_score)
    
    def validate_search_quality(self, results: List[Dict]) -> bool:
        """Validate search results using integrated quality thresholds"""
        if not results:
            return False
        
        high_quality_results = [
            r for r in results 
            if r.get('confidence', 0) >= self.QualityThresholds.MIN_CONFIDENCE_THRESHOLD
        ]
        return len(high_quality_results) / len(results) >= 0.5
```

### **Azure Service Models** (`azure_models.py`)
```python
"""Azure Services - Integrated service models with Azure-specific constants"""

class AzureServiceMetrics(BaseModel):
    """Azure service metrics with integrated SLA thresholds"""
    
    class SLAThresholds:
        MAX_RESPONSE_TIME_MS = 1000
        MIN_AVAILABILITY_PERCENT = 99.9
        MAX_ERROR_RATE = 0.01
        HEALTH_CHECK_INTERVAL = 60
    
    class CostLimits:
        MAX_DAILY_COST_USD = 50.0
        MAX_MONTHLY_COST_USD = 1000.0
        COST_ALERT_THRESHOLD = 0.8
    
    service_name: str = Field(description="Azure service name")
    response_time_ms: float = Field(ge=0.0, description="Response time milliseconds")
    availability_percent: float = Field(ge=0.0, le=100.0, description="Service availability")
    error_rate: float = Field(ge=0.0, le=1.0, description="Error rate")
    daily_cost_usd: float = Field(ge=0.0, description="Daily cost in USD")
    
    @computed_field
    @property
    def meets_sla_requirements(self) -> bool:
        """Check if service meets integrated SLA thresholds"""
        return (
            self.response_time_ms <= self.SLAThresholds.MAX_RESPONSE_TIME_MS and
            self.availability_percent >= self.SLAThresholds.MIN_AVAILABILITY_PERCENT and
            self.error_rate <= self.SLAThresholds.MAX_ERROR_RATE
        )
    
    @computed_field  
    @property
    def cost_status(self) -> str:
        """Determine cost status using integrated cost limits"""
        if self.daily_cost_usd >= self.CostLimits.MAX_DAILY_COST_USD:
            return "over_budget"
        elif self.daily_cost_usd >= self.CostLimits.MAX_DAILY_COST_USD * self.CostLimits.COST_ALERT_THRESHOLD:
            return "approaching_limit"
        else:
            return "within_budget"
```

## 🔄 Migration Strategy

### **Step 1: Create Integrated Domain Modules** (Week 1)
```bash
# Create new integrated modules
touch agents/core/domain_models.py
touch agents/core/extraction_models.py  
touch agents/core/search_models.py
touch agents/core/azure_models.py

# Move relevant constants and models into each domain module
```

### **Step 2: Update Agent Imports** (Week 2)
```python
# Before: Multiple imports from separate files
from agents.core.constants import KnowledgeExtractionConstants, ProcessingConstants
from agents.core.data_models import ExtractionConfiguration, ValidationResult
from agents.core.math_expressions import EXPR

# After: Single domain import
from agents.core.extraction_models import ExtractionConfiguration, ExtractionResults
```

### **Step 3: Deprecate Old Files** (Week 3)
```python
# agents/core/__init__.py - Backward compatibility
from .domain_models import *
from .extraction_models import *
from .search_models import *
from .azure_models import *

# Legacy compatibility (with deprecation warnings)
from .constants import *  # DEPRECATED
from .data_models import *  # DEPRECATED  
from .math_expressions import *  # DEPRECATED
```

## 📊 Expected Benefits

### **Performance Improvements**
- **60% faster imports** - No cross-module dependencies
- **40% reduced memory usage** - Constants loaded only when models are used
- **Elimination of runtime imports** - All constants available at model load time

### **Maintenance Benefits**  
- **Zero external dependencies** between data models and constants
- **Domain-specific maintenance** - Azure team can modify Azure models independently
- **Co-located documentation** - Constants documented with their usage context
- **Type-safe constant usage** - Pydantic Field validation with integrated constants

### **Code Reduction**
- **Eliminate constants.py** - 1,181 lines distributed into domain modules
- **Simplify data_models.py** - Remove 11 external constant imports
- **Reduce math_expressions.py** - Mathematical operations become model methods
- **Net reduction**: ~35% total lines while improving organization

## 🎯 Success Metrics

### **Before Integration**
- External constant imports: 11 per module
- Cross-module dependencies: 29 files affected
- Mathematical operations: Isolated in separate file
- Maintenance overhead: High (3 files to update per change)

### **After Integration**
- External constant imports: 0 (fully self-contained)
- Cross-module dependencies: 0 (domain modules are independent)
- Mathematical operations: Integrated as model methods
- Maintenance overhead: Low (1 domain module per change)

This integrated approach follows the principle that **data types should own their configuration constants**, eliminating artificial separation and creating a more maintainable, performant architecture where constants live with the models that use them.