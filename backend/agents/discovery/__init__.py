"""
Discovery System Components - Essential domain discovery and pattern learning.

This module contains the core discovery components that power the PydanticAI
domain detection and adaptation tools. These components are used internally
by the PydanticAI tools and provide:

- Zero-configuration domain adaptation  
- Advanced pattern learning and evolution
- Dynamic pattern extraction and analysis

Components:
- ZeroConfigAdapter: Automatic domain detection and adaptation
- PatternLearningSystem: Semantic pattern extraction and learning  
- DynamicPatternExtractor: Dynamic pattern analysis and tool generation

Note: These are internal components used by PydanticAI tools.
Direct usage is not recommended - use the PydanticAI tools instead.
"""

# Core discovery components that are actually being used
from .zero_config_adapter import (
    ZeroConfigAdapter,
    DomainDetectionResult,
    AgentAdaptationProfile,
    DomainAdaptationStrategy,
    AdaptationConfidence
)

from .pattern_learning_system import (
    PatternLearningSystem,
    LearningExample,
    PatternEvolutionEvent,
    SemanticCluster,
    LearningSession,
    LearningMode,
    PatternEvolution
)

from .dynamic_pattern_extractor import (
    DynamicPatternExtractor
)

__all__ = [
    # Zero-Config Adapter
    'ZeroConfigAdapter',
    'DomainDetectionResult',
    'AgentAdaptationProfile',
    'DomainAdaptationStrategy',
    'AdaptationConfidence',
    
    # Pattern Learning System
    'PatternLearningSystem',
    'LearningExample',
    'PatternEvolutionEvent',
    'SemanticCluster',
    'LearningSession',
    'LearningMode',
    'PatternEvolution',
    
    # Dynamic Pattern Extractor
    'DynamicPatternExtractor'
]