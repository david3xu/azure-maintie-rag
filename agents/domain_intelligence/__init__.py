# Intelligence consolidation - unified domain intelligence components

"""
Intelligence Module - Consolidated Domain Intelligence Components

This module consolidates domain intelligence functionality from multiple
directories (discovery/ and domain/) into unified, high-performance components:

- DomainAnalyzer: Unified content analysis and domain classification
- PatternEngine: Consolidated pattern extraction and learning system
- Background Processing: Startup optimization and performance enhancement
- Config Generation: Infrastructure and ML configuration management
- PydanticAI Tools: Enterprise integration for PydanticAI agents

Key features preserved:
- Data-driven domain discovery (no hardcoded assumptions)
- Statistical pattern learning and evolution tracking
- High-performance pattern matching and indexing
- Zero-config domain adaptation
- Continuous learning from user interactions
- Background processing for <5ms domain detection
"""

# Import analysis components from organized structure
from .analyzers import (
    UnifiedContentAnalyzer, DomainAnalysisResult, DomainIntelligenceConfig,
    HybridConfigurationGenerator, LLMExtraction,  # ConfigurationRecommendations deleted
    DataDrivenPatternEngine, LearnedPattern, ExtractedPatterns,
    ConfigGenerator, DomainConfig,
    run_startup_background_processing, BackgroundProcessingStats,
)
# Note: DomainAnalyzer and DomainClassification removed - use UnifiedContentAnalyzer directly

# PydanticAI tools moved to toolsets.py following target architecture
# from .pydantic_tools import (
#     analyze_domain_patterns_tool,
#     discover_domain_tool,
#     validate_domain_confidence_tool,
# )

__all__ = [
    # Unified content analysis (primary)
    "UnifiedContentAnalyzer",
    "DomainAnalysisResult", 
    "DomainIntelligenceConfig",
    # Configuration generation
    "HybridConfigurationGenerator",
    # "ConfigurationRecommendations" deleted - generator now returns Dict[str, Any]
    "LLMExtraction",
    # Note: DomainAnalyzer and DomainClassification removed - use UnifiedContentAnalyzer
    # Pattern learning
    "DataDrivenPatternEngine",
    "LearnedPattern",
    "ExtractedPatterns",
    # Background processing
    "run_startup_background_processing",
    "BackgroundProcessingStats",
    # Config generation
    "DomainConfig",
    "ConfigGenerator",
    # PydanticAI Tools moved to toolsets.py following target architecture
    # "discover_domain_tool",
    # "analyze_domain_patterns_tool", 
    # "validate_domain_confidence_tool",
]
