"""
Domain Intelligence Analysis Components

Analysis and processing engines for the domain intelligence system.
"""

from .unified_content_analyzer import (
    UnifiedContentAnalyzer,
    DomainAnalysisResult,
    DomainIntelligenceConfig,
)

from .hybrid_configuration_generator import (
    HybridConfigurationGenerator,
    # ConfigurationRecommendations deleted - now returns Dict[str, Any]
    LLMExtraction,
)

from .pattern_engine import (
    DataDrivenPatternEngine,
)

# Import consolidated data models
from agents.core.data_models import (
    LearnedPattern,
    ExtractedPatterns,
)

from .config_generator import (
    ConfigGenerator,
    DomainConfig,
)

from .background_processor import (
    run_startup_background_processing,
    BackgroundProcessingStats,
)

__all__ = [
    # Unified content analysis
    "UnifiedContentAnalyzer",
    "DomainAnalysisResult",
    "DomainIntelligenceConfig",
    # Configuration generation
    "HybridConfigurationGenerator",
    # "ConfigurationRecommendations" deleted - now returns Dict[str, Any]
    "LLMExtraction",
    # Pattern learning
    "PatternEngine",
    "LearnedPattern",
    "ExtractedPatterns",
    # Config generation
    "ConfigGenerator",
    "DomainConfig",
    # Background processing
    "run_startup_background_processing",
    "BackgroundProcessingStats",
]
