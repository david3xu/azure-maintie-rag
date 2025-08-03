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

# Import consolidated intelligence components
from .domain_analyzer import (
    DomainAnalyzer,
    ContentAnalysis,
    DomainClassification
)

from .pattern_engine import (
    PatternEngine,
    LearnedPattern,
    ExtractedPatterns
)

# Import background processing and config generation
from .background_processor import (
    run_startup_background_processing,
    BackgroundProcessingStats
)

from .config_generator import (
    DomainConfig,
    ConfigGenerator
)

# Import PydanticAI tools
from .pydantic_tools import (
    discover_domain_tool,
    analyze_domain_patterns_tool,
    validate_domain_confidence_tool
)

__all__ = [
    # Domain analysis
    "DomainAnalyzer",
    "ContentAnalysis", 
    "DomainClassification",
    
    # Pattern learning
    "PatternEngine",
    "LearnedPattern",
    "ExtractedPatterns",
    
    # Background processing
    "run_startup_background_processing",
    "BackgroundProcessingStats",
    
    # Config generation
    "DomainConfig",
    "ConfigGenerator",
    
    # PydanticAI Tools
    "discover_domain_tool",
    "analyze_domain_patterns_tool",
    "validate_domain_confidence_tool"
]