"""
PydanticAI Tools for Domain Intelligence Agent
==============================================

This module provides PydanticAI-compatible tools for the Domain Intelligence Agent,
implementing zero-configuration domain discovery with enterprise integration.

Features:
- Zero-configuration domain discovery
- Statistical pattern learning
- ConsolidatedAzureServices integration
- Enterprise error handling and monitoring
"""

import logging
from typing import Any, Dict, Optional

from pydantic_ai import RunContext

from ..core.azure_services import ConsolidatedAzureServices

# Import our consolidated tools and models
from .discovery_tools import (
    DomainDetectionRequest,
    DomainDetectionResponse,
    execute_domain_detection,
)

logger = logging.getLogger(__name__)


async def discover_domain_tool(
    ctx: RunContext,
    text: str,
    azure_services: Optional[ConsolidatedAzureServices] = None,
) -> str:
    """
    Zero-configuration domain discovery using statistical pattern learning.

    This tool provides our competitive advantage of automatic domain detection
    without requiring any predefined configurations or rules.

    Args:
        ctx: PydanticAI run context
        text: Text to analyze for domain detection
        azure_services: Optional ConsolidatedAzureServices instance

    Returns:
        Detected domain name or 'general' as fallback
    """
    try:
        # Extract dependencies from context if available
        confidence_threshold = 0.7
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "azure_services"):
            azure_services = ctx.deps.azure_services
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "app_settings"):
            confidence_threshold = (
                ctx.deps.app_settings.tri_modal_search.competitive_advantage.confidence_threshold
            )

        # Create domain detection request
        detection_request = DomainDetectionRequest(
            text=text,
            confidence_threshold=confidence_threshold,
            include_pattern_analysis=True,
        )

        # Execute domain detection
        result = await execute_domain_detection(ctx, detection_request)
        domain = result.detected_domain or "general"

        logger.info(
            f"PydanticAI Domain Intelligence discovered domain: {domain} (confidence: {result.confidence:.2f})"
        )
        return domain

    except Exception as e:
        logger.error(f"PydanticAI Domain Intelligence domain discovery failed: {e}")
        return "general"  # Graceful fallback


async def analyze_domain_patterns_tool(
    ctx: RunContext,
    text: str,
    domain: Optional[str] = None,
    azure_services: Optional[ConsolidatedAzureServices] = None,
) -> Dict[str, Any]:
    """
    Analyze statistical patterns in text for enhanced domain understanding.

    Args:
        ctx: PydanticAI run context
        text: Text to analyze
        domain: Optional known domain for focused analysis
        azure_services: Optional ConsolidatedAzureServices instance

    Returns:
        Dictionary containing pattern analysis results
    """
    try:
        # Extract dependencies from context if available
        confidence_threshold = 0.7
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "azure_services"):
            azure_services = ctx.deps.azure_services
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "app_settings"):
            confidence_threshold = (
                ctx.deps.app_settings.tri_modal_search.competitive_advantage.confidence_threshold
            )

        # Create enhanced domain detection request
        detection_request = DomainDetectionRequest(
            text=text,
            confidence_threshold=confidence_threshold,
            include_pattern_analysis=True,
            known_domain=domain,
        )

        # Execute domain detection with pattern analysis
        result = await execute_domain_detection(ctx, detection_request)

        # Compile pattern analysis results
        pattern_analysis = {
            "detected_domain": result.detected_domain,
            "confidence": result.confidence,
            "pattern_analysis": getattr(result, "pattern_analysis", {}),
            "statistical_features": getattr(result, "statistical_features", {}),
            "domain_indicators": getattr(result, "domain_indicators", []),
            "competitive_advantage_score": min(
                1.0, result.confidence * 1.2
            ),  # Boost for competitive scoring
            "zero_config_success": result.detected_domain is not None,
        }

        logger.info(
            f"PydanticAI Domain Intelligence pattern analysis completed: "
            f"domain={result.detected_domain}, confidence={result.confidence:.2f}"
        )

        return pattern_analysis

    except Exception as e:
        logger.error(f"PydanticAI Domain Intelligence pattern analysis failed: {e}")
        return {
            "detected_domain": domain or "general",
            "confidence": 0.0,
            "error": str(e),
            "zero_config_success": False,
        }


async def validate_domain_confidence_tool(
    ctx: RunContext,
    domain: str,
    confidence: float,
    text: str,
    required_confidence: float = 0.7,
) -> Dict[str, Any]:
    """
    Validate domain detection confidence and provide enhancement recommendations.

    Args:
        ctx: PydanticAI run context
        domain: Detected domain
        confidence: Detection confidence score
        text: Original text analyzed
        required_confidence: Required confidence threshold

    Returns:
        Validation results and recommendations
    """
    try:
        validation_results = {
            "domain": domain,
            "confidence": confidence,
            "confidence_met": confidence >= required_confidence,
            "required_confidence": required_confidence,
            "text_length": len(text),
            "domain_strength": confidence,
            "enhancement_recommendations": [],
        }

        # Generate enhancement recommendations
        if confidence < required_confidence:
            validation_results["enhancement_recommendations"].extend(
                [
                    "Consider providing more domain-specific text samples",
                    "Increase text length for better pattern detection",
                    "Include more technical terminology if available",
                ]
            )

        if len(text) < 100:
            validation_results["enhancement_recommendations"].append(
                "Text may be too short for reliable domain detection"
            )

        if domain == "general":
            validation_results["enhancement_recommendations"].append(
                "Domain detection resulted in generic classification - consider more specific text"
            )

        # Calculate competitive advantage score
        validation_results["competitive_advantage_score"] = min(
            1.0, confidence * (1.0 + (0.1 if len(text) > 200 else 0))
        )
        validation_results["zero_config_advantage"] = (
            confidence > 0.5
        )  # Success without configuration

        logger.info(
            f"PydanticAI Domain Intelligence validation: domain={domain}, "
            f"confidence={'✅' if validation_results['confidence_met'] else '❌'}"
        )

        return validation_results

    except Exception as e:
        logger.error(f"PydanticAI Domain Intelligence validation failed: {e}")
        return {
            "domain": domain,
            "confidence": 0.0,
            "confidence_met": False,
            "error": str(e),
        }


# Export the tools
__all__ = [
    "discover_domain_tool",
    "analyze_domain_patterns_tool",
    "validate_domain_confidence_tool",
]
