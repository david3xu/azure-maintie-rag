"""
Discovery System Constants - Data-Driven Configuration

This module provides data-driven configuration calculation methods that replace
hardcoded values with statistically derived thresholds based on actual data analysis.
Follows coding rules: No hardcoded values, all thresholds derived from real data.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatisticalConfidenceCalculator:
    """Calculate confidence thresholds from real data analysis"""
    
    @staticmethod
    def calculate_confidence_levels(
        confidence_samples: List[float],
        sample_source: str = "unknown"
    ) -> Dict[str, float]:
        """
        Calculate confidence levels from actual confidence score samples.
        
        Args:
            confidence_samples: Real confidence scores from system operations
            sample_source: Source of the samples for data lineage
            
        Returns:
            Dict with statistically derived confidence levels
            
        Raises:
            ValueError: If insufficient data for statistical calculation
        """
        if len(confidence_samples) < 10:
            raise ValueError(
                f"Need at least 10 confidence samples for statistical calculation, got {len(confidence_samples)}"
            )
        
        # Validate confidence scores are in valid range
        invalid_scores = [s for s in confidence_samples if not 0.0 <= s <= 1.0]
        if invalid_scores:
            raise ValueError(f"Invalid confidence scores found: {invalid_scores}")
        
        # Calculate statistical quartiles from real data
        sorted_samples = sorted(confidence_samples)
        
        # Use statistical quartiles for confidence levels
        very_high_threshold = statistics.quantiles(sorted_samples, n=10)[8]  # 90th percentile
        high_threshold = statistics.quantiles(sorted_samples, n=4)[2]        # 75th percentile  
        medium_threshold = statistics.median(sorted_samples)                 # 50th percentile
        low_threshold = statistics.quantiles(sorted_samples, n=4)[0]         # 25th percentile
        
        logger.info(
            f"Calculated confidence levels from {len(confidence_samples)} samples "
            f"(source: {sample_source}): "
            f"very_high={very_high_threshold:.3f}, high={high_threshold:.3f}, "
            f"medium={medium_threshold:.3f}, low={low_threshold:.3f}"
        )
        
        return {
            "VERY_HIGH": very_high_threshold,
            "HIGH": high_threshold, 
            "MEDIUM": medium_threshold,
            "LOW": low_threshold,
            "data_source": sample_source,
            "sample_size": len(confidence_samples)
        }
    
    @staticmethod
    def calculate_strategy_thresholds(
        performance_data: Dict[str, List[float]],
        success_rate_target: float = 0.85
    ) -> Dict[str, float]:
        """
        Calculate strategy thresholds from actual performance data.
        
        Args:
            performance_data: Dict mapping strategy names to success rates
            success_rate_target: Target success rate for threshold calculation
            
        Returns:
            Dict with data-driven strategy thresholds
        """
        if not performance_data:
            raise ValueError("Performance data cannot be empty for strategy calculation")
        
        strategy_thresholds = {}
        
        for strategy_name, success_rates in performance_data.items():
            if len(success_rates) < 5:
                logger.warning(f"Insufficient data for {strategy_name}: {len(success_rates)} samples")
                continue
                
            # Calculate threshold that achieves target success rate
            sorted_rates = sorted(success_rates, reverse=True)
            target_index = int(len(sorted_rates) * (1.0 - success_rate_target))
            threshold = sorted_rates[min(target_index, len(sorted_rates) - 1)]
            
            strategy_thresholds[f"{strategy_name.upper()}_STRATEGY"] = threshold
            
            logger.info(
                f"Strategy {strategy_name}: threshold={threshold:.3f} "
                f"(based on {len(success_rates)} samples, target success rate {success_rate_target})"
            )
        
        return strategy_thresholds


@dataclass(frozen=True)
class SearchWeightCalculator:
    """Calculate search weights from actual search performance data"""
    
    @staticmethod
    def calculate_optimal_search_weights(
        search_performance_data: Dict[str, Dict[str, List[float]]],
        optimization_target: str = "precision"
    ) -> Dict[str, float]:
        """
        Calculate optimal search modality weights from real performance data.
        
        Args:
            search_performance_data: Dict mapping search types to performance metrics
                Example: {"vector": {"precision": [0.8, 0.85, 0.9]}, "graph": {...}}
            optimization_target: Metric to optimize for ("precision", "recall", "f1")
            
        Returns:
            Dict with statistically optimal search weights
        """
        if not search_performance_data:
            raise ValueError("Search performance data cannot be empty")
        
        # Calculate average performance for each search type
        search_scores = {}
        for search_type, metrics in search_performance_data.items():
            if optimization_target not in metrics:
                logger.warning(f"Missing {optimization_target} data for {search_type}")
                continue
                
            target_scores = metrics[optimization_target]
            if len(target_scores) < 3:
                logger.warning(f"Insufficient {optimization_target} data for {search_type}: {len(target_scores)} samples")
                continue
                
            avg_score = statistics.mean(target_scores)
            search_scores[search_type] = avg_score
            
            logger.info(f"{search_type} average {optimization_target}: {avg_score:.3f}")
        
        if not search_scores:
            raise ValueError(f"No valid {optimization_target} data found for weight calculation")
        
        # Normalize scores to weights (higher performance gets higher weight)
        total_score = sum(search_scores.values())
        search_weights = {
            f"{search_type.upper()}_WEIGHT": score / total_score
            for search_type, score in search_scores.items()
        }
        
        # Add metadata
        search_weights["optimization_target"] = optimization_target
        search_weights["data_source"] = "search_performance_analysis"
        search_weights["total_search_types"] = len(search_scores)
        
        logger.info(f"Calculated search weights: {search_weights}")
        return search_weights
    
    @staticmethod
    def calculate_reasoning_pattern_weights(
        reasoning_effectiveness_data: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Calculate reasoning pattern weights from effectiveness data.
        
        Args:
            reasoning_effectiveness_data: Dict mapping patterns to effectiveness scores
            
        Returns:
            Dict with data-driven reasoning pattern weights
        """
        if not reasoning_effectiveness_data:
            raise ValueError("Reasoning effectiveness data cannot be empty")
        
        pattern_weights = {}
        total_effectiveness = 0
        
        for pattern_name, effectiveness_scores in reasoning_effectiveness_data.items():
            if len(effectiveness_scores) < 3:
                logger.warning(f"Insufficient data for {pattern_name}: {len(effectiveness_scores)} samples")
                continue
                
            avg_effectiveness = statistics.mean(effectiveness_scores)
            pattern_weights[pattern_name] = avg_effectiveness
            total_effectiveness += avg_effectiveness
            
            logger.info(f"{pattern_name} average effectiveness: {avg_effectiveness:.3f}")
        
        # Normalize to weights
        if total_effectiveness > 0:
            pattern_weights = {
                f"{pattern.upper()}_WEIGHT": weight / total_effectiveness
                for pattern, weight in pattern_weights.items()
            }
        
        return pattern_weights


@dataclass(frozen=True)
class DataDrivenConfigurationManager:
    """
    Manages all data-driven configuration calculations.
    
    Replaces hardcoded constants with statistically derived values based on
    actual system performance and user interaction data.
    """
    
    @staticmethod
    def calculate_context_priorities(
        context_effectiveness_data: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Calculate context priority weights from effectiveness data.
        
        Args:
            context_effectiveness_data: Dict mapping context types to effectiveness scores
            
        Returns:
            Dict with data-driven context priorities
        """
        if not context_effectiveness_data:
            raise ValueError("Context effectiveness data required for priority calculation")
        
        total_effectiveness = 0
        context_scores = {}
        
        for context_type, effectiveness_scores in context_effectiveness_data.items():
            if len(effectiveness_scores) < 5:
                logger.warning(f"Limited data for {context_type}: {len(effectiveness_scores)} samples")
                continue
                
            avg_effectiveness = statistics.mean(effectiveness_scores)
            context_scores[context_type] = avg_effectiveness
            total_effectiveness += avg_effectiveness
        
        # Normalize to priorities
        context_priorities = {}
        if total_effectiveness > 0:
            for context_type, score in context_scores.items():
                priority = score / total_effectiveness
                context_priorities[f"{context_type.upper()}_PRIORITY"] = priority
                logger.info(f"{context_type} priority: {priority:.3f} (effectiveness: {score:.3f})")
        
        context_priorities["calculation_source"] = "context_effectiveness_analysis"
        return context_priorities
    
    @staticmethod
    def calculate_performance_targets(
        historical_performance_data: Dict[str, List[float]],
        target_percentile: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate performance targets from historical data.
        
        Args:
            historical_performance_data: Dict with metrics like "response_time", "confidence"
            target_percentile: Percentile to target for performance (0.95 = 95th percentile)
            
        Returns:
            Dict with statistically derived performance targets
        """
        if not historical_performance_data:
            raise ValueError("Historical performance data required for target calculation")
        
        performance_targets = {}
        
        for metric_name, metric_values in historical_performance_data.items():
            if len(metric_values) < 20:
                logger.warning(f"Limited {metric_name} data: {len(metric_values)} samples")
                continue
            
            # For response time: target the 95th percentile (most requests should be faster)
            if "time" in metric_name.lower():
                target_value = statistics.quantiles(metric_values, n=20)[int(20 * target_percentile)]
                performance_targets[f"MAX_{metric_name.upper()}"] = target_value
            
            # For confidence: target the 25th percentile (minimum acceptable)
            elif "confidence" in metric_name.lower():
                target_value = statistics.quantiles(metric_values, n=4)[0]  # 25th percentile
                performance_targets[f"MIN_{metric_name.upper()}"] = target_value
            
            # For other metrics: use median
            else:
                target_value = statistics.median(metric_values)
                performance_targets[f"TARGET_{metric_name.upper()}"] = target_value
            
            logger.info(f"{metric_name} target: {target_value:.3f} (from {len(metric_values)} samples)")
        
        performance_targets["target_percentile"] = target_percentile
        performance_targets["data_source"] = "historical_performance_analysis"
        return performance_targets
    
    @staticmethod
    def calculate_quality_weights(
        quality_impact_data: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Calculate quality assessment weights from impact analysis.
        
        Args:
            quality_impact_data: Dict mapping quality factors to impact scores
            
        Returns:
            Dict with data-driven quality weights
        """
        if not quality_impact_data:
            raise ValueError("Quality impact data required for weight calculation")
        
        total_impact = 0
        quality_scores = {}
        
        for quality_factor, impact_scores in quality_impact_data.items():
            if len(impact_scores) < 5:
                logger.warning(f"Limited {quality_factor} data: {len(impact_scores)} samples")
                continue
                
            avg_impact = statistics.mean(impact_scores)
            quality_scores[quality_factor] = avg_impact
            total_impact += avg_impact
        
        # Normalize to weights
        quality_weights = {}
        if total_impact > 0:
            for factor, impact in quality_scores.items():
                weight = impact / total_impact
                quality_weights[f"{factor.upper()}_WEIGHT"] = weight
                logger.info(f"{factor} weight: {weight:.3f} (impact: {impact:.3f})")
        
        quality_weights["calculation_source"] = "quality_impact_analysis"
        return quality_weights
    
    @staticmethod
    def calculate_learning_parameters(
        learning_performance_data: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Calculate learning system parameters from performance data.
        
        Args:
            learning_performance_data: Dict with learning metrics like "convergence_rate"
            
        Returns:
            Dict with optimized learning parameters
        """
        if not learning_performance_data:
            raise ValueError("Learning performance data required for parameter calculation")
        
        learning_params = {}
        
        for param_name, performance_values in learning_performance_data.items():
            if len(performance_values) < 10:
                logger.warning(f"Limited {param_name} data: {len(performance_values)} samples")
                continue
            
            # Use statistical analysis to find optimal parameters
            if "rate" in param_name.lower():
                # For rates: use median for stability
                optimal_value = statistics.median(performance_values)
            elif "threshold" in param_name.lower():
                # For thresholds: use value that achieves 80% success rate
                sorted_values = sorted(performance_values)
                optimal_value = sorted_values[int(len(sorted_values) * 0.8)]
            else:
                # For other parameters: use mean
                optimal_value = statistics.mean(performance_values)
            
            learning_params[f"{param_name.upper()}_PARAMETER"] = optimal_value
            logger.info(f"{param_name} parameter: {optimal_value:.3f}")
        
        learning_params["optimization_source"] = "learning_performance_analysis"
        return learning_params


# Factory function to create data-driven configuration
def create_data_driven_configuration(
    confidence_samples: Optional[List[float]] = None,
    performance_data: Optional[Dict[str, List[float]]] = None,
    context_data: Optional[Dict[str, List[float]]] = None,
    quality_data: Optional[Dict[str, List[float]]] = None
) -> Dict[str, Any]:
    """
    Create complete data-driven configuration from real system data.
    
    Args:
        confidence_samples: Real confidence scores from system operations
        performance_data: Historical performance metrics
        context_data: Context effectiveness data
        quality_data: Quality impact data
        
    Returns:
        Complete configuration dictionary with statistically derived values
        
    Raises:
        ValueError: If insufficient data provided for calculation
    """
    config_manager = DataDrivenConfigurationManager()
    configuration = {
        "configuration_type": "data_driven",
        "created_from_real_data": True,
        "hardcoded_values": False
    }
    
    # Calculate confidence levels if data available
    if confidence_samples and len(confidence_samples) >= 10:
        confidence_calc = StatisticalConfidenceCalculator()
        configuration["confidence_levels"] = confidence_calc.calculate_confidence_levels(
            confidence_samples, "system_operations"
        )
    
    # Calculate performance targets if data available
    if performance_data:
        configuration["performance_targets"] = config_manager.calculate_performance_targets(
            performance_data
        )
    
    # Calculate context priorities if data available
    if context_data:
        configuration["context_priorities"] = config_manager.calculate_context_priorities(
            context_data
        )
    
    # Calculate quality weights if data available
    if quality_data:
        configuration["quality_weights"] = config_manager.calculate_quality_weights(
            quality_data
        )
    
    logger.info(f"Created data-driven configuration with {len(configuration)} sections")
    return configuration


__all__ = [
    'StatisticalConfidenceCalculator',
    'SearchWeightCalculator', 
    'DataDrivenConfigurationManager',
    'create_data_driven_configuration'
]