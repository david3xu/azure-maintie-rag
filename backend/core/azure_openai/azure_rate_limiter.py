"""
Azure OpenAI Rate Limiting and Cost Optimization Service
Enterprise-grade quota management and cost control
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import logging
from config.settings import azure_settings

logger = logging.getLogger(__name__)

@dataclass
class AzureOpenAIQuota:
    """Azure OpenAI quota configuration"""
    max_tokens_per_minute: int
    max_requests_per_minute: int
    cost_threshold_per_hour: float
    priority_tier: str  # "enterprise", "standard", "cost_optimized"

class AzureOpenAIRateLimiter:
    """Enterprise rate limiting and cost optimization"""

    def __init__(self):
        self.quota_config = self._load_quota_config()
        self.usage_tracker = {
            "tokens_used_this_minute": 0,
            "requests_this_minute": 0,
            "cost_this_hour": 0.0,
            "last_reset_time": time.time()
        }

        # Rate limiting state
        self.rate_limit_state = {
            "last_request_time": 0,
            "min_request_interval": 1.0 / self.quota_config.max_requests_per_minute,
            "backoff_multiplier": 1.0
        }

    def _load_quota_config(self) -> AzureOpenAIQuota:
        """Load quota configuration from Azure settings"""
        return AzureOpenAIQuota(
            max_tokens_per_minute=azure_settings.azure_openai_max_tokens_per_minute,
            max_requests_per_minute=azure_settings.azure_openai_max_requests_per_minute,
            cost_threshold_per_hour=azure_settings.azure_openai_cost_threshold_per_hour,
            priority_tier=azure_settings.azure_openai_priority_tier
        )

    async def execute_with_rate_limiting(
        self,
        extraction_function: Callable,
        estimated_tokens: int,
        priority: str = "standard"
    ) -> Any:
        """
        Execute extraction with enterprise rate limiting
        Includes exponential backoff and cost controls
        """

        # Check quota availability
        if not await self._check_quota_availability(estimated_tokens):
            await self._wait_for_quota_reset()

        # Execute with retry logic
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Track usage before execution
                start_time = time.time()

                # Execute extraction
                result = await extraction_function()

                # Track actual usage
                execution_time = time.time() - start_time
                actual_tokens = self._estimate_tokens_from_result(result)

                await self._update_usage_tracking(actual_tokens, execution_time)

                return result

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Azure OpenAI call failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

    async def _check_quota_availability(self, estimated_tokens: int) -> bool:
        """Check if quota allows the requested operation"""
        current_time = time.time()

        # Reset counters if needed
        await self._reset_counters_if_needed(current_time)

        # Check token quota
        if self.usage_tracker["tokens_used_this_minute"] + estimated_tokens > self.quota_config.max_tokens_per_minute:
            logger.warning(f"Token quota exceeded: {self.usage_tracker['tokens_used_this_minute']}/{self.quota_config.max_tokens_per_minute}")
            return False

        # Check request quota
        if self.usage_tracker["requests_this_minute"] >= self.quota_config.max_requests_per_minute:
            logger.warning(f"Request quota exceeded: {self.usage_tracker['requests_this_minute']}/{self.quota_config.max_requests_per_minute}")
            return False

        # Check cost threshold
        estimated_cost = self._estimate_cost(estimated_tokens)
        if self.usage_tracker["cost_this_hour"] + estimated_cost > self.quota_config.cost_threshold_per_hour:
            logger.warning(f"Cost threshold exceeded: {self.usage_tracker['cost_this_hour']}/{self.quota_config.cost_threshold_per_hour}")
            return False

        return True

    async def _wait_for_quota_reset(self) -> None:
        """Wait for quota reset with exponential backoff"""
        wait_time = 60  # Wait 1 minute for quota reset
        logger.info(f"Waiting {wait_time}s for quota reset...")
        await asyncio.sleep(wait_time)

    async def _reset_counters_if_needed(self, current_time: float) -> None:
        """Reset usage counters based on time intervals"""
        time_since_reset = current_time - self.usage_tracker["last_reset_time"]

        # Reset minute counters
        if time_since_reset >= 60:
            self.usage_tracker["tokens_used_this_minute"] = 0
            self.usage_tracker["requests_this_minute"] = 0
            self.usage_tracker["last_reset_time"] = current_time

        # Reset hour counters
        if time_since_reset >= 3600:
            self.usage_tracker["cost_this_hour"] = 0.0

    async def _update_usage_tracking(self, actual_tokens: int, execution_time: float) -> None:
        """Update usage tracking with actual consumption"""
        current_time = time.time()

        # Update minute counters
        self.usage_tracker["tokens_used_this_minute"] += actual_tokens
        self.usage_tracker["requests_this_minute"] += 1

        # Update cost tracking
        actual_cost = self._estimate_cost(actual_tokens)
        self.usage_tracker["cost_this_hour"] += actual_cost

        # Reset counters if needed
        await self._reset_counters_if_needed(current_time)

        logger.info(f"Usage updated: {actual_tokens} tokens, {actual_cost:.4f} cost")

    def _estimate_tokens_from_result(self, result: Any) -> int:
        """Estimate token usage from extraction result"""
        # This is a simplified estimation
        # In production, you'd parse the actual response to count tokens
        if isinstance(result, dict):
            # Estimate based on result size
            result_str = str(result)
            return len(result_str.split()) * 1.3  # Rough token estimation
        return 100  # Default estimation

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token usage"""
        # Azure OpenAI pricing (approximate)
        # GPT-4: $0.03 per 1K tokens input, $0.06 per 1K tokens output
        # GPT-3.5: $0.0015 per 1K tokens input, $0.002 per 1K tokens output

        # Using GPT-4 pricing as default
        cost_per_1k_tokens = 0.03  # Input tokens
        return (tokens / 1000) * cost_per_1k_tokens

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get current usage summary"""
        return {
            "tokens_used_this_minute": self.usage_tracker["tokens_used_this_minute"],
            "requests_this_minute": self.usage_tracker["requests_this_minute"],
            "cost_this_hour": self.usage_tracker["cost_this_hour"],
            "quota_limits": {
                "max_tokens_per_minute": self.quota_config.max_tokens_per_minute,
                "max_requests_per_minute": self.quota_config.max_requests_per_minute,
                "cost_threshold_per_hour": self.quota_config.cost_threshold_per_hour
            },
            "utilization_percentages": {
                "tokens": (self.usage_tracker["tokens_used_this_minute"] / self.quota_config.max_tokens_per_minute) * 100,
                "requests": (self.usage_tracker["requests_this_minute"] / self.quota_config.max_requests_per_minute) * 100,
                "cost": (self.usage_tracker["cost_this_hour"] / self.quota_config.cost_threshold_per_hour) * 100
            }
        }

    def is_quota_healthy(self) -> bool:
        """Check if quota usage is within healthy limits"""
        summary = self.get_usage_summary()
        utilizations = summary["utilization_percentages"]

        # Consider healthy if all utilizations are below 80%
        return all(util < 80 for util in utilizations.values())

    async def wait_for_healthy_quota(self, max_wait_time: int = 300) -> bool:
        """Wait until quota is healthy or timeout"""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            if self.is_quota_healthy():
                return True

            wait_time = min(30, max_wait_time - (time.time() - start_time))
            logger.info(f"Waiting {wait_time}s for quota to become healthy...")
            await asyncio.sleep(wait_time)

        logger.warning("Timeout waiting for healthy quota")
        return False

    def get_cost_optimization_recommendations(self) -> List[str]:
        """Get recommendations for cost optimization"""
        recommendations = []
        summary = self.get_usage_summary()
        utilizations = summary["utilization_percentages"]

        if utilizations["cost"] > 70:
            recommendations.append("Cost utilization high. Consider reducing batch sizes or using cheaper models")

        if utilizations["tokens"] > 80:
            recommendations.append("Token utilization high. Consider optimizing prompts or using more efficient models")

        if utilizations["requests"] > 80:
            recommendations.append("Request rate high. Consider implementing request batching")

        if not recommendations:
            recommendations.append("Quota usage is healthy. No optimization needed")

        return recommendations