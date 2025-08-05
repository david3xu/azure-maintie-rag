"""
Shared Toolsets - Common PydanticAI Toolset Classes

This module provides common toolset classes that can be used across multiple agents
following the target architecture specification:
- AzureServiceToolset: Common Azure service operations
- PerformanceToolset: Performance monitoring and optimization
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from ..core.azure_service_container import ConsolidatedAzureServices
from ..core.cache_manager import UnifiedCacheManager


class SharedDeps(BaseModel):
    """Shared dependencies for common toolsets"""
    azure_services: Optional[ConsolidatedAzureServices] = None
    cache_manager: Optional[UnifiedCacheManager] = None
    performance_monitor: Optional[Any] = None  # Data-driven performance monitoring
    
    class Config:
        arbitrary_types_allowed = True


class AzureServiceToolset(FunctionToolset):
    """
    Common Azure service operations toolset.
    
    Provides shared Azure service functionality that can be used
    across multiple agents without duplication.
    """
    
    def __init__(self):
        super().__init__()
        
        # Register common Azure service tools
        self.add_function(self.get_service_health, name='get_service_health')
        self.add_function(self.validate_credentials, name='validate_credentials')
        self.add_function(self.get_service_limits, name='get_service_limits')

    async def get_service_health(
        self, ctx: RunContext[SharedDeps]
    ) -> Dict[str, Any]:
        """Get health status of all Azure services"""
        try:
            if ctx.deps and ctx.deps.azure_services:
                status = ctx.deps.azure_services.get_service_status()
                return {
                    "overall_health": status.get("overall_health", "unknown"),
                    "services_ready": status.get("successful_services", 0),
                    "total_services": status.get("total_services", 0),
                    "service_details": status.get("service_status", {})
                }
            else:
                return {
                    "overall_health": "not_initialized",
                    "services_ready": 0,
                    "total_services": 0,
                    "service_details": {}
                }
        except Exception as e:
            return {
                "overall_health": "error",
                "services_ready": 0,
                "total_services": 0,
                "error": str(e)
            }

    async def validate_credentials(
        self, ctx: RunContext[SharedDeps]
    ) -> Dict[str, Any]:
        """Validate Azure credentials and access"""
        try:
            if ctx.deps and ctx.deps.azure_services:
                # Use real Azure credential validation
                service_status = ctx.deps.azure_services.get_service_status()
                overall_health = service_status.get("overall_health", "unknown")
                
                # Credentials are valid if Azure services are healthy
                credentials_valid = overall_health in ["healthy", "partial"]
                
                return {
                    "credentials_valid": credentials_valid,
                    "access_level": "verified" if credentials_valid else "none",
                    "service_health": overall_health,
                    "services_available": service_status.get("successful_services", 0)
                }
            else:
                return {
                    "credentials_valid": False,
                    "access_level": "none",
                    "error": "Azure services not initialized"
                }
        except Exception as e:
            return {
                "credentials_valid": False,
                "access_level": "none",
                "error": str(e)
            }

    async def get_service_limits(
        self, ctx: RunContext[SharedDeps]
    ) -> Dict[str, Any]:
        """Get Azure service limits and quotas"""
        try:
            if ctx.deps and ctx.deps.azure_services:
                # Real Azure service limit checking would be implemented here
                raise NotImplementedError(
                    "Azure service limits checking requires Azure Resource Management API integration. "
                    "This feature is not yet implemented. Configure Azure monitoring to track service quotas."
                )
            else:
                return {
                    "limits_healthy": False,
                    "error": "Azure services not initialized"
                }
        except NotImplementedError:
            raise
        except Exception as e:
            return {
                "limits_healthy": False,
                "error": str(e)
            }


class PerformanceToolset(FunctionToolset):
    """
    Performance monitoring and optimization toolset.
    
    Provides shared performance monitoring functionality that can be used
    across multiple agents for optimization and observability.
    """
    
    def __init__(self):
        super().__init__()
        
        # Register performance monitoring tools
        self.add_function(self.get_performance_metrics, name='get_performance_metrics')
        self.add_function(self.optimize_cache_usage, name='optimize_cache_usage')
        self.add_function(self.monitor_memory_usage, name='monitor_memory_usage')

    async def get_performance_metrics(
        self, ctx: RunContext[SharedDeps]
    ) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            if ctx.deps and ctx.deps.performance_monitor:
                metrics = ctx.deps.performance_monitor.get_current_metrics()
                return {
                    "cpu_usage_percent": metrics.get("cpu_usage", 0.0),
                    "memory_usage_mb": metrics.get("memory_usage", 0.0),
                    "cache_hit_rate": metrics.get("cache_hit_rate", 0.0),
                    "avg_response_time_ms": metrics.get("avg_response_time", 0.0),
                    "active_operations": metrics.get("active_operations", 0)
                }
            else:
                return {
                    "cpu_usage_percent": 0.0,
                    "memory_usage_mb": 0.0,
                    "cache_hit_rate": 0.0,
                    "avg_response_time_ms": 0.0,
                    "active_operations": 0,
                    "monitor_status": "not_initialized"
                }
        except Exception as e:
            return {
                "performance_metrics_error": str(e),
                "monitor_status": "error"
            }

    async def optimize_cache_usage(
        self, ctx: RunContext[SharedDeps]
    ) -> Dict[str, Any]:
        """Optimize cache usage and clear stale entries"""
        try:
            if ctx.deps and ctx.deps.cache_manager:
                # Get cache stats before optimization
                initial_stats = ctx.deps.cache_manager.get_stats()
                
                # Perform cache optimization
                await ctx.deps.cache_manager.cleanup_expired()
                
                # Get stats after optimization
                final_stats = ctx.deps.cache_manager.get_stats()
                
                return {
                    "optimization_completed": True,
                    "entries_before": initial_stats.get("total_entries", 0),
                    "entries_after": final_stats.get("total_entries", 0),
                    "entries_removed": initial_stats.get("total_entries", 0) - final_stats.get("total_entries", 0),
                    "cache_size_mb": final_stats.get("total_size_mb", 0)
                }
            else:
                return {
                    "optimization_completed": False,
                    "error": "Cache manager not available"
                }
        except Exception as e:
            return {
                "optimization_completed": False,
                "error": str(e)
            }

    async def monitor_memory_usage(
        self, ctx: RunContext[SharedDeps]
    ) -> Dict[str, Any]:
        """Monitor and report memory usage"""
        try:
            import psutil
            
            # Get system memory info
            memory = psutil.virtual_memory()
            
            return {
                "total_memory_gb": round(memory.total / (1024**3), 2),
                "available_memory_gb": round(memory.available / (1024**3), 2),
                "used_memory_gb": round(memory.used / (1024**3), 2),
                "memory_percent": memory.percent,
                "memory_status": "healthy" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical"
            }
        except ImportError:
            return {
                "memory_monitoring": "psutil_not_available",
                "memory_status": "unknown"
            }
        except Exception as e:
            return {
                "memory_monitoring_error": str(e),
                "memory_status": "error"
            }


# Create shared toolset instances
azure_service_toolset = AzureServiceToolset()
performance_toolset = PerformanceToolset()

# Export main components
__all__ = [
    "SharedDeps",
    "AzureServiceToolset",
    "PerformanceToolset",
    "azure_service_toolset",
    "performance_toolset",
]