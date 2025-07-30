"""
Performance Monitoring Service
Lightweight performance tracking for query processing
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class QueryPerformanceMetrics:
    """Query performance metrics"""
    query: str
    domain: str
    operation: str
    total_time: float
    cache_hit: bool
    search_times: Dict[str, float]
    result_count: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PerformanceService:
    """Lightweight performance monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_targets = {
            "total_response_time": 3.0,      # < 3 seconds total
            "parallel_search_time": 1.5,     # < 1.5s parallel execution
            "cache_hit_rate": 0.4,           # > 40% cache hits
        }
        self.recent_metrics: List[QueryPerformanceMetrics] = []
        self.max_recent_metrics = 100  # Keep last 100 metrics in memory
    
    async def track_query_performance(
        self, 
        metrics: QueryPerformanceMetrics
    ):
        """Track and log query performance"""
        
        # Add to recent metrics
        self.recent_metrics.append(metrics)
        if len(self.recent_metrics) > self.max_recent_metrics:
            self.recent_metrics.pop(0)  # Remove oldest
        
        # Log performance metrics
        self.logger.info(
            f"Performance [{metrics.operation}]: {metrics.total_time:.2f}s | "
            f"Cache: {'HIT' if metrics.cache_hit else 'MISS'} | "
            f"Results: {metrics.result_count} | "
            f"Query: {metrics.query[:50]}..."
        )
        
        # Check performance targets
        violations = []
        if metrics.total_time > self.performance_targets["total_response_time"]:
            violations.append(
                f"Response time: {metrics.total_time:.2f}s > "
                f"{self.performance_targets['total_response_time']}s"
            )
        
        if violations:
            self.logger.warning(f"Performance Target Violations: {'; '.join(violations)}")
        
        # Log detailed timing breakdown if available
        if metrics.search_times:
            timing_details = ", ".join([
                f"{k}: {v:.2f}s" for k, v in metrics.search_times.items()
            ])
            self.logger.debug(f"Timing breakdown: {timing_details}")
    
    def create_performance_context(self, query: str, domain: str, operation: str = "query"):
        """Create performance tracking context"""
        return PerformanceTracker(query, domain, operation, self)
    
    async def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        if not self.recent_metrics:
            return {
                "summary": "No metrics available",
                "total_queries": 0
            }
        
        # Filter metrics from last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent = [
            m for m in self.recent_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent:
            return {
                "summary": f"No metrics in last {hours} hour(s)",
                "total_queries": 0
            }
        
        # Calculate statistics
        total_queries = len(recent)
        cache_hits = sum(1 for m in recent if m.cache_hit)
        cache_hit_rate = cache_hits / total_queries if total_queries > 0 else 0
        
        response_times = [m.total_time for m in recent]
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        # Performance target compliance
        target_violations = sum(
            1 for rt in response_times 
            if rt > self.performance_targets["total_response_time"]
        )
        target_compliance = ((total_queries - target_violations) / total_queries * 100) if total_queries > 0 else 0
        
        # Operation breakdown
        operations = {}
        for metric in recent:
            op = metric.operation
            if op not in operations:
                operations[op] = {"count": 0, "avg_time": 0, "total_time": 0}
            operations[op]["count"] += 1
            operations[op]["total_time"] += metric.total_time
        
        for op in operations:
            operations[op]["avg_time"] = operations[op]["total_time"] / operations[op]["count"]
            operations[op]["total_time"] = round(operations[op]["total_time"], 2)
            operations[op]["avg_time"] = round(operations[op]["avg_time"], 2)
        
        return {
            "period": f"Last {hours} hour(s)",
            "total_queries": total_queries,
            "cache_performance": {
                "hit_rate": round(cache_hit_rate * 100, 1),
                "hits": cache_hits,
                "misses": total_queries - cache_hits,
                "target": f"> {self.performance_targets['cache_hit_rate'] * 100}%"
            },
            "response_times": {
                "average": round(avg_response_time, 2),
                "min": round(min_response_time, 2),
                "max": round(max_response_time, 2),
                "target": f"< {self.performance_targets['total_response_time']}s"
            },
            "target_compliance": {
                "percentage": round(target_compliance, 1),
                "violations": target_violations,
                "compliant": total_queries - target_violations
            },
            "operations": operations,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_recent_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent slow queries above performance targets"""
        slow_queries = [
            m for m in self.recent_metrics
            if m.total_time > self.performance_targets["total_response_time"]
        ]
        
        # Sort by response time (slowest first)
        slow_queries.sort(key=lambda x: x.total_time, reverse=True)
        
        return [
            {
                "query": metric.query[:100] + "..." if len(metric.query) > 100 else metric.query,
                "domain": metric.domain,
                "operation": metric.operation,
                "response_time": round(metric.total_time, 2),
                "cache_hit": metric.cache_hit,
                "result_count": metric.result_count,
                "timestamp": metric.timestamp.isoformat()
            }
            for metric in slow_queries[:limit]
        ]


class PerformanceTracker:
    """Context manager for tracking query performance"""
    
    def __init__(self, query: str, domain: str, operation: str, service: PerformanceService):
        self.query = query
        self.domain = domain
        self.operation = operation
        self.service = service
        self.start_time = None
        self.cache_hit = False
        self.search_times = {}
        self.result_count = 0
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is None:
            return
            
        total_time = time.time() - self.start_time
        
        metrics = QueryPerformanceMetrics(
            query=self.query,
            domain=self.domain,
            operation=self.operation,
            total_time=total_time,
            cache_hit=self.cache_hit,
            search_times=self.search_times.copy(),
            result_count=self.result_count,
            timestamp=datetime.now()
        )
        
        await self.service.track_query_performance(metrics)
    
    def mark_cache_hit(self):
        """Mark this query as a cache hit"""
        self.cache_hit = True
    
    def record_search_time(self, search_type: str, duration: float):
        """Record timing for a specific search type"""
        self.search_times[search_type] = round(duration, 3)
    
    def set_result_count(self, count: int):
        """Set the number of results returned"""
        self.result_count = count
    
    def record_phase_time(self, phase_name: str, start_time: float):
        """Record time for a processing phase"""
        duration = time.time() - start_time
        self.search_times[phase_name] = round(duration, 3)