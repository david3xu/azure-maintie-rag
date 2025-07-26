"""
Azure Prompt Flow Monitoring and Analytics
Provides comprehensive monitoring for centralized prompt management
Tracks performance, costs, and quality metrics for universal extraction
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PromptFlowExecution:
    """Represents a single Prompt Flow execution with metrics"""
    execution_id: str
    flow_name: str
    domain: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    entities_extracted: int = 0
    relations_extracted: int = 0
    quality_score: float = 0.0
    error_message: Optional[str] = None


@dataclass
class PromptTemplate:
    """Represents a centralized prompt template with usage statistics"""
    template_name: str
    template_path: str
    last_modified: datetime
    version: str = "1.0.0"
    usage_count: int = 0
    avg_performance: float = 0.0
    success_rate: float = 0.0


class PromptFlowMonitor:
    """
    Comprehensive monitoring for Azure Prompt Flow universal extraction
    Tracks performance, costs, quality, and template usage
    """
    
    def __init__(self):
        self.monitoring_enabled = getattr(settings, 'enable_prompt_flow_monitoring', True)
        self.metrics_storage_path = Path(settings.BASE_DIR) / "data" / "metrics" / "prompt_flow"
        self.metrics_storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory tracking
        self.active_executions: Dict[str, PromptFlowExecution] = {}
        self.completed_executions: List[PromptFlowExecution] = []
        self.template_metrics: Dict[str, PromptTemplate] = {}
        
        # Performance thresholds
        self.performance_thresholds = {
            "max_execution_time_seconds": 300,
            "min_quality_score": 0.7,
            "max_cost_per_execution": 5.0,
            "min_success_rate": 0.85
        }
        
        logger.info(f"PromptFlowMonitor initialized - Monitoring enabled: {self.monitoring_enabled}")
    
    def start_execution_tracking(
        self,
        execution_id: str,
        flow_name: str = "universal_knowledge_extraction",
        domain: str = "general"
    ) -> str:
        """Start tracking a new Prompt Flow execution"""
        if not self.monitoring_enabled:
            return execution_id
        
        execution = PromptFlowExecution(
            execution_id=execution_id,
            flow_name=flow_name,
            domain=domain,
            start_time=datetime.now()
        )
        
        self.active_executions[execution_id] = execution
        logger.info(f"Started tracking execution: {execution_id}")
        
        return execution_id
    
    def end_execution_tracking(
        self,
        execution_id: str,
        status: str = "completed",
        results: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Optional[PromptFlowExecution]:
        """End tracking for a Prompt Flow execution"""
        if not self.monitoring_enabled or execution_id not in self.active_executions:
            return None
        
        execution = self.active_executions[execution_id]
        execution.end_time = datetime.now()
        execution.status = status
        execution.error_message = error_message
        
        # Extract metrics from results
        if results:
            execution.entities_extracted = len(results.get("entities", []))
            execution.relations_extracted = len(results.get("relations", []))
            
            quality_assessment = results.get("quality_assessment", {})
            execution.quality_score = quality_assessment.get("overall_score", 0.0)
            
            # Estimate token usage and costs
            execution.input_tokens = self._estimate_input_tokens(results)
            execution.output_tokens = self._estimate_output_tokens(results)
            execution.total_cost = self._estimate_execution_cost(execution)
        
        # Move to completed executions
        self.completed_executions.append(execution)
        del self.active_executions[execution_id]
        
        # Update template metrics
        self._update_template_metrics(execution)
        
        # Check for performance alerts
        self._check_performance_alerts(execution)
        
        logger.info(f"Completed tracking execution: {execution_id} ({status})")
        return execution
    
    def track_template_usage(
        self,
        template_name: str,
        template_path: str,
        performance_score: float = 0.0,
        success: bool = True
    ) -> None:
        """Track usage statistics for centralized prompt templates"""
        if not self.monitoring_enabled:
            return
        
        if template_name not in self.template_metrics:
            self.template_metrics[template_name] = PromptTemplate(
                template_name=template_name,
                template_path=template_path,
                last_modified=datetime.now()
            )
        
        template = self.template_metrics[template_name]
        template.usage_count += 1
        
        # Update performance metrics
        if template.avg_performance == 0:
            template.avg_performance = performance_score
        else:
            template.avg_performance = (template.avg_performance + performance_score) / 2
        
        # Update success rate
        if template.success_rate == 0:
            template.success_rate = 1.0 if success else 0.0
        else:
            success_count = template.usage_count * template.success_rate
            if success:
                success_count += 1
            template.success_rate = success_count / template.usage_count
        
        logger.debug(f"Updated template metrics: {template_name}")
    
    def get_execution_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get execution metrics for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_executions = [
            exec for exec in self.completed_executions
            if exec.start_time >= cutoff_time
        ]
        
        if not recent_executions:
            return {
                "total_executions": 0,
                "avg_execution_time": 0.0,
                "success_rate": 0.0,
                "total_cost": 0.0,
                "avg_quality_score": 0.0,
                "entities_per_execution": 0.0,
                "relations_per_execution": 0.0
            }
        
        # Calculate metrics
        total_executions = len(recent_executions)
        successful_executions = [e for e in recent_executions if e.status == "completed"]
        
        avg_execution_time = sum(
            (e.end_time - e.start_time).total_seconds() 
            for e in recent_executions if e.end_time
        ) / len(recent_executions)
        
        success_rate = len(successful_executions) / total_executions
        total_cost = sum(e.total_cost for e in recent_executions)
        
        avg_quality_score = sum(e.quality_score for e in successful_executions) / len(successful_executions) if successful_executions else 0.0
        
        avg_entities = sum(e.entities_extracted for e in successful_executions) / len(successful_executions) if successful_executions else 0.0
        avg_relations = sum(e.relations_extracted for e in successful_executions) / len(successful_executions) if successful_executions else 0.0
        
        return {
            "time_period_hours": hours,
            "total_executions": total_executions,
            "successful_executions": len(successful_executions),
            "avg_execution_time_seconds": round(avg_execution_time, 2),
            "success_rate": round(success_rate, 3),
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_per_execution": round(total_cost / total_executions, 4) if total_executions > 0 else 0.0,
            "avg_quality_score": round(avg_quality_score, 3),
            "entities_per_execution": round(avg_entities, 1),
            "relations_per_execution": round(avg_relations, 1),
            "total_tokens_used": sum(e.input_tokens + e.output_tokens for e in recent_executions)
        }
    
    def get_template_analytics(self) -> Dict[str, Any]:
        """Get analytics for centralized prompt templates"""
        if not self.template_metrics:
            return {"templates": [], "total_templates": 0}
        
        template_data = []
        for template in self.template_metrics.values():
            template_data.append({
                "name": template.template_name,
                "usage_count": template.usage_count,
                "avg_performance": round(template.avg_performance, 3),
                "success_rate": round(template.success_rate, 3),
                "last_modified": template.last_modified.isoformat(),
                "version": template.version
            })
        
        # Sort by usage count
        template_data.sort(key=lambda x: x["usage_count"], reverse=True)
        
        return {
            "templates": template_data,
            "total_templates": len(template_data),
            "most_used_template": template_data[0]["name"] if template_data else None,
            "avg_template_performance": round(
                sum(t["avg_performance"] for t in template_data) / len(template_data), 3
            ) if template_data else 0.0
        }
    
    def export_metrics(self, output_path: Optional[Path] = None) -> Path:
        """Export comprehensive metrics to JSON file"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.metrics_storage_path / f"prompt_flow_metrics_{timestamp}.json"
        
        metrics_data = {
            "export_timestamp": datetime.now().isoformat(),
            "monitoring_enabled": self.monitoring_enabled,
            "execution_metrics_24h": self.get_execution_metrics(24),
            "execution_metrics_7d": self.get_execution_metrics(168),  # 7 days
            "template_analytics": self.get_template_analytics(),
            "active_executions_count": len(self.active_executions),
            "completed_executions_count": len(self.completed_executions),
            "performance_thresholds": self.performance_thresholds,
            "recent_executions": [
                {
                    "execution_id": exec.execution_id,
                    "flow_name": exec.flow_name,
                    "domain": exec.domain,
                    "status": exec.status,
                    "duration_seconds": (exec.end_time - exec.start_time).total_seconds() if exec.end_time else 0,
                    "entities_extracted": exec.entities_extracted,
                    "relations_extracted": exec.relations_extracted,
                    "quality_score": exec.quality_score,
                    "total_cost": exec.total_cost
                }
                for exec in self.completed_executions[-10:]  # Last 10 executions
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to: {output_path}")
        return output_path
    
    def _estimate_input_tokens(self, results: Dict[str, Any]) -> int:
        """Estimate input tokens based on processed texts"""
        # Simple estimation: ~4 characters per token
        input_text_length = sum(len(str(text)) for text in results.get("input_texts", []))
        return int(input_text_length / 4)
    
    def _estimate_output_tokens(self, results: Dict[str, Any]) -> int:
        """Estimate output tokens based on extracted entities and relations"""
        entities = results.get("entities", [])
        relations = results.get("relations", [])
        
        # Estimate tokens for JSON output
        output_length = sum(len(str(e)) for e in entities) + sum(len(str(r)) for r in relations)
        return int(output_length / 4)
    
    def _estimate_execution_cost(self, execution: PromptFlowExecution) -> float:
        """Estimate cost based on token usage (GPT-4 pricing)"""
        # GPT-4 pricing (approximate, as of 2025)
        input_cost_per_1k = 0.03  # $0.03 per 1K input tokens
        output_cost_per_1k = 0.06  # $0.06 per 1K output tokens
        
        input_cost = (execution.input_tokens / 1000) * input_cost_per_1k
        output_cost = (execution.output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def _update_template_metrics(self, execution: PromptFlowExecution) -> None:
        """Update template metrics based on execution results"""
        template_name = f"{execution.flow_name}_entity_extraction"
        self.track_template_usage(
            template_name=template_name,
            template_path=f"prompt_flows/{execution.flow_name}/entity_extraction.jinja2",
            performance_score=execution.quality_score,
            success=execution.status == "completed"
        )
    
    def _check_performance_alerts(self, execution: PromptFlowExecution) -> None:
        """Check for performance issues and log alerts"""
        if not execution.end_time:
            return
        
        duration = (execution.end_time - execution.start_time).total_seconds()
        
        # Check execution time
        if duration > self.performance_thresholds["max_execution_time_seconds"]:
            logger.warning(f"Slow execution detected: {execution.execution_id} took {duration:.1f}s")
        
        # Check quality score
        if execution.quality_score < self.performance_thresholds["min_quality_score"]:
            logger.warning(f"Low quality score: {execution.execution_id} scored {execution.quality_score:.3f}")
        
        # Check cost
        if execution.total_cost > self.performance_thresholds["max_cost_per_execution"]:
            logger.warning(f"High cost execution: {execution.execution_id} cost ${execution.total_cost:.4f}")


# Global monitor instance
prompt_flow_monitor = PromptFlowMonitor()


# Convenience functions
def start_tracking(execution_id: str, flow_name: str = "universal_knowledge_extraction", domain: str = "general") -> str:
    """Start tracking a Prompt Flow execution"""
    return prompt_flow_monitor.start_execution_tracking(execution_id, flow_name, domain)


def end_tracking(execution_id: str, status: str = "completed", results: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Optional[PromptFlowExecution]:
    """End tracking for a Prompt Flow execution"""
    return prompt_flow_monitor.end_execution_tracking(execution_id, status, results, error)


def get_current_metrics() -> Dict[str, Any]:
    """Get current performance metrics"""
    return prompt_flow_monitor.get_execution_metrics(24)


if __name__ == "__main__":
    # Test the monitoring system
    monitor = PromptFlowMonitor()
    
    # Simulate execution tracking
    exec_id = "test_execution_123"
    monitor.start_execution_tracking(exec_id, "universal_knowledge_extraction", "test")
    
    # Simulate completion
    test_results = {
        "entities": [{"text": "valve"}, {"text": "bearing"}],
        "relations": [{"relation_type": "connected_to"}],
        "quality_assessment": {"overall_score": 0.85}
    }
    
    monitor.end_execution_tracking(exec_id, "completed", test_results)
    
    # Get metrics
    metrics = monitor.get_execution_metrics(24)
    print(json.dumps(metrics, indent=2))