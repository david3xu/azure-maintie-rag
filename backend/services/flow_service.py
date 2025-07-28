"""
Flow Service - Business logic for Azure Prompt Flow integration and monitoring
Consolidated from core/prompt_flow/prompt_flow_integration.py and prompt_flow_monitoring.py
"""

import json
import logging
import asyncio
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

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
    cost_estimate: float = 0.0
    quality_score: Optional[float] = None
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class PromptFlowMetrics:
    """Aggregated metrics for prompt flow performance"""
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_duration_seconds: float
    total_cost: float
    average_quality_score: Optional[float]
    token_usage: Dict[str, int]
    performance_trends: Dict[str, List[float]]


class PromptFlowMonitor:
    """
    Monitoring and analytics for Azure Prompt Flow executions
    Tracks performance, costs, and quality metrics
    """
    
    def __init__(self):
        self.executions: List[PromptFlowExecution] = []
        self.flow_configs: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[Dict]] = {}
        
        logger.info("PromptFlowMonitor initialized")
    
    def start_execution(self, flow_name: str, domain: str, execution_id: str = None) -> str:
        """Start tracking a new prompt flow execution"""
        if not execution_id:
            execution_id = f"{flow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(datetime.now())}"[:32]
        
        execution = PromptFlowExecution(
            execution_id=execution_id,
            flow_name=flow_name,
            domain=domain,
            start_time=datetime.now()
        )
        
        self.executions.append(execution)
        logger.info(f"Started tracking execution: {execution_id}")
        
        return execution_id
    
    def complete_execution(
        self,
        execution_id: str,
        status: str = "completed",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_estimate: float = 0.0,
        quality_score: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Optional[PromptFlowExecution]:
        """Complete a prompt flow execution with metrics"""
        
        execution = self._find_execution(execution_id)
        if not execution:
            logger.warning(f"Execution {execution_id} not found")
            return None
        
        execution.end_time = datetime.now()
        execution.status = status
        execution.input_tokens = input_tokens
        execution.output_tokens = output_tokens
        execution.cost_estimate = cost_estimate
        execution.quality_score = quality_score
        execution.error_message = error_message
        
        # Update performance history
        self._update_performance_history(execution)
        
        logger.info(f"Completed execution {execution_id} with status: {status}")
        return execution
    
    def get_execution_metrics(self, flow_name: Optional[str] = None, hours: int = 24) -> PromptFlowMetrics:
        """Get aggregated metrics for executions"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter executions
        filtered_executions = [
            ex for ex in self.executions
            if ex.start_time >= cutoff_time and (not flow_name or ex.flow_name == flow_name)
        ]
        
        if not filtered_executions:
            return PromptFlowMetrics(
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                average_duration_seconds=0.0,
                total_cost=0.0,
                average_quality_score=None,
                token_usage={"input": 0, "output": 0},
                performance_trends={}
            )
        
        # Calculate metrics
        successful = [ex for ex in filtered_executions if ex.status == "completed"]
        failed = [ex for ex in filtered_executions if ex.status == "failed"]
        
        durations = [ex.duration_seconds for ex in filtered_executions if ex.duration_seconds]
        quality_scores = [ex.quality_score for ex in successful if ex.quality_score is not None]
        
        metrics = PromptFlowMetrics(
            total_executions=len(filtered_executions),
            successful_executions=len(successful),
            failed_executions=len(failed),
            average_duration_seconds=sum(durations) / len(durations) if durations else 0.0,
            total_cost=sum(ex.cost_estimate for ex in filtered_executions),
            average_quality_score=sum(quality_scores) / len(quality_scores) if quality_scores else None,
            token_usage={
                "input": sum(ex.input_tokens for ex in filtered_executions),
                "output": sum(ex.output_tokens for ex in filtered_executions)
            },
            performance_trends=self._calculate_performance_trends(filtered_executions)
        )
        
        return metrics
    
    def _find_execution(self, execution_id: str) -> Optional[PromptFlowExecution]:
        """Find execution by ID"""
        for execution in self.executions:
            if execution.execution_id == execution_id:
                return execution
        return None
    
    def _update_performance_history(self, execution: PromptFlowExecution):
        """Update performance history for trending analysis"""
        flow_name = execution.flow_name
        
        if flow_name not in self.performance_history:
            self.performance_history[flow_name] = []
        
        history_entry = {
            "timestamp": execution.end_time.isoformat() if execution.end_time else None,
            "duration": execution.duration_seconds,
            "cost": execution.cost_estimate,
            "quality": execution.quality_score,
            "tokens": execution.input_tokens + execution.output_tokens,
            "status": execution.status
        }
        
        self.performance_history[flow_name].append(history_entry)
        
        # Keep only last 100 entries per flow
        if len(self.performance_history[flow_name]) > 100:
            self.performance_history[flow_name] = self.performance_history[flow_name][-100:]
    
    def _calculate_performance_trends(self, executions: List[PromptFlowExecution]) -> Dict[str, List[float]]:
        """Calculate performance trends from execution data"""
        trends = {
            "duration_trend": [],
            "cost_trend": [],
            "quality_trend": [],
            "success_rate_trend": []
        }
        
        # Group executions by hour for trending
        hourly_groups = {}
        for execution in executions:
            hour_key = execution.start_time.strftime("%Y-%m-%d-%H")
            if hour_key not in hourly_groups:
                hourly_groups[hour_key] = []
            hourly_groups[hour_key].append(execution)
        
        # Calculate hourly trends
        for hour_key, hour_executions in sorted(hourly_groups.items()):
            durations = [ex.duration_seconds for ex in hour_executions if ex.duration_seconds]
            costs = [ex.cost_estimate for ex in hour_executions]
            qualities = [ex.quality_score for ex in hour_executions if ex.quality_score is not None]
            successful = len([ex for ex in hour_executions if ex.status == "completed"])
            
            trends["duration_trend"].append(sum(durations) / len(durations) if durations else 0)
            trends["cost_trend"].append(sum(costs))
            trends["quality_trend"].append(sum(qualities) / len(qualities) if qualities else 0)
            trends["success_rate_trend"].append(successful / len(hour_executions) if hour_executions else 0)
        
        return trends


class AzurePromptFlowIntegrator:
    """
    Integration service for Azure Prompt Flow with universal knowledge extraction
    Provides centralized prompt management while preserving universal principles
    """
    
    def __init__(self, domain_name: str = "general"):
        self.domain_name = domain_name
        self.monitor = PromptFlowMonitor()
        self.flow_configs = {}
        self.flow_base_path = Path(settings.BASE_DIR) / "prompt_flows"
        
        logger.info(f"AzurePromptFlowIntegrator initialized for domain: {domain_name}")
    
    async def execute_flow(
        self,
        flow_name: str,
        input_data: Dict[str, Any],
        flow_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a prompt flow with monitoring"""
        
        execution_id = self.monitor.start_execution(flow_name, self.domain_name)
        
        try:
            # Validate flow exists
            flow_path = self.flow_base_path / flow_name
            if not flow_path.exists():
                raise FileNotFoundError(f"Prompt flow not found: {flow_name}")
            
            # Execute the flow
            result = await self._execute_prompt_flow(flow_path, input_data, flow_config)
            
            # Calculate metrics
            input_tokens = self._estimate_tokens(str(input_data))
            output_tokens = self._estimate_tokens(str(result.get("output", "")))
            cost_estimate = self._calculate_cost(input_tokens, output_tokens)
            quality_score = self._assess_quality(result)
            
            # Complete monitoring
            self.monitor.complete_execution(
                execution_id=execution_id,
                status="completed",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_estimate=cost_estimate,
                quality_score=quality_score
            )
            
            logger.info(f"Successfully executed flow: {flow_name}")
            return {
                "execution_id": execution_id,
                "status": "success",
                "result": result,
                "metrics": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_estimate": cost_estimate,
                    "quality_score": quality_score
                }
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Flow execution failed: {error_message}")
            
            self.monitor.complete_execution(
                execution_id=execution_id,
                status="failed",
                error_message=error_message
            )
            
            return {
                "execution_id": execution_id,
                "status": "error",
                "error": error_message
            }
    
    async def _execute_prompt_flow(
        self,
        flow_path: Path,
        input_data: Dict[str, Any],
        flow_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the actual prompt flow"""
        
        # Check if prompt flow is enabled
        if not settings.enable_prompt_flow:
            logger.info("Prompt flow disabled, using fallback processing")
            return await self._fallback_processing(input_data)
        
        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_input:
                json.dump(input_data, temp_input)
                temp_input_path = temp_input.name
            
            # Execute prompt flow using pf CLI
            cmd = [
                "pf", "flow", "test",
                "--flow", str(flow_path),
                "--inputs", temp_input_path
            ]
            
            # Add configuration if provided
            if flow_config:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
                    json.dump(flow_config, temp_config)
                    cmd.extend(["--config", temp_config.name])
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Clean up temporary files
            os.unlink(temp_input_path)
            if flow_config:
                os.unlink(temp_config.name)
            
            if result.returncode != 0:
                raise RuntimeError(f"Prompt flow execution failed: {result.stderr}")
            
            # Parse result
            try:
                output = json.loads(result.stdout)
                return {"output": output, "raw_stdout": result.stdout}
            except json.JSONDecodeError:
                return {"output": result.stdout, "raw_stdout": result.stdout}
                
        except Exception as e:
            logger.warning(f"Prompt flow execution failed, using fallback: {str(e)}")
            if settings.prompt_flow_fallback_enabled:
                return await self._fallback_processing(input_data)
            else:
                raise
    
    async def _fallback_processing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing when prompt flow is unavailable"""
        logger.info("Using fallback processing for prompt flow")
        
        # Simple fallback - just return processed input
        return {
            "output": {
                "status": "processed_with_fallback",
                "input_summary": f"Processed {len(str(input_data))} characters of input data",
                "domain": self.domain_name,
                "timestamp": datetime.now().isoformat()
            },
            "fallback_used": True
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for execution"""
        # Example rates (adjust based on actual Azure OpenAI pricing)
        input_rate = 0.00003  # per token
        output_rate = 0.00006  # per token
        
        return (input_tokens * input_rate) + (output_tokens * output_rate)
    
    def _assess_quality(self, result: Dict[str, Any]) -> Optional[float]:
        """Assess quality of flow execution result"""
        try:
            # Simple quality assessment based on result characteristics
            output = result.get("output", {})
            
            if isinstance(output, dict):
                # Check for expected keys, completeness, etc.
                quality_indicators = [
                    "status" in output,
                    len(str(output)) > 10,  # Non-trivial output
                    not result.get("fallback_used", False)  # Didn't use fallback
                ]
                
                return sum(quality_indicators) / len(quality_indicators)
            
            return 0.5  # Default quality for non-dict outputs
            
        except Exception:
            return None
    
    def get_flow_metrics(self, flow_name: Optional[str] = None, hours: int = 24) -> PromptFlowMetrics:
        """Get metrics for flow executions"""
        return self.monitor.get_execution_metrics(flow_name, hours)
    
    def list_available_flows(self) -> List[str]:
        """List available prompt flows"""
        flows = []
        
        if self.flow_base_path.exists():
            for item in self.flow_base_path.iterdir():
                if item.is_dir() and (item / "flow.dag.yaml").exists():
                    flows.append(item.name)
        
        return flows
    
    def get_flow_config(self, flow_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific flow"""
        flow_path = self.flow_base_path / flow_name / "flow.dag.yaml"
        
        if flow_path.exists():
            try:
                import yaml
                with open(flow_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load flow config: {str(e)}")
        
        return None


class FlowService:
    """
    Unified Flow Service
    Provides Azure Prompt Flow integration and monitoring capabilities
    """
    
    def __init__(self, domain_name: str = "general"):
        self.domain_name = domain_name
        self.integrator = AzurePromptFlowIntegrator(domain_name)
        self.monitor = self.integrator.monitor
        
        logger.info(f"FlowService initialized for domain: {domain_name}")
    
    async def execute_flow(
        self,
        flow_name: str,
        input_data: Dict[str, Any],
        flow_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a prompt flow with full monitoring"""
        return await self.integrator.execute_flow(flow_name, input_data, flow_config)
    
    def get_metrics(self, flow_name: Optional[str] = None, hours: int = 24) -> PromptFlowMetrics:
        """Get flow execution metrics"""
        return self.integrator.get_flow_metrics(flow_name, hours)
    
    def list_flows(self) -> List[str]:
        """List available flows"""
        return self.integrator.list_available_flows()
    
    def get_flow_configuration(self, flow_name: str) -> Optional[Dict[str, Any]]:
        """Get flow configuration"""
        return self.integrator.get_flow_config(flow_name)
    
    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        recent_executions = sorted(
            self.monitor.executions,
            key=lambda x: x.start_time,
            reverse=True
        )[:limit]
        
        return [asdict(execution) for execution in recent_executions]
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        overall_metrics = self.get_metrics(hours=168)  # Last week
        recent_metrics = self.get_metrics(hours=24)     # Last day
        
        return {
            "overall_metrics": asdict(overall_metrics),
            "recent_metrics": asdict(recent_metrics),
            "available_flows": self.list_flows(),
            "recent_executions": self.get_execution_history(10),
            "domain": self.domain_name,
            "timestamp": datetime.now().isoformat()
        }