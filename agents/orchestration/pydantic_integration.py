"""
PydanticAI Orchestration Integration
===================================

This module provides integration between PydanticAI agents and the workflow orchestrator,
enabling enterprise-grade agent orchestration with performance monitoring and SLA compliance.

Features:
- PydanticAI agent delegation patterns
- Workflow orchestrator integration
- Performance SLA monitoring
- Multi-agent coordination
- Enterprise error handling and recovery
- Competitive advantage preservation
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from pydantic_ai import Agent, RunContext

# Import our orchestration components
from .workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowRequest,
    WorkflowResults,
    WorkflowProgress,
    WorkflowStage
)

# Import PydanticAI integration
from ..pydantic_ai_integration import (
    # azure_rag_agent,  # Commented out - not available yet
    create_pydantic_ai_agent,
    process_intelligent_query,
    PydanticAIDependencies,
    PydanticAIQueryResponse
)

# Import core services
from ..core.azure_services import ConsolidatedAzureServices
from ..core.pydantic_ai_provider import (
    create_pydantic_agent_async,
    create_azure_pydantic_provider_async
)

logger = logging.getLogger(__name__)


class AgentDelegationStrategy(Enum):
    """Strategy for delegating work to PydanticAI agents"""
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT_PARALLEL = "multi_agent_parallel"
    MULTI_AGENT_SEQUENTIAL = "multi_agent_sequential"
    ADAPTIVE_DELEGATION = "adaptive_delegation"


@dataclass
class PydanticAgentConfig:
    """Configuration for PydanticAI agent in orchestration"""
    agent_name: str
    model_name: str = "gpt-4o"
    system_prompt: Optional[str] = None
    max_response_time: float = 3.0
    confidence_threshold: float = 0.7
    retry_attempts: int = 2


@dataclass
class AgentDelegationRequest:
    """Request for agent delegation through orchestration"""
    query: str
    domain: Optional[str] = None
    delegation_strategy: AgentDelegationStrategy = AgentDelegationStrategy.SINGLE_AGENT
    agent_configs: List[PydanticAgentConfig] = None
    workflow_context: Dict[str, Any] = None
    performance_requirements: Dict[str, Any] = None


@dataclass
class AgentDelegationResult:
    """Result from agent delegation"""
    success: bool
    results: List[PydanticAIQueryResponse]
    execution_time: float
    delegation_strategy_used: AgentDelegationStrategy
    agents_used: List[str]
    competitive_advantages_achieved: Dict[str, bool]
    performance_metrics: Dict[str, Any]
    error_message: Optional[str] = None


class PydanticAIWorkflowIntegration:
    """
    Integration layer between PydanticAI agents and workflow orchestration.
    
    This class provides enterprise-grade integration patterns for delegating
    complex workflows to PydanticAI agents while maintaining performance SLAs
    and competitive advantages.
    """
    
    def __init__(
        self,
        azure_services: ConsolidatedAzureServices,
        workflow_orchestrator: Optional[WorkflowOrchestrator] = None
    ):
        """
        Initialize PydanticAI workflow integration.
        
        Args:
            azure_services: ConsolidatedAzureServices instance
            workflow_orchestrator: Optional workflow orchestrator instance
        """
        self.azure_services = azure_services
        self.workflow_orchestrator = workflow_orchestrator or WorkflowOrchestrator(azure_services)
        self.active_agents: Dict[str, Agent] = {}
        self.delegation_metrics: Dict[str, Any] = {
            "total_delegations": 0,
            "successful_delegations": 0,
            "failed_delegations": 0,
            "average_execution_time": 0.0,
            "sla_compliance_rate": 0.0
        }
        
        logger.info("PydanticAI Workflow Integration initialized")
    
    async def delegate_to_agent(
        self,
        request: AgentDelegationRequest
    ) -> AgentDelegationResult:
        """
        Delegate work to PydanticAI agents using specified strategy.
        
        Args:
            request: Agent delegation request
            
        Returns:
            AgentDelegationResult with delegation outcomes
        """
        start_time = time.time()
        
        logger.info(
            f"PydanticAI delegation starting: strategy={request.delegation_strategy.value}, "
            f"query={request.query[:50]}..."
        )
        
        try:
            # Update metrics
            self.delegation_metrics["total_delegations"] += 1
            
            # Execute delegation based on strategy
            if request.delegation_strategy == AgentDelegationStrategy.SINGLE_AGENT:
                result = await self._single_agent_delegation(request)
            elif request.delegation_strategy == AgentDelegationStrategy.MULTI_AGENT_PARALLEL:
                result = await self._multi_agent_parallel_delegation(request)
            elif request.delegation_strategy == AgentDelegationStrategy.MULTI_AGENT_SEQUENTIAL:
                result = await self._multi_agent_sequential_delegation(request)
            elif request.delegation_strategy == AgentDelegationStrategy.ADAPTIVE_DELEGATION:
                result = await self._adaptive_delegation(request)
            else:
                raise ValueError(f"Unknown delegation strategy: {request.delegation_strategy}")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Update success metrics
            if result.success:
                self.delegation_metrics["successful_delegations"] += 1
                
                # Check SLA compliance
                max_response_time = request.performance_requirements.get("max_response_time", 3.0) if request.performance_requirements else 3.0
                sla_met = execution_time <= max_response_time
                
                if sla_met:
                    result.competitive_advantages_achieved["sub_3s_response_sla"] = True
                
                # Update SLA compliance rate
                total_delegations = self.delegation_metrics["total_delegations"]
                current_compliance = self.delegation_metrics["sla_compliance_rate"] * (total_delegations - 1)
                self.delegation_metrics["sla_compliance_rate"] = (current_compliance + (1 if sla_met else 0)) / total_delegations
            else:
                self.delegation_metrics["failed_delegations"] += 1
            
            # Update average execution time
            total_delegations = self.delegation_metrics["total_delegations"]
            current_avg = self.delegation_metrics["average_execution_time"] * (total_delegations - 1)
            self.delegation_metrics["average_execution_time"] = (current_avg + execution_time) / total_delegations
            
            logger.info(
                f"PydanticAI delegation completed: success={result.success}, "
                f"time={execution_time:.2f}s, agents={len(result.agents_used)}"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.delegation_metrics["failed_delegations"] += 1
            
            logger.error(f"PydanticAI delegation failed: {e}")
            
            return AgentDelegationResult(
                success=False,
                results=[],
                execution_time=execution_time,
                delegation_strategy_used=request.delegation_strategy,
                agents_used=[],
                competitive_advantages_achieved={},
                performance_metrics={"error": str(e)},
                error_message=str(e)
            )
    
    async def _single_agent_delegation(
        self,
        request: AgentDelegationRequest
    ) -> AgentDelegationResult:
        """Execute single agent delegation"""
        try:
            # Create or get agent
            agent_config = request.agent_configs[0] if request.agent_configs else PydanticAgentConfig("primary-agent")
            agent = await self._get_or_create_agent(agent_config)
            
            # Create dependencies
            dependencies = PydanticAIDependencies(
                azure_services=self.azure_services,
                app_settings=None  # Will use defaults
            )
            
            # Execute agent
            result = await agent.run(request.query, deps=dependencies)
            
            return AgentDelegationResult(
                success=True,
                results=[result],
                execution_time=0.0,  # Will be set by caller
                delegation_strategy_used=AgentDelegationStrategy.SINGLE_AGENT,
                agents_used=[agent_config.agent_name],
                competitive_advantages_achieved={
                    "pydantic_ai_integration": True,
                    "enterprise_orchestration": True
                },
                performance_metrics={
                    "confidence": result.confidence,
                    "agents_count": 1
                }
            )
            
        except Exception as e:
            logger.error(f"Single agent delegation failed: {e}")
            raise
    
    async def _multi_agent_parallel_delegation(
        self,
        request: AgentDelegationRequest
    ) -> AgentDelegationResult:
        """Execute multi-agent parallel delegation"""
        try:
            if not request.agent_configs:
                # Create default multi-agent configuration
                request.agent_configs = [
                    PydanticAgentConfig("search-agent", system_prompt="You specialize in search and retrieval."),
                    PydanticAgentConfig("analysis-agent", system_prompt="You specialize in analysis and synthesis."),
                    PydanticAgentConfig("domain-agent", system_prompt="You specialize in domain understanding.")
                ]
            
            # Create agents
            agents = []
            for config in request.agent_configs:
                agent = await self._get_or_create_agent(config)
                agents.append((agent, config))
            
            # Create dependencies
            dependencies = PydanticAIDependencies(
                azure_services=self.azure_services,
                app_settings=None  # Will use defaults
            )
            
            # Execute agents in parallel
            tasks = []
            for agent, config in agents:
                task = agent.run(request.query, deps=dependencies)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_results = []
            failed_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_count += 1
                    logger.error(f"Agent {request.agent_configs[i].agent_name} failed: {result}")
                else:
                    successful_results.append(result)
            
            return AgentDelegationResult(
                success=len(successful_results) > 0,
                results=successful_results,
                execution_time=0.0,  # Will be set by caller
                delegation_strategy_used=AgentDelegationStrategy.MULTI_AGENT_PARALLEL,
                agents_used=[config.agent_name for config in request.agent_configs],
                competitive_advantages_achieved={
                    "pydantic_ai_integration": True,
                    "enterprise_orchestration": True,
                    "multi_agent_coordination": True
                },
                performance_metrics={
                    "successful_agents": len(successful_results),
                    "failed_agents": failed_count,
                    "total_agents": len(agents)
                }
            )
            
        except Exception as e:
            logger.error(f"Multi-agent parallel delegation failed: {e}")
            raise
    
    async def _multi_agent_sequential_delegation(
        self,
        request: AgentDelegationRequest
    ) -> AgentDelegationResult:
        """Execute multi-agent sequential delegation"""
        try:
            if not request.agent_configs:
                # Create default sequential agent configuration
                request.agent_configs = [
                    PydanticAgentConfig("domain-agent", system_prompt="First, identify the domain and context."),
                    PydanticAgentConfig("search-agent", system_prompt="Then, perform comprehensive search."),
                    PydanticAgentConfig("synthesis-agent", system_prompt="Finally, synthesize and format results.")
                ]
            
            # Execute agents sequentially
            results = []
            current_query = request.query
            
            dependencies = PydanticAIDependencies(
                azure_services=self.azure_services,
                app_settings=None  # Will use defaults
            )
            
            for config in request.agent_configs:
                agent = await self._get_or_create_agent(config)
                
                # Execute agent with current query/context
                result = await agent.run(current_query, deps=dependencies)
                results.append(result)
                
                # Use result as context for next agent
                current_query = f"Previous context: {result.results}\n\nContinue with: {request.query}"
            
            return AgentDelegationResult(
                success=True,
                results=results,
                execution_time=0.0,  # Will be set by caller
                delegation_strategy_used=AgentDelegationStrategy.MULTI_AGENT_SEQUENTIAL,
                agents_used=[config.agent_name for config in request.agent_configs],
                competitive_advantages_achieved={
                    "pydantic_ai_integration": True,
                    "enterprise_orchestration": True,
                    "sequential_refinement": True
                },
                performance_metrics={
                    "agents_executed": len(results),
                    "sequential_improvement": True
                }
            )
            
        except Exception as e:
            logger.error(f"Multi-agent sequential delegation failed: {e}")
            raise
    
    async def _adaptive_delegation(
        self,
        request: AgentDelegationRequest
    ) -> AgentDelegationResult:
        """Execute adaptive delegation based on query complexity"""
        try:
            # Analyze query complexity
            query_length = len(request.query)
            domain_complexity = len(request.domain.split()) if request.domain else 0
            
            # Adaptive strategy selection
            if query_length < 50 and domain_complexity < 3:
                # Simple query -> single agent
                request.delegation_strategy = AgentDelegationStrategy.SINGLE_AGENT
                return await self._single_agent_delegation(request)
            elif query_length > 200 or domain_complexity > 5:
                # Complex query -> sequential multi-agent
                request.delegation_strategy = AgentDelegationStrategy.MULTI_AGENT_SEQUENTIAL
                return await self._multi_agent_sequential_delegation(request)
            else:
                # Medium complexity -> parallel multi-agent
                request.delegation_strategy = AgentDelegationStrategy.MULTI_AGENT_PARALLEL
                return await self._multi_agent_parallel_delegation(request)
                
        except Exception as e:
            logger.error(f"Adaptive delegation failed: {e}")
            raise
    
    async def _get_or_create_agent(self, config: PydanticAgentConfig) -> Agent:
        """Get existing agent or create new one"""
        if config.agent_name not in self.active_agents:
            agent = await create_pydantic_agent_async(
                model_name=config.model_name,
                system_prompt=config.system_prompt,
                agent_name=config.agent_name,
                azure_services=self.azure_services
            )
            
            if agent:
                self.active_agents[config.agent_name] = agent
                logger.info(f"Created PydanticAI agent: {config.agent_name}")
            else:
                raise RuntimeError(f"Failed to create agent: {config.agent_name}")
        
        return self.active_agents[config.agent_name]
    
    async def integrate_with_workflow(
        self,
        workflow_request: WorkflowRequest,
        use_pydantic_agents: bool = True
    ) -> WorkflowResults:
        """
        Integrate PydanticAI agents with workflow orchestration.
        
        Args:
            workflow_request: Workflow request to process
            use_pydantic_agents: Whether to use PydanticAI agents for processing
            
        Returns:
            WorkflowResults with PydanticAI integration
        """
        try:
            if not use_pydantic_agents:
                # Fall back to standard workflow orchestration
                return await self.workflow_orchestrator.process_workflow(workflow_request)
            
            # Enhanced workflow with PydanticAI integration
            logger.info(f"Processing workflow with PydanticAI integration: {workflow_request.query[:50]}...")
            
            # Create agent delegation request from workflow request
            delegation_request = AgentDelegationRequest(
                query=workflow_request.query,
                domain=workflow_request.domain,
                delegation_strategy=AgentDelegationStrategy.ADAPTIVE_DELEGATION,
                workflow_context=workflow_request.context,
                performance_requirements=workflow_request.performance_requirements
            )
            
            # Execute agent delegation
            delegation_result = await self.delegate_to_agent(delegation_request)
            
            # Convert to workflow results
            if delegation_result.success and delegation_result.results:
                primary_result = delegation_result.results[0]
                
                workflow_results = WorkflowResults(
                    query=workflow_request.query,
                    results=primary_result.results,
                    confidence=primary_result.confidence,
                    execution_time=delegation_result.execution_time,
                    performance_met=delegation_result.competitive_advantages_achieved.get("sub_3s_response_sla", False),
                    workflow_metadata={
                        "pydantic_ai_integration": True,
                        "delegation_strategy": delegation_result.delegation_strategy_used.value,
                        "agents_used": delegation_result.agents_used,
                        "competitive_advantages": delegation_result.competitive_advantages_achieved
                    }
                )
                
                logger.info(f"Workflow with PydanticAI completed successfully: {delegation_result.execution_time:.2f}s")
                return workflow_results
            else:
                # Handle delegation failure
                logger.error(f"PydanticAI delegation failed: {delegation_result.error_message}")
                
                # Fall back to standard workflow
                return await self.workflow_orchestrator.process_workflow(workflow_request)
                
        except Exception as e:
            logger.error(f"Workflow PydanticAI integration failed: {e}")
            
            # Fall back to standard workflow
            return await self.workflow_orchestrator.process_workflow(workflow_request)
    
    def get_delegation_metrics(self) -> Dict[str, Any]:
        """Get current delegation metrics"""
        return {
            **self.delegation_metrics,
            "active_agents": len(self.active_agents),
            "agent_names": list(self.active_agents.keys())
        }
    
    async def shutdown(self):
        """Shutdown the integration and cleanup resources"""
        logger.info("Shutting down PydanticAI Workflow Integration")
        self.active_agents.clear()


# Export key components
__all__ = [
    'PydanticAIWorkflowIntegration',
    'AgentDelegationRequest',
    'AgentDelegationResult',
    'AgentDelegationStrategy',
    'PydanticAgentConfig'
]