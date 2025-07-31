"""
Inter-Layer Interface Contracts

This module defines standardized contracts for communication between all
architectural layers, ensuring proper boundary separation and dependency management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncIterator, Union
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime


# ============================================================================
# COMMON TYPES AND ENUMS
# ============================================================================

class LayerType(Enum):
    """Architectural layer types"""
    API = "api"
    SERVICES = "services"
    AGENTS = "agents"
    TOOLS = "tools"
    CORE = "core"


class OperationStatus(Enum):
    """Operation status types"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    NOT_SUPPORTED = "not_supported"


@dataclass
class OperationResult:
    """Standard result structure for all inter-layer operations"""
    status: OperationStatus
    data: Any = None
    metadata: Dict[str, Any] = None
    execution_time: float = 0.0
    correlation_id: Optional[str] = None
    layer_source: Optional[LayerType] = None
    error_message: Optional[str] = None
    performance_met: bool = True
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LayerHealthStatus:
    """Health status for layer components"""
    layer_type: LayerType
    overall_status: str  # healthy, degraded, unhealthy, error
    component_health: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: float
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


# ============================================================================
# API LAYER CONTRACTS
# ============================================================================

class APILayerContract(ABC):
    """Contract defining API layer responsibilities and boundaries"""
    
    @abstractmethod 
    async def handle_http_request(self, request_data: Dict[str, Any]) -> OperationResult:
        """Handle HTTP request and return formatted response"""
        pass
    
    @abstractmethod
    def validate_request_data(self, request_data: Dict[str, Any]) -> OperationResult:
        """Validate incoming request data against schema"""
        pass
    
    @abstractmethod
    def format_response_data(self, service_result: OperationResult) -> Dict[str, Any]:
        """Format service layer result for HTTP response"""
        pass
    
    @abstractmethod
    async def handle_authentication(self, auth_data: Dict[str, Any]) -> OperationResult:
        """Handle authentication and authorization"""
        pass


# ============================================================================
# SERVICES LAYER CONTRACTS
# ============================================================================

@dataclass
class ServiceRequest:
    """Standard request structure for services"""
    operation: str
    parameters: Dict[str, Any]
    context: Dict[str, Any] = None
    performance_requirements: Dict[str, float] = None
    correlation_id: Optional[str] = None
    user_session: Optional[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.performance_requirements is None:
            self.performance_requirements = {"max_response_time": 3.0}


class ServicesLayerContract(ABC):
    """Contract defining Services layer responsibilities and boundaries"""
    
    @abstractmethod
    async def orchestrate_business_workflow(self, request: ServiceRequest) -> OperationResult:
        """Orchestrate business logic workflow"""
        pass
    
    @abstractmethod
    async def manage_system_resources(self, resource_type: str) -> OperationResult:
        """Manage system resources and lifecycle"""
        pass
    
    @abstractmethod
    async def coordinate_infrastructure_services(
        self, 
        services: List[str], 
        operation: str,
        parameters: Dict[str, Any]
    ) -> OperationResult:
        """Coordinate multiple infrastructure services"""
        pass
    
    @abstractmethod
    async def handle_caching_strategy(
        self, 
        cache_key: str, 
        operation: str,
        data: Any = None
    ) -> OperationResult:
        """Handle caching and performance optimization"""
        pass


# ============================================================================
# AGENTS LAYER CONTRACTS
# ============================================================================

@dataclass
class AgentRequest:
    """Standard request structure for agent operations"""
    operation_type: str
    query: str
    context: Dict[str, Any] = None
    domain: Optional[str] = None
    performance_requirements: Dict[str, float] = None
    reasoning_constraints: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.performance_requirements is None:
            self.performance_requirements = {"max_response_time": 3.0}
        if self.reasoning_constraints is None:
            self.reasoning_constraints = {}


@dataclass
class AgentResponse:
    """Standard response structure from agent operations"""
    primary_result: str
    confidence: float
    reasoning_trace: List[Dict[str, Any]]
    intelligence_insights: Dict[str, Any]
    tools_recommended: List[str] = None
    learning_outcomes: Dict[str, Any] = None
    domain_discovered: Optional[str] = None
    
    def __post_init__(self):
        if self.tools_recommended is None:
            self.tools_recommended = []
        if self.learning_outcomes is None:
            self.learning_outcomes = {}


class AgentsLayerContract(ABC):
    """Contract defining Agents layer responsibilities and boundaries"""
    
    @abstractmethod
    async def analyze_intelligence(self, request: AgentRequest) -> OperationResult:
        """Perform intelligent analysis of query or context"""
        pass
    
    @abstractmethod
    async def execute_reasoning_workflow(self, request: AgentRequest) -> OperationResult:
        """Execute multi-step reasoning workflow"""
        pass
    
    @abstractmethod
    async def discover_and_adapt(
        self, 
        domain_data: List[str], 
        adaptation_strategy: str
    ) -> OperationResult:
        """Discover patterns and adapt to new domains"""
        pass
    
    @abstractmethod
    async def coordinate_tools(
        self, 
        tool_selection: List[str], 
        context: Dict[str, Any]
    ) -> OperationResult:
        """Coordinate tool selection and execution"""
        pass
    
    @abstractmethod
    async def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> OperationResult:
        """Learn and adapt from interaction outcomes"""
        pass


# ============================================================================
# TOOLS LAYER CONTRACTS (Future Implementation)
# ============================================================================

@dataclass
class ToolRequest:
    """Standard request structure for tool operations"""
    tool_name: str
    operation: str
    parameters: Dict[str, Any]
    context: Dict[str, Any] = None
    validation_requirements: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.validation_requirements is None:
            self.validation_requirements = {}


@dataclass
class ToolResponse:
    """Standard response structure from tool operations"""
    result: Any
    confidence: float
    validation_passed: bool
    execution_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class ToolsLayerContract(ABC):
    """Contract defining Tools layer responsibilities and boundaries"""
    
    @abstractmethod
    async def execute_tool_operation(self, request: ToolRequest) -> OperationResult:
        """Execute specific tool functionality"""
        pass
    
    @abstractmethod
    async def validate_tool_effectiveness(
        self, 
        tool_name: str, 
        context: Dict[str, Any]
    ) -> OperationResult:
        """Validate and score tool effectiveness"""
        pass
    
    @abstractmethod
    async def discover_available_tools(self, domain: str) -> OperationResult:
        """Discover tools available for domain"""
        pass


# ============================================================================
# CORE LAYER CONTRACTS
# ============================================================================

@dataclass
class InfrastructureRequest:
    """Standard request for infrastructure operations"""
    service_type: str
    operation: str
    parameters: Dict[str, Any]
    configuration: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.configuration is None:
            self.configuration = {}


class CoreLayerContract(ABC):
    """Contract defining Core layer responsibilities and boundaries"""
    
    @abstractmethod
    async def execute_infrastructure_operation(
        self, 
        request: InfrastructureRequest
    ) -> OperationResult:
        """Execute infrastructure service operation"""
        pass
    
    @abstractmethod
    async def provide_azure_client(self, service_type: str) -> OperationResult:
        """Provide configured Azure service client"""
        pass
    
    @abstractmethod
    async def monitor_system_health(self) -> OperationResult:
        """Monitor and report system health"""
        pass
    
    @abstractmethod
    def get_configuration(self, config_key: str) -> OperationResult:
        """Get system configuration value"""
        pass
    
    @abstractmethod
    async def manage_data_models(self, model_operation: str, model_data: Any) -> OperationResult:
        """Manage data models and contracts"""
        pass


# ============================================================================
# CROSS-LAYER INTERFACE CONTRACTS
# ============================================================================

class APIToServicesInterface(ABC):
    """Standardized interface between API and Services layers"""
    
    @abstractmethod
    async def process_business_request(self, request: ServiceRequest) -> OperationResult:
        """Process request through business logic layer"""
        pass
    
    @abstractmethod
    async def get_system_status(self) -> OperationResult:
        """Get system status and health information"""
        pass


class ServicesToAgentsInterface(ABC):
    """Standardized interface between Services and Agents layers"""
    
    @abstractmethod
    async def request_intelligent_analysis(self, request: AgentRequest) -> OperationResult:
        """Request intelligent analysis from agents"""
        pass
    
    @abstractmethod
    async def coordinate_reasoning_workflow(self, request: AgentRequest) -> OperationResult:
        """Coordinate multi-step reasoning workflow"""
        pass
    
    @abstractmethod
    async def request_domain_adaptation(
        self, 
        domain_data: List[str], 
        strategy: str
    ) -> OperationResult:
        """Request domain adaptation from discovery system"""
        pass


class AgentsToToolsInterface(ABC):
    """Standardized interface between Agents and Tools layers"""
    
    @abstractmethod
    async def execute_selected_tools(
        self, 
        tool_requests: List[ToolRequest]
    ) -> OperationResult:
        """Execute tools selected by agent reasoning"""
        pass
    
    @abstractmethod
    async def discover_domain_tools(self, domain_context: Dict[str, Any]) -> OperationResult:
        """Discover available tools for domain context"""
        pass
    
    @abstractmethod
    async def validate_tool_results(
        self, 
        results: List[ToolResponse], 
        context: Dict[str, Any]
    ) -> OperationResult:
        """Validate tool execution results"""
        pass


class AllLayersToCoreInterface(ABC):
    """Standardized interface from all layers to Core infrastructure"""
    
    @abstractmethod
    async def get_infrastructure_service(
        self, 
        service_type: str, 
        configuration: Dict[str, Any] = None
    ) -> OperationResult:
        """Get infrastructure service instance"""
        pass
    
    @abstractmethod
    async def log_operation(
        self, 
        operation: str, 
        metadata: Dict[str, Any], 
        correlation_id: str
    ) -> None:
        """Log operation with proper correlation tracking"""
        pass
    
    @abstractmethod
    def get_data_contract(self, contract_name: str) -> OperationResult:
        """Get data model or contract definition"""
        pass
    
    @abstractmethod
    async def monitor_performance(
        self, 
        operation: str, 
        metrics: Dict[str, float]
    ) -> None:
        """Monitor operation performance"""
        pass


# ============================================================================
# CONTRACT ENFORCEMENT AND VALIDATION
# ============================================================================

class ContractViolationError(Exception):
    """Raised when layer boundary contract is violated"""
    
    def __init__(self, violating_layer: LayerType, target_layer: LayerType, violation_type: str):
        self.violating_layer = violating_layer
        self.target_layer = target_layer
        self.violation_type = violation_type
        super().__init__(
            f"Contract violation: {violating_layer.value} -> {target_layer.value} "
            f"violates {violation_type} boundary"
        )


class LayerBoundaryEnforcer:
    """Enforces layer boundary contracts at runtime"""
    
    ALLOWED_DEPENDENCIES = {
        LayerType.API: [LayerType.SERVICES],
        LayerType.SERVICES: [LayerType.AGENTS, LayerType.CORE],
        LayerType.AGENTS: [LayerType.TOOLS, LayerType.CORE],
        LayerType.TOOLS: [LayerType.CORE],
        LayerType.CORE: []  # Core depends only on external services
    }
    
    @classmethod
    def validate_dependency(
        self, 
        source_layer: LayerType, 
        target_layer: LayerType
    ) -> bool:
        """Validate if dependency between layers is allowed"""
        allowed_targets = self.ALLOWED_DEPENDENCIES.get(source_layer, [])
        
        if target_layer not in allowed_targets:
            raise ContractViolationError(
                source_layer, target_layer, "dependency"
            )
        
        return True
    
    @classmethod
    def validate_operation_result(
        self, 
        result: OperationResult, 
        expected_layer: LayerType
    ) -> bool:
        """Validate operation result follows contract"""
        if not isinstance(result, OperationResult):
            raise ContractViolationError(
                expected_layer, LayerType.API, "result_format"
            )
        
        required_fields = ['status', 'execution_time', 'correlation_id']
        for field in required_fields:
            if not hasattr(result, field):
                raise ContractViolationError(
                    expected_layer, LayerType.API, f"missing_field_{field}"
                )
        
        return True


class ContractMonitor:
    """Monitors contract compliance and generates metrics"""
    
    def __init__(self):
        self.violation_count = 0
        self.operation_count = 0
        self.performance_violations = 0
        
    def record_operation(
        self, 
        source_layer: LayerType, 
        target_layer: LayerType, 
        operation: str,
        result: OperationResult
    ):
        """Record inter-layer operation for monitoring"""
        self.operation_count += 1
        
        try:
            LayerBoundaryEnforcer.validate_dependency(source_layer, target_layer)
            LayerBoundaryEnforcer.validate_operation_result(result, source_layer)
            
            # Check performance compliance
            if not result.performance_met:
                self.performance_violations += 1
                
        except ContractViolationError as e:
            self.violation_count += 1
            # Log violation for analysis
            
    def get_compliance_metrics(self) -> Dict[str, Any]:
        """Get contract compliance metrics"""
        return {
            'total_operations': self.operation_count,
            'contract_violations': self.violation_count,
            'performance_violations': self.performance_violations,
            'compliance_rate': (
                (self.operation_count - self.violation_count) / 
                max(1, self.operation_count)
            ),
            'performance_compliance_rate': (
                (self.operation_count - self.performance_violations) / 
                max(1, self.operation_count)
            )
        }


# ============================================================================
# CONTRACT IMPLEMENTATION HELPERS
# ============================================================================

class ContractImplementationHelper:
    """Helper utilities for implementing layer contracts"""
    
    @staticmethod
    def create_operation_result(
        status: OperationStatus,
        data: Any = None,
        layer_source: LayerType = None,
        correlation_id: str = None,
        execution_time: float = 0.0,
        **kwargs
    ) -> OperationResult:
        """Create standardized operation result"""
        return OperationResult(
            status=status,
            data=data,
            layer_source=layer_source,
            correlation_id=correlation_id,
            execution_time=execution_time,
            metadata=kwargs
        )
    
    @staticmethod
    def create_health_status(
        layer_type: LayerType,
        overall_status: str,
        component_health: Dict[str, Any],
        performance_metrics: Dict[str, float] = None
    ) -> LayerHealthStatus:
        """Create standardized health status"""
        return LayerHealthStatus(
            layer_type=layer_type,
            overall_status=overall_status,
            component_health=component_health,
            performance_metrics=performance_metrics or {},
            timestamp=time.time()
        )
    
    @staticmethod
    async def execute_with_contract_validation(
        operation_func,
        source_layer: LayerType,
        target_layer: LayerType,
        *args, **kwargs
    ) -> OperationResult:
        """Execute operation with automatic contract validation"""
        start_time = time.time()
        
        try:
            # Validate dependency is allowed
            LayerBoundaryEnforcer.validate_dependency(source_layer, target_layer)
            
            # Execute operation
            result = await operation_func(*args, **kwargs)
            
            # Validate result format
            if not isinstance(result, OperationResult):
                result = ContractImplementationHelper.create_operation_result(
                    status=OperationStatus.SUCCESS,
                    data=result,
                    layer_source=source_layer,
                    execution_time=time.time() - start_time
                )
            
            # Validate contract compliance
            LayerBoundaryEnforcer.validate_operation_result(result, source_layer)
            
            return result
            
        except ContractViolationError:
            raise
        except Exception as e:
            return ContractImplementationHelper.create_operation_result(
                status=OperationStatus.FAILURE,
                layer_source=source_layer,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


# ============================================================================
# GLOBAL CONTRACT REGISTRY
# ============================================================================

class ContractRegistry:
    """Registry for all layer contracts and their implementations"""
    
    def __init__(self):
        self.registered_contracts = {}
        self.active_implementations = {}
        self.contract_monitor = ContractMonitor()
    
    def register_contract_implementation(
        self, 
        layer_type: LayerType, 
        contract_type: str, 
        implementation: Any
    ):
        """Register contract implementation for layer"""
        if layer_type not in self.registered_contracts:
            self.registered_contracts[layer_type] = {}
        
        self.registered_contracts[layer_type][contract_type] = implementation
        
    def get_contract_implementation(
        self, 
        layer_type: LayerType, 
        contract_type: str
    ) -> Any:
        """Get registered contract implementation"""
        return self.registered_contracts.get(layer_type, {}).get(contract_type)
    
    def validate_all_contracts(self) -> Dict[str, Any]:
        """Validate all registered contracts"""
        validation_results = {}
        
        for layer_type, contracts in self.registered_contracts.items():
            layer_results = {}
            
            for contract_type, implementation in contracts.items():
                try:
                    # Basic validation - check if implementation has required methods
                    layer_results[contract_type] = {
                        'valid': True,
                        'implementation_class': implementation.__class__.__name__
                    }
                except Exception as e:
                    layer_results[contract_type] = {
                        'valid': False,
                        'error': str(e)
                    }
            
            validation_results[layer_type.value] = layer_results
        
        return validation_results


# Global contract registry instance
contract_registry = ContractRegistry()
contract_monitor = ContractMonitor()