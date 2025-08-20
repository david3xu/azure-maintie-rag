"""
Systematic Timeout Configuration for Azure RAG System
Ensures sub-3s response time compliance across all operations
"""

import asyncio
from functools import wraps
from typing import Dict, Optional

from pydantic import BaseModel, Field


class TimeoutConfig(BaseModel):
    """Centralized timeout configuration"""

    # Core search operations (must be < 3s total)
    tri_modal_search: float = Field(default=2.5, description="Tri-modal search timeout")
    vector_search: float = Field(default=1.0, description="Vector search timeout")
    graph_search: float = Field(default=1.0, description="Graph search timeout")
    gnn_search: float = Field(default=1.0, description="GNN search timeout")

    # Agent operations
    agent_tool_execution: float = Field(
        default=2.0, description="Agent tool execution timeout"
    )
    domain_analysis: float = Field(default=1.5, description="Domain analysis timeout")
    pattern_extraction: float = Field(
        default=2.0, description="Pattern extraction timeout"
    )

    # Infrastructure operations
    azure_openai_request: float = Field(
        default=10.0, description="Azure OpenAI API timeout"
    )
    openai_timeout: float = Field(
        default=10.0, description="OpenAI API timeout (alias)"
    )
    azure_search_request: float = Field(
        default=5.0, description="Azure Search API timeout"
    )
    search_timeout: int = Field(
        default=120, description="Search operation timeout in seconds"
    )
    cosmos_db_query: float = Field(default=3.0, description="Cosmos DB query timeout")
    cosmos_timeout: int = Field(default=10, description="Cosmos DB timeout in seconds")
    storage_timeout: int = Field(
        default=60, description="Storage operation timeout in seconds"
    )

    # Circuit breaker settings
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    circuit_breaker_threshold: int = Field(
        default=5, description="Failures before circuit opens"
    )
    circuit_breaker_reset_timeout: float = Field(
        default=60.0, description="Circuit breaker reset time"
    )


# Global timeout configuration instance
timeout_config = TimeoutConfig()


class TimeoutEnforcer:
    """Enforces timeout compliance across all operations"""

    def __init__(self, config: TimeoutConfig = None):
        self.config = config or timeout_config
        self._circuit_breakers: Dict[str, Dict] = {}

    async def enforce_timeout(
        self, operation_name: str, coro, timeout: Optional[float] = None
    ):
        """Enforce timeout for any async operation with circuit breaker"""

        # Get timeout from config or use provided value
        if timeout is None:
            timeout = getattr(self.config, operation_name, 3.0)

        # Check circuit breaker
        if self._is_circuit_open(operation_name):
            raise RuntimeError(f"Circuit breaker open for {operation_name}")

        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            self._record_success(operation_name)
            return result

        except asyncio.TimeoutError:
            self._record_failure(operation_name)
            raise RuntimeError(
                f"Operation '{operation_name}' exceeded {timeout}s timeout - SLA violation"
            )
        except Exception as e:
            self._record_failure(operation_name)
            raise

    def _is_circuit_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open for operation"""
        circuit = self._circuit_breakers.get(
            operation_name, {"failures": 0, "last_failure_time": 0, "state": "closed"}
        )

        if circuit["state"] == "open":
            # Check if reset timeout has passed
            import time

            if (
                time.time() - circuit["last_failure_time"]
                > self.config.circuit_breaker_reset_timeout
            ):
                circuit["state"] = "half_open"
                circuit["failures"] = 0
                self._circuit_breakers[operation_name] = circuit
                return False
            return True

        return False

    def _record_success(self, operation_name: str):
        """Record successful operation"""
        if operation_name in self._circuit_breakers:
            self._circuit_breakers[operation_name]["state"] = "closed"
            self._circuit_breakers[operation_name]["failures"] = 0

    def _record_failure(self, operation_name: str):
        """Record failed operation and update circuit breaker"""
        import time

        circuit = self._circuit_breakers.get(
            operation_name, {"failures": 0, "last_failure_time": 0, "state": "closed"}
        )

        circuit["failures"] += 1
        circuit["last_failure_time"] = time.time()

        if circuit["failures"] >= self.config.circuit_breaker_threshold:
            circuit["state"] = "open"

        self._circuit_breakers[operation_name] = circuit


# Global timeout enforcer instance
timeout_enforcer = TimeoutEnforcer()


def with_timeout(operation_name: str, timeout: Optional[float] = None):
    """Decorator to enforce timeout on async functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await timeout_enforcer.enforce_timeout(
                operation_name, func(*args, **kwargs), timeout
            )

        return wrapper

    return decorator


def get_timeout(operation_name: str) -> float:
    """Get timeout value for an operation"""
    return getattr(timeout_config, operation_name, 3.0)
