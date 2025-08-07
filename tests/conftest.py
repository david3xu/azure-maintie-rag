"""
Test Configuration and Fixtures - CODING_STANDARDS Compliant
Data-driven fixtures using real Azure services.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import pytest_asyncio

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.knowledge_extraction.agent import (
    ExtractionDeps,
)
from agents.knowledge_extraction.agent import agent as knowledge_extraction_agent
from agents.orchestrator import UniversalOrchestrator
from agents.universal_search.agent import SearchDeps
from agents.universal_search.agent import agent as universal_search_agent


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def azure_services():
    """Real Azure services fixture - no mocking"""
    services = ConsolidatedAzureServices()

    # Initialize with real Azure services
    service_status = await services.initialize_all_services()

    # Validate critical services are available
    critical_services = ["ai_foundry", "search", "cosmos"]
    available_services = sum(
        1 for service in critical_services if service_status.get(service, False)
    )

    if available_services < 1:
        pytest.skip(f"Insufficient Azure services available: {available_services}/3")

    yield services


@pytest_asyncio.fixture(scope="session")
async def knowledge_extraction_agent(azure_services):
    """Real Knowledge Extraction Agent with Azure backend"""
    agent = get_knowledge_extraction_agent()
    return agent


@pytest_asyncio.fixture(scope="session")
async def universal_search_agent(azure_services):
    """Real Universal Search Agent with Azure backend"""
    agent = get_universal_search_agent()
    return agent


@pytest.fixture
def sample_documents():
    """Real test documents for domain-agnostic testing"""
    return {
        "programming": """
        Python is a programming language that emphasizes readability and simplicity.
        Functions are defined using the def keyword, and classes use the class keyword.
        Popular frameworks include Django for web development and NumPy for data science.
        """,
        "maintenance": """
        Regular maintenance of HVAC systems is essential for optimal performance.
        Air filters should be replaced every 3 months to ensure proper airflow.
        Refrigerant levels must be checked annually by certified technicians.
        """,
        "legal": """
        Contract law governs agreements between parties in commercial transactions.
        Breach of contract occurs when one party fails to fulfill contractual obligations.
        Remedies may include damages, specific performance, or contract rescission.
        """,
        "medical": """
        Hypertension is a common cardiovascular condition affecting blood pressure.
        ACE inhibitors and beta-blockers are frequently prescribed medications.
        Regular monitoring and lifestyle modifications are essential for management.
        """,
    }


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture for SLA testing"""

    class PerformanceMonitor:
        def __init__(self):
            self.measurements = []

        def measure_operation(self, operation_name: str, sla_target: float = 3.0):
            """Context manager for performance measurement"""

            class OperationMeasurement:
                def __init__(self, monitor, name, target):
                    self.monitor = monitor
                    self.name = name
                    self.target = target
                    self.start_time = None
                    self.end_time = None

                async def __aenter__(self):
                    self.start_time = time.time()
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    self.end_time = time.time()
                    duration = self.end_time - self.start_time

                    measurement = {
                        "operation": self.name,
                        "duration": duration,
                        "sla_target": self.target,
                        "sla_compliant": duration <= self.target,
                        "timestamp": time.time(),
                    }

                    self.monitor.measurements.append(measurement)

                    if not measurement["sla_compliant"]:
                        print(
                            f"⚠️  SLA warning: {self.name} took {duration:.3f}s (target: {self.target}s)"
                        )

            return OperationMeasurement(self, operation_name, sla_target)

    return PerformanceMonitor()


@pytest.fixture
def test_data_directory():
    """Directory for real test data files"""
    test_data_dir = Path(__file__).parent / "fixtures" / "azure_test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    return test_data_dir


@pytest_asyncio.fixture
async def azure_health_check(azure_services):
    """Validate Azure services health before tests"""
    health_status = azure_services.get_service_status()

    if not health_status["overall_health"]:
        pytest.skip(
            f"Azure services unhealthy: {health_status['successful_services']}"
            f"/{health_status['total_services']} services available"
        )

    return health_status


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for Azure testing"""
    config.addinivalue_line("markers", "azure: mark test as requiring Azure services")
    config.addinivalue_line("markers", "performance: mark test as performance/SLA test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on location"""
    for item in items:
        # Auto-mark tests in integration/ directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.azure)
            item.add_marker(pytest.mark.integration)

        # Auto-mark tests in performance/ directory
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.azure)
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Auto-mark tests in azure_validation/ directory
        if "azure_validation" in str(item.fspath):
            item.add_marker(pytest.mark.azure)
