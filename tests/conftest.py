"""
Comprehensive Integration Test Configuration for Azure Universal RAG System
=================================================================================

Real Azure services fixtures with proper environment configuration.
No mocks - all tests use actual deployed Azure infrastructure.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import pytest
import pytest_asyncio
from dotenv import load_dotenv

# CRITICAL: Load environment variables before any imports that need them
load_dotenv()

# CRITICAL: Configure authentication for test environment
# Tests can use either CLI authentication (development) or managed identity (production)
test_use_managed_identity = os.getenv("TEST_USE_MANAGED_IDENTITY", "false").lower() == "true"
os.environ["USE_MANAGED_IDENTITY"] = str(test_use_managed_identity).lower()
os.environ["COSMOS_USE_MANAGED_IDENTITY"] = str(test_use_managed_identity).lower()
os.environ["AZURE_USE_MANAGED_IDENTITY"] = str(test_use_managed_identity).lower()

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.core.universal_deps import UniversalDeps, get_universal_deps
from agents.domain_intelligence.agent import domain_intelligence_agent
from agents.knowledge_extraction.agent import knowledge_extraction_agent
from agents.universal_search.agent import universal_search_agent


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def azure_services():
    """Real Azure services fixture - no mocking"""
    # Validate environment configuration first
    required_env_vars = [
        "OPENAI_API_KEY", 
        "AZURE_OPENAI_ENDPOINT", 
        "OPENAI_MODEL_DEPLOYMENT",
        "EMBEDDING_MODEL_DEPLOYMENT"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        pytest.fail(f"Missing required environment variables: {missing_vars}")
    
    # Set OPENAI_BASE_URL for PydanticAI compatibility
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    
    # Initialize universal dependencies with real Azure services
    services = await get_universal_deps()
    service_status = await services.initialize_all_services()
    
    # Log service status
    print(f"\n📊 Azure Services Status:")
    for service, status in service_status.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {service}: {'Available' if status else 'Failed'}")
    
    # Validate critical services with detailed error reporting
    critical_services = ["openai"]
    failed_critical = [svc for svc in critical_services if not service_status.get(svc, False)]
    
    if failed_critical:
        # Get authentication details for debugging
        test_use_managed_identity = os.getenv("TEST_USE_MANAGED_IDENTITY", "false").lower() == "true"
        auth_method = "Managed Identity" if test_use_managed_identity else "CLI/API Key"
        
        error_details = {
            "failed_services": failed_critical,
            "all_services": service_status,
            "auth_method": auth_method,
            "endpoint": os.getenv('AZURE_OPENAI_ENDPOINT', 'NOT SET'),
            "deployment": os.getenv('OPENAI_MODEL_DEPLOYMENT', 'NOT SET'),
            "environment": os.getenv('AZURE_ENV_NAME', 'NOT SET')
        }
        
        pytest.fail(f"Critical Azure services failed. Details: {error_details}")
    
    yield services


@pytest_asyncio.fixture(scope="session")
async def domain_intelligence_agent_fixture(azure_services):
    """Real Domain Intelligence Agent with Azure OpenAI backend"""
    return domain_intelligence_agent


@pytest_asyncio.fixture(scope="session")
async def knowledge_extraction_agent_fixture(azure_services):
    """Real Knowledge Extraction Agent with Azure OpenAI backend"""
    return knowledge_extraction_agent


@pytest_asyncio.fixture(scope="session")
async def universal_search_agent_fixture(azure_services):
    """Real Universal Search Agent with Azure OpenAI backend"""
    return universal_search_agent


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
    """Directory for real Azure AI test data files"""
    # Use actual Azure AI Language Service test data
    test_data_dir = Path(__file__).parent.parent / "data" / "raw" / "azure-ai-services-language-service_output"
    
    if not test_data_dir.exists():
        pytest.skip(f"Real test data directory not found: {test_data_dir}")
    
    # Validate test data is available
    markdown_files = list(test_data_dir.glob("*.md"))
    if len(markdown_files) == 0:
        pytest.skip("No Azure AI test data files found")
    
    print(f"\n📁 Test Data: {len(markdown_files)} Azure AI files available")
    return test_data_dir


@pytest.fixture
def comprehensive_azure_health_monitor():
    """Enhanced Azure service health monitoring with detailed metrics"""
    
    class ComprehensiveAzureHealthMonitor:
        def __init__(self):
            self.health_checks = []
            self.service_metrics = {}
            
        async def perform_health_check(self, service_name: str, check_func, timeout: float = 30.0):
            """Perform health check with timeout and detailed metrics"""
            start_time = time.time()
            
            try:
                result = await asyncio.wait_for(check_func(), timeout=timeout)
                duration = time.time() - start_time
                
                health_record = {
                    "service": service_name,
                    "status": "healthy",
                    "response_time": duration,
                    "timestamp": time.time(),
                    "result": result,
                    "error": None
                }
                
                self.health_checks.append(health_record)
                self.service_metrics[service_name] = {
                    "healthy": True,
                    "response_time": duration,
                    "last_check": time.time()
                }
                
                return health_record
                
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                health_record = {
                    "service": service_name,
                    "status": "timeout",
                    "response_time": duration,
                    "timestamp": time.time(),
                    "result": None,
                    "error": f"Timeout after {timeout}s"
                }
                
                self.health_checks.append(health_record)
                self.service_metrics[service_name] = {
                    "healthy": False,
                    "response_time": duration,
                    "last_check": time.time(),
                    "error": "timeout"
                }
                
                return health_record
                
            except Exception as e:
                duration = time.time() - start_time
                health_record = {
                    "service": service_name,
                    "status": "error",
                    "response_time": duration,
                    "timestamp": time.time(),
                    "result": None,
                    "error": str(e)
                }
                
                self.health_checks.append(health_record)
                self.service_metrics[service_name] = {
                    "healthy": False,
                    "response_time": duration,
                    "last_check": time.time(),
                    "error": str(e)
                }
                
                return health_record
        
        def get_health_summary(self) -> Dict[str, Any]:
            """Get comprehensive health summary"""
            total_services = len(self.service_metrics)
            healthy_services = sum(1 for metrics in self.service_metrics.values() if metrics["healthy"])
            
            health_ratio = healthy_services / total_services if total_services > 0 else 0
            avg_response_time = sum(metrics["response_time"] for metrics in self.service_metrics.values()) / total_services if total_services > 0 else 0
            
            critical_services = ["openai"]  # Services required for basic functionality
            critical_healthy = all(self.service_metrics.get(service, {}).get("healthy", False) for service in critical_services)
            
            return {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "health_ratio": health_ratio,
                "critical_services_healthy": critical_healthy,
                "average_response_time": avg_response_time,
                "service_details": self.service_metrics,
                "timestamp": time.time()
            }
        
        def is_production_ready(self) -> bool:
            """Determine if system is production ready based on health metrics"""
            summary = self.get_health_summary()
            
            # Production readiness criteria
            return (
                summary["critical_services_healthy"] and
                summary["health_ratio"] >= 0.6 and  # At least 60% of services healthy
                summary["average_response_time"] < 10.0  # Average response time under 10s
            )
    
    return ComprehensiveAzureHealthMonitor()


@pytest.fixture
def integration_test_data_manager():
    """Enhanced test data management for comprehensive integration testing"""
    
    class IntegrationTestDataManager:
        def __init__(self, test_data_dir: Path):
            self.test_data_dir = test_data_dir
            self.file_analysis_cache = {}
            
        def get_test_files_by_criteria(
            self, 
            min_size: int = 500,
            max_size: int = 10000,
            content_types: List[str] = None,
            limit: int = None
        ) -> List[Path]:
            """Get test files matching specific criteria"""
            all_files = list(self.test_data_dir.glob("*.md"))
            
            filtered_files = []
            for file_path in all_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    content_length = len(content)
                    
                    # Size filter
                    if not (min_size <= content_length <= max_size):
                        continue
                    
                    # Content type filter
                    if content_types:
                        content_lower = content.lower()
                        matches_type = any(
                            content_type in content_lower 
                            for content_type in content_types
                        )
                        if not matches_type:
                            continue
                    
                    filtered_files.append(file_path)
                    
                    if limit and len(filtered_files) >= limit:
                        break
                        
                except Exception:
                    continue
            
            return filtered_files
        
        def analyze_file_quality(self, file_path: Path) -> Dict[str, Any]:
            """Analyze file quality for testing suitability"""
            if file_path in self.file_analysis_cache:
                return self.file_analysis_cache[file_path]
            
            content = file_path.read_text(encoding='utf-8')
            
            analysis = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "size_chars": len(content),
                "size_lines": content.count('\n'),
                "has_headers": content.count('#') > 0,
                "has_code_blocks": '```' in content,
                "has_links": '[' in content and '](' in content,
                "has_api_content": any(word in content.lower() for word in ['api', 'endpoint', 'request', 'response', 'curl']),
                "has_tutorial_content": any(word in content.lower() for word in ['how to', 'tutorial', 'example', 'step']),
                "has_conceptual_content": any(word in content.lower() for word in ['concept', 'overview', 'introduction', 'understand']),
                "has_code_examples": any(word in content.lower() for word in ['code', 'example', 'sample', 'snippet']),
                "complexity_score": self._calculate_complexity_score(content),
                "quality_score": self._calculate_quality_score(content),
                "is_suitable_for_testing": self._is_suitable_for_testing(content)
            }
            
            self.file_analysis_cache[file_path] = analysis
            return analysis
        
        def _calculate_complexity_score(self, content: str) -> float:
            """Calculate content complexity score (0-1)"""
            factors = [
                len(content) > 1000,  # Substantial length
                content.count('\n') > 20,  # Multiple lines
                '```' in content,  # Code blocks
                content.count('http') > 0,  # URLs
                content.count('Azure') > 2,  # Azure-specific content
                any(word in content for word in ['JSON', 'REST', 'API', 'SDK'])  # Technical terms
            ]
            
            return sum(factors) / len(factors)
        
        def _calculate_quality_score(self, content: str) -> float:
            """Calculate content quality score (0-1)"""
            factors = [
                len(content) > 500,  # Minimum substantial content
                content.count('#') > 0,  # Has headers
                content.count('.') > 10,  # Multiple sentences
                not content.count('TODO') > 0,  # Not incomplete
                not content.count('FIXME') > 0,  # Not broken
                len(content.split()) > 100  # Word count
            ]
            
            return sum(factors) / len(factors)
        
        def _is_suitable_for_testing(self, content: str) -> bool:
            """Determine if content is suitable for comprehensive testing"""
            return (
                len(content) > 300 and  # Minimum content
                len(content) < 20000 and  # Not too large
                content.count('\n') > 5 and  # Multi-line
                not content.count('error') > 5  # Not error-heavy
            )
        
        def get_diverse_test_set(self, count: int = 10) -> List[Path]:
            """Get a diverse set of test files for comprehensive testing"""
            all_files = list(self.test_data_dir.glob("*.md"))
            
            # Categorize files
            categories = {
                "api_docs": [],
                "tutorials": [],
                "conceptual": [],
                "code_heavy": [],
                "general": []
            }
            
            for file_path in all_files:
                analysis = self.analyze_file_quality(file_path)
                
                if not analysis["is_suitable_for_testing"]:
                    continue
                
                if analysis["has_api_content"]:
                    categories["api_docs"].append(file_path)
                elif analysis["has_tutorial_content"]:
                    categories["tutorials"].append(file_path)
                elif analysis["has_conceptual_content"]:
                    categories["conceptual"].append(file_path)
                elif analysis["has_code_examples"]:
                    categories["code_heavy"].append(file_path)
                else:
                    categories["general"].append(file_path)
            
            # Select diverse files
            diverse_set = []
            files_per_category = max(1, count // len(categories))
            
            for category, files in categories.items():
                if files:
                    # Sort by quality score and take best files from each category
                    sorted_files = sorted(
                        files, 
                        key=lambda f: self.analyze_file_quality(f)["quality_score"],
                        reverse=True
                    )
                    diverse_set.extend(sorted_files[:files_per_category])
            
            # Fill remaining slots with highest quality files
            if len(diverse_set) < count:
                all_suitable = [
                    f for f in all_files 
                    if self.analyze_file_quality(f)["is_suitable_for_testing"]
                    and f not in diverse_set
                ]
                sorted_remaining = sorted(
                    all_suitable,
                    key=lambda f: self.analyze_file_quality(f)["quality_score"],
                    reverse=True
                )
                diverse_set.extend(sorted_remaining[:count - len(diverse_set)])
            
            return diverse_set[:count]
        
        def generate_test_data_report(self) -> Dict[str, Any]:
            """Generate comprehensive test data quality report"""
            all_files = list(self.test_data_dir.glob("*.md"))
            
            analyses = []
            for file_path in all_files:
                try:
                    analysis = self.analyze_file_quality(file_path)
                    analyses.append(analysis)
                except Exception as e:
                    analyses.append({
                        "file_path": str(file_path),
                        "error": str(e),
                        "is_suitable_for_testing": False
                    })
            
            suitable_files = [a for a in analyses if a.get("is_suitable_for_testing", False)]
            
            report = {
                "total_files": len(analyses),
                "suitable_files": len(suitable_files),
                "suitability_ratio": len(suitable_files) / len(analyses) if analyses else 0,
                "average_size": sum(a.get("size_chars", 0) for a in suitable_files) / len(suitable_files) if suitable_files else 0,
                "content_type_distribution": {
                    "api_docs": sum(1 for a in suitable_files if a.get("has_api_content", False)),
                    "tutorials": sum(1 for a in suitable_files if a.get("has_tutorial_content", False)),
                    "conceptual": sum(1 for a in suitable_files if a.get("has_conceptual_content", False)),
                    "code_examples": sum(1 for a in suitable_files if a.get("has_code_examples", False))
                },
                "quality_metrics": {
                    "average_quality_score": sum(a.get("quality_score", 0) for a in suitable_files) / len(suitable_files) if suitable_files else 0,
                    "average_complexity_score": sum(a.get("complexity_score", 0) for a in suitable_files) / len(suitable_files) if suitable_files else 0,
                    "files_with_code_blocks": sum(1 for a in suitable_files if a.get("has_code_blocks", False)),
                    "files_with_headers": sum(1 for a in suitable_files if a.get("has_headers", False))
                },
                "detailed_analyses": analyses
            }
            
            return report
    
    return IntegrationTestDataManager


@pytest.fixture
def azure_ai_test_files(test_data_directory):
    """Real Azure AI Language Service test files"""
    markdown_files = list(test_data_directory.glob("*.md"))
    return markdown_files[:10]  # Use first 10 files for testing


@pytest.fixture  
def enhanced_test_data_manager(test_data_directory, integration_test_data_manager):
    """Initialize the enhanced test data manager with the test data directory"""
    return integration_test_data_manager(test_data_directory)


@pytest.fixture
def azure_model_deployments():
    """Azure OpenAI model deployments configuration"""
    return {
        "chat_model": os.getenv("OPENAI_MODEL_DEPLOYMENT", "gpt-4o"),
        "embedding_model": os.getenv("EMBEDDING_MODEL_DEPLOYMENT", "text-embedding-ada-002"),
        "api_version": os.getenv("OPENAI_API_VERSION", "2024-02-01"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("OPENAI_API_KEY")
    }


@pytest.fixture
def environment_config():
    """Environment configuration for testing"""
    return {
        "environment": os.getenv("AZURE_ENV_NAME", "prod"),
        "resource_group": os.getenv("AZURE_RESOURCE_GROUP", "rg-maintie-rag-prod"),
        "location": os.getenv("AZURE_LOCATION", "westus2"),
        "use_managed_identity": os.getenv("USE_MANAGED_IDENTITY", "false").lower() == "true"
    }


@pytest_asyncio.fixture
async def azure_health_check(azure_services):
    """Validate Azure services health before tests"""
    service_status = await azure_services._get_service_status()
    
    # Check critical services (OpenAI is minimum requirement)
    critical_services = ["openai"]
    overall_health = all(service_status.get(svc, False) for svc in critical_services)
    
    available_services = [service for service, status in service_status.items() if status]
    
    health_report = {
        "overall_health": overall_health,
        "service_status": service_status,
        "available_services": available_services,
        "critical_services_count": len(critical_services),
        "available_services_count": len(available_services),
        "total_services_count": len(service_status)
    }
    
    if not overall_health:
        print(f"\n⚠️ Health Check Summary:")
        print(f"   Critical services: {critical_services}")
        print(f"   Available services: {available_services}")
        print(f"   Service status: {service_status}")
        pytest.skip(f"Critical Azure services unavailable. Available: {len(available_services)}/{len(service_status)}")
    
    print(f"\n✅ Health Check Passed: {len(available_services)}/{len(service_status)} services available")
    return health_report


# Test data validation functions
def validate_azure_ai_file(file_path: Path) -> Dict[str, Any]:
    """Validate Azure AI test file quality and content"""
    content = file_path.read_text(encoding='utf-8')
    
    return {
        "file_name": file_path.name,
        "size_chars": len(content),
        "size_lines": content.count('\n'),
        "has_headers": content.count('#') > 0,
        "has_code_blocks": '```' in content,
        "has_api_content": any(word in content.lower() for word in ['api', 'endpoint', 'request', 'response']),
        "has_tutorial_content": any(word in content.lower() for word in ['how to', 'tutorial', 'example', 'step']),
        "quality_score": min(len(content) / 500.0, 1.0),  # Quality based on content length
        "is_substantial": len(content) > 500
    }


@pytest.fixture
def test_data_quality_report(azure_ai_test_files):
    """Generate quality report for Azure AI test data files"""
    quality_reports = []
    for file_path in azure_ai_test_files:
        quality_reports.append(validate_azure_ai_file(file_path))
    
    total_files = len(quality_reports)
    substantial_files = sum(1 for report in quality_reports if report['is_substantial'])
    avg_size = sum(report['size_chars'] for report in quality_reports) / max(total_files, 1)
    
    summary = {
        "total_files": total_files,
        "substantial_files": substantial_files,
        "quality_ratio": substantial_files / max(total_files, 1),
        "average_size_chars": avg_size,
        "files_with_api_content": sum(1 for report in quality_reports if report['has_api_content']),
        "files_with_tutorial_content": sum(1 for report in quality_reports if report['has_tutorial_content']),
        "files_with_code_blocks": sum(1 for report in quality_reports if report['has_code_blocks']),
        "individual_reports": quality_reports
    }
    
    return summary


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for Azure testing"""
    config.addinivalue_line("markers", "layer1: Layer 1 Infrastructure Tests")
    config.addinivalue_line("markers", "layer2: Layer 2 PydanticAI Agent Tests") 
    config.addinivalue_line("markers", "layer3: Layer 3 Real Data Processing Tests")
    config.addinivalue_line("markers", "layer4: Layer 4 Integration & Performance Tests")
    config.addinivalue_line("markers", "azure: mark test as requiring Azure services")
    config.addinivalue_line("markers", "performance: mark test as performance/SLA test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on location and test names"""
    for item in items:
        # Auto-mark based on test file names
        test_file = str(item.fspath)
        
        if "test_layer1" in test_file or "test_azure_services" in test_file:
            item.add_marker(pytest.mark.layer1)
            item.add_marker(pytest.mark.azure)
            
        if "test_layer2" in test_file or "test_agents" in test_file:
            item.add_marker(pytest.mark.layer2)
            item.add_marker(pytest.mark.azure)
            
        if "test_layer3" in test_file or "test_data_pipeline" in test_file:
            item.add_marker(pytest.mark.layer3)
            item.add_marker(pytest.mark.azure)
            
        if "test_layer4" in test_file or "test_api_endpoints" in test_file:
            item.add_marker(pytest.mark.layer4)
            item.add_marker(pytest.mark.integration)
            
        # Auto-mark integration tests
        if "integration" in test_file:
            item.add_marker(pytest.mark.azure)
            item.add_marker(pytest.mark.integration)

        # Auto-mark performance tests  
        if "performance" in test_file:
            item.add_marker(pytest.mark.azure)
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)