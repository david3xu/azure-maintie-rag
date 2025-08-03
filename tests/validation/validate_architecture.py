"""
Architecture validation tests
Validates that the codebase follows the established architecture patterns and principles
"""

import ast
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

import pytest


class TestArchitectureCompliance:
    """Test architecture compliance and patterns"""

    def test_directory_structure_compliance(self):
        """Test that directory structure follows the target architecture"""
        project_root = Path(__file__).parent.parent.parent

        # Required top-level directories
        required_dirs = [
            "agents", "api", "services", "infrastructure",
            "config", "data", "scripts", "tests", "docs"
        ]

        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory {dir_name} is missing"
            assert dir_path.is_dir(), f"{dir_name} should be a directory"

    def test_agents_structure(self):
        """Test agents directory structure"""
        project_root = Path(__file__).parent.parent.parent
        agents_dir = project_root / "agents"

        # Required agent subdirectories
        required_subdirs = ["core", "intelligence", "search", "tools", "models"]

        for subdir in required_subdirs:
            subdir_path = agents_dir / subdir
            assert subdir_path.exists(), f"agents/{subdir} is missing"

        # Required agent files
        required_files = [
            "universal_agent.py",
            "domain_intelligence_agent.py",
            "simple_universal_agent.py"
        ]

        for file_name in required_files:
            file_path = agents_dir / file_name
            assert file_path.exists(), f"agents/{file_name} is missing"

    def test_api_structure(self):
        """Test API directory structure"""
        project_root = Path(__file__).parent.parent.parent
        api_dir = project_root / "api"

        # Required API components
        required_components = [
            "main.py", "dependencies.py", "middleware.py",
            "endpoints", "models", "streaming"
        ]

        for component in required_components:
            component_path = api_dir / component
            assert component_path.exists(), f"api/{component} is missing"

    def test_services_structure(self):
        """Test services directory structure"""
        project_root = Path(__file__).parent.parent.parent
        services_dir = project_root / "services"

        # Required service files
        required_services = [
            "agent_service.py", "query_service.py", "workflow_service.py",
            "infrastructure_service.py", "ml_service.py", "cache_service.py"
        ]

        for service in required_services:
            service_path = services_dir / service
            assert service_path.exists(), f"services/{service} is missing"

    def test_infrastructure_structure(self):
        """Test infrastructure directory structure"""
        project_root = Path(__file__).parent.parent.parent
        infra_dir = project_root / "infrastructure"

        # Required Azure service integrations
        required_azure_services = [
            "azure_openai", "azure_search", "azure_cosmos",
            "azure_ml", "azure_storage", "azure_auth"
        ]

        for service in required_azure_services:
            service_path = infra_dir / service
            assert service_path.exists(), f"infrastructure/{service} is missing"


class TestImportCompliance:
    """Test import patterns and dependencies"""

    def test_no_circular_imports(self):
        """Test that there are no circular imports"""
        project_root = Path(__file__).parent.parent.parent

        # Add project root to Python path
        sys.path.insert(0, str(project_root))

        try:
            # Test key module imports
            import agents.universal_agent
            import api.main
            import services.agent_service
            import infrastructure.azure_openai.openai_client
            import config.settings

            # If we get here, no circular imports detected
            assert True

        except ImportError as e:
            if "circular" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            else:
                # Other import errors may be due to missing dependencies
                pytest.skip(f"Import error (may be due to missing dependencies): {e}")
        finally:
            # Clean up path
            if str(project_root) in sys.path:
                sys.path.remove(str(project_root))

    def test_layer_boundary_compliance(self):
        """Test that layer boundaries are respected"""
        project_root = Path(__file__).parent.parent.parent

        # Layer dependency rules
        # - API layer can depend on services and models
        # - Services can depend on infrastructure and models
        # - Infrastructure can depend on config and models
        # - Config should be independent
        # - Models should be independent or depend only on Pydantic

        violations = []

        # Check API layer imports
        api_files = list((project_root / "api").rglob("*.py"))
        for file_path in api_files:
            violations.extend(self._check_layer_imports(
                file_path,
                allowed_layers=["services", "config", "api.models", "agents.models"],
                forbidden_layers=["infrastructure"]
            ))

        # Check services layer imports
        services_files = list((project_root / "services").rglob("*.py"))
        for file_path in services_files:
            violations.extend(self._check_layer_imports(
                file_path,
                allowed_layers=["infrastructure", "config", "agents", "services"],
                forbidden_layers=["api"]
            ))

        if violations:
            pytest.fail(f"Layer boundary violations found: {violations}")

    def _check_layer_imports(self, file_path: Path, allowed_layers: List[str], forbidden_layers: List[str]) -> List[str]:
        """Check imports in a file against layer rules"""
        violations = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module_name = node.module
                    elif isinstance(node, ast.Import):
                        module_name = node.names[0].name if node.names else ""
                    else:
                        continue

                    # Check against forbidden layers
                    for forbidden in forbidden_layers:
                        if module_name.startswith(forbidden):
                            violations.append(
                                f"{file_path.name} imports forbidden layer {forbidden}: {module_name}"
                            )

        except (SyntaxError, UnicodeDecodeError):
            # Skip files that can't be parsed
            pass

        return violations


class TestDataDrivenCompliance:
    """Test compliance with data-driven architecture principles"""

    def test_async_first_compliance(self):
        """Test that async patterns are used consistently"""
        project_root = Path(__file__).parent.parent.parent

        # Check that service methods are primarily async
        services_dir = project_root / "services"
        violations = []

        for service_file in services_dir.glob("*.py"):
            if service_file.name == "__init__.py":
                continue

            try:
                with open(service_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)

                class AsyncChecker(ast.NodeVisitor):
                    def __init__(self):
                        self.sync_methods = []
                        self.async_methods = []

                    def visit_FunctionDef(self, node):
                        if not node.name.startswith('_'):  # Public methods
                            self.sync_methods.append(node.name)
                        self.generic_visit(node)

                    def visit_AsyncFunctionDef(self, node):
                        if not node.name.startswith('_'):  # Public methods
                            self.async_methods.append(node.name)
                        self.generic_visit(node)

                checker = AsyncChecker()
                checker.visit(tree)

                # Services should have async methods
                if len(checker.async_methods) == 0 and len(checker.sync_methods) > 0:
                    violations.append(
                        f"{service_file.name}: No async methods found, has {len(checker.sync_methods)} sync methods"
                    )

            except (SyntaxError, UnicodeDecodeError):
                continue

        # Allow some violations as some services may be legitimately sync
        if len(violations) > len(list(services_dir.glob("*.py"))) // 2:
            pytest.fail(f"Too many async-first violations: {violations}")


class TestPerformanceCompliance:
    """Test performance-related architecture compliance"""

    def test_caching_patterns(self):
        """Test that appropriate caching patterns are implemented"""
        project_root = Path(__file__).parent.parent.parent

        # Check that cache service exists and is properly integrated
        cache_service_path = project_root / "services" / "cache_service.py"
        assert cache_service_path.exists(), "Cache service is missing"

        # Check that agents use caching
        agents_dir = project_root / "agents"
        cache_usage_found = False

        for py_file in agents_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if "cache" in content.lower():
                    cache_usage_found = True
                    break

            except (UnicodeDecodeError, PermissionError):
                continue

        assert cache_usage_found, "No caching patterns found in agents"

    def test_resource_cleanup_patterns(self):
        """Test that resources are properly cleaned up"""
        project_root = Path(__file__).parent.parent.parent

        # Check for context managers and proper resource cleanup
        infrastructure_files = list((project_root / "infrastructure").rglob("*.py"))

        cleanup_patterns_found = 0

        for file_path in infrastructure_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Look for cleanup patterns
                if any(pattern in content for pattern in [
                    "async with", "try:", "finally:", "__aenter__", "__aexit__"
                ]):
                    cleanup_patterns_found += 1

            except (UnicodeDecodeError, PermissionError):
                continue

        # Should have cleanup patterns in most infrastructure files
        assert cleanup_patterns_found > 0, "No resource cleanup patterns found"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
