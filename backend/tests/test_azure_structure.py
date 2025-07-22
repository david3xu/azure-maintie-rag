#!/usr/bin/env python3
"""
Test Azure Universal RAG Structure
Verifies all Azure integration files are properly created and importable
"""

import sys
from pathlib import Path
import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

def test_azure_settings():
    """Test Azure settings configuration"""
    from config.settings import AzureSettings
    azure_settings = AzureSettings()
    assert azure_settings.azure_resource_prefix is not None
    assert azure_settings.azure_environment is not None

def test_azure_structure():
    """Test Azure directory structure"""
    integrations_dir = Path("integrations")
    required_files = [
        "__init__.py",
        "azure_services.py",
        "azure_openai.py"
    ]

    all_exist = True

    for file_name in required_files:
        file_path = integrations_dir / file_name
        if not file_path.exists():
            all_exist = False

    assert all_exist, "Missing one or more required Azure integration files."

def test_infrastructure():
    """Test infrastructure files (Azure Bicep templates)"""
    # Check infrastructure directory at project root
    infra_dir = Path("../infrastructure")
    required_files = [
        "azure-resources-core.bicep",
        "azure-resources-ml-simple.bicep",
        "azure-resources-cosmos.bicep",
        "azure-resources-ml-simple.json"
    ]

    all_exist = True

    for file_name in required_files:
        file_path = infra_dir / file_name
        if not file_path.exists():
            all_exist = False

    assert all_exist, "Missing one or more required Azure infrastructure files."

def test_integrations():
    """Test integrations"""
    integrations_dir = Path("integrations")
    required_files = [
        "azure_services.py",
        "azure_openai.py"
    ]

    all_exist = True

    for file_name in required_files:
        file_path = integrations_dir / file_name
        if not file_path.exists():
            all_exist = False

    assert all_exist, "Missing one or more required Azure integration files."

def test_azure_session_management_config():
    """Test Azure session management and connection config"""
    from config.settings import Settings
    settings = Settings()
    assert settings.azure_session_refresh_minutes > 0
    assert settings.azure_connection_pool_size > 0
    assert settings.azure_health_check_timeout_seconds > 0
    assert settings.azure_circuit_breaker_failure_threshold > 0