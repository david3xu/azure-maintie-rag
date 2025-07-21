#!/usr/bin/env python3
"""Simple test to check Azure ML configuration"""

import os
import sys
from pathlib import Path
import pytest

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

def test_environment_variables():
    """Test if environment variables are set correctly"""
    # Check Azure ML variables
    ml_vars = {
        "AZURE_SUBSCRIPTION_ID": "ccc6af52-5928-4dbe-8ceb-fa794974a30f",
        "AZURE_RESOURCE_GROUP": "maintie-rag-rg",
        "AZURE_ML_WORKSPACE": "maintie-dev-ml-1cdd8e11",
        "AZURE_ML_WORKSPACE_NAME": "maintie-dev-ml-1cdd8e11",
        "AZURE_ML_API_VERSION": "2023-04-01",
        "AZURE_TENANT_ID": "05894af0-cb28-46d8-8716-74cdb46e2226"
    }

    for var_name, expected_value in ml_vars.items():
        actual_value = os.getenv(var_name, "NOT_SET")
        assert actual_value == expected_value, f"{var_name}: Expected {expected_value}, got {actual_value}"

def test_settings_import():
    """Test if settings can be imported correctly"""
    from config.settings import azure_settings

    assert azure_settings.azure_subscription_id is not None
    assert azure_settings.azure_resource_group is not None
    assert azure_settings.azure_ml_workspace is not None
    assert azure_settings.azure_ml_api_version is not None

def test_configuration_matches():
    """Test if configuration matches expected values"""
    from config.settings import azure_settings

    expected_values = {
        "azure_subscription_id": "ccc6af52-5928-4dbe-8ceb-fa794974a30f",
        "azure_resource_group": "maintie-rag-rg",
        "azure_ml_workspace": "maintie-dev-ml-1cdd8e11",
        "azure_ml_workspace_name": "maintie-dev-ml-1cdd8e11",
        "azure_ml_api_version": "2023-04-01",
        "azure_tenant_id": "05894af0-cb28-46d8-8716-74cdb46e2226"
    }

    for setting_name, expected_value in expected_values.items():
        actual_value = getattr(azure_settings, setting_name, "NOT_FOUND")
        assert actual_value == expected_value, f"{setting_name}: Expected {expected_value}, got {actual_value}"