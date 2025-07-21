#!/usr/bin/env python3
"""Test script to check Azure ML workspace configuration"""

import sys
from pathlib import Path
import pytest

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

def test_azure_ml_config():
    """Test Azure ML workspace configuration"""
    from config.settings import azure_settings

    # Check basic settings
    assert azure_settings.azure_subscription_id is not None
    assert azure_settings.azure_resource_group is not None
    assert azure_settings.azure_ml_workspace is not None
    assert azure_settings.azure_ml_workspace_name is not None
    assert azure_settings.azure_ml_api_version is not None
    assert azure_settings.azure_tenant_id is not None

    # Check if all required fields are set
    required_fields = [
        azure_settings.azure_subscription_id,
        azure_settings.azure_resource_group,
        azure_settings.azure_ml_workspace,
        azure_settings.azure_ml_workspace_name
    ]
    assert all(required_fields), "Some required Azure ML fields are missing!"

def test_azure_ml_connection():
    """Test Azure ML workspace connection"""
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    from config.settings import azure_settings

    # Create ML client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=azure_settings.azure_subscription_id,
        resource_group_name=azure_settings.azure_resource_group,
        workspace_name=azure_settings.azure_ml_workspace
    )

    # Test connection by getting workspace info
    workspace = ml_client.workspaces.get(azure_settings.azure_ml_workspace)

    assert workspace.name is not None
    assert workspace.location is not None
    assert workspace._workspace_id is not None
    assert workspace.discovery_url is not None
    assert workspace.mlflow_tracking_uri is not None