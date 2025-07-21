#!/usr/bin/env python3
"""Simple test script to demonstrate storage factory functionality"""

import sys
from pathlib import Path
import pytest

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

def test_storage_factory():
    """Test the storage factory functionality"""
    try:
        from core.azure_storage.storage_factory import get_storage_factory

        # Get storage factory
        storage_factory = get_storage_factory()

        # List available clients
        clients = storage_factory.list_available_clients()

        # Get storage status
        status = storage_factory.get_storage_status()

        assert clients is not None and len(clients) > 0
        assert status is not None

    except Exception as e:
        pytest.fail(f"❌ Storage Factory Test Failed: {e}")