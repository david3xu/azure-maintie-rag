#!/usr/bin/env python3
"""Simple test script to demonstrate storage factory functionality"""

import sys
from pathlib import Path

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
        print("✅ Storage Factory Test Results:")
        print(f"Available clients: {clients}")

        # Get storage status
        status = storage_factory.get_storage_status()
        print(f"Storage status: {status}")

        return True

    except Exception as e:
        print(f"❌ Storage Factory Test Failed: {e}")
        return False

if __name__ == "__main__":
    success = test_storage_factory()
    if success:
        print("\n🎉 Storage factory is working correctly!")
    else:
        print("\n💥 Storage factory needs configuration or dependencies")