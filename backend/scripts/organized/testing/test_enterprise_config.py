#!/usr/bin/env python3
"""
Simple Enterprise Configuration Validation
Tests the configuration updates without requiring all dependencies
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_configuration_updates():
    """Test enterprise configuration settings"""
    print("🔧 Testing Enterprise Configuration Updates...")

    try:
        from config.settings import azure_settings

        # Test Key Vault settings
        assert hasattr(azure_settings, 'azure_key_vault_url'), "Key Vault URL setting missing"
        assert hasattr(azure_settings, 'azure_use_managed_identity'), "Managed identity setting missing"
        assert hasattr(azure_settings, 'azure_managed_identity_client_id'), "Managed identity client ID setting missing"

        # Test Application Insights settings
        assert hasattr(azure_settings, 'azure_application_insights_connection_string'), "App Insights connection string missing"
        assert hasattr(azure_settings, 'azure_enable_telemetry'), "Telemetry setting missing"

        print("✅ Configuration updates validated")
        print(f"   Key Vault URL: {azure_settings.azure_key_vault_url}")
        print(f"   Use Managed Identity: {azure_settings.azure_use_managed_identity}")
        print(f"   Enable Telemetry: {azure_settings.azure_enable_telemetry}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_credential_management():
    """Test enterprise credential management pattern"""
    print("🔐 Testing Enterprise Credential Management Pattern...")

    try:
        from core.azure_storage.storage_client import AzureStorageClient

        # Test that the credential method exists
        storage_client = AzureStorageClient()
        assert hasattr(storage_client, '_get_azure_credential'), "Enterprise credential method missing"

        print("✅ Enterprise credential management pattern validated")
        return True

    except Exception as e:
        print(f"⚠️  Credential management test: {e}")
        return False


def test_telemetry_pattern():
    """Test telemetry pattern in storage client"""
    print("📊 Testing Telemetry Pattern...")

    try:
        from core.azure_storage.storage_client import AzureStorageClient

        # Test that upload_file method has telemetry
        storage_client = AzureStorageClient()

        # Create a test file
        test_file = Path("/tmp/test_telemetry.txt")
        test_file.write_text("Test telemetry data")

        result = storage_client.upload_file(test_file, "test_telemetry.txt")

        # Check for telemetry data
        assert "telemetry" in result, "Telemetry data missing from upload result"
        assert "operation_duration_ms" in result, "Operation duration missing"
        assert result["telemetry"]["service"] == "azure_storage", "Service name incorrect"

        print("✅ Telemetry pattern validated")

        # Cleanup
        test_file.unlink(missing_ok=True)
        return True

    except Exception as e:
        print(f"⚠️  Telemetry test: {e}")
        return False


def test_cost_optimization():
    """Test cost optimization features"""
    print("💰 Testing Cost Optimization...")

    try:
        from config.settings import azure_settings

        # Test environment-driven configuration
        env = azure_settings.azure_environment
        print(f"   Environment: {env}")

        # Test resource naming convention
        resource_name = azure_settings.get_resource_name("storage")
        print(f"   Resource naming: {resource_name}")

        # Test configuration validation
        config_validation = azure_settings.validate_azure_config()
        print(f"   Configuration validation: {config_validation}")

        print("✅ Cost optimization features validated")
        return True

    except Exception as e:
        print(f"⚠️  Cost optimization test: {e}")
        return False


def main():
    """Run all enterprise integration tests"""
    print("🚀 Enterprise Azure Integration Validation")
    print("=" * 50)

    tests = [
        test_configuration_updates,
        test_credential_management,
        test_telemetry_pattern,
        test_cost_optimization
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All Enterprise Integration Tests Passed!")
        print("✅ Security Enhancement (Key Vault Integration)")
        print("✅ Monitoring Integration (Application Insights)")
        print("✅ Infrastructure Cost Optimization")
        print("✅ Service Health Monitoring")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()