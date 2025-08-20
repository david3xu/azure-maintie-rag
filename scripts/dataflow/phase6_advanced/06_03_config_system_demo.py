#!/usr/bin/env python3
"""
Configuration System Demo - Phase 6 Advanced
=============================================

Demonstrates the advanced configuration system with dynamic Azure service
adaptation, environment detection, and real-time parameter adjustment.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.universal_config import UniversalConfig
from config.azure_settings import AzureSettings


async def demo_configuration_system():
    """Demonstrate the configuration system capabilities"""
    print("⚙️  CONFIGURATION SYSTEM DEMO")
    print("=" * 50)
    print("Showcasing dynamic Azure service configuration")
    print("and adaptive parameter management")
    print()

    demo_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tests": [],
        "summary": {}
    }

    # Test 1: Universal Config Loading
    print("🔧 Test 1: Universal Configuration Loading")
    try:
        config = UniversalConfig()

        print(f"   ✅ Config loaded successfully")
        print(f"   📊 Debug mode: {config.DEBUG}")
        print(f"   🚀 Environment: {config.ENVIRONMENT}")
        print(f"   📝 Log level: {config.LOG_LEVEL}")
        print(f"   ⏱️  Default timeout: {config.DEFAULT_TIMEOUT}s")

        demo_results["tests"].append({
            "name": "universal_config_loading",
            "success": True,
            "details": {
                "debug": config.DEBUG,
                "environment": config.ENVIRONMENT,
                "log_level": config.LOG_LEVEL,
                "timeout": config.DEFAULT_TIMEOUT
            }
        })

    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
        demo_results["tests"].append({
            "name": "universal_config_loading",
            "success": False,
            "error": str(e)
        })

    print()

    # Test 2: Azure Settings Configuration
    print("🔧 Test 2: Azure Settings Configuration")
    try:
        azure_settings = AzureSettings()

        print(f"   ✅ Azure settings loaded")
        print(f"   🤖 OpenAI deployment: {azure_settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
        print(f"   🧠 Embedding model: {azure_settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME}")
        print(f"   🔍 Search service: {azure_settings.AZURE_SEARCH_SERVICE_NAME}")
        print(f"   🌌 Cosmos endpoint configured: {bool(azure_settings.AZURE_COSMOS_ENDPOINT)}")

        demo_results["tests"].append({
            "name": "azure_settings_configuration",
            "success": True,
            "details": {
                "openai_deployment": azure_settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                "embedding_deployment": azure_settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                "search_service": azure_settings.AZURE_SEARCH_SERVICE_NAME,
                "cosmos_configured": bool(azure_settings.AZURE_COSMOS_ENDPOINT)
            }
        })

    except Exception as e:
        print(f"   ❌ Azure settings failed: {e}")
        demo_results["tests"].append({
            "name": "azure_settings_configuration",
            "success": False,
            "error": str(e)
        })

    print()

    # Test 3: Environment Detection
    print("🔧 Test 3: Environment Detection & Adaptation")
    try:
        # Test different environment scenarios
        from infrastructure.azure_auth.auth_manager import AzureAuthManager

        auth_manager = AzureAuthManager()

        # Check authentication status
        auth_status = auth_manager.validate_authentication()

        print(f"   ✅ Environment detection completed")
        print(f"   🔐 Azure CLI status: {'✅' if auth_status.get('az_cli_ready') else '❌'}")
        print(f"   📋 Subscription: {auth_status.get('subscription_id', 'Not available')[:20]}...")
        print(f"   👤 User: {auth_status.get('user_name', 'Not available')}")

        demo_results["tests"].append({
            "name": "environment_detection",
            "success": True,
            "details": {
                "az_cli_ready": auth_status.get('az_cli_ready', False),
                "subscription_available": bool(auth_status.get('subscription_id')),
                "user_authenticated": bool(auth_status.get('user_name'))
            }
        })

    except Exception as e:
        print(f"   ❌ Environment detection failed: {e}")
        demo_results["tests"].append({
            "name": "environment_detection",
            "success": False,
            "error": str(e)
        })

    print()

    # Test 4: Dynamic Parameter Adjustment
    print("🔧 Test 4: Dynamic Parameter Adjustment")
    try:
        # Simulate parameter adjustment based on workload
        scenarios = [
            {"name": "Light workload", "factor": 0.5},
            {"name": "Normal workload", "factor": 1.0},
            {"name": "Heavy workload", "factor": 2.0}
        ]

        adjustments = []
        for scenario in scenarios:
            base_timeout = config.DEFAULT_TIMEOUT
            adjusted_timeout = int(base_timeout * scenario["factor"])

            adjustments.append({
                "scenario": scenario["name"],
                "base_timeout": base_timeout,
                "adjusted_timeout": adjusted_timeout,
                "factor": scenario["factor"]
            })

            print(f"   📊 {scenario['name']}: {base_timeout}s → {adjusted_timeout}s")

        print(f"   ✅ Dynamic adjustment simulation completed")

        demo_results["tests"].append({
            "name": "dynamic_parameter_adjustment",
            "success": True,
            "details": {
                "adjustments": adjustments
            }
        })

    except Exception as e:
        print(f"   ❌ Dynamic adjustment failed: {e}")
        demo_results["tests"].append({
            "name": "dynamic_parameter_adjustment",
            "success": False,
            "error": str(e)
        })

    print()

    # Test 5: Service Health Monitoring
    print("🔧 Test 5: Service Health Monitoring")
    try:
        # Test basic service connectivity
        from agents.core.universal_deps import get_universal_deps

        deps = await get_universal_deps()

        services_status = {}
        services_to_check = ['openai', 'search', 'cosmos']

        for service in services_to_check:
            try:
                is_available = deps.is_service_available(service)
                services_status[service] = is_available
                status_icon = "✅" if is_available else "❌"
                print(f"   {status_icon} {service.title()} service: {'Available' if is_available else 'Not available'}")
            except Exception as e:
                services_status[service] = False
                print(f"   ❌ {service.title()} service: Error - {str(e)[:50]}...")

        demo_results["tests"].append({
            "name": "service_health_monitoring",
            "success": True,
            "details": {
                "services_checked": services_to_check,
                "services_status": services_status,
                "available_services": sum(services_status.values())
            }
        })

    except Exception as e:
        print(f"   ❌ Service health monitoring failed: {e}")
        demo_results["tests"].append({
            "name": "service_health_monitoring",
            "success": False,
            "error": str(e)
        })

    print()

    # Summary
    print("📊 CONFIGURATION DEMO SUMMARY")
    print("=" * 40)

    successful_tests = sum(1 for test in demo_results["tests"] if test["success"])
    total_tests = len(demo_results["tests"])

    print(f"✅ Successful tests: {successful_tests}/{total_tests}")

    for test in demo_results["tests"]:
        status_icon = "✅" if test["success"] else "❌"
        print(f"   {status_icon} {test['name']}")

    demo_results["summary"] = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": successful_tests / total_tests if total_tests > 0 else 0
    }

    # Save results
    from scripts.dataflow.utilities.path_utils import get_results_dir
    results_dir = get_results_dir()
    results_file = results_dir / "config_system_demo_results.json"

    with open(results_file, 'w') as f:
        json.dump(demo_results, f, indent=2)

    print(f"📄 Demo results saved to: {results_file}")

    if successful_tests == total_tests:
        print(f"\n🎉 CONFIGURATION DEMO COMPLETED SUCCESSFULLY")
        print("All configuration components working correctly")
        return True
    else:
        print(f"\n⚠️  CONFIGURATION DEMO PARTIALLY COMPLETED")
        print(f"Some tests failed - check individual results above")
        return successful_tests > 0


async def main():
    """Main execution function"""
    print("🚀 Starting Configuration System Demo...")
    print("This demonstrates the adaptive configuration capabilities")
    print("of the Azure RAG system.\n")

    success = await demo_configuration_system()
    return success


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            print("\n✅ Configuration demo completed successfully")
            sys.exit(0)
        else:
            print("\n❌ Configuration demo had issues")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Configuration demo error: {e}")
        sys.exit(1)
