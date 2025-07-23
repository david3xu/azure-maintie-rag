#!/usr/bin/env python3
"""
Azure Enterprise Configuration Validation Service
Comprehensive Azure service configuration validation and health monitoring
Based on existing enterprise infrastructure patterns
"""

import sys
import os
import asyncio
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def print_header(title: str, width: int = 60):
    """Print formatted section header"""
    print("\n" + "=" * width)
    print(f"🔧 {title}")
    print("=" * width)

def print_service_status(service_name: str, status: str, details: str = ""):
    """Print service status with appropriate icons"""
    icons = {
        "healthy": "✅",
        "unhealthy": "❌",
        "disabled": "⚠️",
        "not_configured": "⚠️",
        "timeout": "🕐",
        "unknown": "❓"
    }
    icon = icons.get(status, "❓")
    print(f"   {icon} {service_name}: {status.upper()}")
    if details:
        print(f"      {details}")

class AzureEnterpriseConfigValidator:
    """
    Azure Enterprise Configuration Validation Service
    """
    def __init__(self):
        self.validation_results = {}
        self.start_time = time.time()

    async def validate_core_configuration(self) -> Dict[str, Any]:
        print_header("Azure Core Service Configuration Validation")
        try:
            from config.settings import azure_settings
            core_services = {
                "Azure OpenAI": {
                    "endpoint": azure_settings.openai_api_base,
                    "key_configured": bool(azure_settings.openai_api_key),
                    "deployment": azure_settings.openai_deployment_name,
                    "api_version": azure_settings.openai_api_version
                },
                "Azure Storage": {
                    "account": azure_settings.azure_storage_account,
                    "key_configured": bool(azure_settings.azure_storage_key),
                    "container": azure_settings.azure_blob_container,
                    "connection_configured": bool(azure_settings.azure_storage_connection_string)
                },
                "Azure Cognitive Search": {
                    "service": azure_settings.azure_search_service,
                    "key_configured": bool(azure_settings.azure_search_admin_key),
                    "api_version": azure_settings.azure_search_api_version,
                    "service_name": azure_settings.azure_search_service_name
                },
                "Azure Cosmos DB": {
                    "endpoint": azure_settings.azure_cosmos_endpoint,
                    "key_configured": bool(azure_settings.azure_cosmos_key),
                    "database": azure_settings.azure_cosmos_database,
                    "container": azure_settings.azure_cosmos_container
                }
            }
            validation_status = "healthy"
            for service_name, config in core_services.items():
                service_healthy = True
                details = []
                for key, value in config.items():
                    if key.endswith('_configured'):
                        if not value:
                            service_healthy = False
                            details.append(f"{key.replace('_configured', '')} not configured")
                    elif not value:
                        service_healthy = False
                        details.append(f"{key} missing")
                status = "healthy" if service_healthy else "unhealthy"
                if not service_healthy:
                    validation_status = "unhealthy"
                print_service_status(service_name, status, "; ".join(details))
            self.validation_results['core_configuration'] = {
                "status": validation_status,
                "services_validated": len(core_services),
                "timestamp": datetime.now().isoformat()
            }
            return self.validation_results['core_configuration']
        except Exception as e:
            print_service_status("Configuration Loading", "unhealthy", str(e))
            return {"status": "unhealthy", "error": str(e)}

    async def validate_optional_services(self) -> Dict[str, Any]:
        print_header("Azure Optional Service Configuration Validation")
        try:
            from config.settings import azure_settings
            optional_services = {
                "Azure Text Analytics": {
                    "endpoint": getattr(azure_settings, 'azure_text_analytics_endpoint', ''),
                    "key_configured": bool(getattr(azure_settings, 'azure_text_analytics_key', ''))
                },
                "Azure Application Insights": {
                    "connection_configured": bool(getattr(azure_settings, 'azure_application_insights_connection_string', '')),
                    "telemetry_enabled": getattr(azure_settings, 'azure_enable_telemetry', False)
                },
                "Azure Key Vault": {
                    "vault_url": getattr(azure_settings, 'azure_key_vault_url', ''),
                    "managed_identity": getattr(azure_settings, 'azure_use_managed_identity', False)
                },
                "Azure ML Quality Assessment": {
                    "confidence_endpoint": getattr(azure_settings, 'azure_ml_confidence_endpoint', ''),
                    "completeness_endpoint": getattr(azure_settings, 'azure_ml_completeness_endpoint', '')
                }
            }
            configured_count = 0
            for service_name, config in optional_services.items():
                service_configured = any(bool(value) for value in config.values())
                status = "healthy" if service_configured else "not_configured"
                if service_configured:
                    configured_count += 1
                    details = []
                    for key, value in config.items():
                        if value:
                            details.append(f"{key}: configured")
                    print_service_status(service_name, status, "; ".join(details))
                else:
                    print_service_status(service_name, status, "Optional service - can be enabled later")
            self.validation_results['optional_services'] = {
                "status": "healthy",
                "configured_services": configured_count,
                "total_services": len(optional_services),
                "timestamp": datetime.now().isoformat()
            }
            return self.validation_results['optional_services']
        except Exception as e:
            print_service_status("Optional Services", "unhealthy", str(e))
            return {"status": "unhealthy", "error": str(e)}

    async def validate_service_connectivity(self) -> Dict[str, Any]:
        print_header("Azure Service Connectivity Validation")
        try:
            from integrations.azure_services import AzureServicesManager
            azure_services = AzureServicesManager()
            health_results = azure_services.get_service_health()
            print(f"   🔍 Overall Health Status: {health_results['overall_status'].upper()}")
            print(f"   📊 Service Health Ratio: {health_results['healthy_count']}/{health_results['total_count']}")
            print(f"   ⏱️  Health Check Duration: {health_results['health_check_duration_ms']:.2f}ms")
            if 'circuit_breaker' in health_results:
                cb_status = health_results['circuit_breaker']
                print(f"   🔒 Circuit Breaker: {cb_status.get('status', 'unknown')} (failures: {cb_status.get('consecutive_failures', 0)})")
            print("\n   Service Details:")
            for service_name, status in health_results['services'].items():
                service_status = status.get('status', 'unknown')
                details = []
                if status.get('endpoint'):
                    details.append(f"endpoint: {status['endpoint']}")
                if status.get('error'):
                    details.append(f"error: {status['error']}")
                if status.get('index_count') is not None:
                    details.append(f"indices: {status['index_count']}")
                print_service_status(f"   {service_name.replace('_', ' ').title()}", service_status, "; ".join(details))
            self.validation_results['service_connectivity'] = {
                "status": health_results['overall_status'],
                "healthy_services": health_results['healthy_count'],
                "total_services": health_results['total_count'],
                "check_duration_ms": health_results['health_check_duration_ms'],
                "timestamp": datetime.now().isoformat()
            }
            return self.validation_results['service_connectivity']
        except Exception as e:
            print_service_status("Service Connectivity", "unhealthy", str(e))
            return {"status": "unhealthy", "error": str(e)}

    async def validate_environment_configuration(self) -> Dict[str, Any]:
        print_header("Azure Environment Configuration Validation")
        try:
            from config.settings import azure_settings
            env_config = {
                "Environment Tier": azure_settings.azure_environment,
                "Azure Region": azure_settings.azure_region,
                "Resource Prefix": azure_settings.azure_resource_prefix,
                "Debug Mode": azure_settings.debug,
                "Log Level": getattr(azure_settings, 'log_level', 'INFO')
            }
            performance_config = {
                "OpenAI Max Tokens/Min": azure_settings.azure_openai_max_tokens_per_minute,
                "OpenAI Max Requests/Min": azure_settings.azure_openai_max_requests_per_minute,
                "Extraction Batch Size": azure_settings.extraction_batch_size,
                "Discovery Sample Size": azure_settings.discovery_sample_size
            }
            print("   🏗️ Environment Configuration:")
            for key, value in env_config.items():
                print(f"      ✅ {key}: {value}")
            print("\n   ⚡ Performance Configuration:")
            for key, value in performance_config.items():
                print(f"      ✅ {key}: {value}")
            env = azure_settings.azure_environment
            validation_status = "healthy"
            if env == "dev":
                if azure_settings.azure_openai_max_tokens_per_minute > 15000:
                    print("   ⚠️  Dev environment: Consider reducing token limits for cost optimization")
            elif env == "prod":
                if azure_settings.debug:
                    print("   ⚠️  Production environment: Debug mode should be disabled")
                    validation_status = "warning"
            self.validation_results['environment_configuration'] = {
                "status": validation_status,
                "environment": env,
                "region": azure_settings.azure_region,
                "debug_mode": azure_settings.debug,
                "timestamp": datetime.now().isoformat()
            }
            return self.validation_results['environment_configuration']
        except Exception as e:
            print_service_status("Environment Configuration", "unhealthy", str(e))
            return {"status": "unhealthy", "error": str(e)}

    async def generate_validation_report(self) -> Dict[str, Any]:
        print_header("Azure Enterprise Configuration Validation Report")
        total_duration = (time.time() - self.start_time) * 1000
        total_validations = len(self.validation_results)
        healthy_validations = sum(1 for result in self.validation_results.values()
                                if result.get('status') in ['healthy', 'warning'])
        overall_status = "healthy" if healthy_validations == total_validations else "unhealthy"
        report = {
            "validation_summary": {
                "overall_status": overall_status,
                "total_validations": total_validations,
                "healthy_validations": healthy_validations,
                "validation_duration_ms": total_duration,
                "timestamp": datetime.now().isoformat()
            },
            "validation_results": self.validation_results,
            "recommendations": []
        }
        if self.validation_results.get('core_configuration', {}).get('status') != 'healthy':
            report["recommendations"].append("Configure core Azure services for full functionality")
        if self.validation_results.get('optional_services', {}).get('configured_services', 0) == 0:
            report["recommendations"].append("Consider enabling Azure Text Analytics for enhanced knowledge extraction")
        if self.validation_results.get('service_connectivity', {}).get('status') != 'healthy':
            report["recommendations"].append("Verify Azure service credentials and network connectivity")
        print(f"\n   📊 VALIDATION SUMMARY:")
        print(f"   Overall Status: {overall_status.upper()}")
        print(f"   Validations Passed: {healthy_validations}/{total_validations}")
        print(f"   Validation Duration: {total_duration:.2f}ms")
        if report["recommendations"]:
            print(f"\n   💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"   {i}. {rec}")
        print(f"\n   🎯 OPERATIONAL READINESS: {'✅ READY' if overall_status == 'healthy' else '⚠️ NEEDS ATTENTION'}")
        return report

async def main():
    print("🚀 Azure Enterprise Configuration Validation Service")
    print("=" * 60)
    print("🏗️ Universal RAG → Azure Migration Configuration Validation")
    print("📋 Based on existing enterprise infrastructure patterns")
    validator = AzureEnterpriseConfigValidator()
    try:
        await validator.validate_core_configuration()
        await validator.validate_optional_services()
        await validator.validate_service_connectivity()
        await validator.validate_environment_configuration()
        report = await validator.generate_validation_report()
        report_path = Path("azure_config_validation_report.json")
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Detailed report saved: {report_path}")
        overall_status = report["validation_summary"]["overall_status"]
        sys.exit(0 if overall_status == "healthy" else 1)
    except Exception as e:
        print(f"\n❌ Validation service error: {e}")
        print("💡 Ensure you're running from the backend directory with proper dependencies")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())