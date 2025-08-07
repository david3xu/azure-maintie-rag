#!/usr/bin/env python3
"""
Azure State Check - Production Azure Universal RAG
Comprehensive Azure services validation with enterprise session tracking.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import azure_settings
from infrastructure.azure_cosmos.cosmos_gremlin_client import SimpleCosmosClient
from infrastructure.azure_openai.openai_client import AzureOpenAIClient
from infrastructure.azure_search.search_client import SimpleSearchClient
from infrastructure.azure_storage.storage_client import SimpleStorageClient


async def check_azure_state(
    domain: str = "universal", verbose: bool = False
) -> Dict[str, Any]:
    """Comprehensive Azure services state check with detailed reporting"""
    session_id = f"azure_check_{int(time.time())}"
    print(f"🔍 Azure Universal RAG - State Check (Session: {session_id})")
    print(f"Domain: {domain} | Verbose: {verbose}")
    print("=" * 60)

    check_results = {
        "session_id": session_id,
        "domain": domain,
        "timestamp": time.time(),
        "overall_status": "unknown",
        "service_details": {},
        "data_availability": {},
        "recommendations": [],
    }

    try:
        # Initialize individual Azure service clients
        print("🏗️  Initializing Azure Service Clients...")

        service_status = {
            "individual_services": {},
            "successful_services": 0,
            "total_services": 4,
            "overall_health": "unknown",
        }

        # Test Storage Client
        try:
            storage_client = SimpleStorageClient()
            await storage_client.async_initialize()
            service_status["individual_services"]["storage"] = True
            service_status["successful_services"] += 1
            if verbose:
                print("   ✅ Storage Client: Ready")
        except Exception as e:
            service_status["individual_services"]["storage"] = False
            if verbose:
                print(f"   ❌ Storage Client: {str(e)[:50]}...")

        # Test OpenAI Client
        try:
            openai_client = AzureOpenAIClient()
            await openai_client.async_initialize()
            service_status["individual_services"]["openai"] = True
            service_status["successful_services"] += 1
            if verbose:
                print("   ✅ OpenAI Client: Ready")
        except Exception as e:
            service_status["individual_services"]["openai"] = False
            if verbose:
                print(f"   ❌ OpenAI Client: {str(e)[:50]}...")

        # Test Search Client
        try:
            search_client = SimpleSearchClient()
            await search_client.async_initialize()
            service_status["individual_services"]["search"] = True
            service_status["successful_services"] += 1
            if verbose:
                print("   ✅ Search Client: Ready")
        except Exception as e:
            service_status["individual_services"]["search"] = False
            if verbose:
                print(f"   ❌ Search Client: {str(e)[:50]}...")

        # Test Cosmos Client
        try:
            cosmos_client = SimpleCosmosClient()
            await cosmos_client.async_initialize()
            service_status["individual_services"]["cosmos"] = True
            service_status["successful_services"] += 1
            if verbose:
                print("   ✅ Cosmos Client: Ready")
        except Exception as e:
            service_status["individual_services"]["cosmos"] = False
            if verbose:
                print(f"   ❌ Cosmos Client: {str(e)[:50]}...")

        # Determine overall health
        if service_status["successful_services"] >= 3:
            service_status["overall_health"] = "healthy"
        elif service_status["successful_services"] >= 2:
            service_status["overall_health"] = "partial"
        else:
            service_status["overall_health"] = "unhealthy"

        check_results["service_details"] = service_status

        print(f"\n📊 Azure Services Status:")
        print(f"   Overall Health: {service_status['overall_health']}")
        print(
            f"   Services Ready: {service_status['successful_services']}/{service_status['total_services']}"
        )

        if verbose:
            for service_name, status in service_status.get(
                "individual_services", {}
            ).items():
                status_icon = "✅" if status else "❌"
                print(
                    f"   {status_icon} {service_name}: {'Ready' if status else 'Not Available'}"
                )

        # Check data availability
        print(f"\n📁 Data Availability Check:")
        data_availability = await check_data_availability(verbose)
        check_results["data_availability"] = data_availability

        # Environment validation
        print(f"\n🌍 Environment Validation:")
        env_status = check_environment_config()
        check_results["environment"] = env_status

        # Generate recommendations
        print(f"\n💡 System Recommendations:")
        recommendations = generate_recommendations(
            service_status, data_availability, env_status
        )
        check_results["recommendations"] = recommendations

        for rec in recommendations:
            print(f"   • {rec}")

        # Overall status determination
        if (
            service_status["successful_services"] >= 3
            and data_availability["total_files"] > 0
        ):
            check_results["overall_status"] = "ready"
            print(f"\n✅ System Status: READY FOR PROCESSING")
        elif service_status["successful_services"] >= 2:
            check_results["overall_status"] = "partial"
            print(f"\n⚠️  System Status: PARTIAL - Some services unavailable")
        else:
            check_results["overall_status"] = "not_ready"
            print(f"\n❌ System Status: NOT READY - Critical services unavailable")

        return check_results

    except Exception as e:
        check_results["overall_status"] = "error"
        check_results["error"] = str(e)
        print(f"❌ Azure state check failed: {e}")
        return check_results


async def check_data_availability(verbose: bool = False) -> Dict[str, Any]:
    """Check availability of data files for processing"""
    data_info = {
        "raw_data_exists": False,
        "total_files": 0,
        "file_types": {},
        "largest_file": None,
        "total_size_mb": 0,
    }

    raw_data_dir = Path("data/raw")
    if raw_data_dir.exists():
        data_info["raw_data_exists"] = True

        # Count files by type
        all_files = list(raw_data_dir.rglob("*.*"))
        data_info["total_files"] = len(all_files)

        total_size = 0
        largest_size = 0
        largest_file = None

        for file_path in all_files:
            if file_path.is_file():
                file_size = file_path.stat().st_size
                total_size += file_size

                if file_size > largest_size:
                    largest_size = file_size
                    largest_file = file_path.name

                # Count by extension
                ext = file_path.suffix.lower()
                data_info["file_types"][ext] = data_info["file_types"].get(ext, 0) + 1

        data_info["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        data_info["largest_file"] = largest_file

        print(f"   📂 Raw data directory: Found ({data_info['total_files']} files)")
        print(f"   📊 Total size: {data_info['total_size_mb']} MB")

        if verbose:
            for ext, count in data_info["file_types"].items():
                print(f"   📄 {ext}: {count} files")
    else:
        print(f"   📂 Raw data directory: Not found")

    return data_info


def check_environment_config() -> Dict[str, Any]:
    """Validate environment configuration"""
    env_status = {
        "config_files_exist": False,
        "required_env_vars": {},
        "azure_auth_method": "unknown",
    }

    # Check for config files
    config_files = [
        Path(".env"),
        Path("config/azure_settings.py"),
        Path("config/centralized_config.py"),
    ]

    existing_configs = [f.name for f in config_files if f.exists()]
    env_status["config_files_exist"] = len(existing_configs) > 0
    env_status["existing_configs"] = existing_configs

    print(f"   ⚙️  Configuration files: {len(existing_configs)} found")

    # Check authentication method
    import os

    if os.getenv("USE_MANAGED_IDENTITY") == "true":
        env_status["azure_auth_method"] = "managed_identity"
        print(f"   🔐 Authentication: Managed Identity")
    elif os.getenv("AZURE_CLIENT_ID"):
        env_status["azure_auth_method"] = "service_principal"
        print(f"   🔐 Authentication: Service Principal")
    else:
        env_status["azure_auth_method"] = "default_credential"
        print(f"   🔐 Authentication: Default Credential Chain")

    return env_status


def generate_recommendations(
    service_status: Dict, data_availability: Dict, env_status: Dict
) -> List[str]:
    """Generate actionable recommendations based on system state"""
    recommendations = []

    if service_status["successful_services"] < 2:
        recommendations.append(
            "Run 'make azure-deploy' to deploy missing Azure services"
        )
        recommendations.append(
            "Check Azure authentication with 'az login' and 'azd auth login'"
        )

    if not data_availability["raw_data_exists"]:
        recommendations.append("Add data files to 'data/raw/' directory for processing")
        recommendations.append("Use 'make data-upload' after adding data files")

    if (
        data_availability["total_files"] > 0
        and service_status["successful_services"] >= 3
    ):
        recommendations.append(
            "System ready - run 'make data-prep-full' for complete pipeline"
        )
        recommendations.append(
            "Use 'make unified-search-demo' to test tri-modal search"
        )

    if not env_status["config_files_exist"]:
        recommendations.append(
            "Run 'make sync-env' to synchronize environment configuration"
        )

    if len(recommendations) == 0:
        recommendations.append(
            "System is optimally configured - ready for production workloads"
        )

    return recommendations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Azure Universal RAG - State Check")
    parser.add_argument(
        "--domain", default="universal", help="Domain to check (default: universal)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output with detailed service status",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    # Run the state check
    result = asyncio.run(check_azure_state(args.domain, args.verbose))

    # Handle JSON output
    if args.json or args.output:
        json_output = json.dumps(result, indent=2, default=str)

        if args.output:
            # Save to file
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_output)
            print(f"\n📄 Results saved to: {output_path}")

        if args.json:
            # Print JSON to stdout
            print(f"\n" + "=" * 60)
            print("JSON Output:")
            print(json_output)

    # Exit with appropriate code
    if result["overall_status"] in ["ready", "partial"]:
        sys.exit(0)
    else:
        sys.exit(1)
