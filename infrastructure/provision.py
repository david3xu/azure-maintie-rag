#!/usr/bin/env python3
"""
Azure Universal RAG Infrastructure Provisioning Script
Automates Azure resource creation using Bicep templates
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def run_azure_command(command: str, capture_output: bool = True) -> Dict[str, Any]:
    """Run Azure CLI command and return results"""
    try:
        result = subprocess.run(
            command.split(),
            capture_output=capture_output,
            text=True,
            check=True
        )
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
            "return_code": e.returncode
        }

def check_azure_login() -> bool:
    """Check if user is logged into Azure CLI"""
    result = run_azure_command("az account show")
    return result["success"]

def create_resource_group(name: str, location: str) -> Dict[str, Any]:
    """Create Azure resource group"""
    command = f"az group create --name {name} --location {location}"
    return run_azure_command(command)

def deploy_bicep_template(resource_group: str, template_file: str, parameters_file: str) -> Dict[str, Any]:
    """Deploy Bicep template to resource group"""
    command = f"az deployment group create --resource-group {resource_group} --template-file {template_file} --parameters {parameters_file}"
    return run_azure_command(command)

def get_deployment_outputs(resource_group: str, deployment_name: str) -> Dict[str, Any]:
    """Get deployment outputs"""
    command = f"az deployment group show --resource-group {resource_group} --name {deployment_name}"
    result = run_azure_command(command)
    if result["success"]:
        deployment_data = json.loads(result["stdout"])
        return {
            "success": True,
            "outputs": deployment_data.get("properties", {}).get("outputs", {})
        }
    return result

def main():
    """Main provisioning function"""
    print("🚀 Azure Universal RAG Infrastructure Provisioning")
    print("=" * 60)

    # Configuration
    resource_group_name = "maintie-universal-rag-rg"
    location = "eastus"
    template_file = "infrastructure/azure-resources.bicep"
    parameters_file = "infrastructure/parameters.json"

    # Check Azure login
    print("🔐 Checking Azure CLI login...")
    if not check_azure_login():
        print("❌ Not logged into Azure CLI. Please run 'az login' first.")
        sys.exit(1)
    print("✅ Azure CLI login verified")

    # Create resource group
    print(f"\n📦 Creating resource group: {resource_group_name}")
    rg_result = create_resource_group(resource_group_name, location)
    if not rg_result["success"]:
        print(f"❌ Failed to create resource group: {rg_result.get('error', 'Unknown error')}")
        sys.exit(1)
    print("✅ Resource group created successfully")

    # Deploy Bicep template
    print(f"\n🏗️  Deploying Azure resources...")
    deploy_result = deploy_bicep_template(resource_group_name, template_file, parameters_file)
    if not deploy_result["success"]:
        print(f"❌ Deployment failed: {deploy_result.get('error', 'Unknown error')}")
        print(f"Error details: {deploy_result.get('stderr', 'No error details')}")
        sys.exit(1)

    print("✅ Azure resources deployed successfully")

    # Get deployment outputs
    print(f"\n📊 Getting deployment outputs...")
    outputs_result = get_deployment_outputs(resource_group_name, "azure-resources")
    if outputs_result["success"]:
        outputs = outputs_result["outputs"]
        print("\n🎯 Deployment Outputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value.get('value', 'N/A')}")

    print(f"\n🎉 Azure Universal RAG infrastructure provisioning completed!")
    print(f"📋 Resource Group: {resource_group_name}")
    print(f"🌍 Location: {location}")
    print(f"\n📝 Next Steps:")
    print(f"  1. Configure Azure service credentials in environment variables")
    print(f"  2. Update backend/config/azure_settings.py with service endpoints")
    print(f"  3. Test Azure service connections")
    print(f"  4. Deploy Universal RAG application")

if __name__ == "__main__":
    main()