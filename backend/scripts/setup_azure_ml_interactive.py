#!/usr/bin/env python3
"""
Setup Azure ML with Interactive Authentication (No Service Principal Required)
Uses DefaultAzureCredential which works with your current Azure CLI login
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

def setup_interactive_auth():
    """Setup Azure ML using Interactive/CLI authentication"""
    
    print("🔐 AZURE ML WITH INTERACTIVE AUTH")
    print("=" * 50)
    print("✅ Using your current Azure CLI login (no Service Principal needed)")
    
    try:
        from azure.identity import DefaultAzureCredential
        from azureml.core import Workspace
        
        # Use DefaultAzureCredential (uses your az login session)
        credential = DefaultAzureCredential()
        
        print("🔗 Connecting to Azure ML workspace...")
        
        # Connect to workspace using your CLI credentials
        workspace_name = os.environ.get('AZURE_ML_WORKSPACE_NAME', 'maintie-dev-ml-1cdd8e11')
        
        try:
            workspace = Workspace.get(
                name=workspace_name,
                subscription_id=os.environ['AZURE_SUBSCRIPTION_ID'],
                resource_group=os.environ['AZURE_RESOURCE_GROUP'],
                auth=credential
            )
            print(f"✅ Connected to existing workspace: {workspace_name}")
        except:
            print(f"🔄 Creating workspace: {workspace_name}")
            workspace = Workspace.create(
                name=workspace_name,
                subscription_id=os.environ['AZURE_SUBSCRIPTION_ID'],
                resource_group=os.environ['AZURE_RESOURCE_GROUP'],
                location="East US",
                auth=credential
            )
            print(f"✅ Created workspace: {workspace_name}")
        
        print(f"📍 Workspace details:")
        print(f"   • Name: {workspace.name}")
        print(f"   • Location: {workspace.location}")
        print(f"   • Resource Group: {workspace.resource_group}")
        
        return workspace
        
    except Exception as e:
        print(f"❌ Azure ML connection failed: {e}")
        print(f"\n💡 Make sure you're logged in: az login")
        return None

def create_compute_cluster(workspace):
    """Create Azure ML compute cluster"""
    
    try:
        from azureml.core.compute import ComputeTarget, AmlCompute
        from azureml.core.compute_target import ComputeTargetException
        
        cluster_name = "gnn-training-cluster"
        
        try:
            compute_target = ComputeTarget(workspace=workspace, name=cluster_name)
            print(f"✅ Using existing compute cluster: {cluster_name}")
        except ComputeTargetException:
            print(f"🔄 Creating compute cluster: {cluster_name}")
            
            compute_config = AmlCompute.provisioning_configuration(
                vm_size="Standard_NC6s_v3",  # GPU for deep learning
                min_nodes=0,
                max_nodes=2,
                idle_seconds_before_scaledown=300,
                tier="Dedicated"
            )
            
            compute_target = ComputeTarget.create(workspace, cluster_name, compute_config)
            print(f"⏳ Waiting for cluster provisioning...")
            compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
            print(f"✅ Compute cluster created: {cluster_name}")
        
        return compute_target
        
    except Exception as e:
        print(f"❌ Compute cluster setup failed: {e}")
        return None

def create_training_environment(workspace):
    """Create Azure ML environment"""
    
    try:
        from azureml.core import Environment
        from azureml.core.conda_dependencies import CondaDependencies
        
        env_name = "gnn-training-environment"
        
        try:
            environment = Environment.get(workspace, env_name)
            print(f"✅ Using existing environment: {env_name}")
            return environment
        except:
            pass
        
        print(f"🔄 Creating training environment: {env_name}")
        
        environment = Environment(name=env_name)
        
        # Define conda dependencies
        conda_deps = CondaDependencies()
        conda_deps.add_conda_package("python=3.9")
        conda_deps.add_conda_package("pytorch=1.12.0")
        conda_deps.add_conda_package("torchvision")
        conda_deps.add_conda_package("torchaudio")
        conda_deps.add_conda_package("pytorch-cuda=11.6", channel="pytorch")
        
        # PyTorch Geometric
        conda_deps.add_pip_package("torch-geometric>=2.1.0")
        conda_deps.add_pip_package("torch-scatter")
        conda_deps.add_pip_package("torch-sparse") 
        conda_deps.add_pip_package("torch-cluster")
        
        # ML packages
        conda_deps.add_pip_package("scikit-learn>=1.1.0")
        conda_deps.add_pip_package("numpy>=1.21.0")
        conda_deps.add_pip_package("pandas>=1.4.0")
        
        # Azure ML
        conda_deps.add_pip_package("azureml-sdk")
        conda_deps.add_pip_package("azureml-mlflow")
        
        environment.python.conda_dependencies = conda_deps
        environment.docker.enabled = True
        environment.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04"
        
        environment = environment.register(workspace)
        print(f"✅ Environment registered: {env_name}")
        
        return environment
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return None

def save_azure_config(workspace, compute_target, environment):
    """Save Azure ML configuration"""
    
    config_dir = Path(__file__).parent.parent / "azure_ml_config"
    config_dir.mkdir(exist_ok=True)
    
    # Save workspace config for SDK
    workspace.write_config(path=config_dir)
    
    config = {
        "workspace_name": workspace.name,
        "subscription_id": workspace.subscription_id,
        "resource_group": workspace.resource_group,
        "compute_target": compute_target.name if compute_target else None,
        "environment_name": environment.name if environment else None,
        "datastore_name": workspace.get_default_datastore().name,
        "setup_timestamp": datetime.now().isoformat(),
        "setup_status": "completed",
        "auth_method": "interactive"
    }
    
    config_file = config_dir / "azure_ml_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Azure ML config saved: {config_file}")
    return config_file

def main():
    """Main Azure ML setup with interactive auth"""
    
    print("☁️ AZURE ML SETUP (INTERACTIVE AUTH)")
    print("=" * 50)
    print("🎯 No Service Principal required - using your Azure CLI login")
    print("=" * 50)
    
    # Check required environment variables
    required_vars = [
        'AZURE_SUBSCRIPTION_ID',
        'AZURE_RESOURCE_GROUP'
    ]
    
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        print("💡 These should be in your .env file")
        return
    
    try:
        # Connect to workspace
        workspace = setup_interactive_auth()
        if not workspace:
            return
        
        # Create compute cluster
        compute_target = create_compute_cluster(workspace)
        
        # Create training environment
        environment = create_training_environment(workspace)
        
        # Save configuration
        config_file = save_azure_config(workspace, compute_target, environment)
        
        print(f"\n🎉 AZURE ML SETUP COMPLETED!")
        print(f"📋 Configuration: {config_file}")
        print(f"🏭 Workspace: {workspace.name}")
        print(f"💻 Compute: {compute_target.name if compute_target else 'None'}")
        print(f"🐍 Environment: {environment.name if environment else 'None'}")
        
        print(f"\n🚀 READY FOR TRAINING!")
        print(f"   Next: python scripts/azure_ml_gnn_training_interactive.py --partial")
        
    except Exception as e:
        print(f"❌ Azure ML setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()