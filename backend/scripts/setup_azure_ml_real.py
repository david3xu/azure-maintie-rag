#!/usr/bin/env python3
"""
Setup Real Azure ML Environment
Configure actual Azure ML workspace, compute, and environment for GNN training
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

def check_azure_credentials():
    """Check if Azure credentials are configured"""
    
    required_vars = [
        'AZURE_SUBSCRIPTION_ID',
        'AZURE_RESOURCE_GROUP', 
        'AZURE_TENANT_ID',
        'AZURE_CLIENT_ID',
        'AZURE_CLIENT_SECRET'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing Azure credentials:")
        for var in missing_vars:
            print(f"   export {var}=<your-{var.lower().replace('_', '-')}>")
        print("\n💡 Get these from Azure Portal:")
        print("   1. Create Service Principal: az ad sp create-for-rbac --name gnn-training-sp")
        print("   2. Assign ML permissions: az role assignment create --assignee <client-id> --role 'AzureML Data Scientist'")
        return False
    
    print("✅ Azure credentials configured")
    return True

def install_azure_ml_sdk():
    """Install Azure ML SDK and dependencies"""
    
    try:
        import azureml.core
        print("✅ Azure ML SDK already installed")
        return True
    except ImportError:
        print("📦 Installing Azure ML SDK...")
        
        import subprocess
        
        packages = [
            "azureml-sdk[notebooks,explain,contrib,tensorboard]",
            "azureml-mlflow",
            "torch>=1.12.0",
            "torch-geometric>=2.1.0", 
            "torch-scatter",
            "torch-sparse",
            "torch-cluster"
        ]
        
        for package in packages:
            print(f"   Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to install {package}: {result.stderr}")
                return False
        
        print("✅ Azure ML SDK installed successfully")
        return True

def create_azure_ml_workspace():
    """Create or connect to Azure ML workspace"""
    
    try:
        from azureml.core import Workspace
        from azureml.core.authentication import ServicePrincipalAuthentication
        
        print("🔗 Connecting to Azure ML workspace...")
        
        # Service Principal authentication
        auth = ServicePrincipalAuthentication(
            tenant_id=os.environ['AZURE_TENANT_ID'],
            service_principal_id=os.environ['AZURE_CLIENT_ID'],
            service_principal_password=os.environ['AZURE_CLIENT_SECRET']
        )
        
        workspace_name = "maintie-ml-workspace"
        
        try:
            # Try to connect to existing workspace
            workspace = Workspace(
                subscription_id=os.environ['AZURE_SUBSCRIPTION_ID'],
                resource_group=os.environ['AZURE_RESOURCE_GROUP'],
                workspace_name=workspace_name,
                auth=auth
            )
            print(f"✅ Connected to existing workspace: {workspace_name}")
            
        except Exception:
            # Create new workspace
            print(f"🔄 Creating new workspace: {workspace_name}")
            
            workspace = Workspace.create(
                name=workspace_name,
                subscription_id=os.environ['AZURE_SUBSCRIPTION_ID'],
                resource_group=os.environ['AZURE_RESOURCE_GROUP'],
                location="East US",  # Change as needed
                auth=auth
            )
            print(f"✅ Created workspace: {workspace_name}")
        
        # Save workspace config
        config_dir = Path(__file__).parent.parent / "azure_ml_config"
        config_dir.mkdir(exist_ok=True)
        
        workspace.write_config(path=config_dir)
        print(f"✅ Workspace config saved to: {config_dir}")
        
        return workspace
        
    except Exception as e:
        print(f"❌ Workspace setup failed: {e}")
        return None

def create_compute_cluster(workspace):
    """Create Azure ML compute cluster for training"""
    
    try:
        from azureml.core.compute import ComputeTarget, AmlCompute
        from azureml.core.compute_target import ComputeTargetException
        
        cluster_name = "gnn-training-cluster"
        
        try:
            # Check if cluster exists
            compute_target = ComputeTarget(workspace=workspace, name=cluster_name)
            print(f"✅ Using existing compute cluster: {cluster_name}")
            
        except ComputeTargetException:
            # Create new cluster
            print(f"🔄 Creating compute cluster: {cluster_name}")
            
            compute_config = AmlCompute.provisioning_configuration(
                vm_size="Standard_NC6s_v3",  # GPU for deep learning
                min_nodes=0,
                max_nodes=2,
                idle_seconds_before_scaledown=300,
                tier="Dedicated"  # For consistent performance
            )
            
            compute_target = ComputeTarget.create(
                workspace, 
                cluster_name, 
                compute_config
            )
            
            print("⏳ Waiting for cluster provisioning...")
            compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
            print(f"✅ Compute cluster created: {cluster_name}")
        
        return compute_target
        
    except Exception as e:
        print(f"❌ Compute cluster setup failed: {e}")
        return None

def create_training_environment(workspace):
    """Create Azure ML environment for GNN training"""
    
    try:
        from azureml.core import Environment
        from azureml.core.conda_dependencies import CondaDependencies
        
        env_name = "gnn-training-environment"
        
        # Check if environment exists
        try:
            environment = Environment.get(workspace, env_name)
            print(f"✅ Using existing environment: {env_name}")
            return environment
        except:
            pass
        
        # Create new environment
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
        conda_deps.add_pip_package("matplotlib>=3.5.0")
        conda_deps.add_pip_package("seaborn>=0.11.0")
        
        # Azure ML
        conda_deps.add_pip_package("azureml-sdk")
        conda_deps.add_pip_package("azureml-mlflow")
        
        environment.python.conda_dependencies = conda_deps
        
        # Enable Docker
        environment.docker.enabled = True
        environment.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04"
        
        # Register environment
        environment = environment.register(workspace)
        print(f"✅ Environment registered: {env_name}")
        
        return environment
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return None

def setup_datastore(workspace):
    """Setup Azure ML datastore for training data"""
    
    try:
        from azureml.core import Datastore
        
        # Use default datastore
        datastore = workspace.get_default_datastore()
        print(f"✅ Using datastore: {datastore.name}")
        
        return datastore
        
    except Exception as e:
        print(f"❌ Datastore setup failed: {e}")
        return None

def save_azure_config(workspace, compute_target, environment, datastore):
    """Save Azure ML configuration for training scripts"""
    
    config_dir = Path(__file__).parent.parent / "azure_ml_config"
    config_dir.mkdir(exist_ok=True)
    
    config = {
        "workspace_name": workspace.name,
        "subscription_id": workspace.subscription_id,
        "resource_group": workspace.resource_group,
        "compute_target": compute_target.name,
        "environment_name": environment.name,
        "datastore_name": datastore.name,
        "setup_timestamp": datetime.now().isoformat(),
        "setup_status": "completed"
    }
    
    config_file = config_dir / "azure_ml_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Azure ML config saved: {config_file}")
    return config_file

def main():
    """Main Azure ML setup process"""
    
    print("☁️ REAL AZURE ML SETUP")
    print("=" * 50)
    print("Setting up production Azure ML environment for GNN training")
    print("=" * 50)
    
    try:
        # Check credentials
        if not check_azure_credentials():
            print("\n💡 Setup Azure credentials first:")
            print("   1. Create Service Principal:")
            print("      az ad sp create-for-rbac --name gnn-training-sp --role contributor")
            print("   2. Export credentials as environment variables")
            print("   3. Run this script again")
            return
        
        # Install Azure ML SDK
        if not install_azure_ml_sdk():
            print("❌ Failed to install Azure ML SDK")
            return
        
        # Create workspace
        workspace = create_azure_ml_workspace()
        if not workspace:
            print("❌ Failed to create workspace")
            return
        
        # Create compute cluster
        compute_target = create_compute_cluster(workspace)
        if not compute_target:
            print("❌ Failed to create compute cluster")
            return
        
        # Create training environment
        environment = create_training_environment(workspace)
        if not environment:
            print("❌ Failed to create environment")
            return
        
        # Setup datastore
        datastore = setup_datastore(workspace)
        if not datastore:
            print("❌ Failed to setup datastore")
            return
        
        # Save configuration
        config_file = save_azure_config(workspace, compute_target, environment, datastore)
        
        print(f"\n🎉 AZURE ML SETUP COMPLETED!")
        print(f"📋 Configuration: {config_file}")
        print(f"🏭 Workspace: {workspace.name}")
        print(f"💻 Compute: {compute_target.name}")
        print(f"🐍 Environment: {environment.name}")
        
        print(f"\n🚀 READY FOR TRAINING!")
        print(f"   Next step: python scripts/real_azure_ml_gnn_training.py --partial")
        
    except Exception as e:
        print(f"❌ Azure ML setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()