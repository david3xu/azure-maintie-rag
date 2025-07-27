#!/usr/bin/env python3
"""
Azure ML GNN Training with Interactive Authentication
No Service Principal required - uses your Azure CLI login
"""

import json
import numpy as np
import sys
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

class InteractiveAzureMLGNNTrainer:
    """Azure ML GNN trainer using interactive authentication"""
    
    def __init__(self, use_partial: bool = False):
        self.use_partial = use_partial
        self.workspace = None
        self.experiment = None
        self.compute_target = None
        self.environment = None
        self.datastore = None
        
    def connect_to_workspace(self):
        """Connect to Azure ML workspace using interactive auth"""
        
        try:
            from azure.identity import DefaultAzureCredential
            from azureml.core import Workspace
            
            print("🔗 Connecting to Azure ML workspace...")
            print("✅ Using your Azure CLI login (interactive auth)")
            
            # Use DefaultAzureCredential (your az login session)
            credential = DefaultAzureCredential()
            
            # Load workspace from config or connect directly
            config_dir = Path(__file__).parent.parent / "azure_ml_config"
            
            if (config_dir / "config.json").exists():
                self.workspace = Workspace.from_config(path=config_dir, auth=credential)
            else:
                self.workspace = Workspace.get(
                    name=os.environ['AZURE_ML_WORKSPACE_NAME'],
                    subscription_id=os.environ['AZURE_SUBSCRIPTION_ID'],
                    resource_group=os.environ['AZURE_RESOURCE_GROUP'],
                    auth=credential
                )
            
            print(f"✅ Connected to workspace: {self.workspace.name}")
            print(f"   📍 Location: {self.workspace.location}")
            print(f"   🔗 Resource Group: {self.workspace.resource_group}")
            
            return True
            
        except Exception as e:
            print(f"❌ Workspace connection failed: {e}")
            print(f"💡 Make sure you're logged in: az login")
            return False
    
    def setup_experiment(self):
        """Setup Azure ML experiment"""
        
        try:
            from azureml.core import Experiment
            
            experiment_name = "gnn-knowledge-graph-training-interactive"
            self.experiment = Experiment(workspace=self.workspace, name=experiment_name)
            
            print(f"✅ Experiment: {experiment_name}")
            return True
            
        except Exception as e:
            print(f"❌ Experiment setup failed: {e}")
            return False
    
    def get_compute_target(self):
        """Get Azure ML compute target"""
        
        try:
            from azureml.core.compute import ComputeTarget
            
            # Try to get existing compute target
            compute_targets = self.workspace.compute_targets
            
            # Look for GPU compute targets
            gpu_targets = [name for name, target in compute_targets.items() 
                          if 'nc' in target.vm_size.lower() or 'gpu' in name.lower()]
            
            if gpu_targets:
                compute_name = gpu_targets[0]
                self.compute_target = ComputeTarget(workspace=self.workspace, name=compute_name)
                print(f"✅ Using GPU compute: {compute_name}")
            else:
                # Fallback to any available compute
                if compute_targets:
                    compute_name = list(compute_targets.keys())[0]
                    self.compute_target = ComputeTarget(workspace=self.workspace, name=compute_name)
                    print(f"✅ Using compute: {compute_name}")
                else:
                    print("⚠️  No compute targets found - training will use local compute")
                    return True  # Continue without compute target
            
            print(f"   🖥️  VM Size: {self.compute_target.vm_size}")
            print(f"   📊 Status: {self.compute_target.get_status()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Compute target failed: {e}")
            print("⚠️  Continuing without remote compute target")
            return True  # Continue anyway
    
    def get_environment(self):
        """Get Azure ML training environment"""
        
        try:
            from azureml.core import Environment
            
            # Try to get existing environment
            environments = Environment.list(workspace=self.workspace)
            gnn_envs = [name for name in environments.keys() if 'gnn' in name.lower() or 'torch' in name.lower()]
            
            if gnn_envs:
                env_name = gnn_envs[0]
                self.environment = Environment.get(workspace=self.workspace, name=env_name)
                print(f"✅ Using environment: {env_name}")
            else:
                # Use default environment
                self.environment = Environment.get(workspace=self.workspace, name="AzureML-pytorch-1.12-ubuntu20.04-py38-cuda11.6-gpu")
                print(f"✅ Using default PyTorch environment")
            
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            print("⚠️  Will use default environment")
            return True
    
    def get_datastore(self):
        """Get Azure ML datastore"""
        
        try:
            self.datastore = self.workspace.get_default_datastore()
            print(f"✅ Datastore: {self.datastore.name}")
            return True
            
        except Exception as e:
            print(f"❌ Datastore setup failed: {e}")
            return False
    
    def upload_training_data(self):
        """Upload training data to Azure ML datastore"""
        
        try:
            print("📤 Uploading training data to Azure ML...")
            
            # Find training data
            data_dir = Path(__file__).parent.parent / "data" / "gnn_training"
            data_pattern = "gnn_training_data_partial_*.npz" if self.use_partial else "gnn_training_data_full_*.npz"
            data_files = list(data_dir.glob(data_pattern))
            
            if not data_files:
                raise FileNotFoundError(f"No training data found. Run prepare_gnn_training_features.py first")
            
            latest_data_file = max(data_files, key=lambda x: x.stat().st_mtime)
            print(f"   📄 Uploading: {latest_data_file}")
            
            # Upload to datastore
            target_path = "gnn_training_data/"
            self.datastore.upload_files(
                files=[str(latest_data_file)],
                target_path=target_path,
                overwrite=True,
                show_progress=True
            )
            
            data_reference = f"./gnn_training_data/{latest_data_file.name}"
            print(f"✅ Training data uploaded successfully")
            return data_reference
            
        except Exception as e:
            print(f"❌ Data upload failed: {e}")
            return None
    
    def submit_training_job(self, data_reference: str):
        """Submit training job to Azure ML"""
        
        try:
            from azureml.core import ScriptRunConfig
            
            print("🚀 Submitting Azure ML training job...")
            
            # Create training script path
            script_dir = Path(__file__).parent.parent / "azure_ml_training_scripts"
            script_dir.mkdir(exist_ok=True)
            
            # Copy the real training script
            training_script = script_dir / "gnn_training.py"
            if not training_script.exists():
                # Create a simplified training script
                with open(training_script, 'w') as f:
                    f.write(self._get_training_script_content())
            
            # Create script run configuration
            script_config = ScriptRunConfig(
                source_directory=str(script_dir),
                script="gnn_training.py",
                arguments=[
                    "--data_path", data_reference,
                    "--epochs", "50",
                    "--hidden_dim", "256"
                ],
                compute_target=self.compute_target,
                environment=self.environment
            )
            
            # Submit training job
            run = self.experiment.submit(script_config)
            
            print(f"✅ Training job submitted!")
            print(f"📋 Run ID: {run.id}")
            print(f"🔗 Azure ML Studio: {run.get_portal_url()}")
            
            return run
            
        except Exception as e:
            print(f"❌ Job submission failed: {e}")
            return None
    
    def _get_training_script_content(self):
        """Get simplified training script content"""
        
        return '''import argparse
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from azureml.core import Run

class SimpleGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=256)
    args = parser.parse_args()
    
    # Get Azure ML run context
    run = Run.get_context()
    
    print(f"Loading data from: {args.data_path}")
    data = np.load(args.data_path)
    
    node_features = torch.FloatTensor(data['node_features'])
    edge_index = torch.LongTensor(data['edge_index'])
    node_labels = torch.LongTensor(data['node_labels'])
    
    input_dim = node_features.shape[1]
    output_dim = len(torch.unique(node_labels))
    
    model = SimpleGAT(input_dim, args.hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Training model for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(node_features, edge_index)
        loss = F.nll_loss(out, node_labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            run.log("loss", loss.item())
            run.log("epoch", epoch)
    
    # Save model
    torch.save(model.state_dict(), "outputs/model.pt")
    
    print("Training completed!")
    run.log("final_loss", loss.item())

if __name__ == "__main__":
    main()
'''

def main():
    """Main training process with interactive auth"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true", help="Use partial dataset")
    parser.add_argument("--wait", action="store_true", help="Wait for completion")
    args = parser.parse_args()
    
    print("☁️ AZURE ML GNN TRAINING (INTERACTIVE)")
    print("=" * 50)
    print("🎯 Using your Azure CLI login - no Service Principal needed")
    print(f"📊 Data: {'Partial' if args.partial else 'Full'} dataset")
    print("=" * 50)
    
    try:
        trainer = InteractiveAzureMLGNNTrainer(use_partial=args.partial)
        
        # Connect to workspace
        if not trainer.connect_to_workspace():
            return
        
        # Setup experiment
        if not trainer.setup_experiment():
            return
        
        # Get compute target (optional)
        trainer.get_compute_target()
        
        # Get environment (optional)
        trainer.get_environment()
        
        # Get datastore
        if not trainer.get_datastore():
            return
        
        # Upload training data
        data_reference = trainer.upload_training_data()
        if not data_reference:
            return
        
        # Submit training job
        run = trainer.submit_training_job(data_reference)
        if not run:
            return
        
        if args.wait:
            print(f"\n⏳ Waiting for training completion...")
            run.wait_for_completion(show_output=True)
            
            print(f"\n📈 Training Results:")
            metrics = run.get_metrics()
            for metric, value in metrics.items():
                print(f"   • {metric}: {value}")
        else:
            print(f"\n🚀 Training job running on Azure ML")
            print(f"📊 Monitor at: {run.get_portal_url()}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()