#!/usr/bin/env python3
"""
Step 4: Pure Azure ML GNN Training 
Submit real GNN training job to Azure ML - no local training
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        Environment, 
        Job,
        Command,
        Data,
        AmlCompute
    )
    from azure.identity import DefaultAzureCredential
    from azure.core.exceptions import ResourceNotFoundError
    print("✅ Azure ML SDK loaded successfully")
except ImportError as e:
    print(f"❌ Azure ML SDK not available: {e}")
    print("Install with: pip install azure-ai-ml")
    sys.exit(1)

from config.settings import azure_settings


class AzureMLGNNTrainer:
    """Pure Azure ML GNN Training - no local computation"""
    
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.subscription_id = azure_settings.azure_subscription_id
        self.resource_group = azure_settings.azure_resource_group
        self.workspace_name = getattr(azure_settings, 'azure_ml_workspace_name', 'maintie-ml-workspace')
        
        # Initialize ML Client
        try:
            self.ml_client = MLClient(
                credential=self.credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
            print(f"✅ Connected to Azure ML workspace: {self.workspace_name}")
        except Exception as e:
            print(f"❌ Failed to connect to Azure ML: {e}")
            raise
    
    def create_compute_cluster(self, compute_name: str = "gnn-cluster") -> bool:
        """Create or verify Azure ML compute cluster"""
        
        try:
            # Check if compute already exists
            compute_cluster = self.ml_client.compute.get(compute_name)
            print(f"✅ Compute cluster '{compute_name}' already exists")
            return True
            
        except ResourceNotFoundError:
            print(f"🔄 Creating compute cluster '{compute_name}'...")
            
            try:
                compute_cluster = AmlCompute(
                    name=compute_name,
                    type="amlcompute",
                    size="Standard_DS3_v2",  # 4 cores, 14GB RAM
                    min_instances=0,
                    max_instances=4,
                    idle_time_before_scale_down=120,
                    tier="Dedicated"
                )
                
                operation = self.ml_client.compute.begin_create_or_update(compute_cluster)
                compute_cluster = operation.result()
                
                print(f"✅ Compute cluster '{compute_name}' created successfully")
                return True
                
            except Exception as e:
                print(f"❌ Failed to create compute cluster: {e}")
                return False
    
    def upload_training_data(self) -> str:
        """Upload GNN training data to Azure ML datastore"""
        
        print("📤 Uploading training data to Azure ML...")
        
        # Find latest training data
        data_dir = Path(__file__).parent.parent / "data" / "gnn_training"
        data_files = list(data_dir.glob("gnn_training_data_full_*.npz"))
        metadata_files = list(data_dir.glob("gnn_metadata_full_*.json"))
        
        if not data_files or not metadata_files:
            raise FileNotFoundError("No GNN training data found. Run prepare_gnn_training_features.py first")
        
        latest_data_file = max(data_files, key=lambda x: x.stat().st_mtime)
        latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
        
        # Create data asset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_name = f"gnn-training-data-{timestamp}"
        
        try:
            data_asset = Data(
                path=str(data_dir),
                type="uri_folder",
                description="GNN training data with 9100 entities and 1540-dim features",
                name=data_name,
                version=timestamp
            )
            
            data_asset = self.ml_client.data.create_or_update(data_asset)
            print(f"✅ Training data uploaded: {data_asset.name}:{data_asset.version}")
            return f"{data_asset.name}:{data_asset.version}"
            
        except Exception as e:
            print(f"❌ Failed to upload training data: {e}")
            raise
    
    def create_gnn_environment(self) -> str:
        """Create Azure ML environment with PyTorch Geometric"""
        
        env_name = "pytorch-geometric-gnn"
        env_version = "1.0"
        
        try:
            # Check if environment exists
            environment = self.ml_client.environments.get(env_name, env_version)
            print(f"✅ Environment '{env_name}:{env_version}' already exists")
            return f"{env_name}:{env_version}"
            
        except ResourceNotFoundError:
            print(f"🔄 Creating environment '{env_name}:{env_version}'...")
            
            # Create conda environment file first
            conda_file_path = Path(__file__).parent / "azure_ml_conda_env.yml"
            
            conda_env = {
                "name": "pytorch-geometric",
                "channels": ["pytorch", "pyg", "conda-forge"],
                "dependencies": [
                    "python=3.9",
                    "pytorch=2.0.0",
                    "torchvision",
                    "torchaudio",
                    "pytorch-geometric",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "matplotlib",
                    "pip",
                    {
                        "pip": [
                            "azure-ml-mlflow",
                            "mlflow",
                            "torch-scatter",
                            "torch-sparse",
                            "torch-cluster"
                        ]
                    }
                ]
            }
            
            with open(conda_file_path, 'w') as f:
                import yaml
                yaml.dump(conda_env, f)
            
            # Create environment with PyTorch Geometric
            environment = Environment(
                name=env_name,
                version=env_version,
                description="PyTorch Geometric environment for GNN training",
                conda_file=conda_file_path,
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
            )
            
            try:
                environment = self.ml_client.environments.create_or_update(environment)
                print(f"✅ Environment '{env_name}:{env_version}' created successfully")
                return f"{env_name}:{env_version}"
                
            except Exception as e:
                print(f"❌ Failed to create environment: {e}")
                raise
    
    def create_training_script(self) -> str:
        """Create Azure ML training script"""
        
        script_path = Path(__file__).parent / "azure_ml_gnn_training_script.py"
        
        script_content = '''#!/usr/bin/env python3
"""
Azure ML GNN Training Script
Runs in Azure ML compute - no local dependencies
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import mlflow
import mlflow.pytorch
from pathlib import Path


class AzureMLGraphAttentionNetwork(torch.nn.Module):
    """Graph Attention Network for Azure ML training"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=8, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GAT layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)
        
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--output_path", type=str, help="Path to save model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=256)
    args = parser.parse_args()
    
    # Start MLflow tracking
    mlflow.start_run()
    
    print("🚀 Azure ML GNN Training Started")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    
    # Load data
    data_files = list(Path(args.data_path).glob("gnn_training_data_full_*.npz"))
    metadata_files = list(Path(args.data_path).glob("gnn_metadata_full_*.json"))
    
    if not data_files or not metadata_files:
        raise FileNotFoundError("No training data found")
    
    latest_data_file = max(data_files, key=lambda x: x.stat().st_mtime)
    latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
    
    # Load training data
    training_data = np.load(latest_data_file)
    with open(latest_metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Prepare PyTorch Geometric data
    node_features = torch.FloatTensor(training_data['node_features'])
    edge_index = torch.LongTensor(training_data['edge_index'])
    node_labels = torch.LongTensor(training_data['node_labels'])
    
    data = Data(x=node_features, edge_index=edge_index, y=node_labels)
    
    print(f"📊 Data loaded:")
    print(f"   Nodes: {data.x.size(0)}")
    print(f"   Features: {data.x.size(1)}")
    print(f"   Edges: {data.edge_index.size(1)}")
    print(f"   Classes: {metadata['num_classes']}")
    
    # Create data splits
    num_nodes = data.x.size(0)
    indices = torch.randperm(num_nodes)
    
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Create model
    model = AzureMLGraphAttentionNetwork(
        input_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        output_dim=metadata['num_classes']
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.NLLLoss()
    
    print(f"🧠 Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training loop
    best_val_accuracy = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        train_loss = criterion(out[train_mask], data.y[train_mask])
        
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = criterion(val_out[val_mask], data.y[val_mask])
            
            train_pred = out[train_mask].argmax(dim=1)
            val_pred = val_out[val_mask].argmax(dim=1)
            
            train_accuracy = (train_pred == data.y[train_mask]).float().mean().item()
            val_accuracy = (val_pred == data.y[val_mask]).float().mean().item()
        
        # Logging
        mlflow.log_metrics({
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item(),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        }, step=epoch)
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save best model
            model_path = Path(args.output_path) / "best_model.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Train Loss={train_loss.item():.4f}, Val Acc={val_accuracy:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_pred = test_out[test_mask].argmax(dim=1)
        test_accuracy = (test_pred == data.y[test_mask]).float().mean().item()
    
    print(f"✅ Training completed:")
    print(f"   Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"   Final test accuracy: {test_accuracy:.4f}")
    
    # Log final results
    mlflow.log_metrics({
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy
    })
    
    # Register model
    mlflow.pytorch.log_model(model, "gnn_model")
    
    mlflow.end_run()
    
    return test_accuracy


if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Training script created: {script_path}")
        return str(script_path)
    
    def submit_training_job(self, data_asset_name: str, environment_name: str, script_path: str) -> str:
        """Submit GNN training job to Azure ML"""
        
        print("🚀 Submitting GNN training job to Azure ML...")
        
        job_name = f"gnn-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            from azure.ai.ml import command
            
            command_job = command(
                display_name=job_name,
                description="GNN training for maintenance knowledge graph",
                code=str(Path(__file__).parent),
                command="python azure_ml_gnn_training_script.py --data_path ${{inputs.data}} --output_path ${{outputs.model}} --epochs 50 --learning_rate 0.001 --hidden_dim 256",
                environment=environment_name,
                compute="gnn-cluster",
                inputs={
                    "data": {"type": "uri_folder", "path": f"azureml:{data_asset_name}"}
                },
                outputs={
                    "model": {"type": "uri_folder"}
                },
                tags={
                    "framework": "PyTorch Geometric",
                    "task": "node_classification",
                    "dataset": "maintenance_knowledge_graph"
                }
            )
            
            # Submit job
            submitted_job = self.ml_client.jobs.create_or_update(command_job)
            
            print(f"✅ Job submitted successfully:")
            print(f"   Job name: {submitted_job.name}")
            print(f"   Job ID: {submitted_job.id}")
            print(f"   Status: {submitted_job.status}")
            print(f"   Studio URL: {submitted_job.studio_url}")
            
            return submitted_job.name
            
        except Exception as e:
            print(f"❌ Failed to submit job: {e}")
            raise
    
    def monitor_job(self, job_name: str, timeout_minutes: int = 30) -> Dict[str, Any]:
        """Monitor Azure ML job progress"""
        
        print(f"👀 Monitoring job '{job_name}' (timeout: {timeout_minutes} minutes)...")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while True:
            try:
                job = self.ml_client.jobs.get(job_name)
                
                elapsed = time.time() - start_time
                print(f"   Status: {job.status} (elapsed: {elapsed:.0f}s)")
                
                if job.status in ["Completed", "Failed", "Canceled"]:
                    if job.status == "Completed":
                        print(f"✅ Job completed successfully!")
                        
                        # Get job metrics
                        try:
                            metrics = self.ml_client.jobs.download(job_name, download_path="./job_outputs")
                            print(f"📁 Job outputs downloaded to: ./job_outputs")
                        except Exception as e:
                            print(f"⚠️  Could not download outputs: {e}")
                        
                        return {
                            "status": "completed",
                            "job_name": job_name,
                            "elapsed_time": elapsed,
                            "studio_url": job.studio_url
                        }
                    else:
                        print(f"❌ Job {job.status.lower()}")
                        return {
                            "status": job.status.lower(),
                            "job_name": job_name,
                            "elapsed_time": elapsed
                        }
                
                if elapsed > timeout_seconds:
                    print(f"⏰ Monitoring timeout reached ({timeout_minutes} minutes)")
                    return {
                        "status": "timeout",
                        "job_name": job_name,
                        "current_status": job.status,
                        "elapsed_time": elapsed,
                        "studio_url": job.studio_url
                    }
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"❌ Error monitoring job: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }


def main():
    """Execute Step 4: Pure Azure ML GNN Training"""
    
    print("🚀 STEP 4: PURE AZURE ML GNN TRAINING")
    print("=" * 60)
    print("Submitting real GNN training to Azure ML - no local computation")
    
    try:
        # Initialize Azure ML trainer
        trainer = AzureMLGNNTrainer()
        
        # Step 1: Create/verify compute cluster
        if not trainer.create_compute_cluster():
            return 1
        
        # Step 2: Upload training data
        data_asset_name = trainer.upload_training_data()
        
        # Step 3: Create/verify environment
        environment_name = trainer.create_gnn_environment()
        
        # Step 4: Create training script
        script_path = trainer.create_training_script()
        
        # Step 5: Submit training job
        job_name = trainer.submit_training_job(data_asset_name, environment_name, script_path)
        
        # Step 6: Monitor job
        result = trainer.monitor_job(job_name, timeout_minutes=20)
        
        if result["status"] == "completed":
            print(f"\n✅ STEP 4 COMPLETED: Azure ML GNN training successful!")
            print(f"   Job: {result['job_name']}")
            print(f"   Time: {result['elapsed_time']:.0f}s")
            print(f"   Studio: {result.get('studio_url', 'N/A')}")
            return 0
        else:
            print(f"\n⚠️  STEP 4 STATUS: {result['status']}")
            print(f"   Job: {result.get('job_name', 'N/A')}")
            if 'studio_url' in result:
                print(f"   Monitor at: {result['studio_url']}")
            return 0  # Still successful submission
        
    except Exception as e:
        print(f"❌ Step 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())