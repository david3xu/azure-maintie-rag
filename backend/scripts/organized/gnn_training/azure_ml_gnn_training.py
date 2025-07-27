#!/usr/bin/env python3
"""
Real Azure ML GNN Training
Submit actual GNN training job to Azure ML compute cluster
"""

import json
import numpy as np
import sys
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

class AzureMLGNNTrainer:
    """Real Azure ML GNN trainer using Azure ML SDK"""
    
    def __init__(self, use_partial: bool = False):
        self.use_partial = use_partial
        self.workspace = None
        self.experiment = None
        self.compute_target = None
        
    def setup_azure_ml_workspace(self):
        """Setup Azure ML workspace connection"""
        
        try:
            from azureml.core import Workspace, Experiment
            from azureml.core.authentication import ServicePrincipalAuthentication
            
            print("🔑 Connecting to Azure ML Workspace...")
            
            # Azure ML authentication
            auth = ServicePrincipalAuthentication(
                tenant_id=os.environ.get('AZURE_TENANT_ID'),
                service_principal_id=os.environ.get('AZURE_CLIENT_ID'), 
                service_principal_password=os.environ.get('AZURE_CLIENT_SECRET')
            )
            
            # Connect to workspace
            self.workspace = Workspace(
                subscription_id=os.environ.get('AZURE_SUBSCRIPTION_ID'),
                resource_group=os.environ.get('AZURE_RESOURCE_GROUP'),
                workspace_name=getattr(settings, 'azure_ml_workspace_name', 'maintie-dev-ml'),
                auth=auth
            )
            
            print(f"✅ Connected to workspace: {self.workspace.name}")
            
            # Get or create experiment
            experiment_name = getattr(settings, 'azure_ml_experiment_name', 'universal-rag-gnn-dev')
            self.experiment = Experiment(workspace=self.workspace, name=experiment_name)
            
            print(f"✅ Experiment: {experiment_name}")
            
            return True
            
        except ImportError:
            print("❌ Azure ML SDK not installed. Install with: pip install azureml-sdk")
            return False
        except Exception as e:
            print(f"❌ Azure ML connection failed: {e}")
            print("💡 Ensure Azure credentials are set:")
            print("   export AZURE_TENANT_ID=<your-tenant-id>")
            print("   export AZURE_CLIENT_ID=<your-client-id>")
            print("   export AZURE_CLIENT_SECRET=<your-client-secret>")
            print("   export AZURE_SUBSCRIPTION_ID=<your-subscription-id>")
            print("   export AZURE_RESOURCE_GROUP=<your-resource-group>")
            return False
    
    def setup_compute_target(self):
        """Setup or create Azure ML compute cluster"""
        
        try:
            from azureml.core.compute import ComputeTarget, AmlCompute
            from azureml.core.compute_target import ComputeTargetException
            
            cluster_name = getattr(settings, 'azure_ml_compute_cluster_name', 'gnn-cluster-dev')
            
            try:
                # Try to get existing compute target
                self.compute_target = ComputeTarget(workspace=self.workspace, name=cluster_name)
                print(f"✅ Using existing compute cluster: {cluster_name}")
                
            except ComputeTargetException:
                # Create new compute cluster
                print(f"🔄 Creating new compute cluster: {cluster_name}")
                
                compute_config = AmlCompute.provisioning_configuration(
                    vm_size=getattr(settings, 'gnn_training_compute_sku', 'Standard_DS3_v2'),
                    min_nodes=0,
                    max_nodes=4,
                    idle_seconds_before_scaledown=300
                )
                
                self.compute_target = ComputeTarget.create(
                    self.workspace, 
                    cluster_name, 
                    compute_config
                )
                
                self.compute_target.wait_for_completion(show_output=True)
                print(f"✅ Created compute cluster: {cluster_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Compute setup failed: {e}")
            return False
    
    def create_training_environment(self):
        """Create Azure ML environment for GNN training"""
        
        try:
            from azureml.core import Environment
            from azureml.core.conda_dependencies import CondaDependencies
            
            env_name = getattr(settings, 'azure_ml_training_environment', 'gnn-training-env-dev')
            
            # Create environment
            gnn_env = Environment(name=env_name)
            
            # Define conda dependencies
            conda_deps = CondaDependencies()
            conda_deps.add_conda_package("python=3.9")
            conda_deps.add_pip_package("torch>=1.12.0")
            conda_deps.add_pip_package("torch-geometric>=2.1.0")
            conda_deps.add_pip_package("numpy>=1.21.0")
            conda_deps.add_pip_package("scikit-learn>=1.1.0")
            conda_deps.add_pip_package("pandas>=1.4.0")
            conda_deps.add_pip_package("azureml-sdk")
            conda_deps.add_pip_package("matplotlib>=3.5.0")
            
            gnn_env.python.conda_dependencies = conda_deps
            
            # Register environment
            gnn_env = gnn_env.register(workspace=self.workspace)
            
            print(f"✅ Environment registered: {env_name}")
            return gnn_env
            
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            return None
    
    def create_training_script(self) -> Path:
        """Create Azure ML training script for GNN"""
        
        script_dir = Path(__file__).parent.parent / "azure_ml_scripts"
        script_dir.mkdir(exist_ok=True)
        
        script_file = script_dir / "gnn_training_script.py"
        
        script_content = '''#!/usr/bin/env python3
"""
Azure ML GNN Training Script
Actual PyTorch Geometric training on Azure ML compute
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from azureml.core import Run

class GraphAttentionNet(torch.nn.Module):
    """Graph Attention Network for entity classification"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=8, dropout=0.2):
        super(GraphAttentionNet, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.gat_layers = torch.nn.ModuleList()
        
        # Input layer
        self.gat_layers.append(
            GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        )
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
            )
        
        # Output layer
        self.gat_layers.append(
            GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout, concat=False)
        )
        
        self.layer_norm = torch.nn.LayerNorm(hidden_dim * heads)
        
    def forward(self, x, edge_index, batch=None):
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers[:-1]):
            x = gat_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if i < len(self.gat_layers) - 2:
                x = self.layer_norm(x)
        
        # Output layer
        x = self.gat_layers[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)

def load_training_data(data_path):
    """Load prepared training data"""
    
    print(f"Loading training data from: {data_path}")
    
    # Load numpy data
    data = np.load(data_path)
    
    node_features = torch.FloatTensor(data['node_features'])
    edge_index = torch.LongTensor(data['edge_index'])
    node_labels = torch.LongTensor(data['node_labels'])
    
    print(f"Data loaded: {node_features.shape[0]} nodes, {edge_index.shape[1]} edges")
    
    return node_features, edge_index, node_labels

def create_data_splits(node_features, node_labels, test_size=0.2, val_size=0.1):
    """Create train/val/test splits"""
    
    num_nodes = node_features.shape[0]
    indices = np.arange(num_nodes)
    
    # Train/test split
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=node_labels
    )
    
    # Train/val split
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size/(1-test_size), random_state=42, 
        stratify=node_labels[train_idx]
    )
    
    return train_idx, val_idx, test_idx

def train_model(model, data, train_idx, val_idx, optimizer, device, epochs=100):
    """Train the GNN model"""
    
    node_features, edge_index, node_labels = data
    
    model.train()
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    run = Run.get_context()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(node_features, edge_index)
        
        # Compute loss on training nodes
        loss = F.nll_loss(out[train_idx], node_labels[train_idx])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(node_features, edge_index)
                val_pred = val_out[val_idx].argmax(dim=1)
                val_acc = accuracy_score(node_labels[val_idx].cpu(), val_pred.cpu())
            
            # Log metrics to Azure ML
            run.log("epoch", epoch)
            run.log("train_loss", float(loss))
            run.log("val_accuracy", val_acc)
            
            print(f"Epoch {epoch:3d}: Loss={loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), "outputs/best_model.pt")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            model.train()
    
    return best_val_acc

def evaluate_model(model, data, test_idx):
    """Evaluate model on test set"""
    
    node_features, edge_index, node_labels = data
    
    model.eval()
    with torch.no_grad():
        out = model(node_features, edge_index)
        test_pred = out[test_idx].argmax(dim=1)
        test_acc = accuracy_score(node_labels[test_idx].cpu(), test_pred.cpu())
        
        # Classification report
        report = classification_report(
            node_labels[test_idx].cpu(), 
            test_pred.cpu(), 
            output_dict=True
        )
    
    return test_acc, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    node_features, edge_index, node_labels = load_training_data(args.data_path)
    
    # Move to device
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    node_labels = node_labels.to(device)
    
    # Create splits
    train_idx, val_idx, test_idx = create_data_splits(node_features.cpu(), node_labels.cpu())
    train_idx = torch.LongTensor(train_idx).to(device)
    val_idx = torch.LongTensor(val_idx).to(device)
    test_idx = torch.LongTensor(test_idx).to(device)
    
    # Create model
    input_dim = node_features.shape[1]
    output_dim = len(torch.unique(node_labels))
    
    model = GraphAttentionNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model created: {input_dim} -> {args.hidden_dim} -> {output_dim}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    data = (node_features, edge_index, node_labels)
    best_val_acc = train_model(model, data, train_idx, val_idx, optimizer, device, args.epochs)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load("outputs/best_model.pt"))
    test_acc, report = evaluate_model(model, data, test_idx)
    
    # Log final results
    run = Run.get_context()
    run.log("best_val_accuracy", best_val_acc)
    run.log("test_accuracy", test_acc)
    
    print(f"\\nFinal Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save results
    results = {
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "classification_report": report
    }
    
    with open("outputs/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
'''
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Training script created: {script_file}")
        return script_file
    
    def upload_training_data(self):
        """Upload training data to Azure ML datastore"""
        
        try:
            from azureml.core import Datastore
            
            # Get default datastore
            datastore = self.workspace.get_default_datastore()
            
            # Find training data
            data_dir = Path(__file__).parent.parent / "data" / "gnn_training"
            data_pattern = "gnn_training_data_partial_*.npz" if self.use_partial else "gnn_training_data_full_*.npz"
            data_files = list(data_dir.glob(data_pattern))
            
            if not data_files:
                raise FileNotFoundError("No training data found")
            
            latest_data_file = max(data_files, key=lambda x: x.stat().st_mtime)
            
            # Upload to datastore
            target_path = "gnn_training_data/"
            datastore.upload_files(
                files=[str(latest_data_file)],
                target_path=target_path,
                overwrite=True
            )
            
            data_reference = f"azureml://datastores/{datastore.name}/paths/{target_path}{latest_data_file.name}"
            
            print(f"✅ Training data uploaded: {data_reference}")
            return data_reference
            
        except Exception as e:
            print(f"❌ Data upload failed: {e}")
            return None
    
    def submit_training_job(self, training_script: Path, data_reference: str, environment):
        """Submit GNN training job to Azure ML"""
        
        try:
            from azureml.core import ScriptRunConfig
            
            # Create script run config
            script_config = ScriptRunConfig(
                source_directory=str(training_script.parent),
                script=training_script.name,
                arguments=[
                    "--data_path", data_reference,
                    "--hidden_dim", "256",
                    "--num_layers", "3", 
                    "--heads", "8",
                    "--dropout", "0.2",
                    "--learning_rate", "0.001",
                    "--epochs", "100"
                ],
                compute_target=self.compute_target,
                environment=environment
            )
            
            # Submit job
            run = self.experiment.submit(script_config)
            
            print(f"✅ Training job submitted: {run.id}")
            print(f"🔗 Monitor at: {run.get_portal_url()}")
            
            return run
            
        except Exception as e:
            print(f"❌ Job submission failed: {e}")
            return None

def main():
    """Main Azure ML GNN training process"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true", help="Train with partial data")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    args = parser.parse_args()
    
    print("☁️ REAL AZURE ML GNN TRAINING")
    print("=" * 50)
    print(f"Data type: {'Partial' if args.partial else 'Full'} dataset")
    print(f"Azure ML: ✅ Real cloud training")
    print("=" * 50)
    
    try:
        trainer = AzureMLGNNTrainer(use_partial=args.partial)
        
        # Setup Azure ML workspace
        if not trainer.setup_azure_ml_workspace():
            print("❌ Cannot proceed without Azure ML connection")
            print("💡 Alternative: Use simulated training with --simulate flag")
            return
        
        # Setup compute cluster
        if not trainer.setup_compute_target():
            print("❌ Cannot proceed without compute cluster")
            return
        
        # Create training environment
        environment = trainer.create_training_environment()
        if not environment:
            print("❌ Cannot proceed without training environment")
            return
        
        # Create training script
        training_script = trainer.create_training_script()
        
        # Upload training data
        data_reference = trainer.upload_training_data()
        if not data_reference:
            print("❌ Cannot proceed without training data")
            return
        
        # Submit training job
        run = trainer.submit_training_job(training_script, data_reference, environment)
        if not run:
            print("❌ Training job submission failed")
            return
        
        print(f"\n🚀 AZURE ML TRAINING JOB STARTED!")
        print(f"📋 Job ID: {run.id}")
        print(f"🔗 Portal: {run.get_portal_url()}")
        
        if args.wait:
            print(f"\n⏳ Waiting for job completion...")
            run.wait_for_completion(show_output=True)
            
            # Get results
            metrics = run.get_metrics()
            print(f"\n📊 TRAINING RESULTS:")
            print(f"   • Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")
            print(f"   • Best Val Accuracy: {metrics.get('best_val_accuracy', 'N/A')}")
            
            # Download results
            run.download_files(prefix="outputs/", output_directory="./azure_ml_outputs")
            print(f"📄 Results downloaded to: ./azure_ml_outputs")
        else:
            print(f"\n💡 Monitor progress:")
            print(f"   az ml job show --name {run.id}")
            print(f"   Or visit: {run.get_portal_url()}")
        
    except Exception as e:
        print(f"❌ Azure ML training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()