#!/usr/bin/env python3
"""
Real Azure ML GNN Training - No Simulation
Submit actual PyTorch Geometric training job to Azure ML
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

class RealAzureMLGNNTrainer:
    """Real Azure ML GNN trainer - no simulation"""
    
    def __init__(self, use_partial: bool = False):
        self.use_partial = use_partial
        self.workspace = None
        self.experiment = None
        self.compute_target = None
        self.environment = None
        self.datastore = None
        
    def load_azure_config(self):
        """Load Azure ML configuration"""
        
        config_file = Path(__file__).parent.parent / "azure_ml_config" / "azure_ml_config.json"
        
        if not config_file.exists():
            print("❌ Azure ML not configured. Run setup_azure_ml_real.py first")
            return False
        
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        print(f"✅ Azure ML config loaded: {config_file}")
        return True
        
    def connect_to_workspace(self):
        """Connect to Azure ML workspace"""
        
        try:
            from azureml.core import Workspace
            
            config_dir = Path(__file__).parent.parent / "azure_ml_config"
            
            # Load workspace from config
            self.workspace = Workspace.from_config(path=config_dir)
            
            print(f"✅ Connected to workspace: {self.workspace.name}")
            print(f"   📍 Location: {self.workspace.location}")
            print(f"   🔗 Resource Group: {self.workspace.resource_group}")
            
            return True
            
        except Exception as e:
            print(f"❌ Workspace connection failed: {e}")
            return False
    
    def setup_experiment(self):
        """Setup Azure ML experiment"""
        
        try:
            from azureml.core import Experiment
            
            experiment_name = "gnn-knowledge-graph-training"
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
            
            compute_name = self.config["compute_target"]
            self.compute_target = ComputeTarget(workspace=self.workspace, name=compute_name)
            
            print(f"✅ Compute target: {compute_name}")
            print(f"   🖥️  VM Size: {self.compute_target.vm_size}")
            print(f"   📊 Status: {self.compute_target.get_status()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Compute target failed: {e}")
            return False
    
    def get_environment(self):
        """Get Azure ML training environment"""
        
        try:
            from azureml.core import Environment
            
            env_name = self.config["environment_name"]
            self.environment = Environment.get(workspace=self.workspace, name=env_name)
            
            print(f"✅ Environment: {env_name}")
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False
    
    def get_datastore(self):
        """Get Azure ML datastore"""
        
        try:
            self.datastore = self.workspace.get_default_datastore()
            
            print(f"✅ Datastore: {self.datastore.name}")
            return True
            
        except Exception as e:
            print(f"❌ Datastore setup failed: {e}")
            return False
    
    def create_training_script(self) -> Path:
        """Create real PyTorch Geometric training script"""
        
        script_dir = Path(__file__).parent.parent / "azure_ml_training_scripts"
        script_dir.mkdir(exist_ok=True)
        
        script_file = script_dir / "real_gnn_training.py"
        
        script_content = '''#!/usr/bin/env python3
"""
Real PyTorch Geometric GNN Training on Azure ML
No simulation - actual training with GPU acceleration
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import os
import logging
from azureml.core import Run
import mlflow
import mlflow.pytorch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealGraphAttentionNet(torch.nn.Module):
    """Real Graph Attention Network - Production Implementation"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=8, dropout=0.2):
        super(RealGraphAttentionNet, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers with batch normalization
        self.gat_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Input layer
        self.gat_layers.append(
            GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        )
        self.batch_norms.append(BatchNorm(hidden_dim * heads))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
            )
            self.batch_norms.append(BatchNorm(hidden_dim * heads))
        
        # Output layer
        self.gat_layers.append(
            GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout, concat=False)
        )
        
        # Residual connections
        self.residual_layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.residual_layers.append(torch.nn.Linear(input_dim, hidden_dim * heads))
            else:
                self.residual_layers.append(torch.nn.Linear(hidden_dim * heads, hidden_dim * heads))
        
    def forward(self, x, edge_index, batch=None):
        # Store input for residual connections
        residual = x
        
        # Apply GAT layers with residual connections
        for i, (gat_layer, batch_norm) in enumerate(zip(self.gat_layers[:-1], self.batch_norms)):
            x = gat_layer(x, edge_index)
            
            # Batch normalization
            x = batch_norm(x)
            
            # Residual connection
            if i < len(self.residual_layers):
                residual = self.residual_layers[i](residual)
                x = x + residual
                residual = x
            
            # Activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.gat_layers[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)

def load_training_data(data_path):
    """Load context-aware training data"""
    
    logger.info(f"Loading training data from: {data_path}")
    
    # Load numpy data
    data = np.load(data_path)
    
    node_features = torch.FloatTensor(data['node_features'])
    edge_index = torch.LongTensor(data['edge_index'])
    node_labels = torch.LongTensor(data['node_labels'])
    
    logger.info(f"Data loaded:")
    logger.info(f"  • Nodes: {node_features.shape[0]}")
    logger.info(f"  • Features: {node_features.shape[1]}")
    logger.info(f"  • Edges: {edge_index.shape[1]}")
    logger.info(f"  • Classes: {len(torch.unique(node_labels))}")
    
    return node_features, edge_index, node_labels

def create_data_splits(node_features, node_labels, test_size=0.2, val_size=0.1):
    """Create stratified train/val/test splits"""
    
    num_nodes = node_features.shape[0]
    indices = np.arange(num_nodes)
    
    # Ensure we have enough samples per class
    unique_labels, counts = np.unique(node_labels, return_counts=True)
    min_samples = min(counts)
    
    if min_samples < 3:
        logger.warning(f"Some classes have very few samples (min: {min_samples})")
    
    # Train/test split
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42, 
        stratify=node_labels if min_samples >= 2 else None
    )
    
    # Train/val split
    if len(train_idx) > 10:  # Only create val set if we have enough data
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size/(1-test_size), random_state=42,
            stratify=node_labels[train_idx] if min_samples >= 3 else None
        )
    else:
        val_idx = train_idx[:len(train_idx)//2]
        train_idx = train_idx[len(train_idx)//2:]
    
    logger.info(f"Data splits:")
    logger.info(f"  • Training: {len(train_idx)} nodes")
    logger.info(f"  • Validation: {len(val_idx)} nodes")
    logger.info(f"  • Test: {len(test_idx)} nodes")
    
    return train_idx, val_idx, test_idx

def train_real_model(model, data, train_idx, val_idx, optimizer, scheduler, device, epochs=200):
    """Real model training with MLflow tracking"""
    
    node_features, edge_index, node_labels = data
    
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    
    # Get Azure ML run context
    run = Run.get_context()
    
    # Start MLflow run
    mlflow.start_run()
    
    logger.info("Starting real GNN training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(node_features, edge_index)
        
        # Compute loss on training nodes
        loss = F.nll_loss(out[train_idx], node_labels[train_idx])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        # Validation phase
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(node_features, edge_index)
                val_pred = val_out[val_idx].argmax(dim=1)
                val_acc = accuracy_score(node_labels[val_idx].cpu(), val_pred.cpu())
                val_f1 = f1_score(node_labels[val_idx].cpu(), val_pred.cpu(), average='weighted')
            
            # Log metrics to Azure ML and MLflow
            run.log("epoch", epoch)
            run.log("train_loss", float(loss))
            run.log("val_accuracy", val_acc)
            run.log("val_f1_score", val_f1)
            run.log("learning_rate", optimizer.param_groups[0]['lr'])
            
            mlflow.log_metric("train_loss", float(loss), step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_f1_score", val_f1, step=epoch)
            
            logger.info(f"Epoch {epoch:3d}: Loss={loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'val_f1_score': val_f1
                }, "outputs/best_gnn_model.pt")
                
                logger.info(f"New best model saved! Val Acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break
            
            model.train()
        
        # Learning rate scheduling
        scheduler.step()
    
    mlflow.end_run()
    return best_val_acc

def evaluate_real_model(model, data, test_idx):
    """Real model evaluation with comprehensive metrics"""
    
    node_features, edge_index, node_labels = data
    
    # Load best model
    checkpoint = torch.load("outputs/best_gnn_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    with torch.no_grad():
        out = model(node_features, edge_index)
        test_pred = out[test_idx].argmax(dim=1)
        
        test_acc = accuracy_score(node_labels[test_idx].cpu(), test_pred.cpu())
        test_f1 = f1_score(node_labels[test_idx].cpu(), test_pred.cpu(), average='weighted')
        
        # Detailed classification report
        report = classification_report(
            node_labels[test_idx].cpu(), 
            test_pred.cpu(), 
            output_dict=True,
            zero_division=0
        )
    
    logger.info(f"Final test results:")
    logger.info(f"  • Test Accuracy: {test_acc:.4f}")
    logger.info(f"  • Test F1 Score: {test_f1:.4f}")
    
    return test_acc, test_f1, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Setup device - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load context-aware training data
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
    
    # Create real model
    input_dim = node_features.shape[1]
    output_dim = len(torch.unique(node_labels))
    
    model = RealGraphAttentionNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Real GNN model created:")
    logger.info(f"  • Input dim: {input_dim}")
    logger.info(f"  • Hidden dim: {args.hidden_dim}")
    logger.info(f"  • Output classes: {output_dim}")
    logger.info(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Train real model
    data = (node_features, edge_index, node_labels)
    best_val_acc = train_real_model(model, data, train_idx, val_idx, optimizer, scheduler, device, args.epochs)
    
    # Evaluate real model
    test_acc, test_f1, report = evaluate_real_model(model, data, test_idx)
    
    # Log final results to Azure ML
    run = Run.get_context()
    run.log("final_test_accuracy", test_acc)
    run.log("final_test_f1", test_f1)
    run.log("best_val_accuracy", best_val_acc)
    
    # Save comprehensive results
    results = {
        "model_info": {
            "model_type": "RealGraphAttentionNetwork",
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": output_dim,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device)
        },
        "training_results": {
            "best_val_accuracy": float(best_val_acc),
            "final_test_accuracy": float(test_acc),
            "final_test_f1": float(test_f1),
            "epochs_trained": args.epochs
        },
        "data_info": {
            "total_nodes": int(node_features.shape[0]),
            "feature_dimension": int(node_features.shape[1]),
            "num_edges": int(edge_index.shape[1]),
            "num_classes": int(output_dim),
            "train_nodes": int(len(train_idx)),
            "val_nodes": int(len(val_idx)),
            "test_nodes": int(len(test_idx))
        },
        "classification_report": report
    }
    
    with open("outputs/real_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Real GNN training completed successfully!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Final test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
'''
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Real training script created: {script_file}")
        return script_file
    
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
            
            # Create data reference for training script
            data_reference = f"./gnn_training_data/{latest_data_file.name}"
            
            print(f"✅ Training data uploaded successfully")
            return data_reference
            
        except Exception as e:
            print(f"❌ Data upload failed: {e}")
            return None
    
    def submit_real_training_job(self, training_script: Path, data_reference: str):
        """Submit real training job to Azure ML with no simulation"""
        
        try:
            from azureml.core import ScriptRunConfig
            
            print("🚀 Submitting REAL Azure ML training job...")
            
            # Create script run configuration
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
                    "--epochs", "200",
                    "--weight_decay", "1e-4"
                ],
                compute_target=self.compute_target,
                environment=self.environment
            )
            
            # Submit real training job
            run = self.experiment.submit(script_config)
            
            print(f"✅ REAL training job submitted!")
            print(f"📋 Run ID: {run.id}")
            print(f"🔗 Azure ML Studio: {run.get_portal_url()}")
            print(f"💻 Compute: {self.compute_target.name} ({self.compute_target.vm_size})")
            print(f"🐍 Environment: {self.environment.name}")
            
            return run
            
        except Exception as e:
            print(f"❌ Real job submission failed: {e}")
            return None
    
    def monitor_training_job(self, run, wait_for_completion=True):
        """Monitor real Azure ML training job"""
        
        try:
            print(f"\n📊 MONITORING REAL TRAINING JOB")
            print("=" * 50)
            print(f"Run ID: {run.id}")
            print(f"Status: {run.get_status()}")
            print(f"Portal: {run.get_portal_url()}")
            
            if wait_for_completion:
                print(f"\n⏳ Waiting for real training completion...")
                print("   (This will take several minutes with real GPU training)")
                
                run.wait_for_completion(show_output=True)
                
                # Get real training results
                print(f"\n📈 REAL TRAINING RESULTS:")
                print("=" * 30)
                
                metrics = run.get_metrics()
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"   • {metric_name}: {value:.4f}")
                
                # Download real model and results
                output_dir = Path(__file__).parent.parent / "azure_ml_outputs"
                output_dir.mkdir(exist_ok=True)
                
                run.download_files(prefix="outputs/", output_directory=str(output_dir))
                
                print(f"\n💾 Real model downloaded to: {output_dir}")
                
                # Load and display detailed results
                results_file = output_dir / "outputs" / "real_training_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    print(f"\n🎯 FINAL RESULTS:")
                    print(f"   • Test Accuracy: {results['training_results']['final_test_accuracy']:.4f}")
                    print(f"   • Test F1 Score: {results['training_results']['final_test_f1']:.4f}")
                    print(f"   • Best Val Accuracy: {results['training_results']['best_val_accuracy']:.4f}")
                    print(f"   • Model Parameters: {results['model_info']['num_parameters']:,}")
                    
                return results
            else:
                print(f"\n💡 Monitor training progress:")
                print(f"   Azure ML Studio: {run.get_portal_url()}")
                print(f"   Or run with --wait flag")
                
                return None
                
        except Exception as e:
            print(f"❌ Training monitoring failed: {e}")
            return None

def main():
    """Main real Azure ML training process"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true", help="Train with partial data")
    parser.add_argument("--wait", action="store_true", help="Wait for training completion")
    args = parser.parse_args()
    
    print("☁️ REAL AZURE ML GNN TRAINING")
    print("=" * 50)
    print("🚀 NO SIMULATION - REAL CLOUD TRAINING")
    print(f"📊 Data: {'Partial' if args.partial else 'Full'} dataset")
    print(f"💻 Compute: Azure ML GPU cluster")
    print(f"🔬 Framework: PyTorch Geometric")
    print("=" * 50)
    
    try:
        trainer = RealAzureMLGNNTrainer(use_partial=args.partial)
        
        # Load Azure configuration
        if not trainer.load_azure_config():
            print("❌ Run setup_azure_ml_real.py first")
            return
        
        # Connect to workspace
        if not trainer.connect_to_workspace():
            print("❌ Failed to connect to Azure ML workspace")
            return
        
        # Setup experiment
        if not trainer.setup_experiment():
            print("❌ Failed to setup experiment")
            return
        
        # Get compute target
        if not trainer.get_compute_target():
            print("❌ Failed to get compute target")
            return
        
        # Get environment
        if not trainer.get_environment():
            print("❌ Failed to get environment")
            return
        
        # Get datastore
        if not trainer.get_datastore():
            print("❌ Failed to get datastore")
            return
        
        # Create real training script
        training_script = trainer.create_training_script()
        
        # Upload training data
        data_reference = trainer.upload_training_data()
        if not data_reference:
            print("❌ Failed to upload training data")
            return
        
        # Submit real training job
        run = trainer.submit_real_training_job(training_script, data_reference)
        if not run:
            print("❌ Failed to submit training job")
            return
        
        # Monitor training
        results = trainer.monitor_training_job(run, wait_for_completion=args.wait)
        
        if results:
            print(f"\n🎉 REAL AZURE ML TRAINING COMPLETED!")
            print(f"🏆 This is a PRODUCTION-READY model trained on Azure ML!")
        else:
            print(f"\n🚀 Real training job is running on Azure ML")
            print(f"📊 Monitor at: {run.get_portal_url()}")
        
    except Exception as e:
        print(f"❌ Real Azure ML training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()