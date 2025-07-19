# GNN Implementation Plan with Azure ML

## 🎯 **Plan Overview**

Based on the Azure ML tutorial structure, here's how to implement the missing GNN training:

## 📁 **File Structure Plan:**

```
backend/
├── src/
│   ├── gnn/
│   │   ├── model.py              # GNN model architecture
│   │   ├── train.py              # GNN training script
│   │   ├── data_loader.py        # Graph data loading
│   │   └── utils.py              # GNN utilities
├── scripts/
│   ├── train_comprehensive_gnn.py # Azure ML control script
│   ├── example_comprehensive_gnn_config.json
│   └── README_comprehensive_gnn.md
└── gnn-env.yml                   # Conda environment for GNN
```

## 🔧 **Implementation Steps:**

### **Step 1: Create GNN Model Architecture**

**File: `backend/src/gnn/model.py`**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv

class UniversalGNN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=128):
        super(UniversalGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)

        # Global pooling and classification
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Global mean pooling
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
```

### **Step 2: Create GNN Training Script**

**File: `backend/src/gnn/train.py`**
```python
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from azureml.core import Run
from model import UniversalGNN
from data_loader import load_graph_data

# Get Azure ML run context
run = Run.get_context()

def train_gnn():
    # Load graph data from Cosmos DB
    train_loader, val_loader = load_graph_data()

    # Initialize model
    model = UniversalGNN(
        num_node_features=train_loader.dataset[0].num_node_features,
        num_classes=len(set(train_loader.dataset[0].y))
    )

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(100):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log metrics to Azure ML
        avg_loss = total_loss / len(train_loader)
        run.log('training_loss', avg_loss)
        run.log('epoch', epoch)

        print(f'Epoch {epoch}: Loss = {avg_loss:.4f}')

    print("GNN Training Complete")

if __name__ == "__main__":
    train_gnn()
```

### **Step 3: Create Graph Data Loader**

**File: `backend/src/gnn/data_loader.py`**
```python
import torch
from torch_geometric.data import Data, DataLoader
from azure.cosmos import CosmosClient
import networkx as nx

def load_graph_data():
    """Load graph data from Azure Cosmos DB Gremlin"""

    # Connect to Cosmos DB
    client = CosmosClient.from_connection_string(
        "your_cosmos_connection_string"
    )

    # Query entities and relations from Gremlin
    # Convert to PyTorch Geometric format
    graph_data = []

    # Process each graph in the database
    for graph in get_graphs_from_cosmos():
        # Convert to PyTorch Geometric Data
        data = Data(
            x=graph['node_features'],
            edge_index=graph['edge_index'],
            y=graph['labels']
        )
        graph_data.append(data)

    # Split into train/val
    train_size = int(0.8 * len(graph_data))
    train_dataset = graph_data[:train_size]
    val_dataset = graph_data[train_size:]

    return (
        DataLoader(train_dataset, batch_size=32, shuffle=True),
        DataLoader(val_dataset, batch_size=32)
    )
```

### **Step 4: Create Azure ML Control Script**

**File: `backend/scripts/train_comprehensive_gnn.py`**
```python
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
import argparse
import json

def run_comprehensive_gnn_training(config_path=None, n_trials=10, k_folds=3):
    """Run GNN training with Azure ML"""

    # Load workspace
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='universal-rag-gnn')

    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "model_type": "gnn",
            "hyperparameters": {
                "learning_rate": 0.001,
                "hidden_dim": 128,
                "num_layers": 3
            }
        }

    # Create script config
    script_config = ScriptRunConfig(
        source_directory='./src/gnn',
        script='train.py',
        compute_target='cpu-cluster',  # or 'gpu-cluster' for GNN
        arguments=[
            '--config', json.dumps(config),
            '--n_trials', str(n_trials),
            '--k_folds', str(k_folds)
        ]
    )

    # Set up GNN environment
    env = Environment.from_conda_specification(
        name='gnn-env',
        file_path='gnn-env.yml'
    )
    script_config.run_config.environment = env

    # Submit run
    run = experiment.submit(script_config)
    print(f"Submitted GNN training job: {run.get_portal_url()}")

    return run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--k_folds', type=int, default=3)

    args = parser.parse_args()
    run_comprehensive_gnn_training(args.config, args.n_trials, args.k_folds)
```

### **Step 5: Create Conda Environment**

**File: `backend/gnn-env.yml`**
```yaml
name: gnn-env
channels:
  - defaults
  - pytorch
  - conda-forge
dependencies:
  - python=3.9
  - pytorch>=2.0.0
  - torch-geometric>=2.3.0
  - networkx>=3.2.0
  - numpy>=1.23.5
  - pandas>=2.0.0
  - scikit-learn>=1.3.0
  - pip
  - pip:
    - azureml-sdk
    - azure-cosmos
    - optuna>=3.0.0
    - wandb>=0.16.0
```

## 🚀 **Usage Examples:**

### **CLI Usage:**
```bash
# Train with default config
python backend/scripts/train_comprehensive_gnn.py

# Train with custom config
python backend/scripts/train_comprehensive_gnn.py \
    --config backend/scripts/example_comprehensive_gnn_config.json \
    --n_trials 20 \
    --k_folds 5
```

### **API Usage:**
```python
from scripts.train_comprehensive_gnn import run_comprehensive_gnn_training

# Run training
run = run_comprehensive_gnn_training(
    config_path="config.json",
    n_trials=15,
    k_folds=3
)
```

## 📊 **Integration with Current Architecture:**

### **Updated Flow:**
```
Raw Text Data → Azure Blob Storage → Knowledge Extraction → Entity/Relation Graph → GNN Training (Azure ML) → Azure Cosmos DB Gremlin Graph → Query Processing → Response Generation
```

### **Benefits:**
- ✅ **Learned graph representations** for better entity understanding
- ✅ **Enhanced similarity search** in graph space
- ✅ **Azure ML integration** for experiment tracking
- ✅ **Hyperparameter optimization** with Optuna
- ✅ **Model versioning** and deployment

## 🎯 **Next Steps:**

1. **Create the file structure** as outlined above
2. **Implement the GNN model** with PyTorch Geometric
3. **Connect to Cosmos DB** for graph data loading
4. **Set up Azure ML environment** and dependencies
5. **Test locally** before submitting to Azure ML
6. **Integrate with existing RAG pipeline**

**This plan follows the Azure ML tutorial structure and integrates seamlessly with our existing Azure Universal RAG system!** 🚀