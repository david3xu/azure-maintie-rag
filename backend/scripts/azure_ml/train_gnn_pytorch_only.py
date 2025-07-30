#!/usr/bin/env python3
"""
Real GNN training script using only standard PyTorch (no torch_geometric dependency)
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import json
import argparse
import mlflow
import os
import sys
import random

class SimpleGCNLayer(nn.Module):
    """Simple Graph Convolution Layer using standard PyTorch"""
    def __init__(self, input_dim, output_dim):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, adj_matrix):
        # Simple message passing: aggregate neighbor features
        # adj_matrix should be normalized adjacency matrix
        aggregated = torch.mm(adj_matrix, x)  # Aggregate neighbor features
        return self.linear(aggregated)

class GNNModel(nn.Module):
    """GNN model using only standard PyTorch"""
    def __init__(self, input_dim=768, hidden_dim=128, output_dim=64):
        super(GNNModel, self).__init__()
        self.conv1 = SimpleGCNLayer(input_dim, hidden_dim)
        self.conv2 = SimpleGCNLayer(hidden_dim, output_dim)
        self.classifier = nn.Linear(output_dim, 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, adj_matrix):
        x = F.relu(self.conv1(x, adj_matrix))
        x = self.dropout(x)
        x = self.conv2(x, adj_matrix)
        return self.classifier(x)

def create_adjacency_matrix(edge_list, num_nodes):
    """Create normalized adjacency matrix from edge list"""
    adj = torch.zeros(num_nodes, num_nodes)
    
    # Add edges
    for edge in edge_list:
        i, j = edge[0], edge[1]
        if i < num_nodes and j < num_nodes:
            adj[i, j] = 1.0
            adj[j, i] = 1.0  # Undirected graph
    
    # Add self-loops
    adj += torch.eye(num_nodes)
    
    # Normalize adjacency matrix (simple row normalization)
    row_sums = adj.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    adj_normalized = adj / row_sums
    
    return adj_normalized

def load_cosmos_data():
    """Load REAL data from Cosmos DB - not synthetic data!"""
    print("🔄 Loading REAL data from Azure Cosmos DB...")
    
    # Fix Python path for Azure ML environment
    import sys
    import os
    
    # Fix Python path for training environment
    # Try multiple possible backend locations
    possible_backend_paths = [
        os.path.join(os.getcwd(), 'backend'),
        os.path.join(os.getcwd(), '..', '..'),  # Go up from scripts/azure_ml to backend
        '/workspace/azure-maintie-rag/backend',  # Absolute path
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),  # Relative to this script
    ]
    
    for backend_path in possible_backend_paths:
        if os.path.exists(backend_path) and backend_path not in sys.path:
            sys.path.insert(0, backend_path)
            print(f"📁 Added to Python path: {backend_path}")
    
    # Also add current directory
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        # Import Azure Cosmos client
        try:
            from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
        except ImportError as import_error:
            print(f"❌ Import error: {import_error}")
            print(f"🔍 Python path: {sys.path[:3]}...")  # Show first 3 paths
            # Try alternative import paths
            sys.path.insert(0, os.path.join(os.getcwd(), '.'))
            from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
        
        print("✅ Successfully imported AzureCosmosGremlinClient")
        
        # Create client with explicit configuration for training environment
        cosmos_config = {
            'endpoint': os.environ.get('AZURE_COSMOS_ENDPOINT', 'https://cosmos-maintie-rag-staging-oeeopj3ksgnlo.documents.azure.com:443/'),
            'database': os.environ.get('COSMOS_DATABASE_NAME', 'maintie-rag-staging'),
            'container': os.environ.get('COSMOS_GRAPH_NAME', 'knowledge-graph-staging')
        }
        cosmos_client = AzureCosmosGremlinClient(config=cosmos_config)
        print("✅ Created Cosmos DB client with training configuration")
        
        # Get REAL graph data optimized for training
        print("🔍 Fetching real graph data from Cosmos DB...")
        graph_data = cosmos_client.export_graph_for_training("maintenance")
        
        if graph_data.get('success'):
            entities = graph_data.get('entities', [])
            edges_data = graph_data.get('relations', [])
            
            print(f"✅ Found {len(entities)} REAL entities and {len(edges_data)} REAL relationships")
            
            if len(entities) > 10:  # Need minimum data for training
                # Process REAL entities into features
                print("🧠 Processing real entity features...")
                
                # Create entity ID mapping
                entity_id_map = {}
                entity_features = []
                entity_labels = []
                
                for i, entity in enumerate(entities[:285]):  # Cap for memory
                    # Extract entity properties
                    entity_id = entity.get('id', [f'entity_{i}'])[0] if isinstance(entity.get('id'), list) else entity.get('id', f'entity_{i}')
                    entity_text = entity.get('text', [f'entity_{i}'])[0] if isinstance(entity.get('text'), list) else entity.get('text', f'entity_{i}')
                    entity_type = entity.get('entity_type', ['unknown'])[0] if isinstance(entity.get('entity_type'), list) else entity.get('entity_type', 'unknown')
                    
                    entity_id_map[entity_id] = i
                    
                    # Create simple text-based features (could be enhanced with embeddings)
                    feature_vector = torch.zeros(768)
                    
                    # Simple feature encoding based on entity properties
                    text_hash = hash(str(entity_text)) % 768
                    type_hash = hash(str(entity_type)) % 768
                    
                    feature_vector[text_hash % 768] = 1.0
                    feature_vector[type_hash % 768] = 0.5
                    
                    # Add some randomness for diversity
                    feature_vector += torch.randn(768) * 0.01
                    
                    entity_features.append(feature_vector)
                    
                    # Create labels based on entity type
                    label = 1 if entity_type in ['equipment', 'component', 'action'] else 0
                    entity_labels.append(label)
                
                x = torch.stack(entity_features)
                y = torch.tensor(entity_labels, dtype=torch.long)
                
                print(f"✅ Created {x.shape[0]} real entity features with {x.shape[1]} dimensions")
                
                # Process REAL relationships into adjacency matrix
                print("🔗 Processing real relationships...")
                num_nodes = len(entity_features)
                edges = []
                
                for edge_data in edges_data:
                    source_id = edge_data.get('source')
                    target_id = edge_data.get('target')
                    
                    if source_id in entity_id_map and target_id in entity_id_map:
                        source_idx = entity_id_map[source_id]
                        target_idx = entity_id_map[target_id]
                        
                        if source_idx < num_nodes and target_idx < num_nodes:
                            edges.append([source_idx, target_idx])
                            edges.append([target_idx, source_idx])  # Make undirected
                
                print(f"✅ Created {len(edges)} real edges from Cosmos DB relationships")
                
                # Create adjacency matrix from REAL relationships
                adj_matrix = create_adjacency_matrix(edges, num_nodes)
                
                print(f"🎉 SUCCESS: Using REAL Cosmos DB data - {num_nodes} nodes, {len(edges)} edges")
                return x, adj_matrix, y
        
        error_msg = f"❌ CRITICAL ERROR: Insufficient real data found in Cosmos DB - only {len(entities_data)} entities"
        print(error_msg)
        raise ValueError(f"Training requires sufficient real data but only found {len(entities_data)} entities. Cannot proceed without real data.")
        
    except ImportError as e:
        error_msg = f"❌ CRITICAL ERROR: Import error: {e}"
        print(error_msg)
        print(f"🔍 Python path: {sys.path[:3]}...")  # Show first 3 paths
        raise ImportError(f"Cannot import required Cosmos DB client modules: {e}")
        
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR: Cosmos DB connection error: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # NO FALLBACK - ALWAYS FAIL if we can't get real data
        raise RuntimeError(f"GNN training REQUIRES real Cosmos DB data. Cannot proceed with synthetic fallback. Error: {e}")
    
    # This line should never be reached - all paths above either return or raise

def train_gnn():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()
    
    # Start MLflow run
    mlflow.start_run()
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("learning_rate", args.lr)
    mlflow.log_param("hidden_dim", args.hidden_dim)
    
    print(f"🚀 Starting REAL GNN training for {args.epochs} epochs with Azure Cosmos DB data...")
    
    # Load real graph data
    x, adj_matrix, y = load_cosmos_data()
    num_nodes = x.shape[0]
    print(f"Graph: {num_nodes} nodes, adjacency matrix shape: {adj_matrix.shape}")
    
    # Split data
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Ensure each split has at least some nodes
    train_end = max(int(0.6 * num_nodes), 1)
    val_end = max(int(0.8 * num_nodes), train_end + 1)
    
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True
    
    print(f"Data split: {train_mask.sum()} train, {val_mask.sum()} val, {test_mask.sum()} test")
    
    # Initialize model
    model = GNNModel(input_dim=768, hidden_dim=args.hidden_dim, output_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(x, adj_matrix)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation every 5 epochs for better monitoring
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(x, adj_matrix)
                pred = out.argmax(dim=1)
                
                train_acc = (pred[train_mask] == y[train_mask]).float().mean()
                val_acc = (pred[val_mask] == y[val_mask]).float().mean()
                
                print(f"Epoch {epoch+1:03d}: Loss={loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}")
                
                mlflow.log_metric("loss", loss.item(), step=epoch)
                mlflow.log_metric("train_accuracy", train_acc.item(), step=epoch)
                mlflow.log_metric("val_accuracy", val_acc.item(), step=epoch)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc.item()
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        out = model(x, adj_matrix)
        pred = out.argmax(dim=1)
        test_acc = (pred[test_mask] == y[test_mask]).float().mean()
    
    print(f"Final Results:")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "gnn_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc.item(),
        'num_nodes': num_nodes,
        'model_type': 'pytorch_only_gnn'
    }, model_path)
    
    mlflow.log_metric("best_val_accuracy", best_val_acc)
    mlflow.log_metric("test_accuracy", test_acc.item())
    mlflow.log_artifact(model_path)
    
    print(f"Model saved to: {model_path}")
    mlflow.end_run()
    
    return {
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc.item(),
        "model_path": model_path
    }

if __name__ == "__main__":
    train_gnn()