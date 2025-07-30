#!/usr/bin/env python3
"""
Standalone GNN training script for Azure ML - NO external dependencies
This script includes all necessary components inline to work in Azure ML cloud environment
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
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# ============================================================================
# STANDALONE AZURE COSMOS DB CLIENT (Inline)
# ============================================================================

class StandaloneCosmosClient:
    """Standalone Cosmos DB client with no external dependencies"""
    
    def __init__(self, config: Dict[str, str]):
        self.endpoint = config['endpoint']
        self.database = config['database'] 
        self.container = config['container']
        self.gremlin_client = None
        self._initialized = False
        
    def _initialize_client(self):
        """Initialize Gremlin client"""
        try:
            from gremlin_python.driver import client, serializer
            from azure.identity import DefaultAzureCredential
            import os
            
            print(f"🔧 Initializing Cosmos client...")
            print(f"   Endpoint: {self.endpoint}")
            print(f"   Database: {self.database}")
            print(f"   Container: {self.container}")
            
            # Extract account name from endpoint
            account_name = self.endpoint.replace('https://', '').replace('.documents.azure.com:443/', '')
            gremlin_endpoint = f"wss://{account_name}.gremlin.cosmosdb.azure.com:443/"
            print(f"   Gremlin endpoint: {gremlin_endpoint}")
            
            # Get Azure credential with multiple fallback methods
            print(f"🔑 Getting Azure credentials...")
            token = None
            
            # Method 1: Try DefaultAzureCredential (works locally and in most Azure services)
            try:
                print(f"🔄 Trying DefaultAzureCredential...")
                credential = DefaultAzureCredential()
                print(f"✅ DefaultAzureCredential created")
                
                print(f"🎫 Getting Cosmos access token...")
                token = credential.get_token("https://cosmos.azure.com/.default")
                print(f"✅ Token acquired (expires: {token.expires_on})")
                
            except Exception as cred_error:
                print(f"⚠️ DefaultAzureCredential failed: {cred_error}")
                
                # Method 2: Try ManagedIdentityCredential (for Azure ML compute)
                try:
                    print(f"🔄 Trying ManagedIdentityCredential for Azure ML...")
                    from azure.identity import ManagedIdentityCredential
                    credential = ManagedIdentityCredential()
                    token = credential.get_token("https://cosmos.azure.com/.default")
                    print(f"✅ Managed Identity token acquired")
                    
                except Exception as mi_error:
                    print(f"⚠️ ManagedIdentityCredential failed: {mi_error}")
                    
                    # Method 3: Try Environment-based credential
                    try:
                        print(f"🔄 Trying EnvironmentCredential...")
                        from azure.identity import EnvironmentCredential
                        credential = EnvironmentCredential()
                        token = credential.get_token("https://cosmos.azure.com/.default")
                        print(f"✅ Environment credential token acquired")
                        
                    except Exception as env_error:
                        print(f"❌ All credential methods failed:")
                        print(f"   DefaultAzureCredential: {cred_error}")
                        print(f"   ManagedIdentityCredential: {mi_error}")
                        print(f"   EnvironmentCredential: {env_error}")
                        raise Exception("No valid authentication method found for Cosmos DB")
            
            if not token:
                raise Exception("Failed to acquire access token for Cosmos DB")
            
            # Create Gremlin client
            username = f"/dbs/{self.database}/colls/{self.container}"
            print(f"🔌 Creating Gremlin client with username: {username}")
            
            self.gremlin_client = client.Client(
                gremlin_endpoint,
                'g',
                username=username,
                password=token.token,
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            
            self._initialized = True
            print("✅ Standalone Cosmos DB client initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Cosmos client: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
    
    def _execute_query_safe(self, query: str, timeout_seconds: int = 30):
        """Execute Gremlin query safely"""
        try:
            if not self._initialized:
                print(f"🔄 Client not initialized, initializing now...")
                if not self._initialize_client():
                    print(f"❌ Client initialization failed, returning empty result")
                    return []
            
            print(f"🔍 Executing query: {query[:100]}{'...' if len(query) > 100 else ''}")
            result = self.gremlin_client.submit(query)
            query_result = result.all().result(timeout=timeout_seconds)
            print(f"✅ Query executed, got {len(query_result) if query_result else 0} results")
            return query_result
            
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"🔑 This appears to be an authentication issue")
            return []
    
    def get_all_entities(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Get all entities from graph"""
        try:
            # First check what properties exist
            sample_query = "g.V().limit(1).valueMap()"
            sample_result = self._execute_query_safe(sample_query)
            print(f"📊 Sample vertex properties: {sample_result}")
            
            # Get vertices with ID and properties using project()
            query = f"g.V().limit({limit}).project('id', 'label', 'properties').by(id()).by(label()).by(valueMap())"
            entities = self._execute_query_safe(query)
            
            # Convert results to proper format
            result_entities = []
            for i, entity in enumerate(entities):
                if isinstance(entity, dict):
                    # Extract ID, label, and properties
                    vertex_id = entity.get("id", f"vertex_{i}")
                    vertex_label = entity.get("label", "unknown")
                    properties = entity.get("properties", {})
                    
                    # Extract text property
                    text = properties.get("text", [""])
                    text = text[0] if isinstance(text, list) else str(text)
                    
                    # Use label as entity_type if no explicit type
                    entity_type = properties.get("entity_type", properties.get("type", [vertex_label]))
                    entity_type = entity_type[0] if isinstance(entity_type, list) else str(entity_type)
                    
                    result_entities.append({
                        "id": str(vertex_id),
                        "text": str(text),
                        "entity_type": str(entity_type),
                        "confidence": 1.0,
                        "label": str(vertex_label)
                    })
            
            print(f"✅ Retrieved {len(result_entities)} entities from Cosmos DB")
            return result_entities
            
        except Exception as e:
            print(f"❌ Failed to get entities: {e}")
            return []
    
    def get_all_relations(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all relations from graph"""
        try:
            # Check edge properties
            sample_edge_query = "g.E().limit(1).valueMap()"
            sample_result = self._execute_query_safe(sample_edge_query)
            print(f"📊 Sample edge properties: {sample_result}")
            
            # Get edges using flexible query
            query = f"g.E().limit({limit}).project('source', 'target', 'label').by(outV().id()).by(inV().id()).by(label())"
            relations = self._execute_query_safe(query)
            
            # Convert results
            result_relations = []
            for rel in relations:
                if isinstance(rel, dict):
                    result_relations.append({
                        "source_entity": str(rel.get("source", "")),
                        "target_entity": str(rel.get("target", "")),
                        "relation_type": str(rel.get("label", "related"))
                    })
            
            print(f"✅ Retrieved {len(result_relations)} relations from Cosmos DB")
            return result_relations
            
        except Exception as e:
            print(f"❌ Failed to get relations: {e}")
            return []


# ============================================================================
# GNN MODEL CLASSES (Standalone)
# ============================================================================

class SimpleGCNLayer(nn.Module):
    """Simple Graph Convolution Layer using standard PyTorch"""
    def __init__(self, input_dim, output_dim):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, adj_matrix):
        # Simple message passing: aggregate neighbor features
        aggregated = torch.mm(adj_matrix, x)
        return self.linear(aggregated)

class GNNModel(nn.Module):
    """GNN model using only standard PyTorch"""
    def __init__(self, input_dim=768, hidden_dim=128, output_dim=64):
        super(GNNModel, self).__init__()
        self.conv1 = SimpleGCNLayer(input_dim, hidden_dim)
        self.conv2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.conv3 = SimpleGCNLayer(hidden_dim, output_dim)
        self.classifier = nn.Linear(output_dim, 2)  # Binary classification
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, adj_matrix):
        # Layer 1
        x = self.conv1(x, adj_matrix)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2  
        x = self.conv2(x, adj_matrix)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.conv3(x, adj_matrix)
        x = F.relu(x)
        
        # Classification
        return self.classifier(x)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_adjacency_matrix(edges, num_nodes):
    """Create normalized adjacency matrix"""
    adj = torch.zeros(num_nodes, num_nodes)
    
    # Add edges
    for edge in edges:
        if len(edge) >= 2:
            i, j = edge[0], edge[1]
            if 0 <= i < num_nodes and 0 <= j < num_nodes:
                adj[i, j] = 1
                adj[j, i] = 1  # Undirected
    
    # Add self-loops
    adj += torch.eye(num_nodes)
    
    # Normalize
    row_sums = adj.sum(dim=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    adj_normalized = adj / row_sums.unsqueeze(1)
    
    return adj_normalized

def load_from_local_export():
    """Load data from local knowledge graph export file"""
    print("🔄 Loading data from local knowledge graph export...")
    
    import json
    from pathlib import Path
    
    # Path to exported knowledge graph
    export_file = Path(__file__).parent / "knowledge_graph_export.json"
    
    if not export_file.exists():
        raise FileNotFoundError(f"Local knowledge graph export not found: {export_file}")
    
    print(f"📁 Loading from: {export_file}")
    
    with open(export_file, 'r') as f:
        data = json.load(f)
    
    entities = data.get('entities', [])
    relationships = data.get('relationships', [])
    
    print(f"📊 Loaded from export: {len(entities)} entities, {len(relationships)} relationships")
    
    if len(entities) < 10:
        raise ValueError(f"Insufficient entities in export file: {len(entities)} (need at least 10)")
    
    # Process entities into features (same logic as Cosmos version)
    entity_features = []
    entity_labels = []
    entity_id_map = {}
    
    for i, entity in enumerate(entities[:285]):  # Cap for memory
        entity_id_map[entity["id"]] = i
        
        # Create feature vector from entity properties
        feature_vector = torch.zeros(768)
        
        # Hash-based encoding
        text_hash = hash(str(entity["text"])) % 768
        type_hash = hash(str(entity["entity_type"])) % 768
        
        feature_vector[text_hash] = 1.0
        feature_vector[type_hash] = 0.5
        
        # Add small random noise
        feature_vector += torch.randn(768) * 0.01
        
        entity_features.append(feature_vector)
        
        # Create labels based on actual entity types found in data
        entity_type = entity.get("entity_type", entity.get("label", "unknown"))
        
        # Binary classification: maintenance-related (1) vs other (0)
        maintenance_types = ['maintenance_issue', 'evidence_report', 'maintenance', 'equipment', 'component', 'action']
        label = 1 if any(mt in str(entity_type).lower() for mt in maintenance_types) else 0
        entity_labels.append(label)
    
    x = torch.stack(entity_features)
    y = torch.tensor(entity_labels, dtype=torch.long)
    
    # Process relations into adjacency matrix
    num_nodes = len(entity_features)
    edges = []
    
    for rel in relationships:
        source_id = rel["source"]
        target_id = rel["target"]
        
        if source_id in entity_id_map and target_id in entity_id_map:
            source_idx = entity_id_map[source_id]
            target_idx = entity_id_map[target_id]
            
            if source_idx < num_nodes and target_idx < num_nodes:
                edges.append([source_idx, target_idx])
    
    # If no edges, create minimal connectivity
    if not edges:
        # Create ring connectivity for basic structure
        for i in range(num_nodes):
            edges.append([i, (i + 1) % num_nodes])
    
    adj_matrix = create_adjacency_matrix(edges, num_nodes)
    
    print(f"🎉 LOCAL DATA LOADED: {num_nodes} nodes, {len(edges)} edges")
    return x, adj_matrix, y


def load_real_cosmos_data():
    """Load REAL data from Azure Cosmos DB - standalone version"""
    print("🔄 Loading REAL data from Azure Cosmos DB (standalone)...")
    
    try:
        # Configuration for Cosmos DB
        cosmos_config = {
            'endpoint': os.environ.get('AZURE_COSMOS_ENDPOINT', 'https://cosmos-maintie-rag-staging-oeeopj3ksgnlo.documents.azure.com:443/'),
            'database': os.environ.get('COSMOS_DATABASE_NAME', 'maintie-rag-staging'),
            'container': os.environ.get('COSMOS_GRAPH_NAME', 'knowledge-graph-staging')
        }
        
        print(f"📋 Cosmos config: {cosmos_config}")
        
        # Create standalone client
        cosmos_client = StandaloneCosmosClient(cosmos_config)
        
        # Get real data
        entities = cosmos_client.get_all_entities(limit=500)
        relations = cosmos_client.get_all_relations(limit=1000)
        
        # Validate data quality for GNN training
        print(f"\n📊 DATA VALIDATION:")
        print(f"   Entities found: {len(entities)}")
        print(f"   Relations found: {len(relations)}")
        
        # Check entity quality
        entities_with_text = sum(1 for e in entities if e.get('text', '').strip())
        print(f"   Entities with text: {entities_with_text}")
        
        # Check entity types/labels
        entity_types = {}
        for e in entities:
            etype = e.get('entity_type', e.get('label', 'unknown'))
            entity_types[etype] = entity_types.get(etype, 0) + 1
        print(f"   Entity types: {entity_types}")
        
        if len(entities) >= 10:  # Minimum viable data
            print(f"✅ SUFFICIENT DATA: Found {len(entities)} real entities and {len(relations)} real relations")
            
            if len(relations) == 0:
                print(f"⚠️ NO RELATIONSHIPS: Will create synthetic connectivity for GNN training")
            else:
                print(f"✅ REAL RELATIONSHIPS: Using {len(relations)} actual connections")
            
            # Process entities into features
            entity_features = []
            entity_labels = []
            entity_id_map = {}
            
            for i, entity in enumerate(entities[:285]):  # Cap for memory
                entity_id_map[entity["id"]] = i
                
                # Create feature vector from entity properties
                feature_vector = torch.zeros(768)
                
                # Hash-based encoding
                text_hash = hash(str(entity["text"])) % 768
                type_hash = hash(str(entity["entity_type"])) % 768
                
                feature_vector[text_hash] = 1.0
                feature_vector[type_hash] = 0.5
                
                # Add small random noise
                feature_vector += torch.randn(768) * 0.01
                
                entity_features.append(feature_vector)
                
                # Create labels based on actual entity types found in data
                # From our data: maintenance_issue, Entity, evidence_report
                entity_type = entity.get("entity_type", entity.get("label", "unknown"))
                
                # Binary classification: maintenance-related (1) vs other (0)
                maintenance_types = ['maintenance_issue', 'evidence_report', 'maintenance', 'equipment', 'component', 'action']
                label = 1 if any(mt in str(entity_type).lower() for mt in maintenance_types) else 0
                entity_labels.append(label)
            
            x = torch.stack(entity_features)
            y = torch.tensor(entity_labels, dtype=torch.long)
            
            # Process relations into adjacency matrix
            num_nodes = len(entity_features)
            edges = []
            
            for rel in relations:
                source_id = rel["source_entity"]
                target_id = rel["target_entity"]
                
                if source_id in entity_id_map and target_id in entity_id_map:
                    source_idx = entity_id_map[source_id]
                    target_idx = entity_id_map[target_id]
                    
                    if source_idx < num_nodes and target_idx < num_nodes:
                        edges.append([source_idx, target_idx])
            
            # If no edges, create minimal connectivity
            if not edges:
                # Create ring connectivity for basic structure
                for i in range(num_nodes):
                    edges.append([i, (i + 1) % num_nodes])
            
            adj_matrix = create_adjacency_matrix(edges, num_nodes)
            
            print(f"🎉 REAL DATA LOADED: {num_nodes} nodes, {len(edges)} edges")
            return x, adj_matrix, y
        
        else:
            error_msg = f"❌ CRITICAL ERROR: Insufficient real data found - only {len(entities)} entities"
            print(error_msg)
            raise ValueError(f"Training requires at least 10 entities but only found {len(entities)}. Cannot proceed without real data.")
            
    except Exception as e:
        error_msg = f"❌ COSMOS DB ACCESS FAILED: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # Try to load from local exported file as fallback
        print(f"\n🔄 FALLBACK: Attempting to load from local knowledge graph export...")
        try:
            return load_from_local_export()
        except Exception as fallback_error:
            print(f"❌ FALLBACK FAILED: {fallback_error}")
            raise RuntimeError(f"GNN training failed: Cosmos DB inaccessible and local fallback failed. Cosmos error: {e}, Fallback error: {fallback_error}")
    
    # This line should never be reached - all paths above either return or raise

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_gnn():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train GNN with real Azure Cosmos DB data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimensions')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Debug environment information
    print(f"🔍 ENVIRONMENT DEBUG:")
    print(f"   Python: {sys.version}")
    print(f"   Working dir: {os.getcwd()}")
    
    # Check key environment variables
    env_vars = ['AZURE_COSMOS_ENDPOINT', 'COSMOS_DATABASE_NAME', 'COSMOS_GRAPH_NAME', 
                'AZURE_CLIENT_ID', 'AZURE_TENANT_ID', 'MSI_ENDPOINT']
    for var in env_vars:
        value = os.environ.get(var, 'NOT_SET')
        if 'secret' in var.lower() or 'key' in var.lower():
            print(f"   {var}: {'*' * 8 if value != 'NOT_SET' else 'NOT_SET'}")
        else:
            print(f"   {var}: {value}")
    
    # Start MLflow tracking
    mlflow.start_run()
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("learning_rate", args.lr)
    mlflow.log_param("hidden_dim", args.hidden_dim)
    
    print(f"\n🚀 Starting STANDALONE GNN training for {args.epochs} epochs...")
    
    # Load real graph data
    x, adj_matrix, y = load_real_cosmos_data()
    num_nodes = x.shape[0]
    print(f"Graph: {num_nodes} nodes, adjacency matrix shape: {adj_matrix.shape}")
    
    # Create train/val/test splits
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = indices[:train_size]
    val_mask = indices[train_size:train_size + val_size]
    test_mask = indices[train_size + val_size:]
    
    print(f"Data split: {len(train_mask)} train, {len(val_mask)} val, {len(test_mask)} test")
    
    # Initialize model
    model = GNNModel(input_dim=768, hidden_dim=args.hidden_dim, output_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(x, adj_matrix)
        loss = criterion(out[train_mask], y[train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
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
    
    print("Final Results:")
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
        'model_type': 'standalone_gnn',
        'real_data_used': True
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