import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from pathlib import Path
import numpy as np

# --- Configuration ---
GRAPH_DIR = Path(__file__).parent.parent / 'data' / 'graph'
NODES_FILE = GRAPH_DIR / 'nodes.csv'
EDGES_FILE = GRAPH_DIR / 'edges.csv'
MODEL_OUT = GRAPH_DIR / 'model.pt'

# --- Load nodes and edges ---
print(f"Loading nodes from {NODES_FILE}")
df_nodes = pd.read_csv(NODES_FILE)
print(f"Loading edges from {EDGES_FILE}")
df_edges = pd.read_csv(EDGES_FILE)

# --- Build node index mapping ---
node_ids = df_nodes['id'].tolist()
node_id_map = {nid: i for i, nid in enumerate(node_ids)}

# --- Build edge index ---
edge_index = [
    [node_id_map.get(src, -1), node_id_map.get(tgt, -1)]
    for src, tgt in zip(df_edges['source'], df_edges['target'])
    if src in node_id_map and tgt in node_id_map
]
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape [2, num_edges]

# --- Dummy node features (all ones) ---
x = torch.ones((len(node_ids), 8))  # 8-dim features for demo

# --- Dummy labels (random classes for demonstration) ---
num_classes = 3
labels = torch.tensor(np.random.randint(0, num_classes, size=len(node_ids)), dtype=torch.long)

# --- Build PyG Data object ---
data = Data(x=x, edge_index=edge_index, y=labels)

# --- Define a simple GCN model ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_channels=8, hidden_channels=16, out_channels=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Training loop (dummy, for demonstration) ---
print("Starting GNN training (dummy labels)...")
model.train()
for epoch in range(1, 21):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        pred = out.argmax(dim=1)
        acc = int((pred == data.y).sum()) / len(data.y)
        print(f"Epoch {epoch:02d}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

# --- Save the trained model ---
torch.save(model.state_dict(), MODEL_OUT)
print(f"✅ GNN model saved to {MODEL_OUT}")