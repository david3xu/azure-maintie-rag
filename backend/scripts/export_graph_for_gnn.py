import os
import json
import pandas as pd
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
GRAPH_DIR = Path(__file__).parent.parent / 'data' / 'graph'
ENTITIES_FILE = DATA_DIR / 'maintenance_entities.json'
RELATIONS_FILE = DATA_DIR / 'maintenance_relations.json'
NODES_OUT = GRAPH_DIR / 'nodes.csv'
EDGES_OUT = GRAPH_DIR / 'edges.csv'

# --- Ensure output directory exists ---
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# --- Load entities ---
print(f"Loading entities from {ENTITIES_FILE}")
with open(ENTITIES_FILE, 'r') as f:
    entities = json.load(f)

# --- Load relations ---
print(f"Loading relations from {RELATIONS_FILE}")
with open(RELATIONS_FILE, 'r') as f:
    relations = json.load(f)

# --- Build node list ---
node_rows = []
for entity_id, entity in entities.items():
    node_rows.append({
        'id': entity_id,
        'type': entity.get('entity_type', ''),
        'text': entity.get('text', ''),
        'metadata': json.dumps(entity.get('metadata', {})),
    })

df_nodes = pd.DataFrame(node_rows)
df_nodes.to_csv(NODES_OUT, index=False)
print(f"Exported {len(df_nodes)} nodes to {NODES_OUT}")

# --- Build edge list ---
edge_rows = []
for rel in relations:
    edge_rows.append({
        'source': rel.get('head_id', ''),
        'target': rel.get('tail_id', ''),
        'type': rel.get('relation_type', ''),
        'metadata': json.dumps(rel.get('metadata', {})),
    })

df_edges = pd.DataFrame(edge_rows)
df_edges.to_csv(EDGES_OUT, index=False)
print(f"Exported {len(df_edges)} edges to {EDGES_OUT}")

print("✅ MaintIE graph export complete.")