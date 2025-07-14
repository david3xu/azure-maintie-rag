"""
PyTorch Dataset wrapper for MaintIE GNN data
Professional implementation with proper data loading
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, List

class MaintIEGraphDataset(Dataset):
    """PyTorch Dataset for MaintIE graph data"""

    def __init__(self, gnn_data_dict: Dict[str, Any], split: str = 'train'):
        """Initialize dataset with prepared GNN data"""
        self.data_dict = gnn_data_dict
        self.split = split

        # Get the appropriate data split
        if split == 'train':
            self.data = gnn_data_dict['train_data']
        elif split == 'val':
            self.data = gnn_data_dict['val_data']
        elif split == 'test':
            self.data = gnn_data_dict['test_data']
        else:
            self.data = gnn_data_dict['full_data']

        self.node_labels = gnn_data_dict['node_labels']
        self.edge_labels = gnn_data_dict['edge_labels']

    def __len__(self) -> int:
        """Return number of nodes (for node-level tasks)"""
        return self.data.num_nodes if self.data else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item for training (node or edge level)"""
        return {
            'node_features': self.data.x[idx] if self.data and self.data.x is not None else None,
            'node_label': self.node_labels[idx] if self.node_labels is not None else None,
            'node_idx': idx
        }

    def get_full_graph(self):
        """Return the full graph for graph-level operations"""
        return self.data

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.data:
            return {}

        return {
            'num_nodes': self.data.num_nodes,
            'num_edges': self.data.edge_index.shape[1] if self.data.edge_index is not None else 0,
            'num_features': self.data.x.shape[1] if self.data.x is not None else 0,
            'num_classes': len(torch.unique(self.node_labels)) if self.node_labels is not None else 0
        }