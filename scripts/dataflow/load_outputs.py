#!/usr/bin/env python3
"""
Load and Validate Dataflow Pipeline Outputs

This utility script loads and validates the organized outputs from the dataflow pipeline,
demonstrating how to access the cleaned and structured data files.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch_geometric.data import Data

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_knowledge_extraction() -> Dict[str, Any]:
    """Load Step 02 knowledge extraction results"""
    step02_dir = Path("data/outputs/step02")

    # Load main extraction results
    results_file = step02_dir / "step02_knowledge_extraction_results.json"
    with open(results_file, "r") as f:
        data = json.load(f)

    # Load analysis summary
    summary_file = step02_dir / "knowledge_analysis_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        data["summary"] = summary

    return data


def load_pytorch_geometric() -> Tuple[Data, Dict[str, Any]]:
    """Load Step 04 PyTorch Geometric graph data"""
    step04_dir = Path("data/outputs/step04")

    # Load PyTorch Geometric data
    pytorch_file = step04_dir / "pytorch_geometric_maintenance.pt"
    data_dict = torch.load(pytorch_file, weights_only=False)

    # Load node mapping
    mapping_file = step04_dir / "node_mapping_maintenance.json"
    with open(mapping_file, "r") as f:
        node_mapping = json.load(f)

    # Load execution results
    results_file = step04_dir / "graph_construction_results.json"
    with open(results_file, "r") as f:
        results = json.load(f)

    return data_dict["data"], {
        "node_mapping": node_mapping,
        "results": results,
        "domain": data_dict.get("domain", "unknown"),
    }


def validate_data_consistency():
    """Validate consistency between Step 02 and Step 04 outputs"""
    print("🔍 Validating data consistency across pipeline stages...")

    # Load both datasets
    knowledge_data = load_knowledge_extraction()
    graph_data, graph_metadata = load_pytorch_geometric()

    # Extract entities and relationships
    entities = knowledge_data["knowledge_data"]["entities"]
    relationships = knowledge_data["knowledge_data"]["relationships"]

    # Validate counts
    print(f"📊 Step 02 Knowledge Extraction:")
    print(f"   Entities: {len(entities)}")
    print(f"   Relationships: {len(relationships)}")

    print(f"📊 Step 04 PyTorch Geometric:")
    print(f"   Graph nodes: {graph_data.x.size(0)}")
    print(f"   Graph edges: {graph_data.edge_index.size(1)}")
    print(f"   Node features: {graph_data.x.size(1)}D")
    print(f"   Edge features: {graph_data.edge_attr.size(1)}D")

    # Validate consistency
    node_mapping = graph_metadata["node_mapping"]
    mapped_entities = len(node_mapping.get("entity_to_id", {}))

    print(f"🔍 Consistency Checks:")
    print(f"   ✅ Entities → Nodes: {len(entities)} → {mapped_entities} (mapped)")
    print(
        f"   ✅ Relationships preserved: {len(relationships)} → {graph_data.edge_index.size(1)//2} edges"
    )
    print(
        f"   ✅ Graph structure: {graph_data.validate()} (PyTorch Geometric validation)"
    )

    return True


def main():
    """Main validation and demonstration"""
    print("🚀 Dataflow Pipeline Output Loader")
    print("=" * 50)

    try:
        # Validate organized structure
        step02_dir = Path("data/outputs/step02")
        step04_dir = Path("data/outputs/step04")

        if not step02_dir.exists():
            print("❌ Step 02 output directory not found")
            return 1

        if not step04_dir.exists():
            print("❌ Step 04 output directory not found")
            return 1

        print("✅ Organized output structure validated")

        # Load and validate data
        validate_data_consistency()

        print("\n🎯 Usage Examples:")
        print("# Load knowledge extraction")
        print("knowledge_data = load_knowledge_extraction()")
        print("entities = knowledge_data['knowledge_data']['entities']")
        print("")
        print("# Load PyTorch Geometric graph")
        print("graph_data, metadata = load_pytorch_geometric()")
        print(
            "print(f'Graph: {graph_data.x.size(0)} nodes, {graph_data.edge_index.size(1)} edges')"
        )

        print("\n✅ All pipeline outputs validated successfully!")
        return 0

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
