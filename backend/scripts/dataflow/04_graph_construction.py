#!/usr/bin/env python3
"""
Graph Construction - Stage 4 of README Data Flow
Entities/Relations → PyTorch Geometric Format for Direct GNN Training

This script implements the Draft KG → GNN Learning approach:
- Reads knowledge extraction results from Azure Blob Storage
- Transforms entities and relationships into PyTorch Geometric Data format
- Creates node features, edge indices, and edge attributes for GNN training
- Outputs both PyTorch Geometric files and JSON statistics for reference
- Eliminates Cosmos DB dependency for streamlined pipeline
"""

import sys
import asyncio
import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from torch_geometric.data import Data
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService
from config.domain_patterns import DomainPatternManager

logger = logging.getLogger(__name__)

class GraphConstructionStage:
    """Stage 4: Entities/Relations → PyTorch Geometric Format for Direct GNN Training"""
    
    def __init__(self):
        self.infrastructure = InfrastructureService()
        
    async def execute(self, extraction_container: str, domain: str = "maintenance") -> Dict[str, Any]:
        """
        Execute graph construction stage - read knowledge extraction from Azure, create PyTorch Geometric format
        
        Args:
            extraction_container: Azure Blob Storage container with knowledge extraction results
            domain: Domain for graph construction
            
        Returns:
            Dict with graph construction results and PyTorch Geometric file paths
        """
        print("🧠 Stage 4: Graph Construction - Azure Storage → PyTorch Geometric Format")
        print("=" * 70)
        
        start_time = asyncio.get_event_loop().time()
        
        results = {
            "stage": "04_graph_construction",
            "source_container": extraction_container,
            "domain": domain,
            "pytorch_geometric_created": False,
            "success": False
        }
        
        try:
            # Get required services
            storage_client = self.infrastructure.storage_client
            
            if not storage_client:
                raise RuntimeError("❌ Azure Storage client not initialized")
            
            # Find knowledge extraction results in Azure Blob Storage
            print(f"📦 Reading knowledge extraction results from container: {extraction_container}")
            blobs_result = await storage_client.list_blobs(extraction_container)
            
            if not blobs_result.get('success'):
                raise RuntimeError(f"Failed to list blobs: {blobs_result.get('error')}")
            
            blobs = blobs_result.get('data', {}).get('blobs', [])
            print(f"📄 Found {len(blobs)} files in container")
            
            # Find the knowledge extraction JSON file
            extraction_blob = None
            for blob in blobs:
                blob_name = blob.get('name', '')
                if 'knowledge_extraction' in blob_name and blob_name.endswith('.json'):
                    extraction_blob = blob_name
                    break
            
            if not extraction_blob:
                raise ValueError(f"No knowledge extraction JSON file found in container: {extraction_container}")
            
            print(f"📖 Processing extraction file: {extraction_blob}")
            
            # Try to download from Azure, fallback to local Step 02 data for testing
            extraction_data = None
            
            try:
                download_result = await storage_client.download_file(extraction_blob, extraction_container)
                if download_result.get('success'):
                    content = download_result.get('data', {}).get('content', '')
                    if content.strip():
                        extraction_data = json.loads(content)
                        print(f"✅ Using Azure data from: {extraction_blob}")
                    else:
                        raise ValueError(f"Empty content in {extraction_blob}")
                else:
                    raise RuntimeError(f"Azure download failed: {download_result.get('error')}")
            except Exception as azure_error:
                print(f"⚠️  Azure download failed: {azure_error}")
                print("🔄 Falling back to local Step 02 data for testing...")
                
                # Fallback to local Step 02 data
                from pathlib import Path
                import json
                
                local_file = Path("data/outputs/step02/step02_knowledge_extraction_results.json")
                if local_file.exists():
                    with open(local_file, 'r') as f:
                        step02_data = json.load(f)
                    
                    # Extract the knowledge data in the expected format
                    extraction_data = {
                        'entities': step02_data.get('knowledge_data', {}).get('entities', []),
                        'relationships': step02_data.get('knowledge_data', {}).get('relationships', [])
                    }
                    print(f"✅ Using local Step 02 data: {len(extraction_data['entities'])} entities, {len(extraction_data['relationships'])} relationships")
                else:
                    raise RuntimeError("No Azure data available and no local Step 02 backup found")
            
            if not extraction_data:
                raise ValueError("No extraction data available from Azure or local sources")
            
            # Extract entities and relationships
            entities = extraction_data.get('entities', [])
            relationships = extraction_data.get('relationships', [])
            
            print(f"📊 Loaded {len(entities)} entities and {len(relationships)} relationships")
            
            if not entities and not relationships:
                raise ValueError("No entities or relationships found in extraction results")
            
            print(f"🧠 Converting to PyTorch Geometric format: {len(entities)} entities, {len(relationships)} relationships")
            
            # Get domain-specific PyTorch Geometric configuration
            pytorch_config = DomainPatternManager.get_pytorch_geometric(domain)
            
            # Transform to PyTorch Geometric format
            pytorch_data = self._convert_to_pytorch_geometric(entities, relationships, domain, pytorch_config)
            
            if pytorch_data is None:
                raise ValueError("Failed to convert to PyTorch Geometric format")
            
            # Create organized output files
            output_dir = Path("data/outputs/step04")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save PyTorch Geometric data
            pytorch_file = output_dir / f"pytorch_geometric_{domain}.pt"
            torch.save(pytorch_data, pytorch_file)
            print(f"💾 PyTorch Geometric data saved: {pytorch_file}")
            
            # Save node mapping for reference
            mapping_file = output_dir / f"node_mapping_{domain}.json"
            with open(mapping_file, 'w') as f:
                json.dump(pytorch_data['node_mapping'], f, indent=2)
            print(f"📋 Node mapping saved: {mapping_file}")
            
            # Create detailed statistics for local reference
            entity_types = {}
            relationship_types = {}
            
            for entity in entities:
                entity_type = entity.get('type', 'unknown')
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            for relationship in relationships:
                rel_type = relationship.get('relation', 'unknown')
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            # Update results with comprehensive data
            results.update({
                "pytorch_geometric_created": True,
                "pytorch_file": str(pytorch_file),
                "mapping_file": str(mapping_file),
                "extraction_file": extraction_blob,
                "pytorch_data_info": {
                    "num_nodes": pytorch_data['data'].x.size(0),
                    "num_edges": pytorch_data['data'].edge_index.size(1),
                    "node_feature_dim": pytorch_data['data'].x.size(1),
                    "edge_feature_dim": pytorch_data['data'].edge_attr.size(1) if pytorch_data['data'].edge_attr is not None else 0,
                    "num_classes": len(set(pytorch_data['data'].y.numpy()))
                },
                "statistics": {
                    "total_entities": len(entities),
                    "total_relationships": len(relationships),
                    "unique_entity_types": len(entity_types),
                    "unique_relationship_types": len(relationship_types),
                    "entity_type_distribution": dict(sorted(entity_types.items(), key=lambda x: x[1], reverse=True)),
                    "relationship_type_distribution": dict(sorted(relationship_types.items(), key=lambda x: x[1], reverse=True))
                }
            })
            
            # Success
            duration = asyncio.get_event_loop().time() - start_time
            results["duration_seconds"] = round(duration, 2)
            results["success"] = True
            
            # Save execution results and statistics
            results_file = output_dir / "graph_construction_results.json"
            
            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Results saved: {results_file}")
            except Exception as e:
                print(f"⚠️  Failed to save results: {e}")
            
            print(f"✅ Stage 4 Complete:")
            print(f"   📊 Entities processed: {len(entities)}")
            print(f"   🔗 Relationships processed: {len(relationships)}")
            print(f"   🧠 PyTorch Geometric nodes: {pytorch_data['data'].x.size(0)}")
            print(f"   🔗 PyTorch Geometric edges: {pytorch_data['data'].edge_index.size(1)}")
            print(f"   📁 PyTorch file: {pytorch_file}")
            print(f"   ⏱️  Duration: {results['duration_seconds']}s")
            
            return results
            
        except Exception as e:
            results["error"] = str(e)
            results["duration_seconds"] = round(asyncio.get_event_loop().time() - start_time, 2)
            print(f"❌ Stage 4 Failed: {e}")
            logger.error(f"Graph construction failed: {e}", exc_info=True)
            return results

    def _convert_to_pytorch_geometric(self, entities: List[Dict], relationships: List[Dict], domain: str, config) -> Dict[str, Any]:
        """
        Convert entities and relationships to PyTorch Geometric Data format
        
        Args:
            entities: List of entity dictionaries from knowledge extraction
            relationships: List of relationship dictionaries from knowledge extraction  
            domain: Domain name for feature encoding
            config: PyTorchGeometricPatterns configuration for the domain
            
        Returns:
            Dict containing PyTorch Geometric Data object and node mapping
        """
        try:
            print("🔄 Creating node mapping...")
            
            # Create entity to node ID mapping
            entity_to_id = {}
            node_features = []
            node_labels = []
            entity_texts = []
            
            # Process entities to create nodes
            for i, entity in enumerate(entities):
                entity_text = entity.get('text', f"entity_{i}")
                entity_type = entity.get('type', 'unknown')
                
                # Map entity text to node ID
                entity_to_id[entity_text] = i
                entity_texts.append(entity_text)
                
                # Create node features using domain configuration
                features = self._create_entity_features(entity, domain, config)
                node_features.append(features)
                
                # Create node label (encoded entity type)
                label = self._encode_entity_type(entity_type, config)
                node_labels.append(label)
                
                if (i + 1) % config.entity_progress_interval == 0:
                    print(f"    📊 Processed {i + 1}/{len(entities)} entities")
            
            print(f"🔗 Creating edge structure from {len(relationships)} relationships...")
            
            # Create edge indices and features
            edge_indices = []
            edge_features = []
            valid_edges = 0
            
            for relationship in relationships:
                source_text = relationship.get('source', '').strip()
                target_text = relationship.get('target', '').strip()
                relation_type = relationship.get('relation', 'related')
                
                # Find corresponding node IDs
                if source_text in entity_to_id and target_text in entity_to_id:
                    source_id = entity_to_id[source_text]
                    target_id = entity_to_id[target_text]
                    
                    # Add edge (bidirectional for undirected graph)
                    edge_indices.append([source_id, target_id])
                    edge_indices.append([target_id, source_id])
                    
                    # Create edge features using domain configuration
                    edge_feature = self._create_relationship_features(relationship, domain, config)
                    edge_features.append(edge_feature)
                    edge_features.append(edge_feature)  # Same features for both directions
                    
                    valid_edges += 1
            
            print(f"    ✅ Created {valid_edges} valid edges ({len(edge_indices)} directed edges)")
            
            if not node_features:
                raise ValueError("No valid entities found for node creation")
            if not edge_indices:
                raise ValueError("No valid relationships found for edge creation")
            
            # Convert to PyTorch tensors
            x = torch.tensor(node_features, dtype=torch.float32)
            y = torch.tensor(node_labels, dtype=torch.long)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y
            )
            
            print(f"✅ PyTorch Geometric conversion complete:")
            print(f"   📊 Nodes: {data.x.size(0)} ({data.x.size(1)}D features)")
            print(f"   🔗 Edges: {data.edge_index.size(1)} ({data.edge_attr.size(1)}D features)")
            print(f"   🏷️  Classes: {len(set(y.numpy()))}")
            
            # Create node mapping for reference
            node_mapping = {
                'entity_to_id': entity_to_id,
                'id_to_entity': {v: k for k, v in entity_to_id.items()},
                'entity_types': [entities[i].get('type', 'unknown') for i in range(len(entities))],
                'total_nodes': len(entity_texts),
                'total_edges': len(edge_indices),
                'feature_dimensions': {
                    'node_features': data.x.size(1),
                    'edge_features': data.edge_attr.size(1),
                    'num_classes': len(set(y.numpy()))
                }
            }
            
            return {
                'data': data,
                'node_mapping': node_mapping,
                'domain': domain
            }
            
        except Exception as e:
            print(f"❌ PyTorch Geometric conversion failed: {e}")
            logger.error(f"PyTorch Geometric conversion failed: {e}", exc_info=True)
            return None

    def _create_entity_features(self, entity: Dict[str, Any], domain: str, config) -> List[float]:
        """Create feature vector for entity using domain configuration"""
        features = []
        
        # Entity type one-hot encoding
        entity_type = entity.get('type', 'unknown').lower()
        for etype in config.entity_types:
            features.append(1.0 if entity_type == etype else 0.0)
        
        # Text length features (normalized using config)
        text = entity.get('text', '')
        features.append(min(len(text) / config.text_length_normalization, 1.0))  # Normalized text length
        features.append(len(text.split()) / config.word_count_normalization if text else 0.0)  # Word count
        
        # Context features
        context = entity.get('context', '')
        features.append(min(len(context) / config.context_length_normalization, 1.0))  # Context length
        
        # Confidence and quality features
        features.append(entity.get('confidence', 1.0))
        
        # Domain-specific equipment features
        for keyword in config.equipment_keywords:
            features.append(1.0 if keyword in text.lower() else 0.0)
        
        # Domain-specific issue features
        features.append(1.0 if any(term in text.lower() for term in config.issue_keywords) else 0.0)
        
        # Pad to exactly the configured dimension
        while len(features) < config.node_feature_dim:
            features.append(0.0)
        
        return features[:config.node_feature_dim]

    def _create_relationship_features(self, relationship: Dict[str, Any], domain: str, config) -> List[float]:
        """Create feature vector for relationship using domain configuration"""
        features = []
        
        # Relationship type one-hot encoding
        relation_type = relationship.get('relation', 'related').lower()
        for rtype in config.relationship_types:
            features.append(1.0 if relation_type == rtype else 0.0)
        
        # Relationship strength/confidence
        features.append(relationship.get('confidence', 1.0))
        
        # Context length feature (using config normalization)
        context = relationship.get('context', '')
        features.append(min(len(context) / config.context_length_normalization, 1.0))
        
        # Source and target text length features (using config normalization)
        source = relationship.get('source', '')
        target = relationship.get('target', '')
        features.append(min(len(source) / config.source_target_length_normalization, 1.0))
        features.append(min(len(target) / config.source_target_length_normalization, 1.0))
        
        # Domain-specific maintenance relationship features
        for keyword in config.maintenance_keywords:
            features.append(1.0 if keyword in relation_type else 0.0)
        
        # Pad to exactly the configured dimension
        while len(features) < config.edge_feature_dim:
            features.append(0.0)
        
        return features[:config.edge_feature_dim]

    def _encode_entity_type(self, entity_type: str, config) -> int:
        """Encode entity type to integer label for classification using domain configuration"""
        entity_type_lower = entity_type.lower()
        try:
            return config.entity_types.index(entity_type_lower)
        except ValueError:
            # Return index of 'unknown' or last index if 'unknown' not found
            unknown_idx = len(config.entity_types) - 1
            if 'unknown' in config.entity_types:
                unknown_idx = config.entity_types.index('unknown')
            return unknown_idx


async def main():
    """Main entry point for graph construction stage"""
    parser = argparse.ArgumentParser(
        description="Stage 4: Graph Construction - Azure Storage → PyTorch Geometric Format"
    )
    parser.add_argument(
        "--container", 
        required=True,
        help="Azure Blob Storage container with knowledge extraction results"
    )
    parser.add_argument(
        "--domain", 
        default="maintenance",
        help="Domain for graph construction"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Execute stage
    stage = GraphConstructionStage()
    results = await stage.execute(
        extraction_container=args.container,
        domain=args.domain
    )
    
    # Save results if requested
    if args.output and results.get("success"):
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {args.output}")
    
    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))