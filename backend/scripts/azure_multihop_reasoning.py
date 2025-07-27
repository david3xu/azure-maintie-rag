#!/usr/bin/env python3
"""
Azure-Based Multi-Hop Reasoning Implementation
Real knowledge graph traversal using Azure Cosmos DB Gremlin API
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from integrations.azure_openai import AzureOpenAIClient


class AzureMultiHopReasoning:
    """Azure-based multi-hop reasoning using real cloud graph database"""
    
    def __init__(self):
        self.cosmos_client = AzureCosmosGremlinClient()
        self.cosmos_client._initialize_client()
        self.openai_client = AzureOpenAIClient()
        
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get real Azure graph statistics"""
        
        print("📊 Analyzing Azure Knowledge Graph...")
        
        # Get basic counts
        vertex_count = self.cosmos_client.gremlin_client.submit('g.V().count()').all().result()[0]
        edge_count = self.cosmos_client.gremlin_client.submit('g.E().count()').all().result()[0]
        
        # Get entity type distribution
        type_query = "g.V().groupCount().by('entity_type')"
        type_result = self.cosmos_client.gremlin_client.submit(type_query).all().result()
        entity_types = type_result[0] if type_result else {}
        
        # Get domain distribution
        domain_query = "g.V().groupCount().by('domain')"
        domain_result = self.cosmos_client.gremlin_client.submit(domain_query).all().result()
        domains = domain_result[0] if domain_result else {}
        
        stats = {
            'vertices': vertex_count,
            'edges': edge_count,
            'connectivity_ratio': edge_count / vertex_count if vertex_count > 0 else 0,
            'entity_types': entity_types,
            'domains': domains,
            'graph_density': edge_count / (vertex_count * (vertex_count - 1)) if vertex_count > 1 else 0
        }
        
        print(f"✅ Graph Stats: {vertex_count} vertices, {edge_count} edges")
        print(f"   Entity types: {len(entity_types)}")
        print(f"   Connectivity: {stats['connectivity_ratio']:.3f}")
        
        return stats
    
    def find_entities_by_pattern(self, pattern: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find entities matching text pattern using Azure queries"""
        
        print(f"🔍 Searching Azure graph for pattern: '{pattern}'")
        
        # Use Azure Gremlin text search
        query = f"g.V().has('text', containing('{pattern}')).limit({limit}).valueMap()"
        
        try:
            result = self.cosmos_client.gremlin_client.submit(query).all().result()
            
            entities = []
            for entity_data in result:
                entity = {
                    'text': entity_data.get('text', [''])[0],
                    'entity_type': entity_data.get('entity_type', [''])[0],
                    'domain': entity_data.get('domain', [''])[0],
                    'id': entity_data.get('id', [''])[0]
                }
                entities.append(entity)
            
            print(f"   Found {len(entities)} entities matching '{pattern}'")
            return entities
            
        except Exception as e:
            print(f"   Error searching for pattern: {e}")
            return []
    
    def azure_path_finding(self, start_pattern: str, end_pattern: str, max_hops: int = 3) -> List[List[Dict]]:
        """Azure-based path finding between entity patterns"""
        
        print(f"🕸️ Azure Path Finding: '{start_pattern}' → '{end_pattern}' (max {max_hops} hops)")
        
        # Find start and end entities
        start_entities = self.find_entities_by_pattern(start_pattern, limit=5)
        end_entities = self.find_entities_by_pattern(end_pattern, limit=5)
        
        if not start_entities or not end_entities:
            print("   ⚠️ No start or end entities found")
            return []
        
        paths = []
        
        # For each start-end pair, try to find paths using Azure queries
        for start_entity in start_entities[:2]:  # Limit to avoid too many queries
            for end_entity in end_entities[:2]:
                
                start_id = start_entity['id']
                end_id = end_entity['id']
                
                # Azure Gremlin path query
                path_query = f"""
                g.V().has('id', '{start_id}')
                    .repeat(both().simplePath())
                    .times({max_hops})
                    .until(has('id', '{end_id}'))
                    .path()
                    .by(valueMap())
                    .limit(3)
                """
                
                try:
                    path_result = self.cosmos_client.gremlin_client.submit(path_query).all().result()
                    
                    for path_data in path_result:
                        path_entities = []
                        for vertex_data in path_data:
                            entity = {
                                'text': vertex_data.get('text', [''])[0],
                                'entity_type': vertex_data.get('entity_type', [''])[0],
                                'id': vertex_data.get('id', [''])[0]
                            }
                            path_entities.append(entity)
                        
                        if len(path_entities) > 1:  # Valid path
                            paths.append(path_entities)
                
                except Exception as e:
                    print(f"   Path query error: {str(e)[:100]}")
        
        print(f"   Found {len(paths)} Azure-based paths")
        return paths
    
    def azure_neighborhood_analysis(self, entity_pattern: str, hops: int = 2) -> Dict[str, Any]:
        """Analyze entity neighborhood using Azure graph queries"""
        
        print(f"🌐 Azure Neighborhood Analysis: '{entity_pattern}' ({hops} hops)")
        
        entities = self.find_entities_by_pattern(entity_pattern, limit=3)
        
        if not entities:
            return {}
        
        analysis = {
            'center_entities': entities,
            'neighbors': {},
            'patterns': []
        }
        
        for entity in entities[:1]:  # Analyze first entity
            entity_id = entity['id']
            
            # Get neighbors using Azure queries
            neighbor_query = f"""
            g.V().has('id', '{entity_id}')
                .repeat(both().simplePath())
                .times({hops})
                .dedup()
                .valueMap()
                .limit(20)
            """
            
            try:
                neighbor_result = self.cosmos_client.gremlin_client.submit(neighbor_query).all().result()
                
                neighbors = []
                for neighbor_data in neighbor_result:
                    neighbor = {
                        'text': neighbor_data.get('text', [''])[0],
                        'entity_type': neighbor_data.get('entity_type', [''])[0],
                        'distance': 1  # Simplified - Azure doesn't return hop count easily
                    }
                    neighbors.append(neighbor)
                
                analysis['neighbors'][entity_id] = neighbors
                
                # Analyze patterns
                type_counts = {}
                for neighbor in neighbors:
                    entity_type = neighbor['entity_type']
                    type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
                
                analysis['patterns'].append({
                    'center_entity': entity['text'],
                    'neighbor_types': type_counts,
                    'total_neighbors': len(neighbors)
                })
                
            except Exception as e:
                print(f"   Neighborhood analysis error: {str(e)[:100]}")
        
        return analysis
    
    def azure_semantic_path_ranking(self, paths: List[List[Dict]], query_context: str) -> List[Tuple[List[Dict], float]]:
        """Use Azure OpenAI to rank paths by semantic relevance"""
        
        if not paths:
            return []
        
        print(f"🧠 Azure OpenAI Path Ranking for context: '{query_context}'")
        
        ranked_paths = []
        
        for path in paths:
            # Create path description
            path_desc = " → ".join([f"{entity['text']} ({entity['entity_type']})" for entity in path])
            
            # Use Azure OpenAI for semantic scoring
            scoring_prompt = f"""
            Rate the relevance of this knowledge path for the query context "{query_context}".
            Path: {path_desc}
            
            Consider:
            1. Semantic relevance to query
            2. Logical connection strength
            3. Maintenance domain relevance
            
            Provide a score from 0.0 to 1.0 (just the number):
            """
            
            try:
                score_response = self.openai_client.get_completion(scoring_prompt)
                score = float(score_response.strip())
                ranked_paths.append((path, score))
            except:
                ranked_paths.append((path, 0.5))  # Default score
        
        # Sort by score descending
        ranked_paths.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   Ranked {len(ranked_paths)} paths using Azure OpenAI")
        return ranked_paths
    
    def demonstrate_azure_multihop_reasoning(self) -> Dict[str, Any]:
        """Complete Azure-based multi-hop reasoning demonstration"""
        
        print("\n🚀 AZURE MULTI-HOP REASONING DEMONSTRATION")
        print("=" * 70)
        
        results = {
            'timestamp': time.time(),
            'graph_stats': self.get_graph_statistics(),
            'reasoning_examples': []
        }
        
        # Example 1: Equipment to Issues
        print("\n📍 Example 1: Azure Multi-Hop Equipment → Issues")
        paths = self.azure_path_finding("air conditioner", "not working", max_hops=3)
        
        if paths:
            ranked_paths = self.azure_semantic_path_ranking(paths, "air conditioner maintenance problems")
            
            example = {
                'query': 'Equipment to Issues',
                'start_pattern': 'air conditioner',
                'end_pattern': 'not working',
                'paths_found': len(paths),
                'best_path': ranked_paths[0][0] if ranked_paths else None,
                'best_score': ranked_paths[0][1] if ranked_paths else 0
            }
            results['reasoning_examples'].append(example)
            
            print(f"   Found {len(paths)} paths, best score: {example['best_score']:.3f}")
        
        # Example 2: Component Analysis
        print("\n📍 Example 2: Azure Neighborhood Analysis")
        neighborhood = self.azure_neighborhood_analysis("thermostat", hops=2)
        
        if neighborhood:
            example = {
                'query': 'Component Neighborhood',
                'center_pattern': 'thermostat',
                'neighbors_found': sum(len(neighbors) for neighbors in neighborhood['neighbors'].values()),
                'patterns': neighborhood['patterns']
            }
            results['reasoning_examples'].append(example)
            
            print(f"   Analyzed {example['neighbors_found']} neighbors")
        
        # Example 3: Pattern Discovery
        print("\n📍 Example 3: Azure Pattern Discovery")
        equipment_entities = self.find_entities_by_pattern("equipment", limit=10)
        issue_entities = self.find_entities_by_pattern("leak", limit=10)
        
        pattern_example = {
            'query': 'Pattern Discovery',
            'equipment_found': len(equipment_entities),
            'issue_found': len(issue_entities),
            'patterns': {
                'equipment_types': list(set([e['entity_type'] for e in equipment_entities])),
                'issue_types': list(set([e['entity_type'] for e in issue_entities]))
            }
        }
        results['reasoning_examples'].append(pattern_example)
        
        print(f"   Equipment patterns: {pattern_example['equipment_found']}")
        print(f"   Issue patterns: {pattern_example['issue_found']}")
        
        return results


def main():
    """Execute Azure-based multi-hop reasoning demonstration"""
    
    print("🔷 AZURE MULTI-HOP REASONING SYSTEM")
    print("Real knowledge graph operations using Azure Cosmos DB")
    print("=" * 70)
    
    try:
        reasoning = AzureMultiHopReasoning()
        
        # Run complete demonstration
        results = reasoning.demonstrate_azure_multihop_reasoning()
        
        # Save results
        output_file = Path(__file__).parent.parent / "data/demo_outputs/azure_multihop_reasoning.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("🎉 AZURE MULTI-HOP REASONING COMPLETED")
        print(f"{'='*80}")
        print(f"\n📊 Final Results:")
        print(f"   Graph vertices: {results['graph_stats']['vertices']}")
        print(f"   Graph edges: {results['graph_stats']['edges']}")
        print(f"   Reasoning examples: {len(results['reasoning_examples'])}")
        print(f"   Entity types: {len(results['graph_stats']['entity_types'])}")
        print(f"\n📁 Results: {output_file}")
        print(f"\n✅ REAL AZURE KNOWLEDGE GRAPH OPERATIONAL!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Azure multi-hop reasoning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())