#!/usr/bin/env python3
"""
Step 6: Multi-hop Reasoning Integration
Integrate all components for end-to-end multi-hop reasoning demo
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from integrations.azure_openai import AzureOpenAIClient


class MultiHopReasoningDemo:
    """Complete multi-hop reasoning demonstration"""
    
    def __init__(self):
        self.cosmos_client = AzureCosmosGremlinClient()
        self.openai_client = AzureOpenAIClient()
        
    def load_demo_data_direct(self) -> Tuple[List[Dict], List[Dict]]:
        """Load demo data directly from quality dataset (bypassing Cosmos DB issues)"""
        
        print("📊 Loading demo data from quality dataset...")
        
        data_file = Path(__file__).parent.parent / "data/extraction_outputs/full_dataset_extraction_9100_entities_5848_relationships.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Quality dataset not found: {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Create meaningful demo subset
        entities = data['entities'][:100]  # First 100 entities
        entity_ids = {entity['entity_id'] for entity in entities}
        
        # Filter relationships to include only those between demo entities
        relationships = []
        for rel in data['relationships']:
            if (rel['source_entity_id'] in entity_ids and 
                rel['target_entity_id'] in entity_ids and 
                len(relationships) < 50):
                relationships.append(rel)
        
        print(f"✅ Demo data loaded: {len(entities)} entities, {len(relationships)} relationships")
        return entities, relationships
    
    def create_entity_graph(self, entities: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Create in-memory graph for demo reasoning"""
        
        print("🕸️ Creating in-memory knowledge graph...")
        
        # Build entity lookup
        entity_lookup = {entity['entity_id']: entity for entity in entities}
        
        # Build adjacency graph for multi-hop traversal
        graph = {}
        for entity in entities:
            graph[entity['entity_id']] = {
                'entity': entity,
                'outgoing': [],
                'incoming': []
            }
        
        # Add relationships
        for rel in relationships:
            source_id = rel['source_entity_id']
            target_id = rel['target_entity_id']
            
            if source_id in graph and target_id in graph:
                graph[source_id]['outgoing'].append({
                    'target_id': target_id,
                    'relation_type': rel['relation_type'],
                    'confidence': rel.get('confidence', 1.0)
                })
                
                graph[target_id]['incoming'].append({
                    'source_id': source_id,
                    'relation_type': rel['relation_type'],
                    'confidence': rel.get('confidence', 1.0)
                })
        
        print(f"✅ Knowledge graph created with {len(graph)} nodes")
        return graph
    
    def find_multi_hop_paths(self, graph: Dict[str, Any], start_entity_text: str, 
                           end_entity_text: str, max_hops: int = 3) -> List[List[Dict]]:
        """Find multi-hop paths between entities"""
        
        print(f"🔍 Finding paths: '{start_entity_text}' → '{end_entity_text}' (max {max_hops} hops)")
        
        # Find entity IDs by text
        start_id = None
        end_id = None
        
        for entity_id, node in graph.items():
            if node['entity']['text'].lower() == start_entity_text.lower():
                start_id = entity_id
            if node['entity']['text'].lower() == end_entity_text.lower():
                end_id = entity_id
        
        if not start_id or not end_id:
            print(f"⚠️  Entities not found in graph")
            return []
        
        # Multi-hop BFS
        paths = []
        queue = [([start_id], [])]  # (entity_path, relation_path)
        
        while queue and len(paths) < 5:  # Limit to 5 paths for demo
            current_path, relation_path = queue.pop(0)
            current_entity = current_path[-1]
            
            if len(current_path) > max_hops:
                continue
            
            if current_entity == end_id:
                # Found a path
                path_details = []
                for i in range(len(current_path)):
                    entity_info = {
                        'entity_id': current_path[i],
                        'text': graph[current_path[i]]['entity']['text'],
                        'type': graph[current_path[i]]['entity']['entity_type']
                    }
                    path_details.append(entity_info)
                    
                    if i < len(relation_path):
                        path_details.append({
                            'relation_type': relation_path[i]['relation_type'],
                            'confidence': relation_path[i]['confidence']
                        })
                
                paths.append(path_details)
                continue
            
            # Explore neighbors
            for outgoing in graph[current_entity]['outgoing']:
                target_id = outgoing['target_id']
                
                if target_id not in current_path:  # Avoid cycles
                    new_path = current_path + [target_id]
                    new_relations = relation_path + [outgoing]
                    queue.append((new_path, new_relations))
        
        print(f"✅ Found {len(paths)} multi-hop paths")
        return paths
    
    def generate_reasoning_explanation(self, paths: List[List[Dict]], 
                                     start_entity: str, end_entity: str) -> str:
        """Generate natural language explanation of multi-hop reasoning"""
        
        if not paths:
            return f"No reasoning path found between '{start_entity}' and '{end_entity}'."
        
        explanation = f"Multi-hop reasoning from '{start_entity}' to '{end_entity}':\\n\\n"
        
        for i, path in enumerate(paths, 1):
            explanation += f"Path {i} ({len([x for x in path if 'entity_id' in x])} hops):\\n"
            
            entities = [x for x in path if 'entity_id' in x]
            relations = [x for x in path if 'relation_type' in x]
            
            for j, entity in enumerate(entities):
                explanation += f"  • {entity['text']} ({entity['type']})"
                
                if j < len(relations):
                    rel = relations[j]
                    explanation += f" --[{rel['relation_type']}]--> "
                else:
                    explanation += "\\n"
            
            explanation += "\\n"
        
        return explanation
    
    def demonstrate_context_engineering(self, reasoning_text: str) -> str:
        """Demonstrate context engineering vs simple prompting"""
        
        print("🎯 Demonstrating context engineering approach...")
        
        # Context-aware prompt (vs simple prompt)
        context_prompt = f"""
        You are analyzing maintenance knowledge graphs with multi-hop reasoning capabilities.
        
        Context: Industrial maintenance domain with equipment, components, issues, and procedures.
        Task: Provide actionable maintenance insights based on the reasoning path.
        
        Reasoning Analysis:
        {reasoning_text}
        
        Please provide:
        1. Key maintenance insights from this reasoning path
        2. Potential preventive actions
        3. Risk assessment based on the relationships
        
        Focus on practical maintenance value, not just entity connections.
        """
        
        try:
            response = self.openai_client.get_completion(context_prompt)
            return response
        except Exception as e:
            return f"Context engineering demo error: {e}"
    
    def run_complete_demo(self) -> Dict[str, Any]:
        """Run complete multi-hop reasoning demonstration"""
        
        print("🚀 MULTI-HOP REASONING DEMONSTRATION")
        print("=" * 60)
        
        demo_results = {
            "timestamp": time.time(),
            "steps": [],
            "entities_processed": 0,
            "relationships_processed": 0,
            "paths_found": 0,
            "reasoning_examples": []
        }
        
        try:
            # Step 1: Load demo data
            entities, relationships = self.load_demo_data_direct()
            demo_results["entities_processed"] = len(entities)
            demo_results["relationships_processed"] = len(relationships)
            demo_results["steps"].append("✅ Demo data loaded from quality dataset")
            
            # Step 2: Create knowledge graph
            graph = self.create_entity_graph(entities, relationships)
            demo_results["steps"].append("✅ In-memory knowledge graph created")
            
            # Step 3: Multi-hop reasoning examples
            reasoning_examples = [
                ("air conditioner", "thermostat"),
                ("filter", "maintenance"),
                ("pump", "failure")
            ]
            
            for start_entity, end_entity in reasoning_examples:
                print(f"\\n{'='*40}")
                
                # Find paths
                paths = self.find_multi_hop_paths(graph, start_entity, end_entity, max_hops=3)
                
                if paths:
                    demo_results["paths_found"] += len(paths)
                    
                    # Generate explanation
                    explanation = self.generate_reasoning_explanation(paths, start_entity, end_entity)
                    print(f"📝 Reasoning paths found:")
                    print(explanation)
                    
                    # Context engineering demonstration
                    context_insights = self.demonstrate_context_engineering(explanation)
                    print(f"🎯 Context-engineered insights:")
                    print(context_insights[:300] + "..." if len(context_insights) > 300 else context_insights)
                    
                    demo_results["reasoning_examples"].append({
                        "start_entity": start_entity,
                        "end_entity": end_entity,
                        "paths_found": len(paths),
                        "explanation": explanation,
                        "context_insights": context_insights[:500]  # Truncate for storage
                    })
                    
                    demo_results["steps"].append(f"✅ Multi-hop reasoning: {start_entity} → {end_entity}")
                else:
                    print(f"⚠️  No paths found between '{start_entity}' and '{end_entity}'")
                    demo_results["steps"].append(f"⚠️  No paths: {start_entity} → {end_entity}")
            
            demo_results["status"] = "completed"
            demo_results["steps"].append("✅ Multi-hop reasoning demonstration completed")
            
        except Exception as e:
            demo_results["status"] = "failed"
            demo_results["error"] = str(e)
            demo_results["steps"].append(f"❌ Demo failed: {e}")
        
        return demo_results


def main():
    """Execute Step 6: Multi-hop Reasoning Integration"""
    
    print("🚀 STEP 6: MULTI-HOP REASONING INTEGRATION")
    print("=" * 60)
    print("Demonstrating end-to-end multi-hop reasoning with quality dataset")
    
    try:
        # Initialize demo
        demo = MultiHopReasoningDemo()
        
        # Run complete demonstration
        results = demo.run_complete_demo()
        
        # Save results
        output_file = Path(__file__).parent.parent / "data/demo_outputs/multi_hop_reasoning_demo.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Final summary
        print(f"\\n{'='*80}")
        print("🎉 STEP 6 COMPLETED: MULTI-HOP REASONING DEMONSTRATION")
        print(f"{'='*80}")
        print(f"\\n📊 Demo Results:")
        print(f"   Status: {results['status']}")
        print(f"   Entities processed: {results['entities_processed']:,}")
        print(f"   Relationships processed: {results['relationships_processed']:,}")
        print(f"   Multi-hop paths found: {results['paths_found']}")
        print(f"   Reasoning examples: {len(results['reasoning_examples'])}")
        print(f"\\n📁 Results saved to: {output_file}")
        
        if results["status"] == "completed":
            print(f"\\n🚀 Ready for supervisor demo with real multi-hop reasoning!")
            return 0
        else:
            print(f"\\n⚠️  Demo completed with issues: {results.get('error', 'Unknown error')}")
            return 1
        
    except Exception as e:
        print(f"❌ Step 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())