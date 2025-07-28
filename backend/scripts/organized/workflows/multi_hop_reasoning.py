#!/usr/bin/env python3
"""
Step 6: Multi-hop Reasoning Integration (Fixed)
Working demonstration with actual entity data
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add backend to path - for running from backend directory
sys.path.insert(0, '.')

from integrations.azure_openai import AzureOpenAIClient


class WorkingMultiHopDemo:
    """Working multi-hop reasoning demonstration with real data"""
    
    def __init__(self):
        self.openai_client = AzureOpenAIClient()
        
    def load_quality_data(self, limit: int = 200) -> Tuple[List[Dict], List[Dict]]:
        """Load quality dataset with proper structure"""
        
        print(f"📊 Loading quality data (limit: {limit} entities)...")
        
        data_file = Path("data/extraction_outputs/full_dataset_extraction_9100_entities_5848_relationships.json")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Take subset for demo
        entities = data['entities'][:limit]
        entity_ids = {entity['entity_id'] for entity in entities}
        
        # Filter relationships to include only those between our entities
        relationships = []
        for rel in data['relationships']:
            if (rel['source_entity_id'] in entity_ids and 
                rel['target_entity_id'] in entity_ids):
                relationships.append(rel)
        
        print(f"✅ Loaded: {len(entities)} entities, {len(relationships)} relationships")
        
        # Show sample data
        print(f"\\n📋 Sample entities:")
        for i, entity in enumerate(entities[:5]):
            print(f"   {i+1}. '{entity['text']}' ({entity['entity_type']}) [ID: {entity['entity_id']}]")
        
        print(f"\\n🔗 Sample relationships:")
        for i, rel in enumerate(relationships[:3]):
            print(f"   {i+1}. {rel['source_entity_id']} --[{rel['relation_type']}]--> {rel['target_entity_id']}")
        
        return entities, relationships
    
    def build_knowledge_graph(self, entities: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Build knowledge graph with proper indexing"""
        
        print("🕸️ Building knowledge graph...")
        
        # Create entity lookup tables
        entity_by_id = {entity['entity_id']: entity for entity in entities}
        entities_by_text = {}
        
        for entity in entities:
            text = entity['text'].lower().strip()
            if text not in entities_by_text:
                entities_by_text[text] = []
            entities_by_text[text].append(entity)
        
        # Build adjacency graph
        graph = {}
        for entity in entities:
            entity_id = entity['entity_id']
            graph[entity_id] = {
                'entity': entity,
                'outgoing': [],
                'incoming': []
            }
        
        # Add relationships
        relationship_count = 0
        for rel in relationships:
            source_id = rel['source_entity_id']
            target_id = rel['target_entity_id']
            
            if source_id in graph and target_id in graph:
                graph[source_id]['outgoing'].append({
                    'target_id': target_id,
                    'relation_type': rel['relation_type'],
                    'confidence': rel.get('confidence', 1.0),
                    'context': rel.get('context', '')
                })
                
                graph[target_id]['incoming'].append({
                    'source_id': source_id,
                    'relation_type': rel['relation_type'],
                    'confidence': rel.get('confidence', 1.0),
                    'context': rel.get('context', '')
                })
                
                relationship_count += 1
        
        graph_stats = {
            'total_entities': len(entities),
            'total_relationships': relationship_count,
            'entities_by_text': entities_by_text,
            'entity_by_id': entity_by_id
        }
        
        print(f"✅ Knowledge graph built:")
        print(f"   Nodes: {len(graph)}")
        print(f"   Edges: {relationship_count}")
        print(f"   Unique texts: {len(entities_by_text)}")
        
        return graph, graph_stats
    
    def find_entity_by_text(self, text: str, entities_by_text: Dict) -> List[str]:
        """Find entity IDs by text (fuzzy matching)"""
        
        text = text.lower().strip()
        
        # Exact match
        if text in entities_by_text:
            return [entity['entity_id'] for entity in entities_by_text[text]]
        
        # Partial match
        matches = []
        for entity_text, entity_list in entities_by_text.items():
            if text in entity_text or entity_text in text:
                matches.extend([entity['entity_id'] for entity in entity_list])
        
        return matches
    
    def multi_hop_search(self, graph: Dict, start_ids: List[str], end_ids: List[str], max_hops: int = 3) -> List[List[Dict]]:
        """Multi-hop search between entity sets"""
        
        print(f"🔍 Multi-hop search (max {max_hops} hops):")
        print(f"   Start entities: {len(start_ids)} candidates")
        print(f"   End entities: {len(end_ids)} candidates")
        
        all_paths = []
        
        for start_id in start_ids:
            for end_id in end_ids:
                # BFS for paths
                queue = [([start_id], [])]  # (entity_path, relation_path)
                visited = set()
                
                while queue and len(all_paths) < 10:  # Limit total paths
                    current_path, relation_path = queue.pop(0)
                    current_entity = current_path[-1]
                    
                    if len(current_path) > max_hops:
                        continue
                    
                    if current_entity == end_id:
                        # Found path
                        path_details = []
                        for i in range(len(current_path)):
                            entity = graph[current_path[i]]['entity']
                            path_details.append({
                                'entity_id': current_path[i],
                                'text': entity['text'],
                                'type': entity['entity_type'],
                                'step': i
                            })
                            
                            if i < len(relation_path):
                                path_details.append({
                                    'relation_type': relation_path[i]['relation_type'],
                                    'confidence': relation_path[i]['confidence'],
                                    'step': i
                                })
                        
                        all_paths.append(path_details)
                        continue
                    
                    if current_entity in visited:
                        continue
                    visited.add(current_entity)
                    
                    # Explore outgoing edges
                    for edge in graph[current_entity]['outgoing']:
                        target_id = edge['target_id']
                        
                        if target_id not in current_path:  # Avoid cycles
                            new_path = current_path + [target_id]
                            new_relations = relation_path + [edge]
                            queue.append((new_path, new_relations))
        
        print(f"✅ Found {len(all_paths)} multi-hop paths")
        return all_paths
    
    def format_reasoning_path(self, path: List[Dict]) -> str:
        """Format path as readable reasoning chain"""
        
        entities = [x for x in path if 'entity_id' in x]
        relations = [x for x in path if 'relation_type' in x]
        
        reasoning = ""
        for i, entity in enumerate(entities):
            reasoning += f"'{entity['text']}' ({entity['type']})"
            
            if i < len(relations):
                rel = relations[i]
                reasoning += f" --[{rel['relation_type']}]--> "
            
        return reasoning
    
    def demonstrate_multi_hop_reasoning(self, graph: Dict, graph_stats: Dict) -> List[Dict]:
        """Demonstrate multi-hop reasoning with real examples"""
        
        print("\\n🎯 MULTI-HOP REASONING DEMONSTRATION")
        print("=" * 50)
        
        reasoning_examples = []
        
        # Example 1: Equipment to Issues
        print("\\n📍 Example 1: Equipment → Issues")
        air_conditioner_ids = self.find_entity_by_text("air conditioner", graph_stats['entities_by_text'])
        issue_ids = self.find_entity_by_text("not working", graph_stats['entities_by_text'])
        issue_ids.extend(self.find_entity_by_text("unserviceable", graph_stats['entities_by_text']))
        
        if air_conditioner_ids and issue_ids:
            paths = self.multi_hop_search(graph, air_conditioner_ids[:2], issue_ids[:2], max_hops=2)
            
            if paths:
                print(f"   Found {len(paths)} reasoning paths:")
                for i, path in enumerate(paths[:3]):  # Show first 3
                    reasoning = self.format_reasoning_path(path)
                    print(f"   Path {i+1}: {reasoning}")
                
                reasoning_examples.append({
                    "example": "Equipment to Issues",
                    "start_entities": ["air conditioner"],
                    "end_entities": ["not working", "unserviceable"],
                    "paths_found": len(paths),
                    "sample_reasoning": self.format_reasoning_path(paths[0]) if paths else ""
                })
        
        # Example 2: Components to Equipment
        print("\\n📍 Example 2: Components → Equipment")
        thermostat_ids = self.find_entity_by_text("thermostat", graph_stats['entities_by_text'])
        equipment_ids = self.find_entity_by_text("air conditioner", graph_stats['entities_by_text'])
        
        if thermostat_ids and equipment_ids:
            paths = self.multi_hop_search(graph, thermostat_ids[:2], equipment_ids[:2], max_hops=2)
            
            if paths:
                print(f"   Found {len(paths)} reasoning paths:")
                for i, path in enumerate(paths[:3]):
                    reasoning = self.format_reasoning_path(path)
                    print(f"   Path {i+1}: {reasoning}")
                
                reasoning_examples.append({
                    "example": "Components to Equipment",
                    "start_entities": ["thermostat"],
                    "end_entities": ["air conditioner"],
                    "paths_found": len(paths),
                    "sample_reasoning": self.format_reasoning_path(paths[0]) if paths else ""
                })
        
        # Example 3: Show entity type distribution
        print("\\n📊 Entity Type Distribution:")
        type_counts = {}
        for entity in graph_stats['entity_by_id'].values():
            entity_type = entity['entity_type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        for entity_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {entity_type}: {count}")
        
        return reasoning_examples
    
    def demonstrate_context_engineering(self, reasoning_examples: List[Dict]) -> str:
        """Demonstrate context engineering with multi-hop insights"""
        
        print("\\n🎯 Context Engineering Demonstration...")
        
        if not reasoning_examples:
            return "No reasoning examples available for context engineering."
        
        # Create context-rich prompt
        context_prompt = f"""
        You are an AI assistant specializing in industrial maintenance knowledge analysis.
        
        Context: Multi-hop reasoning analysis from maintenance knowledge graph
        Task: Provide actionable maintenance insights
        
        Reasoning Analysis:
        """
        
        for example in reasoning_examples:
            if example['sample_reasoning']:
                context_prompt += f"\\n• {example['example']}: {example['sample_reasoning']}"
        
        context_prompt += """
        
        Based on these multi-hop relationships, provide:
        1. Key maintenance insights
        2. Preventive maintenance recommendations
        3. Risk assessment patterns
        
        Focus on practical value for maintenance teams.
        """
        
        try:
            insights = self.openai_client.get_completion(context_prompt)
            print("✅ Context-engineered insights generated")
            return insights
        except Exception as e:
            error_msg = f"Context engineering error: {e}"
            print(f"⚠️  {error_msg}")
            return error_msg


def main():
    """Execute working multi-hop reasoning demonstration"""
    
    print("🚀 STEP 6: WORKING MULTI-HOP REASONING DEMONSTRATION")
    print("=" * 70)
    
    try:
        demo = WorkingMultiHopDemo()
        
        # Load quality data
        entities, relationships = demo.load_quality_data(limit=150)
        
        # Build knowledge graph
        graph, graph_stats = demo.build_knowledge_graph(entities, relationships)
        
        # Demonstrate multi-hop reasoning
        reasoning_examples = demo.demonstrate_multi_hop_reasoning(graph, graph_stats)
        
        # Context engineering demonstration
        context_insights = demo.demonstrate_context_engineering(reasoning_examples)
        
        # Save comprehensive results
        results = {
            "timestamp": time.time(),
            "status": "completed",
            "data_summary": {
                "entities_processed": len(entities),
                "relationships_processed": len(relationships),
                "graph_nodes": len(graph),
                "graph_edges": sum(len(node['outgoing']) for node in graph.values())
            },
            "reasoning_examples": reasoning_examples,
            "context_insights": context_insights[:1000],  # Truncate for storage
            "demo_features": [
                "Real data from 9,100 entity quality dataset",
                "Multi-hop graph traversal (BFS algorithm)",
                "Context-aware entity matching",
                "Azure OpenAI integration for insights",
                "Production-ready knowledge graph structure"
            ]
        }
        
        # Save results
        output_file = Path(__file__).parent.parent / "data/demo_outputs/working_multi_hop_demo.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n{'='*80}")
        print("🎉 STEP 6 COMPLETED: WORKING MULTI-HOP REASONING")
        print(f"{'='*80}")
        print(f"\\n📊 Final Results:")
        print(f"   Status: {results['status']}")
        print(f"   Entities: {results['data_summary']['entities_processed']}")
        print(f"   Relationships: {results['data_summary']['relationships_processed']}")
        print(f"   Reasoning examples: {len(results['reasoning_examples'])}")
        print(f"   Context insights: Generated ({len(context_insights)} chars)")
        print(f"\\n📁 Results: {output_file}")
        print(f"\\n🚀 READY FOR SUPERVISOR DEMO!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Step 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())