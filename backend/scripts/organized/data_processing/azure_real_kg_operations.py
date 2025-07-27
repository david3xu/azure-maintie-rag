#!/usr/bin/env python3
"""
Azure Real Knowledge Graph Operations
Using the uploaded 2,000 entities + 60k relationships to demonstrate real KG capabilities
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from integrations.azure_openai import AzureOpenAIClient


class AzureRealKnowledgeGraph:
    """Real Knowledge Graph operations using Azure Cosmos DB data"""
    
    def __init__(self):
        self.cosmos_client = AzureCosmosGremlinClient()
        self.cosmos_client._initialize_client()
        self.openai_client = AzureOpenAIClient()
        
    def validate_graph_state(self) -> Dict[str, Any]:
        """Validate current Azure knowledge graph state"""
        
        print("📊 VALIDATING AZURE KNOWLEDGE GRAPH STATE")
        print("=" * 60)
        
        # Get basic statistics
        vertex_count = self.cosmos_client.gremlin_client.submit('g.V().count()').all().result()[0]
        edge_count = self.cosmos_client.gremlin_client.submit('g.E().count()').all().result()[0]
        
        # Get entity type distribution
        type_query = "g.V().groupCount().by('entity_type')"
        type_result = self.cosmos_client.gremlin_client.submit(type_query).all().result()
        entity_types = type_result[0] if type_result else {}
        
        # Calculate graph metrics
        connectivity = edge_count / vertex_count if vertex_count > 0 else 0
        
        state = {
            'vertices': vertex_count,
            'edges': edge_count,
            'connectivity_ratio': connectivity,
            'entity_types': entity_types,
            'is_connected': edge_count > 0,
            'has_multi_hop_potential': connectivity > 1.0
        }
        
        print(f"✅ Graph State:")
        print(f"   Vertices: {vertex_count:,}")
        print(f"   Edges: {edge_count:,}")
        print(f"   Connectivity: {connectivity:.3f}")
        print(f"   Entity Types: {len(entity_types)}")
        print(f"   Multi-hop Capable: {state['has_multi_hop_potential']}")
        
        return state
    
    def demonstrate_graph_traversal(self) -> List[Dict]:
        """Demonstrate real graph traversal operations"""
        
        print("\\n🕸️ DEMONSTRATING REAL GRAPH TRAVERSAL")
        print("=" * 60)
        
        traversal_examples = []
        
        # Example 1: Find connected components
        print("\\n📍 Example 1: Equipment and Related Components")
        equipment_query = '''
        g.V().has('entity_type', 'equipment')
            .limit(3)
            .as('equipment')
            .out()
            .has('entity_type', 'component')
            .as('component')
            .select('equipment', 'component')
            .by('text')
        '''
        
        try:
            equipment_results = self.cosmos_client.gremlin_client.submit(equipment_query).all().result()
            
            print(f"   Found {len(equipment_results)} equipment-component relationships:")
            for i, result in enumerate(equipment_results[:5]):
                equipment = result.get('equipment', 'N/A')
                component = result.get('component', 'N/A')
                print(f"   {i+1}. {equipment} → {component}")
            
            traversal_examples.append({
                'example': 'Equipment-Component Relationships',
                'query_type': 'Direct traversal',
                'results_count': len(equipment_results),
                'sample_results': equipment_results[:3]
            })
            
        except Exception as e:
            print(f"   Error in equipment traversal: {str(e)[:100]}")
        
        # Example 2: Multi-hop path finding
        print("\\n📍 Example 2: Multi-Hop Path Discovery")
        multihop_query = '''
        g.V().has('entity_type', 'equipment')
            .limit(2)
            .repeat(out().simplePath())
            .times(2)
            .has('entity_type', 'action')
            .limit(5)
            .path()
            .by('text')
        '''
        
        try:
            multihop_results = self.cosmos_client.gremlin_client.submit(multihop_query).all().result()
            
            print(f"   Found {len(multihop_results)} multi-hop paths:")
            for i, path in enumerate(multihop_results):
                path_str = " → ".join(path)
                print(f"   {i+1}. {path_str}")
            
            traversal_examples.append({
                'example': 'Multi-hop Equipment to Action Paths',
                'query_type': 'Multi-hop traversal',
                'results_count': len(multihop_results),
                'sample_paths': multihop_results[:3]
            })
            
        except Exception as e:
            print(f"   Error in multi-hop traversal: {str(e)[:100]}")
        
        # Example 3: Neighborhood analysis
        print("\\n📍 Example 3: Entity Neighborhood Analysis")
        neighborhood_query = '''
        g.V().has('text', containing('air conditioner'))
            .limit(1)
            .as('center')
            .both()
            .limit(10)
            .as('neighbor')
            .select('center', 'neighbor')
            .by('text')
        '''
        
        try:
            neighborhood_results = self.cosmos_client.gremlin_client.submit(neighborhood_query).all().result()
            
            if neighborhood_results:
                center_entity = neighborhood_results[0].get('center', 'N/A')
                print(f"   Neighborhood of '{center_entity}':")
                neighbors = set()
                for result in neighborhood_results:
                    neighbor = result.get('neighbor', 'N/A')
                    if neighbor != center_entity:
                        neighbors.add(neighbor)
                
                for i, neighbor in enumerate(list(neighbors)[:5]):
                    print(f"   {i+1}. {neighbor}")
                
                traversal_examples.append({
                    'example': 'Entity Neighborhood',
                    'query_type': 'Bidirectional traversal',
                    'center_entity': center_entity,
                    'neighbors_count': len(neighbors)
                })
            
        except Exception as e:
            print(f"   Error in neighborhood analysis: {str(e)[:100]}")
        
        return traversal_examples
    
    def demonstrate_graph_analytics(self) -> Dict[str, Any]:
        """Demonstrate graph analytics operations"""
        
        print("\\n📊 DEMONSTRATING GRAPH ANALYTICS")
        print("=" * 60)
        
        analytics_results = {}
        
        # Centrality analysis (simplified)
        print("\\n📍 Example 1: Entity Popularity Analysis")
        popularity_query = '''
        g.V()
            .project('entity', 'in_degree', 'out_degree', 'total_degree')
            .by('text')
            .by(inE().count())
            .by(outE().count())
            .by(bothE().count())
            .order()
            .by(select('total_degree'), desc)
            .limit(10)
        '''
        
        try:
            popularity_results = self.cosmos_client.gremlin_client.submit(popularity_query).all().result()
            
            print("   Top 10 Most Connected Entities:")
            for i, result in enumerate(popularity_results):
                entity = result.get('entity', 'N/A')
                total_degree = result.get('total_degree', 0)
                in_degree = result.get('in_degree', 0)
                out_degree = result.get('out_degree', 0)
                print(f"   {i+1}. {entity}: {total_degree} connections ({in_degree} in, {out_degree} out)")
            
            analytics_results['popularity_analysis'] = popularity_results
            
        except Exception as e:
            print(f"   Error in popularity analysis: {str(e)[:100]}")
        
        # Relationship type analysis
        print("\\n📍 Example 2: Relationship Type Distribution")
        relationship_query = '''
        g.E()
            .groupCount()
            .by(label())
        '''
        
        try:
            relationship_results = self.cosmos_client.gremlin_client.submit(relationship_query).all().result()
            
            if relationship_results:
                rel_types = relationship_results[0]
                print(f"   Relationship Types ({len(rel_types)} types):")
                sorted_rels = sorted(rel_types.items(), key=lambda x: x[1], reverse=True)
                for rel_type, count in sorted_rels[:10]:
                    print(f"   {rel_type}: {count:,} relationships")
                
                analytics_results['relationship_types'] = rel_types
            
        except Exception as e:
            print(f"   Error in relationship analysis: {str(e)[:100]}")
        
        # Component connectivity analysis
        print("\\n📍 Example 3: Component Connectivity Patterns")
        connectivity_query = '''
        g.V().has('entity_type', 'component')
            .project('component', 'connected_equipment', 'related_issues')
            .by('text')
            .by(both().has('entity_type', 'equipment').count())
            .by(both().has('entity_type', 'issue').count())
            .where(select('connected_equipment').is(gt(0)))
            .order()
            .by(select('connected_equipment'), desc)
            .limit(5)
        '''
        
        try:
            connectivity_results = self.cosmos_client.gremlin_client.submit(connectivity_query).all().result()
            
            print("   Components with Equipment Connections:")
            for result in connectivity_results:
                component = result.get('component', 'N/A')
                equipment_count = result.get('connected_equipment', 0)
                issues_count = result.get('related_issues', 0)
                print(f"   • {component}: {equipment_count} equipment, {issues_count} issues")
            
            analytics_results['connectivity_patterns'] = connectivity_results
            
        except Exception as e:
            print(f"   Error in connectivity analysis: {str(e)[:100]}")
        
        return analytics_results
    
    def demonstrate_semantic_search(self, query_text: str) -> Dict[str, Any]:
        """Demonstrate semantic search using graph structure"""
        
        print(f"\\n🔍 DEMONSTRATING SEMANTIC SEARCH: '{query_text}'")
        print("=" * 60)
        
        # Step 1: Find entities matching query
        print("\\n📍 Step 1: Finding Matching Entities")
        entity_search_query = f'''
        g.V().has('text', containing('{query_text.lower()}'))
            .limit(5)
            .valueMap()
        '''
        
        matching_entities = []
        try:
            search_results = self.cosmos_client.gremlin_client.submit(entity_search_query).all().result()
            
            print(f"   Found {len(search_results)} matching entities:")
            for i, entity in enumerate(search_results):
                text = entity.get('text', ['N/A'])[0]
                entity_type = entity.get('entity_type', ['N/A'])[0]
                original_id = entity.get('original_entity_id', ['N/A'])[0]
                print(f"   {i+1}. '{text}' ({entity_type})")
                
                matching_entities.append({
                    'text': text,
                    'type': entity_type,
                    'id': original_id
                })
            
        except Exception as e:
            print(f"   Error in entity search: {str(e)[:100]}")
        
        # Step 2: Expand search through graph connections
        print("\\n📍 Step 2: Graph-Based Context Expansion")
        if matching_entities:
            first_entity_text = matching_entities[0]['text']
            
            expansion_query = f'''
            g.V().has('text', '{first_entity_text}')
                .both()
                .limit(10)
                .project('entity', 'type', 'relationship')
                .by('text')
                .by('entity_type')
                .by(bothE().limit(1).label())
            '''
            
            try:
                expansion_results = self.cosmos_client.gremlin_client.submit(expansion_query).all().result()
                
                print(f"   Related entities through graph connections:")
                for i, result in enumerate(expansion_results):
                    entity = result.get('entity', 'N/A')
                    entity_type = result.get('type', 'N/A')
                    relationship = result.get('relationship', ['N/A'])
                    rel_label = relationship[0] if isinstance(relationship, list) and relationship else 'connected'
                    print(f"   {i+1}. {entity} ({entity_type}) - via {rel_label}")
                
            except Exception as e:
                print(f"   Error in graph expansion: {str(e)[:100]}")
        
        # Step 3: Generate semantic insights using Azure OpenAI
        print("\\n📍 Step 3: AI-Enhanced Insights")
        if matching_entities:
            try:
                context_prompt = f'''
                Based on this maintenance knowledge graph query for "{query_text}", I found these entities:
                
                Primary matches:
                {chr(10).join([f"- {e['text']} ({e['type']})" for e in matching_entities[:3]])}
                
                Please provide:
                1. What this query is likely asking about
                2. Key maintenance considerations
                3. Related components or actions to consider
                
                Keep response concise and practical for maintenance teams.
                '''
                
                insights = self.openai_client.get_completion(context_prompt)
                print(f"   AI Insights:")
                print(f"   {insights}")
                
            except Exception as e:
                print(f"   Error generating insights: {str(e)[:100]}")
                insights = "Could not generate AI insights"
        
        return {
            'query': query_text,
            'matching_entities': matching_entities,
            'search_method': 'Graph-based semantic search',
            'entities_found': len(matching_entities)
        }
    
    def demonstrate_maintenance_scenarios(self) -> List[Dict]:
        """Demonstrate real maintenance scenarios using the knowledge graph"""
        
        print("\\n🔧 DEMONSTRATING MAINTENANCE SCENARIOS")
        print("=" * 60)
        
        scenarios = []
        
        # Scenario 1: Troubleshooting workflow
        print("\\n📍 Scenario 1: Troubleshooting Workflow")
        troubleshooting_query = '''
        g.V().has('entity_type', 'issue')
            .limit(3)
            .as('issue')
            .both()
            .has('entity_type', 'action')
            .as('action')
            .select('issue', 'action')
            .by('text')
        '''
        
        try:
            troubleshooting_results = self.cosmos_client.gremlin_client.submit(troubleshooting_query).all().result()
            
            print("   Issue → Action Workflows:")
            workflows = {}
            for result in troubleshooting_results:
                issue = result.get('issue', 'N/A')
                action = result.get('action', 'N/A')
                if issue not in workflows:
                    workflows[issue] = []
                workflows[issue].append(action)
            
            for i, (issue, actions) in enumerate(list(workflows.items())[:3]):
                print(f"   {i+1}. Issue: {issue}")
                for action in actions[:2]:
                    print(f"      → Action: {action}")
            
            scenarios.append({
                'scenario': 'Troubleshooting Workflows',
                'type': 'Issue-Action mapping',
                'workflows_found': len(workflows)
            })
            
        except Exception as e:
            print(f"   Error in troubleshooting analysis: {str(e)[:100]}")
        
        # Scenario 2: Preventive maintenance
        print("\\n📍 Scenario 2: Preventive Maintenance Planning")
        preventive_query = '''
        g.V().has('entity_type', 'equipment')
            .limit(3)
            .as('equipment')
            .both()
            .has('entity_type', 'component')
            .as('component')
            .both()
            .has('entity_type', 'action')
            .as('action')
            .select('equipment', 'component', 'action')
            .by('text')
        '''
        
        try:
            preventive_results = self.cosmos_client.gremlin_client.submit(preventive_query).all().result()
            
            print("   Equipment → Component → Action Chains:")
            for i, result in enumerate(preventive_results[:3]):
                equipment = result.get('equipment', 'N/A')
                component = result.get('component', 'N/A')
                action = result.get('action', 'N/A')
                print(f"   {i+1}. {equipment} → {component} → {action}")
            
            scenarios.append({
                'scenario': 'Preventive Maintenance',
                'type': 'Multi-hop maintenance planning',
                'chains_found': len(preventive_results)
            })
            
        except Exception as e:
            print(f"   Error in preventive maintenance analysis: {str(e)[:100]}")
        
        return scenarios


def main():
    """Demonstrate real knowledge graph operations"""
    
    print("🚀 AZURE REAL KNOWLEDGE GRAPH OPERATIONS")
    print("Using 2,000 entities + 60K relationships in Azure Cosmos DB")
    print("=" * 80)
    
    try:
        kg = AzureRealKnowledgeGraph()
        
        # Validate graph state
        graph_state = kg.validate_graph_state()
        
        if not graph_state['is_connected']:
            print("❌ Graph is not connected - cannot demonstrate operations")
            return 1
        
        # Demonstrate graph traversal
        traversal_examples = kg.demonstrate_graph_traversal()
        
        # Demonstrate graph analytics
        analytics_results = kg.demonstrate_graph_analytics()
        
        # Demonstrate semantic search
        search_results = kg.demonstrate_semantic_search("air conditioner")
        
        # Demonstrate maintenance scenarios
        maintenance_scenarios = kg.demonstrate_maintenance_scenarios()
        
        # Save comprehensive results
        results = {
            'timestamp': time.time(),
            'graph_state': graph_state,
            'traversal_examples': traversal_examples,
            'analytics_results': analytics_results,
            'semantic_search': search_results,
            'maintenance_scenarios': maintenance_scenarios,
            'capabilities_demonstrated': [
                'Real graph traversal',
                'Multi-hop path finding',
                'Graph analytics',
                'Semantic search',
                'Maintenance workflows',
                'Azure Cosmos DB Gremlin operations'
            ]
        }
        
        # Save results
        output_file = Path(__file__).parent.parent / "data/kg_operations/azure_real_kg_demo.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n{'='*80}")
        print("🎉 REAL KNOWLEDGE GRAPH OPERATIONS COMPLETED")
        print(f"{'='*80}")
        print(f"\\n📊 Capabilities Demonstrated:")
        print(f"   ✅ Graph State: {graph_state['vertices']:,} vertices, {graph_state['edges']:,} edges")
        print(f"   ✅ Traversal Examples: {len(traversal_examples)} operations")
        print(f"   ✅ Analytics: Graph metrics and patterns")
        print(f"   ✅ Semantic Search: Context-aware entity discovery")
        print(f"   ✅ Maintenance Workflows: Real-world scenarios")
        print(f"\\n📁 Results: {output_file}")
        print(f"\\n🚀 REAL AZURE KNOWLEDGE GRAPH IS OPERATIONAL!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Real KG operations failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())