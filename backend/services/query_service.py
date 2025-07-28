"""
Query Service
Handles all query processing and response generation
Consolidates: search operations, query analysis, response synthesis, RAG workflows
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from core.azure_unified import UnifiedAzureOpenAIClient, UnifiedSearchClient, UnifiedCosmosClient
from .graph_service import GraphService

logger = logging.getLogger(__name__)


class QueryService:
    """High-level service for query processing and response generation"""
    
    def __init__(self):
        self.openai_client = UnifiedAzureOpenAIClient()
        self.search_client = UnifiedSearchClient()
        self.cosmos_client = UnifiedCosmosClient()
        self.graph_service = GraphService()
        
    # === MAIN QUERY PROCESSING ===
    
    async def process_universal_query(self, query: str, domain: str = "maintenance", 
                                    max_results: int = 10) -> Dict[str, Any]:
        """Process query using full RAG pipeline"""
        try:
            start_time = datetime.now()
            
            # Step 1: Analyze query
            query_analysis = self.search_client.analyze_query(query)
            enhanced_query = query_analysis.get('enhanced_query', query)
            
            # Step 2: Multi-source retrieval
            retrieval_tasks = [
                self._search_documents(enhanced_query, max_results),
                self._search_knowledge_graph(query, domain),
                self._find_related_entities(query, domain)
            ]
            
            search_results, graph_results, entity_results = await asyncio.gather(
                *retrieval_tasks, return_exceptions=True
            )
            
            # Step 3: Consolidate context
            context = self._consolidate_context(search_results, graph_results, entity_results)
            
            # Step 4: Generate response
            response = await self._generate_response(query, context, domain)
            
            # Step 5: Create final result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'operation': 'process_universal_query',
                'data': {
                    'query': query,
                    'domain': domain,
                    'response': response,
                    'context_sources': len(context['sources']),
                    'processing_time': processing_time,
                    'query_analysis': query_analysis,
                    'timestamp': start_time.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Universal query processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'process_universal_query'
            }
    
    async def semantic_search(self, query: str, search_type: str = "hybrid", 
                            filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform semantic search across multiple sources"""
        try:
            results = {}
            
            if search_type in ["hybrid", "documents"]:
                # Document search
                doc_results = await self.search_client.search_documents(
                    query, 
                    top=20, 
                    filters=filters.get('document_filters') if filters else None
                )
                results['documents'] = doc_results['data'] if doc_results['success'] else []
            
            if search_type in ["hybrid", "graph"]:
                # Graph search
                graph_results = await self._search_knowledge_graph(query, filters.get('domain', 'general'))
                results['graph'] = graph_results
            
            if search_type in ["hybrid", "entities"]:
                # Entity search
                entity_results = await self._find_related_entities(query, filters.get('domain', 'general'))
                results['entities'] = entity_results
            
            return {
                'success': True,
                'operation': 'semantic_search',
                'data': {
                    'query': query,
                    'search_type': search_type,
                    'results': results,
                    'total_sources': sum(len(v) if isinstance(v, list) else 1 for v in results.values())
                }
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'semantic_search'
            }
    
    # === SPECIALIZED QUERIES ===
    
    async def reasoning_query(self, start_concept: str, target_concept: str, 
                            max_hops: int = 3) -> Dict[str, Any]:
        """Multi-hop reasoning query between concepts"""
        try:
            # Find entities matching concepts
            start_entities = await self._find_entities_by_concept(start_concept)
            target_entities = await self._find_entities_by_concept(target_concept)
            
            if not start_entities or not target_entities:
                return {
                    'success': False,
                    'error': 'Could not find entities for given concepts',
                    'operation': 'reasoning_query'
                }
            
            # Find reasoning paths
            reasoning_result = await self.graph_service.find_reasoning_paths(
                start_entities[:3],  # Limit to top 3 matches
                target_entities[:3],
                max_hops=max_hops
            )
            
            if reasoning_result['success']:
                # Generate reasoning explanation
                explanation = await self._explain_reasoning_paths(
                    reasoning_result['data']['paths'],
                    start_concept,
                    target_concept
                )
                
                reasoning_result['data']['explanation'] = explanation
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Reasoning query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'reasoning_query'
            }
    
    async def maintenance_query(self, equipment: str, issue: str = None) -> Dict[str, Any]:
        """Specialized maintenance domain query"""
        try:
            # Build maintenance-specific query
            if issue:
                query = f"{equipment} {issue} maintenance repair troubleshooting"
            else:
                query = f"{equipment} maintenance procedures components"
            
            # Enhanced search with maintenance context
            results = await self.process_universal_query(query, domain="maintenance")
            
            if results['success']:
                # Add maintenance-specific analysis
                maintenance_analysis = self._analyze_maintenance_context(
                    equipment, issue, results['data']['response']
                )
                results['data']['maintenance_analysis'] = maintenance_analysis
            
            return results
            
        except Exception as e:
            logger.error(f"Maintenance query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'maintenance_query'
            }
    
    # === RETRIEVAL METHODS ===
    
    async def _search_documents(self, query: str, max_results: int) -> List[Dict]:
        """Search documents using Azure Cognitive Search"""
        try:
            result = await self.search_client.search_documents(query, top=max_results)
            return result['data']['documents'] if result['success'] else []
        except Exception as e:
            logger.warning(f"Document search failed: {e}")
            return []
    
    async def _search_knowledge_graph(self, query: str, domain: str) -> List[Dict]:
        """Search knowledge graph for relevant entities and relationships"""
        try:
            # Extract key terms from query for entity matching
            key_terms = self._extract_key_terms(query)
            
            graph_results = []
            for term in key_terms[:5]:  # Limit to top 5 terms
                # This would query Cosmos DB for entities matching the term
                # Simplified implementation
                entities = await self._find_entities_by_term(term)
                graph_results.extend(entities)
            
            return graph_results[:10]  # Limit results
            
        except Exception as e:
            logger.warning(f"Knowledge graph search failed: {e}")
            return []
    
    async def _find_related_entities(self, query: str, domain: str) -> List[Dict]:
        """Find entities related to query concepts"""
        try:
            # Extract concepts and find related entities
            concepts = self._extract_concepts(query, domain)
            
            related_entities = []
            for concept in concepts:
                entities = await self._find_entities_by_concept(concept)
                related_entities.extend(entities)
            
            return related_entities[:15]  # Limit results
            
        except Exception as e:
            logger.warning(f"Related entity search failed: {e}")
            return []
    
    # === CONTEXT AND RESPONSE GENERATION ===
    
    def _consolidate_context(self, search_results: List[Dict], graph_results: List[Dict], 
                           entity_results: List[Dict]) -> Dict[str, Any]:
        """Consolidate retrieval results into unified context"""
        context = {
            'sources': [],
            'entities': [],
            'relationships': [],
            'documents': []
        }
        
        # Process search results
        if isinstance(search_results, list):
            for doc in search_results:
                context['documents'].append({
                    'content': doc.get('content', ''),
                    'title': doc.get('title', ''),
                    'score': doc.get('score', 0),
                    'source_type': 'document'
                })
                context['sources'].append('document_search')
        
        # Process graph results
        if isinstance(graph_results, list):
            for entity in graph_results:
                context['entities'].append({
                    'text': entity.get('text', ''),
                    'type': entity.get('type', ''),
                    'source_type': 'knowledge_graph'
                })
                context['sources'].append('knowledge_graph')
        
        # Process entity results
        if isinstance(entity_results, list):
            context['entities'].extend(entity_results)
            context['sources'].extend(['entity_search'] * len(entity_results))
        
        return context
    
    async def _generate_response(self, query: str, context: Dict[str, Any], domain: str) -> str:
        """Generate response using Azure OpenAI with retrieved context"""
        try:
            # Build context-aware prompt
            context_text = self._build_context_text(context)
            
            prompt = f"""You are an expert assistant for {domain} domain queries.

Context from knowledge base:
{context_text}

User Query: {query}

Based on the retrieved context, provide a comprehensive and accurate response. 
If the context contains relevant information, use it to inform your answer.
If insufficient context is available, clearly state the limitations.

Response:"""
            
            response = await self.openai_client.get_completion(
                prompt, 
                model="gpt-4",
                temperature=0.7,
                max_tokens=500
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    def _build_context_text(self, context: Dict[str, Any]) -> str:
        """Build formatted context text for prompt"""
        context_parts = []
        
        # Add documents
        for i, doc in enumerate(context.get('documents', [])[:3]):  # Top 3 documents
            context_parts.append(f"Document {i+1}: {doc.get('content', '')[:200]}...")
        
        # Add entities
        entities = context.get('entities', [])[:10]  # Top 10 entities
        if entities:
            entity_text = ", ".join([e.get('text', '') for e in entities])
            context_parts.append(f"Related entities: {entity_text}")
        
        return "\\n\\n".join(context_parts)
    
    # === UTILITY METHODS ===
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Simple keyword extraction
        import re
        words = re.findall(r'\\b\\w+\\b', query.lower())
        
        # Filter common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        key_terms = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return key_terms[:10]  # Return top 10
    
    def _extract_concepts(self, query: str, domain: str) -> List[str]:
        """Extract domain-specific concepts from query"""
        concepts = []
        
        if domain == "maintenance":
            # Maintenance-specific concept extraction
            equipment_terms = ['pump', 'motor', 'valve', 'conditioner', 'thermostat', 'cylinder']
            issue_terms = ['broken', 'failed', 'leaking', 'not working', 'damaged']
            action_terms = ['repair', 'replace', 'fix', 'maintenance', 'service']
            
            query_lower = query.lower()
            
            for term in equipment_terms + issue_terms + action_terms:
                if term in query_lower:
                    concepts.append(term)
        else:
            # General concept extraction
            concepts = self._extract_key_terms(query)
        
        return concepts[:5]  # Return top 5 concepts
    
    async def _find_entities_by_term(self, term: str) -> List[Dict]:
        """Find entities matching a specific term"""
        # Simplified entity search - would query Cosmos DB in production
        return [
            {'text': term, 'type': 'unknown', 'score': 0.8}
        ]
    
    async def _find_entities_by_concept(self, concept: str) -> List[str]:
        """Find entity IDs matching a concept"""
        # Simplified - would query actual graph database
        return [f"entity_{concept}_1", f"entity_{concept}_2"]
    
    async def _explain_reasoning_paths(self, paths: List[Dict], start_concept: str, 
                                     target_concept: str) -> str:
        """Generate explanation of reasoning paths"""
        if not paths:
            return f"No reasoning paths found between {start_concept} and {target_concept}."
        
        explanation = f"Found {len(paths)} reasoning paths from {start_concept} to {target_concept}:\\n\\n"
        
        for i, path in enumerate(paths[:3]):  # Top 3 paths
            hops = path.get('hops', 0)
            explanation += f"Path {i+1}: {hops}-hop connection\\n"
        
        return explanation
    
    def _analyze_maintenance_context(self, equipment: str, issue: str, response: str) -> Dict[str, Any]:
        """Analyze maintenance-specific context"""
        return {
            'equipment_identified': equipment.lower() in response.lower(),
            'issue_addressed': issue.lower() in response.lower() if issue else False,
            'maintenance_keywords': len([word for word in ['repair', 'replace', 'fix', 'maintenance', 'service'] 
                                       if word in response.lower()]),
            'response_relevance': 'high' if equipment.lower() in response.lower() else 'medium'
        }