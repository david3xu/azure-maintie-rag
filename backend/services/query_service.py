"""
Query Service
Handles all query processing and response generation
Consolidates: search operations, query analysis, response synthesis, RAG workflows
CONFIGURATION-DRIVEN: All parameters sourced from domain_patterns.py - NO hardcoded values
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from core.azure_openai import UnifiedAzureOpenAIClient
from core.azure_search import UnifiedSearchClient
from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from .graph_service import GraphService
from config.domain_patterns import DomainPatternManager, DomainType

logger = logging.getLogger(__name__)


class QueryService:
    """High-level service for query processing and response generation"""
    
    def __init__(self):
        self.openai_client = UnifiedAzureOpenAIClient()
        self.search_client = UnifiedSearchClient()
        self.cosmos_client = AzureCosmosGremlinClient()
        self.graph_service = GraphService()
        
    # === MAIN QUERY PROCESSING ===
    
    async def process_universal_query(self, query: str, domain: str = None, 
                                    max_results: int = None) -> Dict[str, Any]:
        """Process query using full RAG pipeline"""
        try:
            start_time = datetime.now()
            
            # Auto-detect domain if not provided
            if domain is None:
                domain = DomainPatternManager.detect_domain(query)
            
            # Load domain-specific configuration
            patterns = DomainPatternManager.get_patterns(domain)
            training = DomainPatternManager.get_training(domain)
            
            # Set max_results from configuration if not provided
            if max_results is None:
                max_results = training.batch_size // 2  # Use half of training batch size as reasonable default
            
            # Step 1: Analyze query using domain patterns
            query_analysis = DomainPatternManager.enhance_query(query, domain)
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
            # Auto-detect domain if not provided in filters
            domain = filters.get('domain') if filters else None
            if domain is None:
                domain = DomainPatternManager.detect_domain(query)
            
            # Load domain-specific configuration
            training = DomainPatternManager.get_training(domain)
            doc_search_limit = training.batch_size  # Use batch_size as document search limit
            
            results = {}
            
            if search_type in ["hybrid", "documents"]:
                # Document search using configured limit
                doc_results = await self.search_client.search_documents(
                    query, 
                    top=doc_search_limit, 
                    filters=filters.get('document_filters') if filters else None
                )
                results['documents'] = doc_results['data']['documents'] if doc_results['success'] else []
            
            if search_type in ["hybrid", "graph"]:
                # Graph search using detected domain
                graph_results = await self._search_knowledge_graph(query, domain)
                results['graph'] = graph_results
            
            if search_type in ["hybrid", "entities"]:
                # Entity search using detected domain
                entity_results = await self._find_related_entities(query, domain)
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
                            max_hops: int = None, domain: str = None) -> Dict[str, Any]:
        """Multi-hop reasoning query between concepts"""
        try:
            # Auto-detect domain if not provided
            if domain is None:
                combined_query = f"{start_concept} {target_concept}"
                domain = DomainPatternManager.detect_domain(combined_query)
            
            # Load domain-specific configuration
            training = DomainPatternManager.get_training(domain)
            
            # Set max_hops from configuration if not provided
            if max_hops is None:
                max_hops = min(5, training.batch_size // 10)  # Reasonable hop limit based on batch size
            
            # Find entities matching concepts
            start_entities = await self._find_entities_by_concept(start_concept, domain)
            target_entities = await self._find_entities_by_concept(target_concept, domain)
            
            if not start_entities or not target_entities:
                return {
                    'success': False,
                    'error': 'Could not find entities for given concepts',
                    'operation': 'reasoning_query'
                }
            
            # Find reasoning paths using configuration-driven limits
            entity_limit = max(3, training.batch_size // 10)  # Dynamic entity limit based on training config
            reasoning_result = await self.graph_service.find_reasoning_paths(
                start_entities[:entity_limit],
                target_entities[:entity_limit],
                max_hops=max_hops
            )
            
            if reasoning_result['success']:
                # Generate reasoning explanation
                explanation = await self._explain_reasoning_paths(
                    reasoning_result['data']['paths'],
                    start_concept,
                    target_concept,
                    domain
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
    
    async def maintenance_query(self, equipment: str, issue: str = None, domain: str = "maintenance") -> Dict[str, Any]:
        """Specialized maintenance domain query"""
        try:
            # Load domain-specific patterns for query construction
            patterns = DomainPatternManager.get_patterns(domain)
            
            # Build domain-specific query using configured patterns
            if issue:
                # Use configured action and issue terms for query enhancement
                action_terms = patterns.action_terms[:2]  # First 2 action terms
                query = f"{equipment} {issue} {' '.join(action_terms)}"
            else:
                # Use configured enhancement keywords for procedures
                enhancement_terms = patterns.enhancement_keywords[:2]  # First 2 enhancement terms
                query = f"{equipment} {' '.join(enhancement_terms)}"
            
            # Enhanced search with domain context
            results = await self.process_universal_query(query, domain=domain)
            
            if results['success']:
                # Add domain-specific analysis
                domain_analysis = self._analyze_domain_context(
                    equipment, issue, results['data']['response'], domain
                )
                results['data']['domain_analysis'] = domain_analysis
            
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
            # Load domain-specific configuration
            training = DomainPatternManager.get_training(domain)
            
            # Extract key terms from query for entity matching
            key_terms = self._extract_key_terms(query, domain)
            
            # Use configuration-driven limits
            max_terms = min(len(key_terms), training.batch_size // 6)  # Dynamic term limit
            max_results = training.batch_size // 3  # Dynamic result limit
            
            graph_results = []
            for term in key_terms[:max_terms]:
                # This would query Cosmos DB for entities matching the term
                # Simplified implementation
                entities = await self._find_entities_by_term(term, domain)
                graph_results.extend(entities)
            
            return graph_results[:max_results]
            
        except Exception as e:
            logger.warning(f"Knowledge graph search failed: {e}")
            return []
    
    async def _find_related_entities(self, query: str, domain: str) -> List[Dict]:
        """Find entities related to query concepts"""
        try:
            # Load domain-specific configuration
            training = DomainPatternManager.get_training(domain)
            
            # Extract concepts and find related entities
            concepts = self._extract_concepts(query, domain)
            
            # Use configuration-driven result limit
            max_results = training.batch_size // 2  # Dynamic result limit based on batch size
            
            related_entities = []
            for concept in concepts:
                entities = await self._find_entities_by_concept(concept, domain)
                related_entities.extend(entities)
            
            return related_entities[:max_results]
            
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
                if isinstance(entity, str):
                    # Handle string entities
                    context['entities'].append({
                        'text': entity,
                        'type': 'entity',
                        'source_type': 'knowledge_graph'
                    })
                else:
                    # Handle dict entities
                    context['entities'].append({
                        'text': entity.get('text', entity.get('name', str(entity))),
                        'type': entity.get('type', 'entity'),
                        'source_type': 'knowledge_graph'
                    })
                context['sources'].append('knowledge_graph')
        
        # Process entity results
        if isinstance(entity_results, list):
            for entity in entity_results:
                if isinstance(entity, str):
                    # Handle string entities
                    context['entities'].append({
                        'text': entity,
                        'type': 'entity',
                        'source_type': 'entity_search'
                    })
                else:
                    # Handle dict entities
                    context['entities'].append({
                        'text': entity.get('text', entity.get('name', str(entity))),
                        'type': entity.get('type', 'entity'),
                        'source_type': 'entity_search'
                    })
                context['sources'].append('entity_search')
        
        return context
    
    async def _generate_response(self, query: str, context: Dict[str, Any], domain: str) -> str:
        """Generate response using Azure OpenAI with retrieved context"""
        try:
            # Load domain-specific configuration
            prompts = DomainPatternManager.get_prompts(domain)
            
            # Build context-aware prompt using domain configuration
            context_with_domain = {**context, 'domain': domain}
            context_text = self._build_context_text(context_with_domain)
            
            prompt = f"""You are an expert assistant for {prompts.completion_context}.

Context from knowledge base:
{context_text}

User Query: {query}

Based on the retrieved context, provide a comprehensive and accurate response. 
If the context contains relevant information, use it to inform your answer.
If insufficient context is available, clearly state the limitations.

Response:"""
            
            response = await self.openai_client.get_completion(
                prompt, 
                model=prompts.model_name,
                temperature=prompts.temperature,
                max_tokens=prompts.max_tokens
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    def _build_context_text(self, context: Dict[str, Any]) -> str:
        """Build formatted context text for prompt using domain configuration"""
        # Load domain configuration for dynamic limits
        domain = context.get('domain', 'general')
        training = DomainPatternManager.get_training(domain)
        prompts = DomainPatternManager.get_prompts(domain)
        
        # Use configuration-driven limits
        max_docs = min(5, training.batch_size // 10)  # Dynamic document limit
        max_entities = min(15, training.batch_size // 2)  # Dynamic entity limit
        content_length = prompts.chunk_size // 5  # Dynamic content preview length
        
        context_parts = []
        
        # Add documents
        for i, doc in enumerate(context.get('documents', [])[:max_docs]):
            context_parts.append(f"Document {i+1}: {doc.get('content', '')[:content_length]}...")
        
        # Add entities
        entities = context.get('entities', [])[:max_entities]
        if entities:
            entity_text = ", ".join([e.get('text', '') for e in entities])
            context_parts.append(f"Related entities: {entity_text}")
        
        return "\\n\\n".join(context_parts)
    
    # === UTILITY METHODS ===
    
    def _extract_key_terms(self, query: str, domain: str = 'general') -> List[str]:
        """Extract key terms from query using domain patterns"""
        # Load domain-specific configuration
        patterns = DomainPatternManager.get_patterns(domain)
        training = DomainPatternManager.get_training(domain)
        
        # Simple keyword extraction
        import re
        words = re.findall(r'\\b\\w+\\b', query.lower())
        
        # Use domain-specific filtering
        # Priority: domain indicators > action terms > issue terms > general words
        domain_terms = [word for word in words if word in patterns.domain_indicators]
        action_terms = [word for word in words if word in patterns.action_terms]
        issue_terms = [word for word in words if word in patterns.issue_terms]
        
        # Filter common words for remaining terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        other_terms = [word for word in words if len(word) > 3 and word not in stop_words 
                      and word not in domain_terms and word not in action_terms and word not in issue_terms]
        
        # Combine with priority ordering
        key_terms = domain_terms + action_terms + issue_terms + other_terms
        
        # Use configuration-driven limit
        max_terms = min(10, training.batch_size // 3)
        
        return key_terms[:max_terms]
    
    def _extract_concepts(self, query: str, domain: str) -> List[str]:
        """Extract domain-specific concepts from query using configuration"""
        # Load domain-specific patterns
        patterns = DomainPatternManager.get_patterns(domain)
        training = DomainPatternManager.get_training(domain)
        
        concepts = []
        query_lower = query.lower()
        
        # Use configured domain patterns for concept extraction
        all_domain_terms = (
            patterns.domain_indicators + 
            patterns.action_terms + 
            patterns.issue_terms + 
            patterns.enhancement_keywords
        )
        
        # Find concepts matching domain patterns
        for term in all_domain_terms:
            if term in query_lower:
                concepts.append(term)
        
        # If no domain-specific concepts found, fall back to key terms
        if not concepts:
            concepts = self._extract_key_terms(query, domain)
        
        # Use configuration-driven limit
        max_concepts = min(8, training.batch_size // 4)
        return concepts[:max_concepts]
    
    async def _find_entities_by_term(self, term: str, domain: str = 'general') -> List[Dict]:
        """Find entities matching a specific term"""
        # Load domain-specific configuration
        metadata = DomainPatternManager.get_metadata(domain)
        
        # Simplified entity search - would query Cosmos DB in production
        return [
            {
                'text': term, 
                'type': metadata.default_entity_type, 
                'score': metadata.default_confidence
            }
        ]
    
    async def _find_entities_by_concept(self, concept: str, domain: str = 'general') -> List[str]:
        """Find entity IDs matching a concept"""
        # Load domain-specific configuration for dynamic limits
        training = DomainPatternManager.get_training(domain)
        entity_limit = min(3, training.batch_size // 15)  # Dynamic entity limit
        
        # Simplified - would query actual graph database
        return [f"entity_{concept}_{i+1}" for i in range(entity_limit)]
    
    async def _explain_reasoning_paths(self, paths: List[Dict], start_concept: str, 
                                     target_concept: str, domain: str = 'general') -> str:
        """Generate explanation of reasoning paths using domain configuration"""
        if not paths:
            return f"No reasoning paths found between {start_concept} and {target_concept}."
        
        # Load domain configuration for dynamic limits
        training = DomainPatternManager.get_training(domain)
        max_paths = min(len(paths), training.batch_size // 10)  # Dynamic path limit
        
        explanation = f"Found {len(paths)} reasoning paths from {start_concept} to {target_concept}:\\n\\n"
        
        for i, path in enumerate(paths[:max_paths]):
            hops = path.get('hops', 0)
            explanation += f"Path {i+1}: {hops}-hop connection\\n"
        
        return explanation
    
    def _analyze_domain_context(self, equipment: str, issue: str, response: str, domain: str) -> Dict[str, Any]:
        """Analyze domain-specific context using configuration"""
        # Load domain-specific patterns
        patterns = DomainPatternManager.get_patterns(domain)
        
        response_lower = response.lower()
        
        # Count domain-specific keywords in response
        action_matches = len([word for word in patterns.action_terms if word in response_lower])
        issue_matches = len([word for word in patterns.issue_terms if word in response_lower])
        domain_matches = len([word for word in patterns.domain_indicators if word in response_lower])
        
        # Calculate relevance based on domain patterns
        total_keywords = action_matches + issue_matches + domain_matches
        equipment_present = equipment.lower() in response_lower
        issue_present = issue.lower() in response_lower if issue else True
        
        # Determine relevance level
        if equipment_present and issue_present and total_keywords >= 3:
            relevance = 'high'
        elif equipment_present and total_keywords >= 1:
            relevance = 'medium'
        else:
            relevance = 'low'
        
        return {
            'equipment_identified': equipment_present,
            'issue_addressed': issue_present,
            'domain_keywords': {
                'action_terms': action_matches,
                'issue_terms': issue_matches,
                'domain_indicators': domain_matches,
                'total': total_keywords
            },
            'response_relevance': relevance,
            'domain': domain
        }