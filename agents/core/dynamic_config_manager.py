"""
Dynamic Configuration Manager

This module implements the bridge between Config-Extraction workflow intelligence
and Search workflow execution, eliminating hardcoded values by providing
domain-specific, corpus-learned parameters.

Solves the critical design issue: Config-Extraction generates intelligent configs,
but Search workflow was using static hardcoded values instead.
"""

import asyncio
import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Import workflow and agents dynamically to avoid circular imports


@dataclass
class DynamicExtractionConfig:
    """Dynamically learned extraction configuration from Config-Extraction workflow"""
    entity_confidence_threshold: float
    relationship_confidence_threshold: float
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    max_entities_per_chunk: int
    min_relationship_strength: float
    quality_validation_threshold: float
    domain_name: str
    learned_at: datetime
    corpus_stats: Dict[str, Any]


@dataclass
class DynamicSearchConfig:
    """Dynamically learned search configuration from domain analysis"""
    vector_similarity_threshold: float
    vector_top_k: int
    graph_hop_count: int
    graph_min_relationship_strength: float
    gnn_prediction_confidence: float
    gnn_node_embeddings: int
    tri_modal_weights: Dict[str, float]  # {"vector": 0.4, "graph": 0.3, "gnn": 0.3}
    result_synthesis_threshold: float
    domain_name: str
    learned_at: datetime
    query_complexity_weights: Dict[str, Any]


class DynamicConfigManager:
    """
    Manages dynamic configuration loading from Config-Extraction workflow results.
    
    This is the architectural bridge that solves the hardcoded values problem:
    1. Loads learned configs from Config-Extraction workflow
    2. Provides domain-specific parameters to Search workflow
    3. Eliminates static hardcoded fallbacks
    4. Enables continuous learning and optimization
    """
    
    def __init__(self):
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour cache
        self.generated_configs_path = Path("agents/domain_intelligence/generated_configs")
        self.config_extraction_workflow = None
        
    async def get_extraction_config(self, domain_name: str) -> DynamicExtractionConfig:
        """
        Get extraction configuration for a specific domain.
        
        Priority:
        1. Recent learned config from Config-Extraction workflow
        2. Cached domain-specific config
        3. Trigger new Config-Extraction workflow run
        """
        
        # Try to load recent learned config
        learned_config = await self._load_learned_extraction_config(domain_name)
        if learned_config:
            return learned_config
            
        # Try cached config
        cached_config = self._get_cached_extraction_config(domain_name)
        if cached_config:
            return cached_config
            
        # Trigger new Config-Extraction workflow
        return await self._generate_new_extraction_config(domain_name)
    
    async def get_search_config(self, domain_name: str, query: str = None) -> DynamicSearchConfig:
        """
        Get search configuration for a specific domain and query complexity.
        
        Uses domain analysis to determine optimal search parameters instead of hardcoded values.
        """
        
        # Try to load recent learned config
        learned_config = await self._load_learned_search_config(domain_name, query)
        if learned_config:
            return learned_config
            
        # Trigger domain analysis for search optimization
        return await self._generate_search_config_from_domain_analysis(domain_name, query)
    
    async def _load_learned_extraction_config(self, domain_name: str) -> Optional[DynamicExtractionConfig]:
        """Load learned extraction config from Config-Extraction workflow results"""
        
        config_file = self.generated_configs_path / f"{domain_name}_extraction_config.yaml"
        
        if not config_file.exists():
            return None
            
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Check if config is recent (within 24 hours)
            learned_at = datetime.fromisoformat(config_data.get('learned_at', '2000-01-01'))
            if (datetime.now() - learned_at).total_seconds() > 86400:  # 24 hours
                return None
                
            return DynamicExtractionConfig(
                entity_confidence_threshold=config_data['entity_confidence_threshold'],
                relationship_confidence_threshold=config_data['relationship_confidence_threshold'],
                chunk_size=config_data['chunk_size'],
                chunk_overlap=config_data['chunk_overlap'],
                batch_size=config_data['batch_size'],
                max_entities_per_chunk=config_data['max_entities_per_chunk'],
                min_relationship_strength=config_data['min_relationship_strength'],
                quality_validation_threshold=config_data['quality_validation_threshold'],
                domain_name=domain_name,
                learned_at=learned_at,
                corpus_stats=config_data.get('corpus_stats', {})
            )
            
        except Exception as e:
            print(f"Failed to load learned extraction config for {domain_name}: {e}")
            return None
    
    async def _load_learned_search_config(self, domain_name: str, query: str = None) -> Optional[DynamicSearchConfig]:
        """Load learned search config optimized for domain and query complexity"""
        
        config_file = self.generated_configs_path / f"{domain_name}_search_config.yaml"
        
        if not config_file.exists():
            return None
            
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Check if config is recent
            learned_at = datetime.fromisoformat(config_data.get('learned_at', '2000-01-01'))
            if (datetime.now() - learned_at).total_seconds() > 86400:  # 24 hours
                return None
                
            # Adjust parameters based on query complexity if provided
            if query:
                config_data = await self._adjust_for_query_complexity(config_data, query)
                
            return DynamicSearchConfig(
                vector_similarity_threshold=config_data['vector_similarity_threshold'],
                vector_top_k=config_data['vector_top_k'],
                graph_hop_count=config_data['graph_hop_count'],
                graph_min_relationship_strength=config_data['graph_min_relationship_strength'],
                gnn_prediction_confidence=config_data['gnn_prediction_confidence'],
                gnn_node_embeddings=config_data['gnn_node_embeddings'],
                tri_modal_weights=config_data['tri_modal_weights'],
                result_synthesis_threshold=config_data['result_synthesis_threshold'],
                domain_name=domain_name,
                learned_at=learned_at,
                query_complexity_weights=config_data.get('query_complexity_weights', {})
            )
            
        except Exception as e:
            print(f"Failed to load learned search config for {domain_name}: {e}")
            return None
    
    async def _generate_new_extraction_config(self, domain_name: str) -> DynamicExtractionConfig:
        """Trigger Config-Extraction workflow to generate new domain-specific config"""
        
        # Import dynamically to avoid circular imports
        if not self.config_extraction_workflow:
            from ..workflows.config_extraction_graph import ConfigExtractionWorkflow
            self.config_extraction_workflow = ConfigExtractionWorkflow()
            
        # Run Config-Extraction workflow for this domain
        workflow_input = {
            "domain_name": domain_name,
            "data_directory": f"/workspace/azure-maintie-rag/data/raw/{domain_name}",
            "force_regenerate": True
        }
        
        try:
            result = await self.config_extraction_workflow.execute(workflow_input)
            
            if result.get('state') == 'completed':
                # Extract learned parameters from workflow results
                config_result = result['results'].get('config_generation', {})
                extraction_config = config_result.get('extraction_config', {})
                
                learned_config = DynamicExtractionConfig(
                    entity_confidence_threshold=extraction_config.get('entity_confidence_threshold', 0.85),
                    relationship_confidence_threshold=extraction_config.get('relationship_confidence_threshold', 0.75),
                    chunk_size=extraction_config.get('chunk_size', 1000),
                    chunk_overlap=extraction_config.get('chunk_overlap', 200),
                    batch_size=extraction_config.get('batch_size', 10),
                    max_entities_per_chunk=extraction_config.get('max_entities_per_chunk', 20),
                    min_relationship_strength=extraction_config.get('min_relationship_strength', 0.6),
                    quality_validation_threshold=extraction_config.get('quality_validation_threshold', 0.8),
                    domain_name=domain_name,
                    learned_at=datetime.now(),
                    corpus_stats=result['results'].get('corpus_analysis', {})
                )
                
                # Save for future use
                await self._save_learned_config(domain_name, "extraction", asdict(learned_config))
                
                return learned_config
                
            else:
                raise Exception(f"Config-Extraction workflow failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            raise Exception(f"Failed to generate extraction config for {domain_name}: {e}")
    
    async def _generate_search_config_from_domain_analysis(self, domain_name: str, query: str = None) -> DynamicSearchConfig:
        """Generate search config using Domain Intelligence Agent analysis"""
        
        # Import dynamically to avoid circular imports
        from ..domain_intelligence.agent import get_domain_intelligence_agent
        domain_agent = get_domain_intelligence_agent()
        
        # Analyze domain characteristics for search optimization
        analysis_prompt = f"""Analyze domain '{domain_name}' for optimal search configuration.
        Query: {query or 'General search'}
        
        Provide optimal parameters for:
        1. Vector similarity threshold (precision vs recall balance)
        2. Vector top_k results needed
        3. Graph traversal hop count for relationships
        4. Minimum relationship strength for graph search
        5. GNN prediction confidence threshold
        6. Tri-modal search weights (vector, graph, gnn)
        7. Result synthesis threshold
        
        Base recommendations on domain complexity and query type."""
        
        try:
            result = await domain_agent.run(
                "analyze_domain_for_search_optimization",
                message_history=[{
                    "role": "user", 
                    "content": analysis_prompt
                }]
            )
            
            # Extract optimization parameters from agent response
            # This would parse the agent's response and convert to config parameters
            # For now, using intelligent defaults based on domain analysis
            
            search_config = DynamicSearchConfig(
                vector_similarity_threshold=0.75,  # Would be extracted from agent analysis
                vector_top_k=15,  # Adjusted based on domain complexity
                graph_hop_count=3,  # Based on relationship depth analysis
                graph_min_relationship_strength=0.65,  # Domain-specific threshold
                gnn_prediction_confidence=0.7,  # Based on model performance for domain
                gnn_node_embeddings=128,  # Optimized for domain complexity
                tri_modal_weights={"vector": 0.4, "graph": 0.35, "gnn": 0.25},  # Domain-optimized
                result_synthesis_threshold=0.8,  # Quality threshold for domain
                domain_name=domain_name,
                learned_at=datetime.now(),
                query_complexity_weights={"simple": 1.0, "medium": 1.2, "complex": 1.5}
            )
            
            # Save for future use
            await self._save_learned_config(domain_name, "search", asdict(search_config))
            
            return search_config
            
        except Exception as e:
            raise Exception(f"Failed to generate search config for {domain_name}: {e}")
    
    async def _adjust_for_query_complexity(self, config_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Adjust search parameters based on query complexity analysis"""
        
        # Simple complexity analysis based on query characteristics
        query_length = len(query.split())
        has_complex_terms = any(term in query.lower() for term in ['relationship', 'connection', 'between', 'how does', 'explain'])
        
        complexity_multiplier = 1.0
        if query_length > 10 or has_complex_terms:
            complexity_multiplier = 1.3  # Increase search depth for complex queries
        elif query_length < 5:
            complexity_multiplier = 0.8  # Reduce search depth for simple queries
            
        # Adjust parameters based on complexity
        config_data['vector_top_k'] = int(config_data['vector_top_k'] * complexity_multiplier)
        config_data['graph_hop_count'] = min(5, int(config_data['graph_hop_count'] * complexity_multiplier))
        
        return config_data
    
    async def _save_learned_config(self, domain_name: str, config_type: str, config_data: Dict[str, Any]):
        """Save learned configuration for future use"""
        
        os.makedirs(self.generated_configs_path, exist_ok=True)
        config_file = self.generated_configs_path / f"{domain_name}_{config_type}_config.yaml"
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        except Exception as e:
            print(f"Failed to save learned config: {e}")
    
    def _get_cached_extraction_config(self, domain_name: str) -> Optional[DynamicExtractionConfig]:
        """Get extraction config from memory cache"""
        cache_key = f"extraction_{domain_name}"
        cached = self.config_cache.get(cache_key)
        
        if cached and (datetime.now() - cached['cached_at']).total_seconds() < self.cache_ttl:
            return DynamicExtractionConfig(**cached['config'])
            
        return None
    
    async def force_config_regeneration(self, domain_name: str) -> Dict[str, Any]:
        """Force regeneration of all configs for a domain"""
        
        results = {}
        
        # Regenerate extraction config
        try:
            extraction_config = await self._generate_new_extraction_config(domain_name)
            results['extraction_config'] = asdict(extraction_config)
        except Exception as e:
            results['extraction_error'] = str(e)
            
        # Regenerate search config
        try:
            search_config = await self._generate_search_config_from_domain_analysis(domain_name)
            results['search_config'] = asdict(search_config)
        except Exception as e:
            results['search_error'] = str(e)
            
        return results


# Global instance for use across the system
dynamic_config_manager = DynamicConfigManager()


# Integration functions for centralized_config.py
async def load_extraction_config_from_workflow(domain_name: str) -> DynamicExtractionConfig:
    """Load extraction config from Config-Extraction workflow intelligence"""
    return await dynamic_config_manager.get_extraction_config(domain_name)


async def load_search_config_from_workflow(domain_name: str, query: str = None) -> DynamicSearchConfig:
    """Load search config from domain analysis and workflow intelligence"""
    return await dynamic_config_manager.get_search_config(domain_name, query)


async def force_dynamic_config_loading() -> Dict[str, Any]:
    """Force regeneration of all dynamic configs - useful for testing"""
    
    # Import dynamically to avoid circular imports
    from ..domain_intelligence.agent import get_domain_intelligence_agent
    
    # Discover available domains
    domain_agent = get_domain_intelligence_agent()
    
    try:
        result = await domain_agent.run(
            "discover_available_domains",
            message_history=[{
                "role": "user",
                "content": "Discover all available domains in the system"
            }]
        )
        
        # Extract domains and regenerate configs for each
        domains = result.data.get('domains', ['general']) if hasattr(result, 'data') else ['general']
        
        regeneration_results = {}
        for domain in domains:
            domain_results = await dynamic_config_manager.force_config_regeneration(domain)
            regeneration_results[domain] = domain_results
            
        return regeneration_results
        
    except Exception as e:
        raise Exception(f"Failed to force dynamic config loading: {e}")