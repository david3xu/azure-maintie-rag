"""
Configuration Unit Tests - CODING_STANDARDS Compliant
Tests configuration validation and centralized config behavior.
"""

import pytest
import os
from unittest.mock import patch
from config.centralized_config import (
    get_system_config,
    get_extraction_config,
    get_search_config,
    get_model_config,
    get_cache_config,
    get_processing_config,
    get_workflow_config,
    get_security_config,
    SystemConfiguration,
    ExtractionConfiguration,
    SearchConfiguration,
    ModelConfiguration
)


class TestConfigurationValidation:
    """Test configuration validation and environment handling"""
    
    def test_system_config_defaults(self):
        """Test system configuration with default values"""
        config = get_system_config()
        
        # Validate required fields exist
        assert hasattr(config, 'max_workers')
        assert hasattr(config, 'max_concurrent_requests')
        assert hasattr(config, 'openai_timeout')
        assert hasattr(config, 'max_query_length')
        
        # Validate default ranges
        assert 1 <= config.max_workers <= 16
        assert 1 <= config.max_concurrent_requests <= 100
        assert 30 <= config.openai_timeout <= 300
        assert 100 <= config.max_query_length <= 10000
        
        print(f"✅ System config validation passed: {config.max_workers} workers, {config.openai_timeout}s timeout")
    
    def test_extraction_config_defaults(self):
        """Test extraction configuration validation"""
        config = get_extraction_config()
        
        # Validate confidence thresholds
        assert 0.0 <= config.entity_confidence_threshold <= 1.0
        assert 0.0 <= config.relationship_confidence_threshold <= 1.0
        assert config.relationship_confidence_threshold <= config.entity_confidence_threshold + 0.2  # Reasonable relationship
        
        # Validate chunking parameters
        assert 100 <= config.chunk_size <= 5000
        assert 0.0 <= config.chunk_overlap_ratio <= 0.5
        
        # Validate limits
        assert 1 <= config.max_entities_per_chunk <= 100
        assert 1 <= config.max_relationships_per_entity <= 50
        
        print(f"✅ Extraction config validation passed: {config.entity_confidence_threshold} entity confidence")
    
    def test_search_config_validation(self):
        """Test search configuration validation"""
        config = get_search_config()
        
        # Validate search parameters
        assert 0.0 <= config.vector_similarity_threshold <= 1.0
        assert 1 <= config.vector_top_k <= 100
        assert 1 <= config.graph_max_depth <= 10
        assert 1 <= config.graph_max_entities <= 100
        
        # Validate GNN parameters
        assert 0.0 <= config.gnn_pattern_threshold <= 1.0
        assert 1 <= config.gnn_max_predictions <= 100
        
        # Validate result limits
        assert 1 <= config.max_results_per_modality <= 100
        assert 1 <= config.max_final_results <= 1000
        
        print(f"✅ Search config validation passed: {config.vector_top_k} vector results, {config.graph_max_depth} graph depth")
    
    def test_model_config_validation(self):
        """Test model configuration validation"""
        config = get_model_config()
        
        # Validate deployment names exist
        assert len(config.gpt4o_deployment_name) > 0
        assert len(config.gpt4o_mini_deployment_name) > 0
        assert len(config.text_embedding_deployment_name) > 0
        
        # Validate API version format
        assert config.openai_api_version.count('-') >= 2  # YYYY-MM-DD format
        
        # Validate parameters
        assert 0.0 <= config.default_temperature <= 2.0
        assert 100 <= config.default_max_tokens <= 10000
        
        print(f"✅ Model config validation passed: {config.gpt4o_deployment_name}, API {config.openai_api_version}")
    
    @patch.dict(os.environ, {"MAX_WORKERS": "8", "OPENAI_TIMEOUT": "90"})
    def test_environment_override(self):
        """Test configuration overrides from environment variables"""
        # Clear cached config to test environment override
        import config.centralized_config
        config.centralized_config._system_config = None
        
        config = get_system_config()
        
        # Validate environment overrides work
        assert config.max_workers == 8  # From environment
        assert config.openai_timeout == 90  # From environment
        
        print(f"✅ Environment override validation passed: workers={config.max_workers}, timeout={config.openai_timeout}")
    
    def test_configuration_immutability(self):
        """Test that configurations maintain consistency"""
        config1 = get_system_config()
        config2 = get_system_config()
        
        # Should return same instance (singleton pattern)
        assert config1 is config2
        
        # Validate key fields are identical
        assert config1.max_workers == config2.max_workers
        assert config1.openai_timeout == config2.openai_timeout
        
        print("✅ Configuration immutability validation passed")
    
    def test_legacy_compatibility_functions(self):
        """Test legacy compatibility functions still work"""
        from config.centralized_config import (
            get_ml_hyperparameters_config,
            get_azure_services_config,
            get_entity_processing_config,
            get_relationship_processing_config
        )
        
        # Test legacy functions return appropriate configs
        ml_config = get_ml_hyperparameters_config()
        azure_config = get_azure_services_config()
        entity_config = get_entity_processing_config()
        relationship_config = get_relationship_processing_config()
        
        # Validate they return actual configuration objects
        assert hasattr(ml_config, 'max_workers')
        assert hasattr(azure_config, 'max_concurrent_requests')
        assert hasattr(entity_config, 'entity_confidence_threshold')
        assert hasattr(relationship_config, 'relationship_confidence_threshold')
        
        print("✅ Legacy compatibility validation passed")


class TestConfigurationEdgeCases:
    """Test configuration edge cases and error handling"""
    
    @patch.dict(os.environ, {"MAX_WORKERS": "invalid"})
    def test_invalid_environment_values(self):
        """Test handling of invalid environment values"""
        import config.centralized_config
        config.centralized_config._system_config = None
        
        # Should handle invalid values gracefully and use defaults
        try:
            config = get_system_config()
            # If we get here, it used default value
            assert isinstance(config.max_workers, int)
            assert config.max_workers > 0
            print("✅ Invalid environment value handled gracefully")
        except ValueError:
            # Alternative: might raise ValueError, which is also acceptable
            print("✅ Invalid environment value rejected appropriately")
    
    def test_configuration_boundaries(self):
        """Test configuration boundary conditions"""
        config = get_extraction_config()
        
        # Test that confidence thresholds are within valid ranges
        assert 0.0 <= config.entity_confidence_threshold <= 1.0
        assert 0.0 <= config.relationship_confidence_threshold <= 1.0
        
        # Test that minimum values are reasonable for production
        assert config.min_entity_length >= 1
        assert config.max_entity_length > config.min_entity_length
        assert config.min_relationship_confidence >= 0.0
        assert config.max_relationship_confidence <= 1.0
        
        print("✅ Configuration boundary validation passed")
    
    def test_config_interdependencies(self):
        """Test that related configuration values are consistent"""
        extraction_config = get_extraction_config()
        search_config = get_search_config()
        
        # Validate that search limits don't exceed reasonable bounds
        total_potential_results = (
            search_config.max_results_per_modality * 3  # 3 modalities: vector, graph, GNN
        )
        assert total_potential_results >= search_config.max_final_results or search_config.max_final_results <= 100
        
        # Validate that extraction parameters are compatible
        assert extraction_config.chunk_size > 0
        assert extraction_config.chunk_overlap_ratio < 1.0
        
        print("✅ Configuration interdependency validation passed")