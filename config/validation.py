"""
Configuration validation utilities for MaintIE Enhanced RAG
Ensures all configurable parameters are valid and within acceptable ranges
"""

import logging
from typing import Dict, Any, List
from pathlib import Path

from config.advanced_settings import advanced_settings
from config.settings import settings

logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Validate configuration parameters and provide recommendations"""

    def __init__(self):
        self.validation_results = {}
        self.warnings = []
        self.errors = []

    def validate_all_settings(self) -> Dict[str, Any]:
        """Validate all configuration settings"""
        logger.info("Validating configuration settings...")

        # Validate basic settings
        self._validate_basic_settings()

        # Validate data processing settings
        self._validate_data_settings()

        # Validate query analysis settings
        self._validate_query_settings()

        # Validate retrieval settings
        self._validate_retrieval_settings()

        # Validate generation settings
        self._validate_generation_settings()

        # Validate API settings
        self._validate_api_settings()

        return {
            "valid": len(self.errors) == 0,
            "warnings": self.warnings,
            "errors": self.errors,
            "validation_results": self.validation_results
        }

    def _validate_basic_settings(self) -> None:
        """Validate basic application settings"""
        # Check OpenAI API key
        if not settings.openai_api_key or settings.openai_api_key in ["your_openai_api_key_here", "az-1234abcd5678efgh9012ijkl3456mnop"]:
            self.errors.append("OPENAI_API_KEY is not set or is using default/example value")

        # Check environment
        if settings.environment not in ["development", "staging", "production"]:
            self.warnings.append(f"Environment '{settings.environment}' is not standard")

        # Check data directories
        data_dir = Path(settings.data_dir)
        if not data_dir.exists():
            self.warnings.append(f"Data directory {data_dir} does not exist")

    def _validate_data_settings(self) -> None:
        """Validate data processing settings"""
        # Check confidence bases
        if not (0.0 <= advanced_settings.gold_confidence_base <= 1.0):
            self.errors.append("GOLD_CONFIDENCE_BASE must be between 0.0 and 1.0")

        if not (0.0 <= advanced_settings.silver_confidence_base <= 1.0):
            self.errors.append("SILVER_CONFIDENCE_BASE must be between 0.0 and 1.0")

        # Check file names
        if not advanced_settings.gold_data_filename.endswith('.json'):
            self.warnings.append("GOLD_DATA_FILENAME should end with .json")

        if not advanced_settings.silver_data_filename.endswith('.json'):
            self.warnings.append("SILVER_DATA_FILENAME should end with .json")

    def _validate_query_settings(self) -> None:
        """Validate query analysis settings"""
        # Check limits
        if advanced_settings.max_related_entities <= 0:
            self.errors.append("MAX_RELATED_ENTITIES must be positive")

        if advanced_settings.max_neighbors <= 0:
            self.errors.append("MAX_NEIGHBORS must be positive")

        if advanced_settings.concept_expansion_limit <= 0:
            self.errors.append("CONCEPT_EXPANSION_LIMIT must be positive")

        # Check keyword lists
        if not advanced_settings.troubleshooting_keywords:
            self.warnings.append("TROUBLESHOOTING_KEYWORDS is empty")

        if not advanced_settings.procedural_keywords:
            self.warnings.append("PROCEDURAL_KEYWORDS is empty")

    def _validate_retrieval_settings(self) -> None:
        """Validate retrieval settings"""
        # Check batch size
        if advanced_settings.embedding_batch_size <= 0:
            self.errors.append("EMBEDDING_BATCH_SIZE must be positive")

        if advanced_settings.embedding_batch_size > 128:
            self.warnings.append("EMBEDDING_BATCH_SIZE is large, may cause memory issues")

        # Check similarity threshold
        if not (0.0 <= advanced_settings.similarity_threshold <= 1.0):
            self.errors.append("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")

        # Check FAISS index type
        valid_index_types = ["IndexFlatIP", "IndexFlatL2", "IndexIVFFlat"]
        if advanced_settings.faiss_index_type not in valid_index_types:
            self.warnings.append(f"FAISS_INDEX_TYPE '{advanced_settings.faiss_index_type}' is not standard")

    def _validate_generation_settings(self) -> None:
        """Validate generation settings"""
        # Check LLM parameters
        if not (0.0 <= advanced_settings.llm_top_p <= 1.0):
            self.errors.append("LLM_TOP_P must be between 0.0 and 1.0")

        if not (-2.0 <= advanced_settings.llm_frequency_penalty <= 2.0):
            self.errors.append("LLM_FREQUENCY_PENALTY must be between -2.0 and 2.0")

        if not (-2.0 <= advanced_settings.llm_presence_penalty <= 2.0):
            self.errors.append("LLM_PRESENCE_PENALTY must be between -2.0 and 2.0")

    def _validate_api_settings(self) -> None:
        """Validate API settings"""
        # Check query length limits
        if advanced_settings.query_min_length <= 0:
            self.errors.append("QUERY_MIN_LENGTH must be positive")

        if advanced_settings.query_max_length <= advanced_settings.query_min_length:
            self.errors.append("QUERY_MAX_LENGTH must be greater than QUERY_MIN_LENGTH")

        if advanced_settings.query_max_length > 2000:
            self.warnings.append("QUERY_MAX_LENGTH is very large")

        # Check results limit
        if advanced_settings.max_results_limit <= 0:
            self.errors.append("MAX_RESULTS_LIMIT must be positive")

        if advanced_settings.max_results_limit > 200:
            self.warnings.append("MAX_RESULTS_LIMIT is very large, may impact performance")

    def get_recommendations(self) -> List[str]:
        """Get configuration recommendations"""
        recommendations = []

        # Performance recommendations
        if advanced_settings.embedding_batch_size < 16:
            recommendations.append("Consider increasing EMBEDDING_BATCH_SIZE for better performance")

        if advanced_settings.max_related_entities < 10:
            recommendations.append("Consider increasing MAX_RELATED_ENTITIES for better query expansion")

        # Quality recommendations
        if advanced_settings.gold_confidence_base < 0.8:
            recommendations.append("Consider increasing GOLD_CONFIDENCE_BASE for higher quality data")

        if advanced_settings.similarity_threshold < 0.5:
            recommendations.append("Consider increasing SIMILARITY_THRESHOLD for more relevant results")

        # Safety recommendations
        if advanced_settings.query_max_length > 1000:
            recommendations.append("Consider reducing QUERY_MAX_LENGTH to prevent abuse")

        if advanced_settings.max_results_limit > 100:
            recommendations.append("Consider reducing MAX_RESULTS_LIMIT to prevent resource exhaustion")

        return recommendations

    def validate_for_environment(self, environment: str) -> Dict[str, Any]:
        """Validate settings for specific environment"""
        env_validation = {
            "development": self._validate_development_settings,
            "staging": self._validate_staging_settings,
            "production": self._validate_production_settings
        }

        if environment in env_validation:
            env_validation[environment]()

        return self.validate_all_settings()

    def _validate_development_settings(self) -> None:
        """Validate settings for development environment"""
        if not settings.debug:
            self.warnings.append("DEBUG should be True in development environment")

    def _validate_staging_settings(self) -> None:
        """Validate settings for staging environment"""
        if settings.debug:
            self.warnings.append("DEBUG should be False in staging environment")

    def _validate_production_settings(self) -> None:
        """Validate settings for production environment"""
        if settings.debug:
            self.errors.append("DEBUG must be False in production environment")

        if settings.environment != "production":
            self.errors.append("ENVIRONMENT must be 'production' in production environment")

        if advanced_settings.embedding_batch_size > 64:
            self.warnings.append("Consider reducing EMBEDDING_BATCH_SIZE for production stability")


def validate_configuration(environment: str = None) -> Dict[str, Any]:
    """Convenience function to validate configuration"""
    validator = ConfigurationValidator()

    if environment:
        return validator.validate_for_environment(environment)
    else:
        return validator.validate_all_settings()


if __name__ == "__main__":
    # Test configuration validation
    results = validate_configuration()
    print("Configuration Validation Results:")
    print(f"Valid: {results['valid']}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Errors: {len(results['errors'])}")

    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")

    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")

    validator = ConfigurationValidator()
    recommendations = validator.get_recommendations()
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
