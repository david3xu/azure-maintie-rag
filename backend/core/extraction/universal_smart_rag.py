"""
Universal Smart RAG System
Main class that orchestrates universal RAG functionality for any domain
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from universal.optimized_llm_extractor import OptimizedLLMExtractor
from universal.domain_config_validator import DomainConfigValidator

logger = logging.getLogger(__name__)


class UniversalSmartRAG:
    """Universal Smart RAG system that works with any domain"""

    def __init__(self, domain_name: str, base_config_path: Optional[Path] = None):
        """Initialize Universal Smart RAG for a specific domain"""
        self.domain_name = domain_name
        self.base_config_path = base_config_path or Path("config/domains/enhanced_schema_template.yaml")

        # Initialize components
        self.llm_extractor = OptimizedLLMExtractor(domain_name)
        self.config_validator = DomainConfigValidator()

        # Domain configuration
        self.domain_config: Optional[Dict[str, Any]] = None
        self.schema_config: Optional[Dict[str, Any]] = None

        # Status tracking
        self.is_initialized = False
        self.knowledge_extracted = False

        logger.info(f"UniversalSmartRAG initialized for domain: {domain_name}")

    def create_domain_from_texts(self, texts: List[str], quality_filter: str = "high") -> Dict[str, Any]:
        """Create a complete domain configuration from raw texts"""
        logger.info(f"Creating domain '{self.domain_name}' from {len(texts)} texts")

        try:
            # Step 1: Extract knowledge using LLM
            logger.info("🧠 Step 1: Extracting knowledge from texts...")
            extraction_results = self.llm_extractor.extract_entities_and_relations(texts)

            # Step 2: Generate domain schema
            logger.info("📋 Step 2: Generating domain schema...")
            schema_config = self.llm_extractor.generate_domain_schema(extraction_results)

            # Step 3: Validate configuration
            logger.info("✅ Step 3: Validating domain configuration...")
            validation_results = self.config_validator.validate_domain_config(
                self.domain_name, schema_config
            )

            # Step 4: Create final domain configuration
            logger.info("🔧 Step 4: Creating final domain configuration...")
            domain_config = self._create_final_config(schema_config, extraction_results, validation_results)

            # Store configuration
            self.domain_config = domain_config
            self.schema_config = schema_config
            self.knowledge_extracted = True
            self.is_initialized = True

            result = {
                "success": True,
                "domain_name": self.domain_name,
                "extraction_results": extraction_results,
                "schema_config": schema_config,
                "validation_results": validation_results,
                "domain_config": domain_config,
                "texts_processed": len(texts),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"✅ Domain '{self.domain_name}' created successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to create domain '{self.domain_name}': {e}")
            return {
                "success": False,
                "domain_name": self.domain_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def load_domain_config(self, config_path: Path) -> bool:
        """Load existing domain configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.domain_config = json.load(f)

            self.is_initialized = True
            logger.info(f"Domain configuration loaded from {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load domain config from {config_path}: {e}")
            return False

    def save_domain_config(self, output_path: Path) -> bool:
        """Save domain configuration to file"""
        if not self.domain_config:
            logger.error("No domain configuration to save")
            return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.domain_config, f, indent=2, ensure_ascii=False)

            logger.info(f"Domain configuration saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save domain config to {output_path}: {e}")
            return False

    def get_domain_stats(self) -> Dict[str, Any]:
        """Get statistics about the domain"""
        if not self.domain_config:
            return {"error": "No domain configuration loaded"}

        stats = {
            "domain_name": self.domain_name,
            "is_initialized": self.is_initialized,
            "knowledge_extracted": self.knowledge_extracted,
        }

        if self.domain_config:
            extraction_results = self.domain_config.get("extraction_results", {})
            stats.update({
                "entities_discovered": len(extraction_results.get("entities", [])),
                "relations_discovered": len(extraction_results.get("relations", [])),
                "confidence_score": extraction_results.get("confidence_score", 0.0),
                "texts_processed": self.domain_config.get("texts_processed", 0)
            })

        return stats

    def validate_current_config(self) -> Dict[str, Any]:
        """Validate the current domain configuration"""
        if not self.domain_config:
            return {"valid": False, "error": "No domain configuration loaded"}

        return self.config_validator.validate_domain_config(
            self.domain_name, self.schema_config or {}
        )

    def _create_final_config(self, schema_config: Dict[str, Any],
                           extraction_results: Dict[str, Any],
                           validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create the final domain configuration"""

        return {
            "domain_info": {
                "name": self.domain_name,
                "description": f"Auto-generated domain configuration for {self.domain_name}",
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "type": "universal_rag_domain"
            },
            "schema": schema_config,
            "extraction_metadata": {
                "entities_count": len(extraction_results.get("entities", [])),
                "relations_count": len(extraction_results.get("relations", [])),
                "confidence_score": extraction_results.get("confidence_score", 0.0),
                "extraction_method": "llm_powered"
            },
            "validation_metadata": {
                "is_valid": validation_results.get("valid", False),
                "validation_score": validation_results.get("score", 0.0),
                "issues_found": len(validation_results.get("issues", [])),
                "recommendations": validation_results.get("recommendations", [])
            },
            "configuration": {
                "processing_settings": {
                    "quality_filter": "high",
                    "min_confidence": 0.7,
                    "max_entities": 1000,
                    "max_relations": 500
                },
                "performance_settings": {
                    "caching_enabled": True,
                    "batch_processing": True,
                    "parallel_processing": False
                }
            }
        }

    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        return {
            "llm_extractor": {
                "initialized": hasattr(self, 'llm_extractor') and self.llm_extractor is not None,
                "stats": self.llm_extractor.get_extraction_stats() if hasattr(self, 'llm_extractor') else {}
            },
            "config_validator": {
                "initialized": hasattr(self, 'config_validator') and self.config_validator is not None
            },
            "domain": {
                "name": self.domain_name,
                "initialized": self.is_initialized,
                "knowledge_extracted": self.knowledge_extracted,
                "config_loaded": self.domain_config is not None
            }
        }